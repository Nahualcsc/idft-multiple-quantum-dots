#!/usr/bin/env python3
"""Triple-dot NRG for Fig. 3(b), organised for a batch system.

Every discretisation shift z is an independent task that writes its own
checkpoint, and a final assembly step broadens all of them and writes the
reference file.  Restarting a task that already has a checkpoint is free.

    U_1 = 4, U_2 = 6, U_3 = 8,  U_ij = 2 (all pairs),  v = -10,  T = 1e-5
    (units of gamma; each dot has its own reservoir, no interdot hopping)

The defaults below are the production settings (Lambda = 16, Nkeep = 4000,
48000 states in the first 6 shells, Nz = 8).  How the production run was
launched on the cluster, and the resources it needed, is described in
../input/cluster_run.md.  In short, one task per z shift and one per gate value:

    python 01_run_nrg_tqd.py --z-index K        # K = 1..8, one core each
    python 01_run_nrg_tqd.py --assemble         # -> ../output/tqd_nrg.npz
    python 01_run_nrg_tqd.py --gate-index K --checkpoints ../output/checkpoints_sweep   # K = 1..31
    python 01_run_nrg_tqd.py --assemble-sweep --checkpoints ../output/checkpoints_sweep

Without a batch system, all shifts in one go (N workers, see the memory below):

    python 01_run_nrg_tqd.py --all --n-jobs N

Measured on one core with the production settings: a thermodynamics-only run
(--probe) takes ~7.6 h and ~46 GB; a spectral z shift needs more (the cluster
tasks were given 120 GB, 48 h and ~0.4 TB of scratch for the shell spill,
which is released at the end of each task).  Extra BLAS threads did not speed
this up, so ask for one core per task and get the parallelism from the array.
Check the environment first with  python 01_run_nrg_tqd.py --selftest  (~1 min).
"""
import argparse
import json
import os
import socket
import sys
import time

# ============================================================================
# PRODUCTION SETTINGS
# ============================================================================
NZ = 8                  # discretisation shifts, z = k/NZ, k = 1..NZ
NKEEP = 4000            # states kept per shell
NKEEP_EARLY = 48000     # states kept in the first N_EARLY shells
N_EARLY = 6
LAM = 16.0              # large Lambda: better shell separation for 3 channels
ECUT = 10.0
B = 0.3                 # log-Gaussian broadening used in the assembly
GATE_SWEEP = tuple(range(-25, 6, 1))     # gate values for the inset of Fig. 3(b)

# Convergence at v = -10.  First ladder (forward pass, Lambda = 8, z = 1, few early states):
#     Nkeep 1000 -> n = (1.764, 1.184, 1.005)     3000 -> (1.835, 1.115, 1.003)
#           2000 -> n = (1.856, 1.092, 1.003)     4000 -> (1.837, 1.107, 1.004)
#                                                 6000 -> (1.836, 1.106, 1.004)
# z = 1 converges from ~3000 states on, but the shifts differ (at 4000: z = 0.5
# gives (1.826, 1.138)): n_1 + n_2 is stable, the split between dots 1 and 2 is
# a soft direction set by the first, not yet scale-separated shells.  The
# production settings above come from a --probe ladder in Lambda (8, 16) and in
# the early-shell truncation (24000 / 48000 / 96000 states), comparing
# z = 1/16 with z = 1/2 at Nkeep = 4000: Lambda = 16 with 48000 states in the
# first 6 shells is the setting for which the two shifts nearly agree.

# --- physical parameters (units of gamma)
U_LIST = (4.0, 6.0, 8.0)
U12 = 2.0
V_LINE = -10.0
T_OVER_GAMMA = 1e-5
GAMMA_OVER_D = 0.01     # wide-band limit
# ============================================================================

# one thread per task: threading does not help these block sizes
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "..", "output")
sys.path.insert(0, HERE)            # the NRG engine: nrg_multidot.py in this folder
import nrg_multidot as nm  # noqa: E402

G = GAMMA_OVER_D
T = T_OVER_GAMMA * G
M = len(U_LIST)


def nrg_settings(nkeep, nkeep_early, lam, ecut, n_early):
    nshells = int(np.ceil(2 * np.log(0.5 / (1e-3 * T)) / np.log(lam))) + 2
    return dict(Lam=float(lam), nshells=int(nshells), Ecut=float(ecut), Nkeep=int(nkeep),
                Nkeep_early=int(nkeep_early), n_early=int(n_early), T=T)


def model(v=V_LINE):
    return nm.MultiDot(eps=[v * G] * M, U=[u * G for u in U_LIST], Uij=U12 * G, Gamma=[G] * M)


# ----------------------------------------------------------------------------
# checkpoints: one file per z shift
# ----------------------------------------------------------------------------
def ckpt_path(ckdir, k, nz):
    return os.path.join(ckdir, f"z_{k:02d}_of_{nz:02d}.npz")


def save_ckpt(path, r, z, nrg):
    data = dict(z=z, settings_json=json.dumps(nrg, sort_keys=True), bins=r["bins"],
                n=np.array([r["expect"][("n", i)] for i in range(M)]),
                docc=np.array([r["expect"][("docc", i)] for i in range(M)]),
                nn=np.array([r["expect"][("nn", i, j)] for i in range(M) for j in range(i + 1, M)]),
                nkept=r["nkept"], scale=r["scale"], run_seconds=r["time"])
    for i in range(M):
        for op in ("d", "O"):
            h = r["hist"][(op, i)]
            for name, val in zip(("pos", "neg", "total"), h):
                data[f"{op}{i}_{name}"] = val
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def load_ckpt(path, nrg):
    with np.load(path, allow_pickle=False) as d:
        if str(d["settings_json"]) != json.dumps(nrg, sort_keys=True):
            raise ValueError(f"{path} was written with different settings; use another "
                             f"--checkpoints directory or delete it")
        expect = {("n", i): float(d["n"][i]) for i in range(M)}
        expect.update({("docc", i): float(d["docc"][i]) for i in range(M)})
        k = 0
        for i in range(M):
            for j in range(i + 1, M):
                expect[("nn", i, j)] = float(d["nn"][k]); k += 1
        hist = {(op, i): (d[f"{op}{i}_pos"].copy(), d[f"{op}{i}_neg"].copy(),
                          float(d[f"{op}{i}_total"])) for i in range(M) for op in ("d", "O")}
        return dict(z=float(d["z"]), bins=d["bins"].copy(), hist=hist, expect=expect,
                    nkept=d["nkept"].copy(), scale=d["scale"].copy(), time=float(d["run_seconds"]))


# ----------------------------------------------------------------------------
def run_shift(k, nz, nrg, ckdir, scratch, force=False):
    z = (k) / nz
    path = ckpt_path(ckdir, k, nz)
    if os.path.exists(path) and not force:
        print(f"[z {k}/{nz}] checkpoint exists, skipping", flush=True)
        return path
    print(f"[z {k}/{nz}] z={z:g} starting on {socket.gethostname()} "
          f"(settings {nrg}, scratch {scratch})", flush=True)
    t0 = time.time()
    import tempfile
    old_tmp = tempfile.tempdir
    if scratch:
        os.makedirs(scratch, exist_ok=True)
        tempfile.tempdir = scratch           # nrg_multidot spills the shells here
    try:
        r = nm.run_single_z(model(), z, spectral=True, disk_backed=True, **nrg)
    finally:
        tempfile.tempdir = old_tmp
    save_ckpt(path, r, z, nrg)
    print(f"[z {k}/{nz}] done in {time.time() - t0:.0f}s; "
          f"n = {[round(r['expect'][('n', i)], 5) for i in range(M)]}; saved {os.path.basename(path)}",
          flush=True)
    return path


def assemble(nz, nrg, ckdir, b, out):
    runs, missing = [], []
    for k in range(1, nz + 1):
        p = ckpt_path(ckdir, k, nz)
        if not os.path.exists(p):
            missing.append(k)
            continue
        try:
            runs.append(load_ckpt(p, nrg))
        except ValueError as exc:
            raise SystemExit(str(exc))
    if missing:
        raise SystemExit(f"missing checkpoints for z index {missing}; rerun those tasks "
                         f"(or use --nz to assemble a smaller set)")
    w = nm.default_grid(wmin=0.1 * T, wmax=0.999, per_decade=80)
    res = nm.broaden_and_sigma(model(), runs, w, b, True, T)
    x = w / G
    payload = dict(U_list=np.array(U_LIST), U12=U12, v_line=V_LINE, T_over_gamma=T_OVER_GAMMA,
                   gamma_over_D=G, Lam=nrg["Lam"], Nz=nz, Nkeep=nrg["Nkeep"],
                   Nkeep_early=nrg["Nkeep_early"], n_early=nrg["n_early"], Ecut=nrg["Ecut"],
                   b=b, nshells=nrg["nshells"], zs=np.array([r["z"] for r in runs]),
                   w_over_gamma=x,
                   gA4_nrg=np.array([0.5 * G * res[("A_sigma", i)] for i in range(M)]),
                   gA4_raw=np.array([0.5 * G * res[("A_raw", i)] for i in range(M)]),
                   Sigma_over_gamma=np.array([res[("Sigma", i)] / G for i in range(M)]),
                   sumrule=np.array([res[("sumrule", i)] for i in range(M)]),
                   n_nrg=np.array([res["expect"][("n", i)] for i in range(M)]),
                   n_nrg_z=np.array([[r["expect"][("n", i)] for r in runs] for i in range(M)]),
                   docc_nrg=np.array([res["expect"][("docc", i)] for i in range(M)]),
                   run_times=np.array([r["time"] for r in runs]),
                   hist_bins_D=runs[0]["bins"],
                   hist_pos=np.array([[[r["hist"][(op, i)][0] for op in ("d", "O")]
                                       for i in range(M)] for r in runs]),
                   hist_neg=np.array([[[r["hist"][(op, i)][1] for op in ("d", "O")]
                                       for i in range(M)] for r in runs]))
    for i in range(M):
        for j in range(i + 1, M):
            payload[f"nn_{i}{j}"] = res["expect"][("nn", i, j)]
    if not np.allclose(payload["sumrule"], 1.0, atol=1e-6):
        raise SystemExit("Lehmann sum rule violated; reference file not written")
    i0 = np.argmin(np.abs(x - 1e-3)); j0 = np.argmin(np.abs(x + 1e-3))
    n = payload["n_nrg"]
    for i in range(M):
        A0 = 0.5 * (payload["gA4_nrg"][i][i0] + payload["gA4_nrg"][i][j0])
        print(f"dot {i+1}: n = {n[i]:.5f} (z-spread {np.ptp(payload['n_nrg_z'][i]):.1e}), "
              f"gA/4(0) = {A0:.5f}, Friedel sin^2(pi n/2)/pi = {np.sin(np.pi*n[i]/2)**2/np.pi:.5f}, "
              f"sum rule {payload['sumrule'][i].min():.10f}..{payload['sumrule'][i].max():.10f}", flush=True)
    np.savez_compressed(out, **payload)
    print(f"saved {out}  (Nz = {nz}, b = {b})")


# ----------------------------------------------------------------------------
def run_gate(k, nrg, ckdir):
    v = float(GATE_SWEEP[k - 1])
    path = os.path.join(ckdir, f"gate_{k:03d}.npz")
    if os.path.exists(path):
        print(f"[gate {k}] checkpoint exists, skipping", flush=True)
        return
    t0 = time.time()
    r = nm.run_single_z(model(v), 1.0, spectral=False, **nrg)
    n = [r["expect"][("n", i)] for i in range(M)]
    np.savez_compressed(path + ".tmp.npz", v=v, n=np.array(n),
                        settings_json=json.dumps(nrg, sort_keys=True))
    os.replace(path + ".tmp.npz", path)
    print(f"[gate {k}] v={v:g} done in {time.time() - t0:.0f}s; n = {np.round(n, 5)}", flush=True)


def peak_rss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6      # bytes on macOS, kB on Linux


def run_probe(z, nrg, ckdir):
    """Thermodynamics only (no spectra, no scratch) at v = V_LINE and one z.
    For convergence scans in Lambda / Nkeep / Nkeep_early: prints the
    occupations and the truncation energy E_trunc (highest kept level in units
    of the shell scale).  E_trunc well below Ecut in some shells means Nkeep,
    not Ecut, sets the truncation there."""
    tag = f"probe_L{nrg['Lam']:g}_nk{nrg['Nkeep']}_early{nrg['n_early']}x{nrg['Nkeep_early']}_z{z:.4f}"
    t0 = time.time()
    r = nm.run_single_z(model(), z, spectral=False, **nrg)
    n = np.array([r["expect"][("n", i)] for i in range(M)])
    et = r["etrunc"][:-1]
    np.savez_compressed(os.path.join(ckdir, tag + ".npz"), z=z, n=n, etrunc=r["etrunc"], nkept=r["nkept"],
                        settings_json=json.dumps(nrg, sort_keys=True), run_seconds=time.time() - t0)
    early, late = et[:nrg["n_early"]], et[nrg["n_early"]:]
    print(f"[{tag}] done in {time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.1f} GB\n"
          f"    n = {np.round(n, 5)}\n"
          f"    E_trunc (units of shell scale): early shells min {np.min(early):.2f}, "
          f"later shells min {np.min(late):.2f} / median {np.median(late):.2f}  (Ecut = {nrg['Ecut']:g})",
          flush=True)


def assemble_sweep(ckdir, out):
    vs, ns = [], []
    for k in range(1, len(GATE_SWEEP) + 1):
        p = os.path.join(ckdir, f"gate_{k:03d}.npz")
        if not os.path.exists(p):
            print(f"  (missing gate {k}: v = {GATE_SWEEP[k-1]})")
            continue
        with np.load(p) as d:
            vs.append(float(d["v"])); ns.append(d["n"])
    if not vs:
        raise SystemExit("no gate checkpoints found")
    order = np.argsort(vs)
    np.savez_compressed(out, v_sweep=np.array(vs)[order], n_sweep=np.array(ns)[order],
                        U_list=np.array(U_LIST), U12=U12, T_over_gamma=T_OVER_GAMMA, gamma_over_D=G)
    print(f"saved {out} with {len(vs)} gate values")


def selftest(scratch, nkeep=300, lam=8.0):
    """End-to-end check of the environment and the pipeline.

    Runs two small z shifts through the production code path (disk-backed
    spill, checkpoint, assembly).  PASS/FAIL covers what a small run can prove:
    the exact FDM sum rule, the checkpoint round trip and the assembly.  The
    Friedel value of each dot is printed as a diagnostic only -- it needs
    production truncation to be meaningful (with 300 states the low-energy
    fixed point is not reached yet; try --selftest-nkeep 2000, ~6 min).
    """
    import resource
    import shutil
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="tqd_selftest_")
    nrg = nrg_settings(nkeep, 3 * nkeep, lam, 9.0, 2)
    t0 = time.time()
    ok = True
    try:
        for k in (1, 2):
            run_shift(k, 2, nrg, tmpdir, scratch)
        r = load_ckpt(ckpt_path(tmpdir, 1, 2), nrg)
        print(f"  checkpoint round trip OK (z = {r['z']:g}, "
              f"n = {[round(r['expect'][('n', i)], 4) for i in range(M)]})")
        out = os.path.join(tmpdir, "tqd_selftest.npz")
        assemble(2, nrg, tmpdir, 0.4, out)
        d = np.load(out)
        for i in range(M):
            if not np.allclose(d["sumrule"][i], 1.0, atol=1e-9):
                print(f"  FAIL: dot {i+1} sum rule {d['sumrule'][i]}")
                ok = False
        x, n = d["w_over_gamma"], d["n_nrg"]
        print("  diagnostic (needs production truncation to agree):")
        for i in range(M):
            a0 = np.interp(0.0, x, d["gA4_nrg"][i])
            fr = np.sin(np.pi * n[i] / 2) ** 2 / np.pi
            print(f"    dot {i+1}: A(0) = {a0:.4f} vs Friedel {fr:.4f}")
    except Exception as exc:                      # noqa: BLE001 - report and fail
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"selftest {'PASSED' if ok else 'FAILED'} (Nkeep = {nkeep}) in {time.time() - t0:.0f}s; "
          f"peak RSS {peak_rss_gb():.1f} GB. "
          f"A production z shift needs tens of GB and many hours on one core (../input/cluster_run.md).")
    print("For the physics check of the engine itself run:  python test_nrg_ed.py")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--z-index", type=int, help="run this z shift only (1..NZ); "
                                               "defaults to $SLURM_ARRAY_TASK_ID if set")
    g.add_argument("--all", action="store_true", help="run every z shift here (see --n-jobs)")
    g.add_argument("--assemble", action="store_true", help="broaden all checkpoints -> reference file")
    g.add_argument("--gate-index", type=int, help="run one gate value of the occupation sweep (1..%d)"
                                                  % len(GATE_SWEEP))
    g.add_argument("--all-gates", action="store_true", help="run the whole gate sweep here")
    g.add_argument("--assemble-sweep", action="store_true", help="collect the gate checkpoints")
    g.add_argument("--probe", type=float, metavar="Z",
                   help="occupations + truncation energies only, at v = V_LINE and shift z = Z (0 < Z <= 1)")
    g.add_argument("--selftest", action="store_true",
                   help="two tiny shifts through the full pipeline; run this first on a new machine")
    ap.add_argument("--nz", type=int, default=NZ)
    ap.add_argument("--nkeep", type=int, default=NKEEP)
    ap.add_argument("--nkeep-early", type=int, default=NKEEP_EARLY)
    ap.add_argument("--n-early", type=int, default=N_EARLY)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--ecut", type=float, default=ECUT)
    ap.add_argument("--b", type=float, default=B)
    ap.add_argument("--checkpoints", default=os.path.join(OUTPUT, "checkpoints"))
    ap.add_argument("--scratch", default=os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR"),
                    help="directory for the shell spill (~0.4 TB per spectral task with the production settings)")
    ap.add_argument("--out", default=os.path.join(OUTPUT, "tqd_nrg.npz"))
    ap.add_argument("--out-sweep", default=os.path.join(OUTPUT, "tqd_nrg_sweep.npz"))
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                    help="workers for --all / --all-gates (one core each)")
    ap.add_argument("--selftest-nkeep", type=int, default=300,
                    help="kept states for --selftest (2000 makes the Friedel diagnostic meaningful)")
    ap.add_argument("--force", action="store_true", help="recompute even if a checkpoint exists")
    a = ap.parse_args()

    nrg = nrg_settings(a.nkeep, a.nkeep_early, a.lam, a.ecut, a.n_early)
    os.makedirs(a.checkpoints, exist_ok=True)

    if a.selftest:
        raise SystemExit(selftest(a.scratch, a.selftest_nkeep, a.lam))
    if a.assemble:
        assemble(a.nz, nrg, a.checkpoints, a.b, a.out)
    elif a.assemble_sweep:
        assemble_sweep(a.checkpoints, a.out_sweep)
    elif a.all_gates:
        from joblib import Parallel, delayed
        Parallel(n_jobs=a.n_jobs)(delayed(run_gate)(k, nrg, a.checkpoints)
                                  for k in range(1, len(GATE_SWEEP) + 1))
    elif a.probe is not None:
        run_probe(a.probe, nrg, a.checkpoints)
    elif a.gate_index is not None:
        run_gate(a.gate_index, nrg, a.checkpoints)
    elif a.all:
        from joblib import Parallel, delayed
        Parallel(n_jobs=a.n_jobs)(delayed(run_shift)(k, a.nz, nrg, a.checkpoints, a.scratch, a.force)
                                  for k in range(1, a.nz + 1))
    else:
        k = a.z_index or int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if not 1 <= k <= a.nz:
            ap.error("give --z-index 1..NZ, set SLURM_ARRAY_TASK_ID, or use --all/--assemble")
        run_shift(k, a.nz, nrg, a.checkpoints, a.scratch, a.force)


if __name__ == "__main__":
    main()
