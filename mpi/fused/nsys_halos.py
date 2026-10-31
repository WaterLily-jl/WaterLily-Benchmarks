#!/usr/bin/env python3
"""The halos in an Nsight Systems trace of mpi/fused/halo_diag.jl (DIAG_NVTX=1), exported to SQLite: per halo array,
the mean time of each phase, and the CUDA API calls and memory copies made in each phase, per halo. The calls in
post, wait_recv and wait_send are those of the MPI library (WaterLily makes none there). The other NVTX ranges, such
as those of the ping-pong of mpi/cuda_mpi_test.jl, get the same counts, on every thread, per unit of their payload
(per round trip of the ping-pong).

    python3 -I mpi/fused/nsys_halos.py [--steps=N] [--top=K] trace_0.sqlite [trace_4.sqlite ...]

--steps: the steps in the trace (scaling.jl --steps), to print the halo time per step too; --top: the calls listed
per phase [8].
"""
import bisect, sqlite3, sys
from collections import defaultdict

PHASES = ("pack", "sync", "post", "work", "wait_recv", "unpack", "wait_send")
opts = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
paths = [a for a in sys.argv[1:] if not a.startswith("--")]
steps, top = int(opts.get("steps", 0)), int(opts.get("top", 8))
paths or sys.exit(__doc__)


def tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def report(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    print(f"\n== {path}")
    others(db, names)
    dom = db.execute("SELECT domainId FROM NVTX_EVENTS WHERE eventType=75 AND text='halo'").fetchone()
    if dom is None:
        return
    ranges = [(s, e, tid, t if t is not None else names.get(i, "?")) for s, e, tid, t, i in db.execute(
        "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE eventType=60 AND domainId=?", dom)]
    halos = defaultdict(list)  # thread => [(start, end, array)], sorted
    for s, e, tid, t in ranges:
        if t not in PHASES:
            halos[tid].append((s, e, t))
    for v in halos.values():
        v.sort()
    # the phases, each with the halo it is in
    phases = defaultdict(list)  # thread => [(start, end, phase, array)], sorted
    for s, e, tid, t in ranges:
        if t in PHASES:
            hs = halos[tid]; k = bisect.bisect_right(hs, (s, float("inf"), "")) - 1
            if k >= 0 and hs[k][1] >= e:
                phases[tid].append((s, e, t, hs[k][2]))
    for v in phases.values():
        v.sort()
    starts = {tid: [p[0] for p in v] for tid, v in phases.items()}

    def phase_of(tid, s, e):
        if tid not in starts:
            return None
        k = bisect.bisect_right(starts[tid], s) - 1
        if k >= 0 and phases[tid][k][1] >= e:
            return phases[tid][k][3], phases[tid][k][2]
        return None

    calls = {a: sum(1 for v in halos.values() for h in v if h[2] == a) for a in {h[2] for v in halos.values() for h in v}}
    time = defaultdict(float)  # (array, phase) => ns
    for v in phases.values():
        for s, e, t, a in v:
            time[a, t] += e - s
    api = defaultdict(lambda: [0, 0.0])  # (array, phase, call) => [count, ns]
    corr = {}  # correlationId => (array, phase), of the API calls in a phase
    tab = tables(db)
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tab:
        for s, e, tid, n, c in db.execute("SELECT start, end, globalTid, nameId, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            ap = phase_of(tid, s, e)
            if ap:
                x = api[ap + (names.get(n, "?"),)]; x[0] += 1; x[1] += e - s; corr[c] = ap
    kinds = dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")) if "ENUM_CUDA_MEMCPY_OPER" in tab else {}
    copies = defaultdict(lambda: [0, 0, 0.0])  # (array, phase, kind) => [count, bytes, GPU ns]
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tab:
        for s, e, b, k, c in db.execute("SELECT start, end, bytes, copyKind, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            if c in corr:
                x = copies[corr[c] + (kinds.get(k, str(k)),)]; x[0] += 1; x[1] += b; x[2] += e - s

    arrays = sorted(calls, key=lambda a: (-eval_size(a), a))
    print("Halos: calls, then the mean time of each phase per call (us)" + (f"; ms per step over {steps} steps" if steps else ""))
    print(f"{'array':<22} {'calls':>6}" + "".join(f"{p:>10}" for p in PHASES) + f"{'us/call':>10}" + (f"{'ms/step':>9}" if steps else ""))
    for a in arrays:
        us = [time[a, p] / calls[a] / 1e3 for p in PHASES]
        print(f"{a:<22} {calls[a]:>6}" + "".join(f"{u:>10.1f}" for u in us) + f"{sum(us):>10.1f}" +
              (f"{calls[a] * sum(us) / 1e3 / steps:>9.2f}" if steps else ""))
    print("\nCUDA API calls (count, us) and memory copies (count, KB, GPU us) per halo call, by phase")
    for a in arrays:
        print(f"{a}:")
        for p in PHASES:
            cs = sorted(((n, c, t) for (a_, p_, n), (c, t) in api.items() if a_ == a and p_ == p), key=lambda x: -x[2])
            ms = sorted(((k, c, b, t) for (a_, p_, k), (c, b, t) in copies.items() if a_ == a and p_ == p), key=lambda x: -x[3])
            if not cs and not ms:
                continue
            tot = sum(t for _, _, t in cs) / calls[a] / 1e3
            line = " | ".join(f"{n} {c / calls[a]:.1f} {t / calls[a] / 1e3:.1f}" for n, c, t in cs[:top])
            print(f"  {p:<10} {tot:6.1f} us in {sum(c for _, c, _ in cs) / calls[a]:.1f} calls: {line}")
            for k, c, b, t in ms:
                print(f"  {'':<10} copies {k}: {c / calls[a]:.1f}, {b / calls[a] / 1e3:.1f} KB, {t / calls[a] / 1e3:.1f} us")


def others(db, names):
    """The NVTX ranges of the domains other than 'halo' and Julia's: CUDA API calls and copies per payload unit."""
    rs = list(db.execute(
        "SELECT e.start, e.end, COALESCE(e.text, s.value), COALESCE(e.int64Value, e.uint64Value, 1) FROM NVTX_EVENTS e "
        "LEFT JOIN StringIds s ON e.textId=s.id WHERE e.eventType=60 AND e.domainId NOT IN "
        "(SELECT domainId FROM NVTX_EVENTS WHERE eventType=75 AND text IN ('halo', 'Julia')) ORDER BY e.start"))
    if not rs:
        return
    tab = tables(db)
    starts = [r[0] for r in rs]

    def range_of(s, e):
        k = bisect.bisect_right(starts, s) - 1
        return k if k >= 0 and rs[k][1] >= e else None

    api, corr = defaultdict(lambda: [0, 0.0]), {}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tab:
        for s, e, n, c in db.execute("SELECT start, end, nameId, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            k = range_of(s, e)
            if k is not None:
                x = api[k, names.get(n, "?")]; x[0] += 1; x[1] += e - s; corr[c] = k
    kinds = dict(db.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")) if "ENUM_CUDA_MEMCPY_OPER" in tab else {}
    copies = defaultdict(lambda: [0, 0, 0.0])
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tab:
        for s, e, b, kd, c in db.execute("SELECT start, end, bytes, copyKind, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            if c in corr:
                x = copies[corr[c], kinds.get(kd, str(kd))]; x[0] += 1; x[1] += b; x[2] += e - s
    print("NVTX ranges: time, CUDA API calls (count, us) and memory copies (count, KB, GPU us), per payload unit")
    for k, (s, e, t, u) in enumerate(rs):
        u = max(u, 1)
        cs = sorted(((n, c, d) for (k_, n), (c, d) in api.items() if k_ == k), key=lambda x: -x[2])
        print(f"{t} (payload {u}): {(e - s) / u / 1e3:.1f} us, {sum(d for _, _, d in cs) / u / 1e3:.1f} us in "
              f"{sum(c for _, c, _ in cs) / u:.1f} calls: " + " | ".join(f"{n} {c / u:.1f} {d / u / 1e3:.1f}" for n, c, d in cs[:top]))
        for kd, c, b, d in sorted(((kd, *v) for (k_, kd), v in copies.items() if k_ == k), key=lambda x: -x[3]):
            print(f"  copies {kd}: {c / u:.1f}, {b / u / 1e3:.1f} KB, {d / u / 1e3:.1f} us")
    print()


def eval_size(a):  # the number of elements of an array named "34x66x66x3 Float32"
    n = 1
    for d in a.split()[0].split("x"):
        n *= int(d) if d.isdigit() else 1
    return n


for p in paths:
    report(p)
