#!/usr/bin/env python3
"""analyze.py - turn the logs written by tools/run.py into the numbers and figures for the paper.

usage:  python3 tools/analyze.py runs [--out analysis]

Reads every runs/<exp>/<config>/<policy>/rep<r>/ directory and writes
  analysis/runs.csv      one row per run (all metrics)
  analysis/summary.csv   mean and 95% confidence interval per (exp, config, policy)
  analysis/summary.md    the tables used in the paper, plus Welch t-tests of HARC vs each baseline
  analysis/fig3.png      RQ1 throughput vs workers and RQ3 lost work vs Weibull shape (if matplotlib is installed)
  analysis/fig4.png      RQ4 discovery results (if runs/rq4/discovery.csv exists)

Only the Python standard library is needed for the numbers; matplotlib is optional.
"""
import argparse, csv, glob, json, math, os, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="runs")
ap.add_argument("--out", default="analysis")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# ---------------- statistics without scipy ----------------
def _betacf(x, p, q):
    tiny, c, d = 1e-300, 1.0, 1.0 - (p + q) * x / (p + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny); h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (q - m) * x / ((p + m2 - 1) * (p + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny; h *= d * c
        aa = -(p + m) * (p + q + m) * x / ((p + m2) * (p + m2 + 1))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c if abs(1 + aa / c) > tiny else tiny; de = d * c; h *= de
        if abs(de - 1) < 1e-12:
            break
    return h

def betainc(x, p, q):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = math.lgamma(p + q) - math.lgamma(p) - math.lgamma(q) + p * math.log(x) + q * math.log(1 - x)
    if x < (p + 1) / (p + q + 2):
        return math.exp(lb) * _betacf(x, p, q) / p
    return 1 - math.exp(lb) * _betacf(1 - x, q, p) / q

def t_sf2(t, df):          # two-sided p-value
    return betainc(df / (df + t * t), df / 2, 0.5)

def t_crit(df, alpha=0.05):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf2(mid, df) > alpha: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def mean_ci(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs: return (float("nan"), float("nan"), 0)
    if len(xs) == 1: return (xs[0], float("nan"), 1)
    return (st.mean(xs), t_crit(len(xs) - 1) * st.stdev(xs) / math.sqrt(len(xs)), len(xs))

def welch(x, y):
    x = [v for v in x if v == v]; y = [v for v in y if v == v]
    if len(x) < 2 or len(y) < 2: return float("nan")
    vx, vy = st.variance(x) / len(x), st.variance(y) / len(y)
    if vx + vy == 0: return 1.0 if st.mean(x) == st.mean(y) else 0.0
    t = (st.mean(x) - st.mean(y)) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return t_sf2(abs(t), df)

def pct(xs, q):
    xs = sorted(xs)
    if not xs: return float("nan")
    i = q * (len(xs) - 1); lo = int(math.floor(i)); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

def rcsv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

# ---------------- per-run metrics ----------------
runs = []
for meta_path in sorted(glob.glob(os.path.join(a.root, "*", "*", "*", "rep*", "meta.json"))):
    d = os.path.dirname(meta_path)
    m = json.load(open(meta_path))
    cl = rcsv(os.path.join(d, "client.csv"))
    ev = rcsv(os.path.join(d, "events.csv"))
    ok = [r for r in cl if r["ok"] == "1"]
    lat = [float(r["latency_ms"]) / 1000 for r in ok]
    G = {r["cid"]: int(r["G"]) for r in ok}
    run = dict(exp=m["exp"], config=m["config"], policy=m["policy"], rep=m["rep"], dir=d, nworkers=m["nworkers"],
               k=m["k"], mean=m["mean"], bw=m["bw_mbps"], mix=m["mix"], model=os.path.basename(m["model"]),
               rng=m["rng"], churn=not m["no_churn"], n=len(cl), ok=len(ok), run_seed=m["run_seed"])
    run["success"] = len(ok) / len(cl) if cl else float("nan")
    run["lat_mean_s"] = st.mean(lat) if lat else float("nan")
    run["lat_p50_s"] = pct(lat, 0.5); run["lat_p95_s"] = pct(lat, 0.95)
    if cl:
        span = max(float(r["t_done"]) for r in cl) - min(float(r["t_submit"]) for r in cl)
        run["makespan_s"] = span
        run["tokens_per_s"] = sum(G.values()) / span if span > 0 else float("nan")
    run["failures_per_req"] = st.mean(int(r["failures"]) for r in ok) if ok else float("nan")
    run["share_failed"] = sum(1 for r in ok if int(r["failures"]) > 0) / len(ok) if ok else float("nan")
    reg = [dict(x.split("=") for x in e["info"].split()) for e in ev if e["event"] == "register"]
    tp = st.median(float(x["tp"]) for x in reg) / 1000 if reg else float("nan")
    td = st.median(float(x["td"]) for x in reg) / 1000 if reg else float("nan")
    run["td_ms"] = td * 1000; run["tp_ms"] = tp * 1000
    run["worker_failures"] = sum(1 for e in ev if e["event"] == "worker_down")
    kvb = [int(dict(x.split("=") for x in e["info"].split()).get("kv_bytes", 0)) for e in ev if e["event"] == "done"]
    run["kv_mb_per_req"] = st.mean(kvb) / 1e6 if kvb else float("nan")
    asg = [dict(x.split("=") for x in e["info"].split()) for e in ev if e["event"] == "assign"]
    run["k_hat"] = float(asg[-1]["k"]) if asg else float("nan")
    run["theta_hat"] = float(asg[-1]["theta"]) if asg else float("nan")
    modes = [int(x["mode"]) for x in asg]
    for mi, mn in enumerate(("restart", "tokens", "kv")):
        run["share_mode_" + mn] = modes.count(mi) / len(modes) if modes else float("nan")
    # work logs: every prefill / decode step that any worker actually executed
    dec, pre, first_np = defaultdict(int), defaultdict(int), {}
    wl = glob.glob(os.path.join(d, "work_*.csv"))
    for w in wl:
        for line in open(w):
            f = line.strip().split(",")
            if len(f) != 7: continue
            _, _, rid, att, ph, x0, x1 = f
            if ph == "decode": dec[rid] += 1
            elif ph == "prefill":
                pre[rid] += int(x1) - int(x0)
                if att == "1": first_np[rid] = int(x1)
    if wl and ok:
        rec = [dec[c] - G[c] for c in G]
        rep_ = [pre[c] - first_np.get(c, pre[c]) for c in G]
        run["recomputed_tokens"] = st.mean(rec)
        run["reprefill_tokens"] = st.mean(rep_)
        run["lost_work_s"] = st.mean(r * td + p * tp for r, p in zip(rec, rep_))
    else:
        run["recomputed_tokens"] = run["reprefill_tokens"] = run["lost_work_s"] = float("nan")
    run["_hash"] = {r["cid"]: r["hash"] for r in ok}
    run["_fail"] = {r["cid"]: int(r["failures"]) for r in ok}
    runs.append(run)

# exactness against the failure-free reference with the same requests (same seed, model, mix, rng)
refs = {(r["model"], r["mix"], r["n"], r["rng"], r["run_seed"]): r for r in runs if not r["churn"]}
for r in runs:
    ref = refs.get((r["model"], r["mix"], r["n"], r["rng"], r["run_seed"]))
    r["exact_all"] = r["exact_resumed"] = float("nan"); r["resumed"] = 0
    if ref is None or ref is r: continue
    same = [c for c in r["_hash"] if c in ref["_hash"]]
    res = [c for c in same if r["_fail"][c] > 0]
    r["resumed"] = len(res)
    if same: r["exact_all"] = sum(r["_hash"][c] == ref["_hash"][c] for c in same) / len(same)
    if res: r["exact_resumed"] = sum(r["_hash"][c] == ref["_hash"][c] for c in res) / len(res)

METRICS = ["success", "lat_mean_s", "lat_p50_s", "lat_p95_s", "makespan_s", "tokens_per_s", "failures_per_req",
           "share_failed", "recomputed_tokens", "reprefill_tokens", "lost_work_s", "kv_mb_per_req", "worker_failures",
           "td_ms", "tp_ms", "k_hat", "theta_hat", "share_mode_restart", "share_mode_tokens", "share_mode_kv",
           "exact_all", "exact_resumed", "resumed"]
KEYS = ["exp", "config", "policy", "rep", "nworkers", "k", "mean", "bw", "mix", "model", "rng", "churn", "n", "ok"]
with open(os.path.join(a.out, "runs.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(KEYS + METRICS)
    for r in runs: w.writerow([r[k] for k in KEYS] + [r.get(k, "") for k in METRICS])

groups = defaultdict(list)
for r in runs: groups[(r["exp"], r["config"], r["policy"])].append(r)
summ = {}
with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["exp", "config", "policy", "reps", "nworkers", "k", "mean", "bw", "mix", "model", "rng"] +
               sum([[m, m + "_ci95"] for m in METRICS], []))
    for key, rs in sorted(groups.items()):
        row = {m: mean_ci([r.get(m) for r in rs]) for m in METRICS}
        summ[key] = (rs, row)
        r0 = rs[0]
        w.writerow(list(key) + [len(rs), r0["nworkers"], r0["k"], r0["mean"], r0["bw"], r0["mix"], r0["model"], r0["rng"]] +
                   sum([[round(row[m][0], 6), round(row[m][1], 6)] for m in METRICS], []))

def fmt(v, nd=2):
    m, h, n = v
    if m != m: return "–"
    return ("%.*f" % (nd, m)) + ("" if h != h else " ± %.*f" % (nd, h))

PN = {"restart": "B1 Restart", "yd": "B2 Fixed Y/D", "contkv": "B3 Continuous KV", "random": "B4 Random placement", "harc": "HARC"}
md = ["# HARC experiment summary", "", "Values are means over repetitions ± 95% confidence half-width (Student t).", ""]
for exp in sorted({k[0] for k in summ}):
    cfgs = sorted({k[1] for k in summ if k[0] == exp})
    pols = [p for p in ["restart", "yd", "contkv", "random", "harc"] if any((exp, c, p) in summ for c in cfgs)]
    md += ["## %s" % exp, ""]
    for metric, nd, title in [("lat_p95_s", 2, "p95 latency (s)"), ("lat_mean_s", 2, "mean latency (s)"),
                              ("recomputed_tokens", 1, "recomputed tokens per request"),
                              ("reprefill_tokens", 1, "re-prefilled tokens per request"),
                              ("lost_work_s", 3, "lost work per request (s)"), ("success", 3, "success rate"),
                              ("tokens_per_s", 1, "throughput (tokens/s)"), ("makespan_s", 1, "makespan (s)"),
                              ("worker_failures", 1, "worker failures per run"), ("kv_mb_per_req", 2, "KV checkpoint MB per request"),
                              ("exact_resumed", 3, "share of resumed requests identical to reference"),
                              ("td_ms", 2, "decode ms/token at registration")]:
        md += ["**%s**" % title, "", "| policy | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
        for p in pols:
            md.append("| %s | " % PN[p] + " | ".join(fmt(summ[(exp, c, p)][1][metric], nd) if (exp, c, p) in summ else "" for c in cfgs) + " |")
        md.append("")
    if "harc" in pols:
        md += ["**Welch t-test, HARC vs baseline (two-sided p-values; p95 latency / recomputed tokens)**", "",
               "| baseline | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
        for p in pols:
            if p == "harc": continue
            cells = []
            for c in cfgs:
                if (exp, c, p) in summ and (exp, c, "harc") in summ:
                    A, B = summ[(exp, c, "harc")][0], summ[(exp, c, p)][0]
                    cells.append("%.3g / %.3g" % (welch([r["lat_p95_s"] for r in A], [r["lat_p95_s"] for r in B]),
                                                  welch([r["recomputed_tokens"] for r in A], [r["recomputed_tokens"] for r in B])))
                else: cells.append("")
            md.append("| %s | " % PN[p] + " | ".join(cells) + " |")
        md.append("")

# RQ1 scaling, per model
rq1m = defaultdict(list)
for k, v in summ.items():
    if k[0] == "rq1": rq1m[v[0][0]["model"]].append((v[0][0]["nworkers"], v))
rq1 = []
for model, lst in sorted(rq1m.items()):
    lst.sort(key=lambda x: x[0]); rq1.append((model, lst))
    T1 = next((v[1]["makespan_s"][0] for n, v in lst if n == 1), None)
    md += ["## RQ1 scaling, %s (S(n) from makespan)" % model, "",
           "| workers | tokens/s | p50 (s) | p95 (s) | S(n) | E(n) | f̂ |", "|---|---|---|---|---|---|---|"]
    for n, (rs, row) in lst:
        S = T1 / row["makespan_s"][0] if T1 else float("nan")
        f = (1 - 1 / S) / (1 - 1 / n) if n > 1 and S == S else float("nan")
        md.append("| %d | %s | %s | %s | %.2f | %.2f | %s |" % (n, fmt(row["tokens_per_s"], 1), fmt(row["lat_p50_s"]), fmt(row["lat_p95_s"]),
                                                              S, S / n, "–" if f != f else "%.3f" % f))
    md.append("")

disc = rcsv(os.path.join(a.root, "rq4", "discovery.csv"))
if disc:
    dg = defaultdict(list)
    for r in disc: dg[(r["mode"], int(r["peers"]), int(r["ttl"]))].append(r)
    md += ["## RQ4 discovery", "", "| overlay | peers | TTL | messages/query | hit rate | best-cluster rate | selection latency (ms) |", "|---|---|---|---|---|---|---|"]
    for (mode, P, ttl), rs in sorted(dg.items()):
        md.append("| %s | %d | %d | %s | %s | %s | %s |" % ("HyPEC" if mode == "hypec" else "B5 Flat", P, ttl,
                  fmt(mean_ci([float(r["messages"]) for r in rs]), 1), fmt(mean_ci([float(r["hit"]) for r in rs]), 3),
                  fmt(mean_ci([float(r["best"]) for r in rs]), 3), fmt(mean_ci([float(r["select_ms"]) for r in rs if r["hit"] == "1"]), 1)))
    md.append("")
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ---------------- figures ----------------
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed: skipping figures"); raise SystemExit
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
rq3 = defaultdict(dict)
for (exp, cfg, pol), (rs, row) in summ.items():
    if exp == "rq3": rq3[pol][rs[0]["k"]] = row["lost_work_s"]
if rq1 or rq3:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for (model, lst), mk in zip(rq1, "o^sv"):
        ns = [n for n, _ in lst]; tp_ = [v[1]["tokens_per_s"] for _, v in lst]
        ax[0].errorbar(ns, [t[0] for t in tp_], yerr=[0 if t[1] != t[1] else t[1] for t in tp_], marker=mk, color="k", capsize=2, lw=1,
                       ms=3, label=model.replace(".bin", ""))
        if ns and ns[0] == 1:
            ax[0].plot(ns, [tp_[0][0] * n for n in ns], "k:", lw=0.6)
    if rq1:
        ax[0].set_xlabel("Workers n"); ax[0].set_ylabel("Throughput (tokens/s)"); ax[0].legend(frameon=False); ax[0].set_title("(a) RQ1 scaling (dotted: linear)")
    for pol, mk, ls in (("random", "s", "--"), ("harc", "o", "-")):
        if pol in rq3:
            ks = sorted(rq3[pol]); v = [rq3[pol][k] for k in ks]
            ax[1].errorbar(ks, [x[0] for x in v], yerr=[0 if x[1] != x[1] else x[1] for x in v], marker=mk, ls=ls, color="k", capsize=2, lw=1, label=PN[pol])
    ax[1].set_xlabel("Weibull shape k"); ax[1].set_ylabel("Lost work per request (s)"); ax[1].legend(frameon=False); ax[1].set_title("(b) RQ3 placement")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3.png"), dpi=300)
if disc:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for mode, mk, ls in (("flat", "s", "--"), ("hypec", "o", "-")):
        for ttl in sorted({k[2] for k in dg if k[0] == mode}):
            ps = sorted({k[1] for k in dg if k[0] == mode and k[2] == ttl})
            msgs = [st.mean(float(r["messages"]) for r in dg[(mode, P, ttl)]) for P in ps]
            hits = [st.mean(float(r["best"]) for r in dg[(mode, P, ttl)]) for P in ps]
            lab = "%s, TTL %d" % ("HyPEC" if mode == "hypec" else "Flat", ttl)
            ax[0].plot(ps, msgs, marker=mk, ls=ls, lw=1, ms=3, label=lab); ax[1].plot(ps, hits, marker=mk, ls=ls, lw=1, ms=3, label=lab)
    ax[0].set_xlabel("Peers"); ax[0].set_ylabel("Messages per query"); ax[0].set_title("(a) Cost")
    ax[1].set_xlabel("Peers"); ax[1].set_ylabel("Best cluster chosen (share)"); ax[1].set_title("(b) Selection quality")
    ax[1].legend(frameon=False, fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4.png"), dpi=300)
print("figures written to", a.out)
