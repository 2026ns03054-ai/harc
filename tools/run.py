#!/usr/bin/env python3
"""run.py - run one HARC experiment configuration (one or more repetitions).

Run it on the master VM from the repository root.  For each repetition it starts the
master, starts tools/churn.py on every worker host (over ssh, or locally for "local"),
waits for the workers to register, runs the client, stops everything and collects all
logs into runs/<exp>/<config>/<policy>/rep<r>/.

examples
  # everything on one machine (smoke test)
  python3 tools/run.py --exp smoke --config k0.5 --policy harc --hosts local:3 \
      --model models/stories15M.bin --k 0.5 --mean 30 --n 12 --mix 200:1

  # Azure: master on node1 (10.1.1.4), two workers on each of node2 and node3
  python3 tools/run.py --exp rq2 --config k0.5_bw10 --policy harc --reps 1 \
      --master-ip 10.1.1.4 --hosts 10.1.1.5:2,10.1.1.6:2 --ssh-key ~/.ssh/dc-key.pem \
      --model models/stories42M.bin --k 0.5 --mean 60 --netem "10mbit 20ms" --bw-mbps 10
"""
import argparse, json, os, shlex, shutil, signal, socket, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--policy", default="harc", choices=["restart", "yd", "contkv", "random", "harc"])
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--rep-start", type=int, default=1)
ap.add_argument("--hosts", default="local:3", help="local:N and/or ip:N,... (local = on this machine, next to the master)")
ap.add_argument("--master-ip", default="127.0.0.1")
ap.add_argument("--master-port", type=int, default=7000)
ap.add_argument("--model", default="models/stories15M.bin")
ap.add_argument("--tokenizer", default="models/tokenizer.bin")
ap.add_argument("--k", type=float, default=1.0)
ap.add_argument("--mean", type=float, default=120.0)
ap.add_argument("--down", type=float, default=10.0)
ap.add_argument("--signal", default="KILL", choices=["KILL", "STOP"])
ap.add_argument("--no-churn", action="store_true")
ap.add_argument("--rng", default="counter", choices=["counter", "stateful"])
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--mix", default="256:1")
ap.add_argument("--concurrency", type=int, default=0, help="default: number of workers")
ap.add_argument("--rate", type=float, default=0.0, help="Poisson arrivals per second instead of closed loop")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--tau-ms", type=float, default=1500)
ap.add_argument("--hb-ms", type=int, default=500)
ap.add_argument("--mu0", type=float, default=0, help="initial mean lifetime for the model (default: --mean)")
ap.add_argument("--bw-mbps", type=float, default=1000)
ap.add_argument("--netem", default="", help='e.g. "10mbit 20ms" (applied on all VMs, needs sudo)')
ap.add_argument("--warmup", type=float, default=3.0)
ap.add_argument("--timeout", type=float, default=3600)
ap.add_argument("--ssh-key", default="")
ap.add_argument("--ssh-user", default="azureuser")
ap.add_argument("--remote-dir", default="harc", help="repository path on the worker VMs (relative to home)")
ap.add_argument("--out", default="runs")
ap.add_argument("--force", action="store_true", help="repeat repetitions that already finished")
a = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
hosts = []
for h in a.hosts.split(","):
    ip, n = h.split(":")
    hosts.append((ip, int(n)))
nworkers = sum(n for _, n in hosts)
conc = a.concurrency or nworkers
mu0 = a.mu0 or a.mean
master_ip = "127.0.0.1" if all(ip == "local" for ip, _ in hosts) else a.master_ip   # local workers register with this address too


def ssh(ip, cmd, **kw):
    base = ["ssh", "-n", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
    if a.ssh_key:
        base += ["-i", os.path.expanduser(a.ssh_key)]
    return base + ["%s@%s" % (a.ssh_user, ip), cmd]


def run_quiet(cmd, timeout=300):
    try:
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "timed out: " + " ".join(cmd[-2:]))


STALE = "pkill -f '[t]ools/churn.py'; pkill -9 -f '[b]uild/worker'; true"


def cleanup(rel):
    """remove anything an interrupted run left behind: processes on every host and partial logs"""
    run_quiet(["bash", "-c", "pkill -f '[b]uild/master'; pkill -f '[b]uild/client'; " + STALE +
               "; sudo -n bash tools/netem.sh off >/dev/null 2>&1; true"])
    for ip, _ in hosts:
        if ip != "local":
            run_quiet(ssh(ip, STALE + "; cd %s && sudo -n bash tools/netem.sh off >/dev/null 2>&1; rm -rf %s; true"
                          % (a.remote_dir, shlex.quote(rel))))
    if os.path.isdir(rel):
        shutil.rmtree(rel)
    time.sleep(1)


def status():
    try:
        s = socket.create_connection((master_ip, a.master_port), timeout=2)
        s.sendall(b"STATUS\n")
        data = b""
        while not data.endswith(b"\n") or b"END " not in data:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        s.close()
        return data.decode()
    except OSError:
        return ""


def netem(on):
    if not a.netem:
        return
    args = ("on " + a.netem) if on else "off"
    for ip, _ in hosts:
        if ip != "local":
            print(run_quiet(ssh(ip, "cd %s && sudo bash tools/netem.sh %s" % (a.remote_dir, args))).stdout.strip())
    print(run_quiet(["sudo", "bash", "tools/netem.sh"] + args.split()).stdout.strip())


for rep in range(a.rep_start, a.rep_start + a.reps):
    rel = os.path.join(a.out, a.exp, a.config, a.policy, "rep%d" % rep)
    if os.path.exists(os.path.join(rel, "meta.json")) and not a.force:
        print("--- %s %s rep %d already done, skipped" % (a.config, a.policy, rep), flush=True)
        continue
    cleanup(rel)
    os.makedirs(rel, exist_ok=True)
    seed = a.seed * 100 + rep                    # same requests and churn schedule for every policy
    meta = dict(vars(a), rep=rep, run_seed=seed, nworkers=nworkers, concurrency=conc, mu0=mu0,
                started=time.time(), hostname=socket.gethostname())
    print("=== %s rep %d: policy=%s workers=%d k=%.2f mean=%.0fs churn=%s netem=%s ===" %
          (a.config, rep, a.policy, nworkers, a.k, a.mean, "off" if a.no_churn else a.signal, a.netem or "off"), flush=True)
    netem(True)
    mlog = open(os.path.join(rel, "master.log"), "w")
    master = subprocess.Popen(["build/master", str(a.master_port), "--policy", a.policy, "--tau-ms", str(a.tau_ms),
                               "--mu0", str(mu0), "--bw-mbps", str(a.bw_mbps), "--out", rel, "--seed", str(seed)],
                              stdin=subprocess.DEVNULL, stdout=mlog, stderr=subprocess.STDOUT)
    time.sleep(0.5)
    churners, port = [], 7101
    for hi, (ip, n) in enumerate(hosts):
        label = ("l" if master_ip == "127.0.0.1" else "n" + master_ip.split(".")[-1]) if ip == "local" else "n" + ip.split(".")[-1]
        p0 = port if ip == "local" else 7101
        names = ",".join("%sw%d:%d" % (label, j + 1, p0 + j) for j in range(n))
        if ip == "local":
            port += n
        args = ["--workers", names, "--ip", master_ip if ip == "local" else ip,
                "--master", "%s:%d" % (master_ip, a.master_port), "--model", a.model, "--tokenizer", a.tokenizer,
                "--k", str(a.k), "--mean", str(a.mean), "--down", str(a.down), "--signal", a.signal,
                "--seed", str(seed * 10 + hi), "--rng", a.rng, "--hb-ms", str(a.hb_ms), "--logdir", rel,
                "--tag", "harcrun"] + (["--no-churn"] if a.no_churn else [])
        if ip == "local":
            churners.append((ip, subprocess.Popen([sys.executable, "tools/churn.py"] + args, stdin=subprocess.DEVNULL)))
        else:
            cmd = "cd %s && mkdir -p %s && exec python3 tools/churn.py %s" % (a.remote_dir, shlex.quote(rel), " ".join(shlex.quote(x) for x in args))
            churners.append((ip, subprocess.Popen(ssh(ip, cmd), stdin=subprocess.DEVNULL)))
    t0 = time.time()
    while time.time() - t0 < 120:
        st = status()
        live = sum(1 for l in st.splitlines() if l.startswith("WORKER") and " DOWN " not in l)
        if live >= nworkers:
            break
        time.sleep(0.5)
    else:
        print("warning: only %d of %d workers registered" % (live, nworkers), flush=True)
    time.sleep(a.warmup)
    ccmd = ["build/client", "%s:%d" % (master_ip, a.master_port), "--n", str(a.n), "--mix", a.mix,
            "--seed", str(seed), "--out", os.path.join(rel, "client.csv"), "--timeout", str(int(a.timeout))]
    ccmd += ["--rate", str(a.rate)] if a.rate > 0 else ["--concurrency", str(conc)]
    clog = open(os.path.join(rel, "client.log"), "w")
    t_client = time.time()
    try:
        subprocess.run(ccmd, stdin=subprocess.DEVNULL, stdout=clog, stderr=subprocess.STDOUT, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        print("warning: client timed out", flush=True)
    meta["client_seconds"] = time.time() - t_client
    open(os.path.join(rel, "status.txt"), "w").write(status())
    for ip, p in churners:
        if ip == "local":
            p.send_signal(signal.SIGTERM)
        else:
            run_quiet(ssh(ip, "pkill -TERM -f '[c]hurn.py.*--tag harcrun'"))
    for ip, p in churners:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    for ip, _ in hosts:
        if ip != "local":
            src = "%s@%s:%s/%s/*" % (a.ssh_user, ip, a.remote_dir, rel)
            cmd = ["scp", "-q", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"] + \
                  (["-i", os.path.expanduser(a.ssh_key)] if a.ssh_key else []) + [src, rel + "/"]
            r = run_quiet(cmd)
            if r.returncode:
                print("warning: could not copy logs from %s: %s" % (ip, r.stdout.strip()), flush=True)
    time.sleep(0.5)
    master.send_signal(signal.SIGTERM)
    try:
        master.wait(timeout=10)
    except subprocess.TimeoutExpired:
        master.kill()
    netem(False)
    meta["finished"] = time.time()
    json.dump(meta, open(os.path.join(rel, "meta.json"), "w"), indent=1)
    ok = sum(1 for l in open(os.path.join(rel, "client.csv")).read().splitlines()[1:] if l.split(",")[6] == "1") if os.path.exists(os.path.join(rel, "client.csv")) else 0
    print("    done in %.0f s: %d/%d requests completed -> %s" % (meta["client_seconds"], ok, a.n, rel), flush=True)
