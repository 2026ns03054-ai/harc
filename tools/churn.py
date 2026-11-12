#!/usr/bin/env python3
"""churn.py - start HARC workers on this machine and make them fail on a schedule.

Each worker, once it has registered with the master, runs for a lifetime drawn from a
Weibull(k, theta) distribution with the requested mean, is then killed (SIGKILL, or SIGSTOP to mimic a hang), stays away for
--down seconds and is started again as a fresh session.  With --no-churn the workers
simply run until this script is stopped.  Stop it with SIGTERM or Ctrl-C.

example:
  python3 tools/churn.py --workers n2w1:7101,n2w2:7102 --ip 10.1.1.5 --master 10.1.1.4:7000 \
      --model models/stories42M.bin --tokenizer models/tokenizer.bin \
      --k 0.5 --mean 60 --down 10 --seed 1 --logdir runs/test/rep1
"""
import argparse, math, os, random, signal, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--workers", required=True, help="name:port,name:port,...")
ap.add_argument("--ip", required=True, help="address the master uses to reach these workers")
ap.add_argument("--master", required=True, help="master ip:port")
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--bin", default="build/worker")
ap.add_argument("--k", type=float, default=1.0, help="Weibull shape")
ap.add_argument("--mean", type=float, default=120.0, help="mean session length (s)")
ap.add_argument("--down", type=float, default=10.0, help="time a failed worker stays away (s)")
ap.add_argument("--signal", choices=["KILL", "STOP"], default="KILL")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--no-churn", action="store_true")
ap.add_argument("--rng", default="counter", choices=["counter", "stateful"])
ap.add_argument("--hb-ms", type=int, default=500)
ap.add_argument("--logdir", default=".")
ap.add_argument("--tag", default="harc", help="label so a remote pkill can find this script")
a = ap.parse_args()

os.makedirs(a.logdir, exist_ok=True)
theta = a.mean / math.gamma(1.0 + 1.0 / a.k)
stop = threading.Event()
lock = threading.Lock()
procs = {}
log = open(os.path.join(a.logdir, "churn_%s.csv" % a.ip.replace(".", "_")), "a")
if log.tell() == 0:
    log.write("time,worker,event,value\n")


def note(name, event, value=""):
    with lock:
        log.write("%.6f,%s,%s,%s\n" % (time.time(), name, event, value))
        log.flush()


def start(name, port):
    """start a worker; returns (process, event set once it has registered with the master)"""
    out = open(os.path.join(a.logdir, "worker_%s.log" % name), "a")
    cmd = [a.bin, name, a.ip, str(port), a.master, a.model, a.tokenizer, "--hb-ms", str(a.hb_ms),
           "--log", os.path.join(a.logdir, "work_%s.csv" % name), "--rng", a.rng]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    ready = threading.Event()

    def pump():
        for line in p.stdout:
            out.write(line); out.flush()
            if "registered with master" in line:
                ready.set()
        out.close()
    threading.Thread(target=pump, daemon=True).start()
    with lock:
        procs[name] = p
    note(name, "start", p.pid)
    return p, ready


def life(name, port, idx):
    rnd = random.Random(a.seed * 1000 + idx)
    while not stop.is_set():
        p, ready = start(name, port)
        if a.no_churn:
            stop.wait()
            break
        t = rnd.weibullvariate(theta, a.k)
        while not ready.wait(0.2):              # the session (and its lifetime) starts at registration
            if stop.is_set() or p.poll() is not None:
                break
        note(name, "lifetime", "%.3f" % t)
        if stop.wait(t):
            break
        if a.signal == "STOP":
            p.send_signal(signal.SIGSTOP)
            note(name, "stop")
            if stop.wait(a.down):
                break
            p.kill()
        else:
            p.kill()
            note(name, "kill")
            if stop.wait(a.down):
                p.wait()
                break
        p.wait()


def shutdown(*_):
    stop.set()


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
ws = [w.split(":") for w in a.workers.split(",")]
print("churn: %d workers, %s, k=%.2f mean=%.0fs (theta=%.1fs) down=%.0fs signal=%s" %
      (len(ws), "no churn" if a.no_churn else "Weibull", a.k, a.mean, theta, a.down, a.signal), flush=True)
ts = [threading.Thread(target=life, args=(n, int(p), i), daemon=True) for i, (n, p) in enumerate(ws)]
for t in ts:
    t.start()
while not stop.is_set():
    stop.wait(0.5)
with lock:
    for name, p in procs.items():
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGCONT)
                p.terminate()
            except ProcessLookupError:
                pass
for name, p in list(procs.items()):
    try:
        p.wait(timeout=3)
    except subprocess.TimeoutExpired:
        p.kill()
note("-", "end")
print("churn: stopped", flush=True)
