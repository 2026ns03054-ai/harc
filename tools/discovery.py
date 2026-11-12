#!/usr/bin/env python3
"""discovery.py - RQ4: cost and quality of cluster discovery in the overlay.

Starts P lightweight peer processes on this machine, wires them into either
  hypec : super-peers (the cluster masters plus about one peer in eight) linked to each other,
          every other peer attached to two super-peers; only super-peers forward; HITs carry load
  flat  : baseline B5, a random graph over all peers; every peer forwards; HITs carry no load,
          so the requester takes the first one
and then issues queries from random requesters with random cluster loads.

usage: python3 tools/discovery.py [--peers 8,16,32,64] [--ttl 2,3,4,5] [--clusters 3] [--queries 50]
                                  [--reps 5] [--hop-ms 5] [--delta-ms 200] [--L 256] [--out runs/rq4/discovery.csv]
"""
import argparse, csv, math, os, random, socket, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--peers", default="8,16,32,64")
ap.add_argument("--ttl", default="2,3,4,5")
ap.add_argument("--clusters", type=int, default=3)
ap.add_argument("--queries", type=int, default=50)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--modes", default="hypec,flat")
ap.add_argument("--hop-ms", type=int, default=5)
ap.add_argument("--delta-ms", type=int, default=200)
ap.add_argument("--L", type=int, default=256)
ap.add_argument("--base-port", type=int, default=8100)
ap.add_argument("--out", default="runs/rq4/discovery.csv")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)


def msg(port, line, reply=True):
    for _ in range(50):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            s.sendall((line + "\n").encode())
            data = b""
            while reply and not data.endswith(b"\n"):
                c = s.recv(65536)
                if not c:
                    break
                data += c
            s.close()
            return data.decode().strip()
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("peer on port %d not reachable" % port)


def link(nodes, i, j):
    for x, y in ((i, j), (j, i)):
        msg(nodes[x]["port"], "HELLO %s 127.0.0.1 %d %d" % (nodes[y]["name"], nodes[y]["port"], nodes[y]["super"]))


def random_graph(ids, rnd, chords):
    """ring plus random chords: connected, average degree about 2 + 2*chords"""
    ids = ids[:]
    rnd.shuffle(ids)
    edges = set()
    n = len(ids)
    for k in range(n):
        if n > 1:
            edges.add(tuple(sorted((ids[k], ids[(k + 1) % n]))))
    for k in range(n):
        for _ in range(chords):
            j = rnd.choice(ids)
            if j != ids[k]:
                edges.add(tuple(sorted((ids[k], j))))
    return edges


new = not os.path.exists(a.out)
out = open(a.out, "a", newline="")
w = csv.writer(out)
if new:
    w.writerow(["rep", "mode", "peers", "ttl", "query", "clusters", "messages", "hits", "hit", "all", "best",
                "first_ms", "select_ms", "delta_ms", "hop_ms", "superpeers"])
for rep in range(1, a.reps + 1):
    for mode in a.modes.split(","):
        for P in [int(x) for x in a.peers.split(",")]:
            rnd = random.Random(rep * 1000 + P)
            C = a.clusters
            masters = rnd.sample(range(P), C)
            if mode == "hypec":
                ns = max(C, math.ceil(P / 8))
                supers = masters + rnd.sample([i for i in range(P) if i not in masters], ns - C)
            else:
                supers = []
            nodes = []
            procs = []
            for i in range(P):
                nd = dict(name="p%02d" % i, port=a.base_port + i, super=int(i in supers))
                args = ["build/peer", nd["name"], "127.0.0.1", str(nd["port"]), "--hop-ms", str(a.hop_ms)]
                if i in masters:
                    args += ["--cluster", "C%d" % masters.index(i)]
                elif i in supers:
                    args += ["--super"]
                if mode == "flat":
                    args += ["--flat"]
                nodes.append(nd)
                procs.append(subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            time.sleep(0.3)
            if mode == "hypec":
                edges = random_graph(supers, rnd, 1)
                leaves = [i for i in range(P) if i not in supers]
                for l in leaves:
                    for s_ in rnd.sample(supers, min(2, len(supers))):
                        edges.add(tuple(sorted((l, s_))))
            else:
                edges = random_graph(list(range(P)), rnd, 1)
            for i, j in edges:
                link(nodes, i, j)
            requesters = [i for i in range(P) if i not in supers and i not in masters]
            for ttl in [int(x) for x in a.ttl.split(",")]:
                for q in range(a.queries):
                    loads = {}
                    for ci, mi in enumerate(masters):
                        Q, Wc, td = rnd.randint(0, 8), rnd.randint(1, 6), rnd.randint(20, 40)
                        loads["C%d" % ci] = (Q + 1) * a.L * td / 1000.0 / Wc
                        msg(nodes[mi]["port"], "SETLOAD %d %d %d" % (Q, Wc, td))
                    req = rnd.choice(requesters)
                    before = sum(int(msg(n["port"], "STATS").split()[1]) for n in nodes)
                    r = msg(nodes[req]["port"], "DISCOVER %d %d %d" % (a.L, ttl, a.delta_ms)).split()
                    time.sleep(0.03 + a.hop_ms * ttl / 1000.0)
                    after = sum(int(msg(n["port"], "STATS").split()[1]) for n in nodes)
                    chosen, hits, first, sel = r[2], int(r[3]), float(r[4]), float(r[5])
                    best = min(loads, key=loads.get)
                    w.writerow([rep, mode, P, ttl, q, C, after - before, hits, int(hits > 0), int(hits == C),
                                int(chosen == best), first, sel, a.delta_ms, a.hop_ms, len(supers)])
                out.flush()
                print("rep %d %-5s P=%2d TTL=%d done" % (rep, mode, P, ttl), flush=True)
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()
            time.sleep(0.2)
print("written", a.out)
