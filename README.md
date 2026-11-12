# HARC: Hazard-Aware Resumable Commit for LLM inference on churning edge clusters

This repository holds the prototype and the experiment scripts for the paper
*HARC: Hazard-Aware Resumable Commit for Large Language Model Inference on Churning Edge Micro-Clusters*.

- **HyPEC** is a two-tier system. Requesters and cluster masters form an unstructured peer-to-peer overlay, and the masters act as super-peers. Each master runs a micro-cluster of CPU workers.
- **HARC** is the master's scheduler. It fits a Weibull model to the workers' session lengths, sends long requests to low-hazard workers (Lemma 1), and sets each commit interval from the worker's current hazard (Eq. 8). It picks restart, token-log recovery or KV-checkpoint recovery for each request (Eqs. 9–11), and resumes interrupted requests with exactly the output they would have had (Proposition 2).

Everything is written in C with POSIX sockets and threads. The worker runs [llama2.c](https://github.com/karpathy/llama2.c) (MIT licence), extended with:

- batched prompt processing that is bit-identical to the original;
- counter-based sampling;
- KV-cache export and import.

## Layout

| Path | What it is |
|---|---|
| `src/engine.c`, `src/engine.h` | inference engine on top of llama2.c: batched prefill, counter-based sampler, KV export/import |
| `src/master.c` | micro-cluster master: registry, heartbeats, Weibull fit, HARC and the baseline policies B1–B4 |
| `src/worker.c` | worker: resumes from a committed prefix or a KV checkpoint, streams COMMIT/KV messages |
| `src/client.c` | workload generator (closed loop or Poisson arrivals) |
| `src/peer.c` | HyPEC overlay node: HELLO/BYE, QUERY flooding with TTL and duplicate suppression, HIT with load, Eq. 5 |
| `src/test_engine.c` | exactness tests and the prefill speed-up measurement |
| `src/net.h` | sockets, line protocol, logging (adapted from the CCZG 526 Lab II prototype) |
| `tools/run.py` | runs one experiment configuration end to end and collects all logs |
| `tools/churn.py` | starts workers and kills them at Weibull-distributed times, then restarts them |
| `tools/discovery.py` | RQ4: overlay discovery experiment with 8–64 peer processes |
| `tools/netem.sh` | link emulation with `tc netem` |
| `tools/analyze.py` | metrics, 95% confidence intervals, Welch t-tests, Table IV, Figs. 3 and 4 |
| `scripts/experiments.sh` | the full evaluation plan (RQ1–RQ5) |
| `third_party/llama2.c/` | `run.c`, `tokenizer.bin` and the licence of llama2.c |

## Policies

| `--policy` | Paper name | Behaviour |
|---|---|---|
| `restart` | B1 Restart | no commits; a failed request starts again from the prompt |
| `yd` | B2 Fixed Young/Daly | token commits every √(2cμ) seconds, from the mean lifetime |
| `contkv` | B3 Continuous KV | KV cache streamed to the master after every token |
| `random` | B4 Random placement | HARC commits and recovery, workers chosen at random, requests in arrival order |
| `harc` | HARC | Weibull hazard, risk-matched placement, adaptive commits, per-request recovery mode |

## Build and test

Ubuntu 24.04 with gcc and Python 3 is enough.

```bash
sudo apt update && sudo apt install -y build-essential python3 git wget
git clone <this repository> harc && cd harc
make
bash tools/get_models.sh              # stories15M.bin and stories42M.bin from Hugging Face
./build/test_engine models/stories42M.bin models/tokenizer.bin 120 40 256
```

`test_engine` checks the properties that HARC relies on:

1. Batched prefill gives the same output as token-by-token processing.
2. Resuming from a token log on a fresh process gives the same output as an uninterrupted run.
3. Resuming from a KV checkpoint gives the same output.
4. Resuming from a KV checkpoint that lags the token log gives the same output.

It then prints the prefill speed-up of batched over sequential processing. This number goes in Section V of the paper.

`make PORTABLE=1` builds without CPU-specific instructions and without FMA contraction. Use it when workers run on different CPU models (RQ5).

## Quick local run (one machine)

```bash
python3 tools/run.py --exp demo --config k0.5 --policy harc --hosts local:2 \
    --model models/stories15M.bin --k 0.5 --mean 20 --down 3 --n 12 --mix 128:0.5,200:0.5
python3 tools/analyze.py runs --out analysis_demo && cat analysis_demo/summary.md
```

## Running the experiments on Azure

The scripts assume the three VMs from the pilot study:

- node1 (10.1.1.4) runs the master, the client and the overlay experiment;
- node2 (10.1.1.5) and node3 (10.1.1.6) run two workers each.

1. **Start the VMs** in the Azure portal. Standard_B2ats_v2 is burstable: under sustained load it runs out of CPU credits and slows down. Check *CPU Credits Remaining* in the VM's *Metrics* page. If your subscription allows it, resize to a non-burstable 2-vCPU size for the experiment days. Either way, `analyze.py` reports the decode speed measured at each worker registration (`td_ms`), so you can see whether throttling occurred.
2. **On every VM**, run the build and test steps above, with the repository in `~/harc`.
3. **On node1**, give the master SSH access to the workers. From your laptop, run `scp -i dc-key.pem dc-key.pem azureuser@<node1-public-ip>:~/.ssh/`. Then, on node1:
   ```bash
   chmod 600 ~/.ssh/dc-key.pem
   ssh -i ~/.ssh/dc-key.pem azureuser@10.1.1.5 hostname    # answer "yes" once; repeat for 10.1.1.6
   ```
4. **On node1**, create the configuration and do a short check run:
   ```bash
   cp cluster.env.example cluster.env      # edit if your IPs or user name differ
   bash scripts/experiments.sh check       # about 3 minutes, one short run per policy
   ```
5. **Run the plan** inside `tmux`, so that it keeps running if your SSH session drops:
   ```bash
   tmux new -s exp
   bash scripts/experiments.sh all 2>&1 | tee experiments.log
   # detach with Ctrl-B then D; reattach later with: tmux attach -t exp
   ```
   You can also run the parts one at a time (`ref`, `rq2`, `rq3`, `rq5`, `rq1`, `rq4`). `REPS` in `cluster.env` sets the number of repetitions.
6. **Analyse:** run `python3 tools/analyze.py runs`. It writes `analysis/summary.md`, `summary.csv`, `runs.csv`, `fig3.png` and `fig4.png`; install `python3-matplotlib` for the figures. Pack everything with `tar czf harc-results.tar.gz runs analysis experiments.log`.

### What each experiment measures

| RQ | Command | Varies | Main metrics |
|---|---|---|---|
| RQ1 | `rq1` | 1–6 workers, 15M and 42M models, no churn | tokens/s, p50/p95 latency, S(n), E(n), f̂ |
| RQ2 | `rq2` | Weibull k ∈ {0.5, 1}, 10 Mb/s or 1 Gb/s with 20 ms delay | p95 latency, recomputed tokens, success rate |
| RQ3 | `rq3` | short/long mix, k ∈ {0.5, 0.7, 1} | lost work per request, p95 latency |
| RQ4 | `rq4` | 8–64 peers, TTL 2–5, 3 clusters, HyPEC vs flat | messages per query, hit rate, best-cluster rate, selection latency |
| RQ5 | `rq5` (+ `ref`) | counter vs stateful RNG under churn | share of resumed requests identical to the failure-free run |

The `ref` runs serve the same requests without failures. RQ2, RQ3 and RQ5 use them as the reference outputs for the exactness check.

## Protocol (one text line per message)

```
worker -> master   REGISTER name ip port tp_ms td_ms kv_bytes_per_pos | HB name | BYE name
master -> worker   TASK rid L seed k_c mode n_prefix kv_pos kv_bytes attempt, PROMPT text, <prefix tokens>, <KV bytes>
worker -> master   COMMIT rid g n t1..tn | KV rid a b nbytes + bytes | DONE rid G hash tp td commit_ms n_commits + tokens
client -> master   JOB cid L seed, PROMPT text            -> RESULT cid G hash failures attempts latency_ms queue_ms
peer   <-> peer    HELLO | BYE | QUERY id ttl origin_ip origin_port sender | HIT id cluster Q W td
anyone -> master   STATUS | LOAD
```

## Licence

Our code is released under the MIT licence (see `LICENSE`). `third_party/llama2.c` is © Andrej Karpathy and is also under the MIT licence; see `third_party/llama2.c/LICENSE`. The TinyStories model checkpoints are downloaded from the llama2.c release on Hugging Face and are not included here.
