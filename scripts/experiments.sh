#!/usr/bin/env bash
# experiments.sh - the full evaluation plan of the paper (Section VII, Table III).
#
#   bash scripts/experiments.sh check        # one short run per policy, to test the setup
#   bash scripts/experiments.sh rq1          # scaling, no churn
#   bash scripts/experiments.sh rq2          # recovery cost under churn (Table IV)
#   bash scripts/experiments.sh rq3          # risk-matched placement
#   bash scripts/experiments.sh rq4          # discovery in the overlay (runs on this VM only)
#   bash scripts/experiments.sh rq5          # exact resumption, counter vs stateful RNG
#   bash scripts/experiments.sh ref          # failure-free reference runs used by RQ2, RQ3 and RQ5
#   bash scripts/experiments.sh all          # everything, in the order ref rq2 rq3 rq5 rq1 rq4
#
# Settings come from cluster.env (copy cluster.env.example).  Run it on the master VM inside
# tmux or with nohup, because it takes hours.  Repetitions are the outer loop and policies the
# inner loop in a shuffled order, so slow drifts (for example burstable CPU credits) affect
# every policy alike.  Afterwards: python3 tools/analyze.py runs
set -e
cd "$(dirname "$0")/.."
[ -f cluster.env ] && source cluster.env
MASTER_IP=${MASTER_IP:-127.0.0.1}
HOSTS=${HOSTS:-local:3}                 # workers for RQ2, RQ3, RQ5
SSH_KEY=${SSH_KEY:-}
SSH_USER=${SSH_USER:-azureuser}
REPS=${REPS:-5}
MODEL=${MODEL:-models/stories42M.bin}
MEAN=${MEAN:-60}                        # mean worker session (s)
DOWN=${DOWN:-10}                        # time a failed worker stays away (s)
N=${N:-40}                              # requests per run
MIX=${MIX:-256:0.5,512:0.5}
DELAY=${DELAY:-20ms}
RQ1_HOSTS=${RQ1_HOSTS:-"1=local:1 2=local:2"}

COMMON=(--master-ip "$MASTER_IP" --ssh-user "$SSH_USER" --model "$MODEL" --mean "$MEAN" --down "$DOWN")
[ -n "$SSH_KEY" ] && COMMON+=(--ssh-key "$SSH_KEY")

run() { python3 tools/run.py "${COMMON[@]}" "$@"; }
shuffled() { python3 -c "import random,sys; r=random.Random(int(sys.argv[1])); x=sys.argv[2:]; r.shuffle(x); print(' '.join(x))" "$@"; }

ref() {   # same requests (seeds) as the churn runs, without failures
  for rep in $(seq 1 "$REPS"); do
    run --exp ref --config main    --policy harc --no-churn --hosts "$HOSTS" --n "$N" --mix "$MIX" --rep-start "$rep"
    run --exp ref --config mix     --policy harc --no-churn --hosts "$HOSTS" --n "$N" --mix 64:0.5,512:0.5 --rep-start "$rep"
    run --exp ref --config stateful --policy harc --no-churn --hosts "$HOSTS" --n "$N" --mix "$MIX" --rng stateful --rep-start "$rep"
  done
}

rq2() {
  for rep in $(seq 1 "$REPS"); do
    for k in 0.5 1.0; do
      for bw in 10 1000; do
        for pol in $(shuffled "$rep$k$bw" restart yd contkv harc); do
          run --exp rq2 --config "k${k}_bw${bw}" --policy "$pol" --hosts "$HOSTS" --k "$k" --n "$N" --mix "$MIX" \
              --netem "${bw}mbit $DELAY" --bw-mbps "$bw" --rep-start "$rep"
        done
      done
    done
  done
}

rq3() {
  for rep in $(seq 1 "$REPS"); do
    for k in 0.5 0.7 1.0; do
      for pol in $(shuffled "$rep$k" random harc); do
        run --exp rq3 --config "mix_k${k}" --policy "$pol" --hosts "$HOSTS" --k "$k" --n "$N" --mix 64:0.5,512:0.5 --rep-start "$rep"
      done
    done
  done
}

rq5() {
  for rep in $(seq 1 "$REPS"); do
    run --exp rq5 --config counter  --policy harc --hosts "$HOSTS" --k 0.5 --n "$N" --mix "$MIX" --rng counter  --rep-start "$rep"
    run --exp rq5 --config stateful --policy harc --hosts "$HOSTS" --k 0.5 --n "$N" --mix "$MIX" --rng stateful --rep-start "$rep"
  done
}

rq1() {   # RQ1_HOSTS lists "workers=hosts" pairs, e.g. "1=10.1.1.5:1 2=10.1.1.5:1,10.1.1.6:1 ..."
  for m in ${RQ1_MODELS:-$MODEL}; do
    tag=$(basename "$m" .bin)
    for rep in $(seq 1 "$REPS"); do
      for spec in $RQ1_HOSTS; do
        n=${spec%%=*}; h=${spec#*=}
        python3 tools/run.py "${COMMON[@]}" --model "$m" --exp rq1 --config "${tag}_n${n}" --policy harc --no-churn \
            --hosts "$h" --n "${RQ1_N:-48}" --mix 256:1 --rep-start "$rep"
      done
    done
  done
}

rq4() { python3 tools/discovery.py --reps "$REPS" --out runs/rq4/discovery.csv; }

check() {
  for pol in restart yd contkv random harc; do
    run --exp check --config quick --policy "$pol" --hosts "$HOSTS" --k 0.5 --mean 30 --n 8 --mix 128:1
  done
  python3 tools/analyze.py runs --out analysis_check >/dev/null && echo "check finished: see analysis_check/summary.md"
}

case "${1:-}" in
  check|ref|rq1|rq2|rq3|rq4|rq5) "$1" ;;
  all) ref; rq2; rq3; rq5; rq1; rq4 ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
