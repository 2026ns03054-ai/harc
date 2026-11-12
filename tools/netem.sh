#!/usr/bin/env bash
# netem.sh - emulate an edge link on this VM's network interface (needs sudo)
#   sudo bash tools/netem.sh on  10mbit 20ms     # limit bandwidth and add one-way delay
#   sudo bash tools/netem.sh off                 # remove it
#   bash tools/netem.sh show
# The interface is the one that reaches the virtual network (detected automatically).
set -e
IF=${IF:-$(ip -o route get 10.1.1.1 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')}
IF=${IF:-eth0}
case "$1" in
  on)   tc qdisc replace dev "$IF" root netem delay "${3:-20ms}" rate "${2:-10mbit}"
        echo "netem on $IF: rate ${2:-10mbit}, delay ${3:-20ms}" ;;
  off)  tc qdisc del dev "$IF" root 2>/dev/null || true; echo "netem off on $IF" ;;
  show) tc qdisc show dev "$IF" ;;
  *)    echo "usage: $0 on <rate> <delay> | off | show"; exit 1 ;;
esac
