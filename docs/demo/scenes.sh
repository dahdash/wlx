#!/usr/bin/env bash
# Demo video: one function per scene, run from the release root.
#
#   bash docs/demo/scenes.sh 3          # show scene 3
#   DRY=1 bash docs/demo/scenes.sh all  # check every scene without opening windows
#
# STATE points at the engines' state directory (not part of the repo; default: the
# operator's checkout next to this one). PY is the Python with the package installed.
set -euo pipefail
cd "$(dirname "$0")/../.."
STATE=${STATE:-../wallex/data/state}
PY=${PY:-python}
DRY=${DRY:-0}

show() {  # open a file in the browser (or just check it exists in DRY mode)
  [[ -e $1 ]] || { echo "missing: $1" >&2; exit 1; }
  if [[ $DRY == 1 ]]; then echo "open $1"; else xdg-open "$1" >/dev/null 2>&1 & fi
}

run() { echo "\$ $*"; "$@"; }

scene1() { show docs/demo/dashboard.html; }

scene2() {
  show docs/demo/dashboard.html
  run sed -n '1,30p' RESULTS.md
}

scene3() {
  show README.md
  run sed -n '15,20p' src/wallex/strategy/g.py
}

scene4() {
  show docs/backtest/validation.html
  run head -20 docs/backtest/metrics.md
}

scene5() {
  show docs/demo/dashboard.html
  echo "\$ tail -1 decisions.jsonl | jq"
  tail -1 "$STATE/oil-live-1/decisions.jsonl" |
    jq '{ts, why, gate: .gate.verdict, decision: .decision.decision,
         thesis: .decision.thesis, counter: .decision.counter_evidence_ids,
         lesson: .decision.lesson}'
  run sed -n '1,25p' src/wallex/oil/prompts/analyst-v2.txt
  run grep -n 'min_gap_4h\|def ' src/wallex/oil/gate.py
}

scene6() {
  run sed -n '/^## 5\./,/^## 6\./p' README.md
}

scene7() {
  echo "\$ pytest -q"
  if [[ $DRY == 1 ]]; then echo "(skipped in DRY mode)"; else PYTHONPATH=src "$PY" -m pytest -q; fi
  run head -5 MANIFEST.sha256
  run sed -n '1,20p' docs/decision-log.md
}

PS1='$ '
case ${1:-all} in
  all) for i in 1 2 3 4 5 6 7; do echo "=== scene $i ==="; "scene$i"; done ;;
  [1-7]) clear 2>/dev/null || true; "scene$1" ;;
  *) echo "usage: $0 [1-7|all]" >&2; exit 2 ;;
esac
