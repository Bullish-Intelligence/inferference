#!/usr/bin/env bash
# GATE — ci/runner/gpu-lease.sh places and serializes correctly.
#
# "Two concurrent leases with budgets that fit
# land on different cards; two that do not, serialize. Assert by the card each
# child actually saw, read from CUDA_VISIBLE_DEVICES inside the child, not from
# the parent's intent."
#
# So every assertion below reads a file the CHILD wrote. The parent's placement
# log is never the evidence.
#
#   ci/gates/gpu_lease_gate.sh [<workdir>]
#
# Case B forces the free-VRAM figures with GPU_LEASE_FREE_OVERRIDE. Two idle
# 3060s cannot be filled to within a budget of full reproducibly, and no unit is
# installed to fill them. The forced value is the only forced input;
# ranking, locking, placement and the exit-code path below it are shipped code.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"       # ci/
lease="${here}/runner/gpu-lease.sh"
work="${1:-$(mktemp -d /tmp/gpu-lease-gate.XXXXXX)}"
mkdir -p "$work"
lockdir="${work}/locks"; mkdir -p "$lockdir"
export GPU_LEASE_LOCK_DIR="$lockdir"
export GPU_LEASE_POLL=1

fail=0
_ok()   { echo "   PASS  $*"; }
_bad()  { echo "   FAIL  $*"; fail=1; }

# The child. It records the card IT saw and the interval it ran for, so the
# parent can prove both placement and overlap from the child's own view.
cat > "${work}/child.sh" <<'CHILD'
#!/usr/bin/env bash
set -euo pipefail
out="$1"; hold="$2"
start=$(date +%s.%N)
sleep "$hold"
end=$(date +%s.%N)
printf 'cvd=%s ci_gpu=%s start=%s end=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" "${LLAMA_CI_GPU:-unset}" "$start" "$end" > "$out"
CHILD
chmod +x "${work}/child.sh"

_field() { sed -n "s/.*\b$2=\([^ ]*\).*/\1/p" "$1"; }

# --- case A: two budgets that fit -> two cards, concurrently -------------------
echo ">> case A: two 2000 MiB leases on the real rig"
rm -f "${work}/a0.txt" "${work}/a1.txt"
"$lease" 2000 -- "${work}/child.sh" "${work}/a0.txt" 3 >"${work}/a0.log" 2>&1 &
p0=$!
"$lease" 2000 -- "${work}/child.sh" "${work}/a1.txt" 3 >"${work}/a1.log" 2>&1 &
p1=$!
wait $p0; ra=$?
wait $p1; rb=$?

if [[ $ra -ne 0 || $rb -ne 0 ]]; then
  _bad "a lease exited non-zero (${ra}, ${rb}); see ${work}/a0.log ${work}/a1.log"
else
  c0=$(_field "${work}/a0.txt" cvd); c1=$(_field "${work}/a1.txt" cvd)
  r0=$(_field "${work}/a0.txt" ci_gpu); r1=$(_field "${work}/a1.txt" ci_gpu)
  [[ "$c0" != "$c1" ]] && _ok "different cards: child saw ${c0} and ${c1}" \
                       || _bad "both children saw card ${c0}"
  [[ "$c0" == "$r0" && "$c1" == "$r1" ]] \
    && _ok "CUDA_VISIBLE_DEVICES == LLAMA_CI_GPU in both children (${c0}, ${c1})" \
    || _bad "role variable disagrees with CUDA_VISIBLE_DEVICES: ${c0}/${r0}, ${c1}/${r1}"
  s0=$(_field "${work}/a0.txt" start); e0=$(_field "${work}/a0.txt" end)
  s1=$(_field "${work}/a1.txt" start); e1=$(_field "${work}/a1.txt" end)
  if awk -v s0="$s0" -v e0="$e0" -v s1="$s1" -v e1="$e1" 'BEGIN{exit !(s0<e1 && s1<e0)}'; then
    _ok "the two children overlapped in time — they really ran concurrently"
  else
    _bad "no overlap: ${s0}..${e0} and ${s1}..${e1}. They serialized when they should not have"
  fi
fi

# --- case B: budgets only one card can hold -> serialize on that card ----------
# Forced free: card 0 has 6174 MiB (a ~3B server resident), card 1 has 5112
# (a ~9B server resident). A 6000 MiB job fits
# card 0 and no other, so two of them must serialize on card 0.
echo ">> case B: two 6000 MiB leases, free forced to the two-server table"
export GPU_LEASE_FREE_OVERRIDE="0=6174,1=5112"
rm -f "${work}/b0.txt" "${work}/b1.txt"
"$lease" 6000 -- "${work}/child.sh" "${work}/b0.txt" 3 >"${work}/b0.log" 2>&1 &
q0=$!
sleep 0.5
"$lease" 6000 -- "${work}/child.sh" "${work}/b1.txt" 3 >"${work}/b1.log" 2>&1 &
q1=$!
wait $q0; rc0=$?
wait $q1; rc1=$?
unset GPU_LEASE_FREE_OVERRIDE

if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
  _bad "a lease exited non-zero (${rc0}, ${rc1}); see ${work}/b0.log ${work}/b1.log"
else
  d0=$(_field "${work}/b0.txt" cvd); d1=$(_field "${work}/b1.txt" cvd)
  [[ "$d0" == "0" && "$d1" == "0" ]] \
    && _ok "both children landed on card 0, the only card that fits" \
    || _bad "expected both on card 0, children saw ${d0} and ${d1}"
  t0=$(_field "${work}/b0.txt" start); u0=$(_field "${work}/b0.txt" end)
  t1=$(_field "${work}/b1.txt" start); u1=$(_field "${work}/b1.txt" end)
  if awk -v s0="$t0" -v e0="$u0" -v s1="$t1" -v e1="$u1" 'BEGIN{exit !(s0<e1 && s1<e0)}'; then
    _bad "the two children overlapped: ${t0}..${u0} and ${t1}..${u1}. The lock did not serialize them"
  else
    _ok "no overlap — the second lease waited for the first"
  fi
fi

# --- case C: the exit code is the child's -------------------------------------
echo ">> case C: exit-code propagation"
set +e
"$lease" 1000 -- bash -c 'exit 42' >/dev/null 2>&1; rcx=$?
set -e
[[ $rcx -eq 42 ]] && _ok "child exit 42 propagated" || _bad "expected 42, got ${rcx}"

# --- case D: an unsatisfiable budget fails loudly, it does not hang ------------
echo ">> case D: a budget no card can ever hold"
set +e
out=$("$lease" 99000000 -- true 2>&1); rcy=$?
set -e
if [[ $rcy -eq 2 ]] && grep -q "No card can ever satisfy it" <<<"$out"; then
  _ok "refused loudly with rc=2"
else
  _bad "expected rc=2 and a loud refusal, got rc=${rcy}: ${out}"
fi

# --- case E: gpu-serialized.sh still works, and still pins a card --------------
echo ">> case E: gpu-serialized.sh back-compat"
rm -f "${work}/e0.txt"
set +e
"${here}/runner/gpu-serialized.sh" "${work}/child.sh" "${work}/e0.txt" 1 >"${work}/e0.log" 2>&1; rce=$?
set -e
if [[ $rce -eq 0 ]] && [[ -f "${work}/e0.txt" ]]; then
  ec=$(_field "${work}/e0.txt" cvd)
  [[ "$ec" =~ ^[0-9]+$ ]] && _ok "ran and pinned card ${ec}" || _bad "child saw CUDA_VISIBLE_DEVICES='${ec}'"
else
  _bad "gpu-serialized.sh exited ${rce}; see ${work}/e0.log"
fi

set +e
LLAMA_CI_GPU_LOCK=/tmp/legacy.lock "${here}/runner/gpu-serialized.sh" true >/dev/null 2>&1; rcl=$?
set -e
[[ $rcl -eq 2 ]] && _ok "a retired LLAMA_CI_GPU_LOCK is refused, not ignored" \
                 || _bad "expected rc=2 for LLAMA_CI_GPU_LOCK, got ${rcl}"

# --- case H: the wait is BOUNDED, and it names the reason ---------------------
# Option 5. With an always-on server the old wait-forever default is a hang, not
# a queue: a resident server never releases its card on its own, because
# preemption is not implemented. The lease must give up and say what to stop.
echo ">> case H: a budget no resident card can offer fails, and explains itself"
start=$(date +%s)
set +e
outh=$(GPU_LEASE_FREE_OVERRIDE="0=1000,1=1000" GPU_LEASE_TIMEOUT=6 GPU_LEASE_POLL=2 \
       timeout 60 "$lease" 9000 -- true 2>&1)
rch=$?
set -e
elapsed=$(( $(date +%s) - start ))
if [[ $rch -eq 124 ]]; then
  _bad "the lease hung instead of timing out"
elif [[ $rch -ne 2 ]]; then
  _bad "expected rc=2 on timeout, got ${rch}: ${outh}"
else
  _ok "timed out with rc=2 after ${elapsed}s (bound 6s)"
  grep -q "something resident is holding it" <<<"$outh" \
    && _ok "named the cards that were short of VRAM, not just 'waiting'" \
    || _bad "no per-card reason in the wait message: ${outh}"
  grep -q "preemption is not implemented" <<<"$outh" \
    && _ok "said a resident server will not yield on its own" \
    || _bad "did not say the wait cannot resolve itself"
  grep -qi "stop the server\|systemctl --user stop" <<<"$outh" \
    && _ok "told the operator what to do" \
    || _bad "no actionable remedy in the refusal: ${outh}"
fi

# The default must be BOUNDED. A regression to 0 reintroduces the hang.
grep -qE 'GPU_LEASE_TIMEOUT:-[1-9][0-9]*\}' "$lease" \
  && _ok "the default timeout is non-zero in the script" \
  || _bad "GPU_LEASE_TIMEOUT defaults to 0 (unbounded) — that is the hang this case exists for"

# --- case G: a nested lease reuses the outer one instead of deadlocking --------
# A gate script can call gpu-serialized.sh, so any gate battery run under a
# lease nests one. Before the fix that blocked on a lock its own ancestor held,
# forever. The child must see the OUTER card and the whole thing must finish.
echo ">> case G: a lease inside a lease"
rm -f "${work}/g0.txt"
set +e
timeout 60 "$lease" 2000 -- "$lease" 1000 -- "${work}/child.sh" "${work}/g0.txt" 1 >"${work}/g0.log" 2>&1
rcg=$?
set -e
if [[ $rcg -eq 124 ]]; then
  _bad "the nested lease deadlocked (timed out); see ${work}/g0.log"
elif [[ $rcg -ne 0 ]]; then
  _bad "the nested lease exited ${rcg}; see ${work}/g0.log"
else
  ng=$(_field "${work}/g0.txt" cvd)
  outer=$(sed -n 's/^>> \[gpu-lease\] card \([0-9]*\) .*/\1/p' "${work}/g0.log" | head -1)
  [[ -n "$ng" && "$ng" == "$outer" ]] \
    && _ok "nested child ran on the outer lease's card ${ng}, no second lock" \
    || _bad "outer leased card '${outer}', nested child saw '${ng}'"
  grep -q 'nested on card' "${work}/g0.log" \
    && _ok "the nested call said so rather than locking silently" \
    || _bad "no nested-lease line in the log"
fi

# A nested budget the leased card cannot hold must fail loudly, not over-commit.
set +e
outg=$(GPU_LEASE_CARD=0 GPU_LEASE_BUDGET_MIB=1000 GPU_LEASE_FREE_OVERRIDE="0=500" \
       timeout 30 "$lease" 4000 -- true 2>&1)
rcn=$?
set -e
if [[ $rcn -eq 2 ]] && grep -q 'nested lease asks' <<<"$outg"; then
  _ok "an over-budget nested lease refuses with rc=2"
else
  _bad "expected rc=2 for an over-budget nested lease, got rc=${rcn}: ${outg}"
fi

# --- case F: the NEGATIVE CONTROL ---------------------------------------------
# Case A must be able to fail, or it proves nothing. Run case A's own
# check against the behaviour this change replaces — one exclusive rig-wide flock
# and one pinned card — and require that it FAILS there.
echo ">> case F: negative control, case A against the pre-lease behaviour"
cat > "${work}/legacy.sh" <<LEGACY
#!/usr/bin/env bash
set -euo pipefail
source "${here}/runner/rig-env.sh"
exec 9>"${lockdir}/legacy-rigwide.lock"
flock -x 9
exec "\$@"
LEGACY
chmod +x "${work}/legacy.sh"
rm -f "${work}/f0.txt" "${work}/f1.txt"
"${work}/legacy.sh" "${work}/child.sh" "${work}/f0.txt" 2 >/dev/null 2>&1 &
n0=$!
"${work}/legacy.sh" "${work}/child.sh" "${work}/f1.txt" 2 >/dev/null 2>&1 &
n1=$!
wait $n0 || true; wait $n1 || true
g0=$(_field "${work}/f0.txt" cvd); g1=$(_field "${work}/f1.txt" cvd)
h0=$(_field "${work}/f0.txt" start); i0=$(_field "${work}/f0.txt" end)
h1=$(_field "${work}/f1.txt" start); i1=$(_field "${work}/f1.txt" end)
overlap=$(awk -v s0="$h0" -v e0="$i0" -v s1="$h1" -v e1="$i1" 'BEGIN{print (s0<e1 && s1<e0) ? "yes" : "no"}')
if [[ "$g0" == "$g1" && "$overlap" == "no" ]]; then
  _ok "the old behaviour puts both children on card ${g0} and serializes them — case A discriminates"
else
  _bad "the negative control did not reproduce the old behaviour: cards ${g0}/${g1}, overlap ${overlap}"
fi

echo
if [[ $fail -eq 0 ]]; then echo "[gate] PASS  gpu-lease  (artefacts: ${work})"; else echo "[gate] FAIL  gpu-lease  (artefacts: ${work})"; fi
exit $fail
