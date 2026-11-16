#!/usr/bin/env bash
# One measured load run (ADR 0013), for `make load-measured`. A run counts only when the
# host stayed awake and started from the same state, so this script:
#   1. keeps the host awake for the whole run: it re-runs itself under `caffeinate -i -s`;
#   2. starts from `make load-reset` (conversations cleared, fresh backend) and stops when
#      the reset fails;
#   3. records swap, memory and paging before the reset, at start and at end;
#   4. voids the run when pmset logged a Sleep, Wake or DarkWake between the first
#      snapshot and the last.
# Usage (from the repo root): measured_run.sh <run-name> [VAR=value ...]; the variables go
# to `make load`. The log is loadtest/results/<run-name>.runner.log, never overwritten.
# Exit: 0 counted, 1 reset failed, 2 usage or log exists, 3 host slept (run void),
# otherwise the exit code of `make load`.
set -uo pipefail

if (($# < 1)) || [[ -z $1 ]]; then
	echo "usage: measured_run.sh <run-name> [VAR=value ...]" >&2
	exit 2
fi
run=$1
shift
make_cmd=${MAKE:-make}
log=loadtest/results/$run.runner.log
if [[ -e $log ]]; then
	echo "$log exists: a run's records are never mixed with another's" >&2
	exit 2
fi
if [[ -z ${MEASURED_RUN_CAFFEINATED:-} ]]; then
	MEASURED_RUN_CAFFEINATED=1 exec caffeinate -i -s "$0" "$run" "$@"
fi

mkdir -p "$(dirname "$log")"

now() { date '+%Y-%m-%d %H:%M:%S'; }

snapshot() {
	echo "[$1] $(now)"
	sysctl vm.swapusage
	memory_pressure | grep 'free percentage'
	vm_stat | grep -E 'Pageins|Pageouts|Swapins|Swapouts'
}

# pmset -g log lines: date time zone event ... ; "Wake Requests" are schedules, not wakes.
sleep_events() {
	pmset -g log | awk -v from="$1" -v to="$2" '
		($4 == "Sleep" || $4 == "DarkWake" || ($4 == "Wake" && $5 != "Requests")) {
			t = $1 " " $2
			if (t >= from && t <= to) print
		}'
}

measured_run() {
	local from to rc events
	from=$(now)
	snapshot pre-reset
	if ! "$make_cmd" load-reset; then
		echo "reset failed: run not started"
		return 1
	fi
	snapshot start
	"$make_cmd" load LOAD_RUN="$run" "$@"
	rc=$?
	snapshot end
	to=$(now)
	echo "make load rc=$rc window $from .. $to"
	echo "--- pmset sleep/wake events in window"
	events=$(sleep_events "$from" "$to")
	echo "${events:-none}"
	echo "--- end pmset"
	if [[ -n $events ]]; then
		echo "run void: the host slept or woke inside the run window; rerun it"
		return 3
	fi
	return "$rc"
}

measured_run "$@" 2>&1 | tee "$log"
