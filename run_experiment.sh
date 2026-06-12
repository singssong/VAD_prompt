#!/usr/bin/env bash

# Editable experiment settings.
CATEGORIES=("bottle" "carpet")
REPS_STEP1=15
REPS_STEP2=10
REPS_STEP3=10
REPS_STEP4=15
PER_RUN_TIMEOUT=600
TOTAL_BUDGET=10800
MAX_CONSECUTIVE_FAILS=3

ROOT="${HOME}/vad_pilot"
RESULTS_DIR="${ROOT}/results"
MASTER_LOG="${RESULTS_DIR}/master_log.csv"
EXPERIMENT_START=$(date +%s)

planned_runs=0
for category in "${CATEGORIES[@]}"; do
    for step in 1 2 3 4; do
        case "$step" in
            1) planned_runs=$((planned_runs + REPS_STEP1)) ;;
            2) planned_runs=$((planned_runs + REPS_STEP2)) ;;
            3) planned_runs=$((planned_runs + REPS_STEP3)) ;;
            4) planned_runs=$((planned_runs + REPS_STEP4)) ;;
        esac
    done
done

total_runs=0
passes=0
fails=0
completed=0
skipped=0
progress_index=0
budget_exhausted=false
consecutive_codex_fails=0

mkdir -p "$RESULTS_DIR"
if [[ ! -f "$MASTER_LOG" ]]; then
    printf '%s\n' \
        "category,step,rep,status,duration_sec,image_scores_count,isolation_ok" \
        > "$MASTER_LOG"
fi

reps_for_step() {
    case "$1" in
        1) printf '%s\n' "$REPS_STEP1" ;;
        2) printf '%s\n' "$REPS_STEP2" ;;
        3) printf '%s\n' "$REPS_STEP3" ;;
        4) printf '%s\n' "$REPS_STEP4" ;;
        *) printf '%s\n' "0" ;;
    esac
}

count_image_scores() {
    local scores_file=$1
    python - "$scores_file" <<'PY' 2>/dev/null
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        value = json.load(handle)
    print(len(value) if isinstance(value, (list, dict)) else 0)
except Exception:
    print(0)
PY
}

find_step1_scores_count() {
    local run_dir=$1
    local expected_count=$2
    python - "$run_dir" "$expected_count" <<'PY' 2>/dev/null
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_count = int(sys.argv[2])
preferred = run_dir / "outputs" / "image_scores.json"
files = []
if preferred.is_file():
    files.append(preferred)

try:
    candidates = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}
    )
except OSError:
    candidates = []

files.extend(path for path in candidates if path != preferred)

for path in files:
    try:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            count = len(value) if isinstance(value, (list, dict)) else 0
        else:
            with path.open("r", encoding="utf-8", newline="") as handle:
                sample = handle.read(8192)
                handle.seek(0)
                rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
            has_header = False
            if rows:
                try:
                    has_header = csv.Sniffer().has_header(sample)
                except csv.Error:
                    has_header = False
            count = len(rows) - (1 if has_header else 0)
    except (OSError, UnicodeError, ValueError, csv.Error, json.JSONDecodeError):
        continue

    if count == expected_count:
        print(count)
        break
else:
    print(0)
PY
}

validation_is_pass() {
    local validation_file=$1
    python - "$validation_file" <<'PY' >/dev/null 2>&1
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        result = json.load(handle)
    raise SystemExit(0 if result.get("status") == "PASS" else 1)
except (OSError, ValueError, AttributeError):
    raise SystemExit(1)
PY
}

sample_is_256() {
    local png_file=$1
    python - "$png_file" <<'PY' 2>/dev/null
import struct
import sys

try:
    with open(sys.argv[1], "rb") as handle:
        header = handle.read(24)
    valid = header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) == 24
    width, height = struct.unpack(">II", header[16:24]) if valid else (0, 0)
    print("true" if (width, height) == (256, 256) else "false")
except Exception:
    print("false")
PY
}

for category in "${CATEGORIES[@]}"; do
    for step in 1 2 3 4; do
        reps=$(reps_for_step "$step")

        for ((rep = 1; rep <= reps; rep++)); do
            progress_index=$((progress_index + 1))
            now=$(date +%s)
            elapsed=$((now - EXPERIMENT_START))
            printf '[PROGRESS] Run %s/%s | category=%s step=%s rep=%s | elapsed=%ss | done=%s skipped=%s\n' \
                "$progress_index" "$planned_runs" "$category" "$step" "$rep" \
                "$elapsed" "$completed" "$skipped"

            run_dir="${RESULTS_DIR}/${category}/step${step}/run${rep}"
            validation_file="$run_dir/validation.json"
            if [[ -f "$validation_file" ]] && validation_is_pass "$validation_file"; then
                printf '[SKIP] %s step%s run%s (already PASS)\n' "$category" "$step" "$rep"
                skipped=$((skipped + 1))
                continue
            fi

            if ((elapsed >= TOTAL_BUDGET)); then
                budget_exhausted=true
                break 3
            fi

            prompt_file="${ROOT}/prompts/step${step}.txt"
            run_start=$(date +%s)
            total_runs=$((total_runs + 1))

            # This guard prevents an accidentally empty or malformed path from
            # causing cleanup outside this experiment's results tree.
            expected_prefix="${RESULTS_DIR}/${category}/step${step}/run"
            if [[ "$run_dir" != "${expected_prefix}"* ]]; then
                printf 'Refusing unsafe run directory: %s\n' "$run_dir" >&2
                exit 1
            fi

            rm -rf -- "$run_dir"
            mkdir -p "$run_dir"
            if [[ "$category" == "bottle" ]]; then
                category_data_dir="$ROOT/data"
                category_ground_truth_dir="$ROOT/ground_truth"
            else
                category_data_dir="$ROOT/data_${category}"
                category_ground_truth_dir="$ROOT/ground_truth_${category}"
            fi
            ln -s "$category_data_dir" "$run_dir/data"
            ln -s "$category_ground_truth_dir" "$run_dir/ground_truth"

            expected_count=$(
                find "$run_dir/data/test_images" -maxdepth 1 -type f \
                    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
                    -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \
                    -o -iname '*.webp' \) -printf '.' 2>/dev/null |
                    wc -c
            )

            codex_exit=0
            set +e
            timeout "$PER_RUN_TIMEOUT" \
                codex exec \
                --skip-git-repo-check \
                --ephemeral \
                -C "$run_dir" \
                --dangerously-bypass-approvals-and-sandbox \
                -o "$run_dir/agent_last_message.txt" \
                - < "$prompt_file"
            codex_exit=$?
            set +e

            if [[ ! -f "$run_dir/agent_last_message.txt" ]]; then
                : > "$run_dir/agent_last_message.txt"
            fi

            isolation_ok=true
            while IFS= read -r -d '' py_file; do
                if grep -IqE 'ground_truth|labels\.json' "$py_file"; then
                    isolation_ok=false
                    break
                fi
            done < <(find "$run_dir" -type f -name '*.py' -print0 2>/dev/null)

            image_scores_count=0
            scores_file="$run_dir/outputs/image_scores.json"
            if ((step == 1)); then
                image_scores_count=$(find_step1_scores_count "$run_dir" "$expected_count")
                [[ "$image_scores_count" =~ ^[0-9]+$ ]] || image_scores_count=0
            elif [[ -f "$scores_file" ]]; then
                image_scores_count=$(count_image_scores "$scores_file")
                [[ "$image_scores_count" =~ ^[0-9]+$ ]] || image_scores_count=0
            fi

            pixel_maps_count=0
            sample_png=""
            pixel_dir="$run_dir/outputs/pixel_scores"
            if [[ -d "$pixel_dir" ]]; then
                pixel_maps_count=$(
                    find "$pixel_dir" -maxdepth 1 -type f \
                        \( -iname '*.png' \) -printf '.' 2>/dev/null |
                        wc -c
                )
                sample_png=$(
                    find "$pixel_dir" -maxdepth 1 -type f \
                        \( -iname '*.png' \) -print 2>/dev/null |
                        LC_ALL=C sort |
                        head -n 1
                )
            fi

            all_256=false
            if [[ -n "$sample_png" ]]; then
                all_256=$(sample_is_256 "$sample_png")
                [[ "$all_256" == "true" ]] || all_256=false
            fi

            status=PASS
            reason=""
            if ((codex_exit == 124 || codex_exit == 137)); then
                status=FAIL
                reason=timeout
            elif ((codex_exit != 0)); then
                status=FAIL
                reason="codex_exit_${codex_exit}"
            elif [[ "$isolation_ok" != "true" ]]; then
                status=FAIL
                reason=isolation_violation
            elif ((image_scores_count != expected_count)); then
                status=FAIL
                reason=invalid_image_scores_count
            elif ((step != 1 && pixel_maps_count == 0)); then
                status=FAIL
                reason=missing_pixel_maps
            fi

            rerun_status=skipped
            infer_file="$run_dir/outputs/infer.py"
            if [[ -f "$infer_file" ]]; then
                rerun_status=completed
                original_scores="$run_dir/.initial_image_scores.json"
                had_original=false
                if [[ -f "$scores_file" ]]; then
                    cp "$scores_file" "$original_scores"
                    had_original=true
                fi

                for rerun in 1 2; do
                    rm -f "$scores_file"
                    set +e
                    (
                        cd "$run_dir" &&
                        timeout "$PER_RUN_TIMEOUT" python outputs/infer.py
                    ) > "$run_dir/rerun${rerun}.log" 2>&1
                    rerun_exit=$?
                    set +e

                    if ((rerun_exit == 0)) && [[ -f "$scores_file" ]]; then
                        cp "$scores_file" "$run_dir/rerun${rerun}.json"
                    else
                        rerun_status=failed
                    fi
                done

                if [[ "$had_original" == "true" ]]; then
                    mv "$original_scores" "$scores_file"
                else
                    rm -f "$scores_file" "$original_scores"
                fi
            fi

            VALIDATION_RUN_DIR="$run_dir" \
            VALIDATION_ISOLATION_OK="$isolation_ok" \
            VALIDATION_IMAGE_COUNT="$image_scores_count" \
            VALIDATION_PIXEL_COUNT="$pixel_maps_count" \
            VALIDATION_ALL_256="$all_256" \
            VALIDATION_STATUS="$status" \
            VALIDATION_REASON="$reason" \
            VALIDATION_RERUN="$rerun_status" \
            python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["VALIDATION_RUN_DIR"])
message_path = run_dir / "agent_last_message.txt"
try:
    method_backbone = message_path.read_text(encoding="utf-8", errors="replace")
except OSError:
    method_backbone = ""

result = {
    "isolation_ok": os.environ["VALIDATION_ISOLATION_OK"] == "true",
    "image_scores_count": int(os.environ["VALIDATION_IMAGE_COUNT"]),
    "pixel_maps_count": int(os.environ["VALIDATION_PIXEL_COUNT"]),
    "all_256": os.environ["VALIDATION_ALL_256"] == "true",
    "method_backbone": method_backbone,
    "status": os.environ["VALIDATION_STATUS"],
    "rerun": os.environ["VALIDATION_RERUN"],
}
reason = os.environ["VALIDATION_REASON"]
if reason:
    result["reason"] = reason

with (run_dir / "validation.json").open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, ensure_ascii=True)
    handle.write("\n")
PY

            run_end=$(date +%s)
            duration=$((run_end - run_start))
            printf '%s,%s,%s,%s,%s,%s,%s\n' \
                "$category" "$step" "$rep" "$status" "$duration" \
                "$image_scores_count" "$isolation_ok" >> "$MASTER_LOG"
            git add -A >/dev/null 2>&1
            git commit -q -m "run: ${category} step${step} run${rep} ${status}" >/dev/null 2>&1 || true
            git push -q origin main >/dev/null 2>&1 || true

            if [[ "$status" == "PASS" ]]; then
                passes=$((passes + 1))
            else
                fails=$((fails + 1))
            fi
            completed=$((completed + 1))

            if ((codex_exit != 0)); then
                consecutive_codex_fails=$((consecutive_codex_fails + 1))
            else
                consecutive_codex_fails=0
            fi

            if ((consecutive_codex_fails >= MAX_CONSECUTIVE_FAILS)); then
                printf '[ABORT] %s consecutive codex failures (likely token exhaustion or environment issue). Stopping. Re-run the script later to resume from completed runs.\n' \
                    "$consecutive_codex_fails"
                break 3
            fi
        done
    done
done

total_elapsed=$(($(date +%s) - EXPERIMENT_START))
printf '\nExperiment summary\n'
printf 'Total runs: %s\n' "$total_runs"
printf 'Passes: %s\n' "$passes"
printf 'Fails: %s\n' "$fails"
printf 'Total elapsed time: %s seconds\n' "$total_elapsed"
if [[ "$budget_exhausted" == "true" ]]; then
    printf 'Stopped before launching another run because TOTAL_BUDGET was reached.\n'
fi
