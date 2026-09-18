"""Measure Outlines JSON-schema index construction without loading model weights.

Each case runs in a fresh child process with an address-space limit. The parent samples
the child's resident memory from /proc and preserves every JSON phase event, stderr,
exit status, signal, and timeout in one report. A repeated case can build the same index
more than once in one process to expose retained-state growth.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from assets import output_directory


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_bytes(pid):
    """Read current resident memory for a Linux process, or return None."""

    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def phase_event(phase, started, **fields):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    event = {
        "event": "phase",
        "phase": phase,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "child_peak_rss_bytes": usage.ru_maxrss * 1024,
        **fields,
    }
    print(json.dumps(event, sort_keys=True), flush=True)


def child(args):
    """Build one schema/index case; invoked only by the guarded parent."""

    if len(args.max_length) != 1:
        raise SystemExit("child mode requires exactly one --max-length")
    max_length = args.max_length[0]
    if args.prefer_child_oom_kill:
        Path("/proc/self/oom_score_adj").write_text("1000\n")
    if args.memory_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_bytes, args.memory_bytes))

    started = time.monotonic()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    phase_event(
        "tokenizer_loaded",
        started,
        eos_token=tokenizer.eos_token,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer_class=type(tokenizer).__name__,
        vocabulary_entries=len(tokenizer.get_vocab()),
    )

    started = time.monotonic()
    from outlines.backends.outlines_core import OutlinesCoreBackend
    from outlines.models.transformers import TransformerTokenizer

    wrapped = TransformerTokenizer(tokenizer)
    vocabulary = OutlinesCoreBackend.create_outlines_core_vocabulary(
        wrapped.get_vocab(),
        wrapped.eos_token_id,
        wrapped.eos_token,
        wrapped.convert_token_to_string,
    )
    phase_event("vocabulary_built", started)

    started = time.monotonic()
    from text_albumentations.tasks.qa_pairs import QaPairAugmentation

    schema_type = QaPairAugmentation(
        max_question_length=max_length,
        max_answer_length=max_length,
    ).get_schema()
    schema = json.dumps(schema_type.model_json_schema(), ensure_ascii=False)
    phase_event(
        "schema_built",
        started,
        schema_bytes=len(schema.encode()),
        schema_sha256=hashlib.sha256(schema.encode()).hexdigest(),
    )

    started = time.monotonic()
    from outlines_core.json_schema import build_regex_from_schema

    regex = build_regex_from_schema(schema)
    phase_event(
        "regex_built",
        started,
        regex_characters=len(regex),
        regex_sha256=hashlib.sha256(regex.encode()).hexdigest(),
    )

    from outlines_core import Index

    indexes = []
    for repetition in range(1, args.repetitions + 1):
        started = time.monotonic()
        indexes.append(Index(regex, vocabulary))
        phase_event("index_built", started, repetition=repetition)

    print(json.dumps({"event": "complete", "indexes_retained": len(indexes)}), flush=True)


def parse_events(stdout):
    events = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event"):
            events.append(value)
    return events


def run_case(script, args, max_length, repetitions):
    command = [
        sys.executable,
        str(script),
        "--child",
        "--tokenizer",
        args.tokenizer,
        "--max-length",
        str(max_length),
        "--repetitions",
        str(repetitions),
        "--memory-bytes",
        str(int(args.memory_gib * 1024**3)),
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.prefer_child_oom_kill:
        command.append("--prefer-child-oom-kill")

    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    peak_rss = 0
    timed_out = False
    while process.poll() is None:
        sample = rss_bytes(process.pid)
        if sample is not None:
            peak_rss = max(peak_rss, sample)
        if time.monotonic() - started > args.timeout_seconds:
            timed_out = True
            process.kill()
            break
        time.sleep(args.sample_interval)
    stdout, stderr = process.communicate()
    sample = rss_bytes(process.pid)
    if sample is not None:
        peak_rss = max(peak_rss, sample)
    elapsed = time.monotonic() - started
    returncode = process.returncode
    return {
        "max_length": max_length,
        "repetitions": repetitions,
        "command": command,
        "elapsed_seconds": round(elapsed, 6),
        "parent_sampled_peak_rss_bytes": peak_rss or None,
        "returncode": returncode,
        "signal": -returncode if returncode is not None and returncode < 0 else None,
        "signal_name": (
            signal.Signals(-returncode).name
            if returncode is not None and returncode < 0
            else None
        ),
        "timed_out": timed_out,
        "events": parse_events(stdout),
        "stdout": stdout,
        "stderr": stderr,
        "succeeded": returncode == 0 and not timed_out,
    }


def tokenizer_files(tokenizer):
    root = Path(tokenizer)
    if not root.is_dir():
        return []
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "config.json",
    )
    return [
        {"path": name, "sha256": sha256_file(root / name), "bytes": (root / name).stat().st_size}
        for name in names
        if (root / name).is_file()
    ]


def parent(args):
    out = output_directory(args.output)
    script = Path(__file__).resolve()
    report = {
        "scope": "CPU-only tokenizer/schema/regex/index construction; no model weights or generation",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "script_sha256": sha256_file(script),
        "tokenizer": args.tokenizer,
        "revision": args.revision,
        "tokenizer_files": tokenizer_files(args.tokenizer),
        "memory_limit_bytes": int(args.memory_gib * 1024**3),
        "prefer_child_oom_kill": args.prefer_child_oom_kill,
        "timeout_seconds": args.timeout_seconds,
        "sample_interval_seconds": args.sample_interval,
        "stop_peak_bytes": int(args.stop_peak_gib * 1024**3),
        "cases_requested": args.max_length,
        "cases": [],
    }

    for max_length in args.max_length:
        case = run_case(script, args, max_length, 1)
        report["cases"].append(case)
        (out / "schema_index_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in case.items() if k not in {"stdout", "stderr", "command"}}, indent=2))
        if not case["succeeded"]:
            report["stopped_reason"] = f"max_length={max_length} did not succeed"
            break
        if (case["parent_sampled_peak_rss_bytes"] or 0) >= report["stop_peak_bytes"]:
            report["stopped_reason"] = f"max_length={max_length} crossed the configured peak-RSS stop"
            break

    if args.repeat_successful and report["cases"]:
        successful = [
            case
            for case in report["cases"]
            if case["succeeded"]
            and (case["parent_sampled_peak_rss_bytes"] or 0) < report["stop_peak_bytes"]
        ]
        if successful:
            repeat_length = successful[-1]["max_length"]
            repeat_case = run_case(script, args, repeat_length, args.repeat_successful)
            repeat_case["kind"] = "same_process_repeat"
            report["cases"].append(repeat_case)

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "schema_index_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out / 'schema_index_probe.json'}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tokenizer", required=True, help="Pinned local tokenizer directory or Hub ID")
    result.add_argument("--revision")
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--max-length", type=int, nargs="+", required=True)
    result.add_argument(
        "--memory-gib", type=float, default=20.0, help="Child address-space limit; 0 disables it"
    )
    result.add_argument(
        "--prefer-child-oom-kill",
        action="store_true",
        help="Make the compiler child the kernel's preferred OOM-kill target",
    )
    result.add_argument("--stop-peak-gib", type=float, default=16.0)
    result.add_argument("--timeout-seconds", type=float, default=600.0)
    result.add_argument("--sample-interval", type=float, default=0.1)
    result.add_argument("--repeat-successful", type=int, default=0)
    result.add_argument("--output")
    result.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--memory-bytes", type=int, help=argparse.SUPPRESS)
    result.add_argument("--repetitions", type=int, default=1, help=argparse.SUPPRESS)
    return result


def main():
    args = parser().parse_args()
    if args.child:
        child(args)
    else:
        if not args.output:
            raise SystemExit("--output is required")
        if args.repeat_successful == 1:
            raise SystemExit("--repeat-successful must be 0 or at least 2")
        parent(args)


if __name__ == "__main__":
    main()
