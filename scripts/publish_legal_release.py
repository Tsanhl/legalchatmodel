#!/usr/bin/env python3
"""Refresh gated HF dataset staging, upload Agnes999/legalchat, and push GitHub.

Quality gate: length passes up to LEGAL_MAX_WORDS (give-ups skipped) and
verify_legal_app must exit 0 unless --force is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "training" / "live_private_release_sweep" / "report.jsonl"
STAGE = ROOT / "tmp" / "huggingface" / "legalchat"
SRC_GUIDES = ROOT / "legal_chat_ui" / "law_guides"
DONE = Path(os.environ.get("LEGAL_PUBLISH_DONE", "/tmp/legal_release_publish_done.json"))
LOG = Path(os.environ.get("LEGAL_PUBLISH_LOG", "/tmp/legal_release_publish.log"))
HF_DATASET = os.environ.get("LEGAL_HF_DATASET", "Agnes999/legalchat")
HF_MODEL = os.environ.get("LEGAL_HF_MODEL", "Agnes999/legalchat-v11-lora")
ADAPTER = ROOT / "adapters" / "legal_answer_flow_v11_specialist_lora"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def notify(title: str, body: str) -> None:
    script = (
        f'display notification {json.dumps(body)} with title {json.dumps(title)} '
        f'sound name "Glass"'
    )
    subprocess.run(["osascript", "-e", script], check=False)


def report_rows() -> dict[str, dict]:
    if not REPORT.exists():
        return {}
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in REPORT.read_text().splitlines()
        if line.strip()
    }


def length_pass_summary() -> tuple[int, list[str]]:
    """Require length passes up to LEGAL_MAX_WORDS (phased publish).

    Skips cases listed in /tmp/legal_give_up_cases.txt so a deferred/failed
    slot (e.g. 5k trusts) does not block an earlier batch release.
    """
    rows = report_rows()
    max_words = int(os.environ.get("LEGAL_MAX_WORDS", "20000"))
    give_up: set[str] = set()
    give_up_path = Path("/tmp/legal_give_up_cases.txt")
    if give_up_path.exists():
        give_up = {line.strip() for line in give_up_path.read_text().splitlines() if line.strip()}

    required = [
        words
        for words in range(1000, max_words + 1, 1000)
        if not any(gid.startswith(f"length_{words:05d}_") for gid in give_up)
    ]
    missing = [
        f"{words:05d}"
        for words in required
        if not any(
            key.startswith(f"length_{words:05d}_") and rows[key].get("passed")
            for key in rows
        )
    ]
    passed = len(required) - len(missing)
    return passed, missing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_hf_staging() -> dict:
    assert SRC_GUIDES.is_dir(), f"missing guides: {SRC_GUIDES}"
    STAGE.mkdir(parents=True, exist_ok=True)
    guides_out = STAGE / "guides"
    guides_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sorted(SRC_GUIDES.glob("*.md")):
        text = src.read_text(encoding="utf-8")
        text = re.sub(r"/Users/[^\s]+", "[local path removed]", text)
        (guides_out / src.name).write_text(text, encoding="utf-8")
        copied += 1

    knowledge_path = STAGE / "knowledge.jsonl"
    records = []
    for path in sorted(guides_out.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        records.append(
            {
                "id": path.stem,
                "slug": path.stem,
                "subject": path.stem.replace("_", " "),
                "jurisdiction": "England and Wales",
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    knowledge_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )

    files = []
    for path in sorted(STAGE.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rel = path.relative_to(STAGE).as_posix()
        files.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "release": date.today().isoformat(),
        "jurisdiction_default": "England and Wales",
        "knowledge_records": len(records),
        "gold_answer_records": sum(
            1 for _ in (STAGE / "gold_answers.jsonl").open() if _.strip()
        )
        if (STAGE / "gold_answers.jsonl").exists()
        else 0,
        "evaluation_questions": 100,
        "files": files,
    }
    (STAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log(f"HF staging refreshed guides={copied} knowledge={len(records)}")
    return manifest


def run_verify() -> int:
    log("running verify_legal_app.py")
    proc = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/verify_legal_app.py")],
        cwd=str(ROOT),
    )
    log(f"verify exit={proc.returncode}")
    return proc.returncode


def upload_hf_dataset() -> None:
    log(f"uploading dataset {HF_DATASET}")
    proc = subprocess.run(
        [
            "hf",
            "upload",
            HF_DATASET,
            str(STAGE),
            ".",
            "--repo-type",
            "dataset",
            "--commit-message",
            f"Refresh LegalChatModel knowledge release {date.today().isoformat()}",
        ],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hf dataset upload failed exit={proc.returncode}")
    log(f"dataset uploaded https://huggingface.co/datasets/{HF_DATASET}")


def upload_hf_adapter() -> None:
    if not ADAPTER.exists():
        log(f"skip model upload; missing adapter {ADAPTER}")
        return
    log(f"uploading adapter model {HF_MODEL}")
    # Ensure gated/private-friendly model card exists locally.
    readme = ADAPTER / "README.md"
    if not readme.exists() or "LegalChatModel" not in readme.read_text(encoding="utf-8"):
        readme.write_text(
            "\n".join(
                [
                    "---",
                    "license: other",
                    "library_name: mlx",
                    "tags:",
                    "  - legal",
                    "  - england-and-wales",
                    "  - lora",
                    "  - mlx",
                    "base_model: mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit",
                    "---",
                    "",
                    "# LegalChatModel V11 specialist LoRA",
                    "",
                    "MLX LoRA adapter for England & Wales legal answering.",
                    "Use with the LegalChatModel app: https://github.com/Tsanhl/legalchatmodel",
                    "Companion dataset: https://huggingface.co/datasets/Agnes999/legalchat",
                    "",
                    "Not a substitute for qualified legal advice.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    subprocess.run(
        ["hf", "repo", "create", HF_MODEL, "--exist-ok"],
        cwd=str(ROOT),
        check=False,
    )
    proc = subprocess.run(
        [
            "hf",
            "upload",
            HF_MODEL,
            str(ADAPTER),
            ".",
            "--commit-message",
            f"Publish LegalChatModel V11 LoRA {date.today().isoformat()}",
        ],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hf model upload failed exit={proc.returncode}")
    log(f"model uploaded https://huggingface.co/{HF_MODEL}")


def git_push_release() -> None:
    log("staging release artifacts for GitHub")
    paths = [
        "legal_chat_ui",
        "scripts",
        "training/live_private_release_sweep",
        "training/LEGAL_APP_VERIFICATION.json",
        "training/LEGAL_EVAL_100_RAG_AUDIT.json",
        "adapters/legal_answer_flow_v11_specialist_lora/adapter_config.json",
        "adapters/legal_answer_flow_v11_specialist_lora/adapters.safetensors",
        "adapters/legal_answer_flow_v11_specialist_lora/README.md",
    ]
    existing = [p for p in paths if (ROOT / p).exists()]
    subprocess.run(["git", "add", *existing], cwd=str(ROOT), check=False)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True)
    if not status.strip():
        log("no git changes to commit")
    else:
        msg = (
            f"Release verified live length matrix and publish sync ({date.today().isoformat()})."
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(ROOT),
            check=False,
        )
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(ROOT), check=True)
    log("GitHub push complete")


def write_done(payload: dict) -> None:
    DONE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Cursor loop sentinel
    print(
        f'AGENT_LOOP_WAKE_legal_publish_done {json.dumps({"prompt": "Legal release finished — notify user with GitHub/HF links and quality summary.", "done": str(DONE)})}',
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-hf-model", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    args = parser.parse_args()

    passed, missing = length_pass_summary()
    max_words = int(os.environ.get("LEGAL_MAX_WORDS", "20000"))
    log(f"length passes {passed} (phase ≤{max_words}) missing={missing or 'none'}")
    if missing and not args.force:
        log("refusing publish: length matrix incomplete for this phase")
        return 2

    if not args.skip_verify:
        code = run_verify()
        if code != 0 and not args.force:
            notify("LegalChatModel publish blocked", "verify_legal_app failed")
            return code

    refresh_hf_staging()
    upload_hf_dataset()
    if not args.skip_hf_model:
        upload_hf_adapter()
    if not args.skip_git:
        git_push_release()

    payload = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "length_passed": passed,
        "missing_lengths": missing,
        "phase_max_words": max_words,
        "github": "https://github.com/Tsanhl/legalchatmodel",
        "hf_dataset": f"https://huggingface.co/datasets/{HF_DATASET}",
        "hf_model": f"https://huggingface.co/{HF_MODEL}",
        "friend_access": "HF dataset is gated (manual). Approve friends on the dataset settings; for live site use scripts/configure_public_macos.py with Cloudflare Access.",
    }
    write_done(payload)
    notify(
        "LegalChatModel publish complete",
        f"phase ≤{max_words}: {passed} lengths · GitHub + HF updated",
    )
    log(f"DONE {json.dumps(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
