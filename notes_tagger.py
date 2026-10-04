#!/usr/bin/env python3
# Notes Organizer - LLM tagging, filing and rewriting of Markdown notes.
# Copyright (C) 2026 The Notes Organizer contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version. It is distributed WITHOUT ANY
# WARRANTY; see the LICENSE file or <https://www.gnu.org/licenses/>.
"""Notes Auto-Tagger / Organizer / Rewriter.

Scans a folder of markdown notes and, using an LLM (local or hosted), can:
  * tag       - add tags to notes that have no tags yet (tags = a line of `#tag` tokens at the top)
  * organize  - move recently created, unfiled notes into the best existing folder
                (or any note carrying the manual trigger tag #llmtomove)
  * rewrite   - re-format / fix grammar of new notes (or any note carrying #llmtoprocess);
                the original is archived and both notes are cross-linked with [[wikilinks]]

Python does all file handling, selection, validation and linking; the LLM is only asked
for tags, a folder choice, or rewritten text. Notes edited in the last `min_idle_minutes`
(default 60) are never touched. Standard library only.

Settings precedence: command line > config.json > built-in defaults.
API keys are read from environment variables only (never from config files).
"""
import argparse
import datetime as dt
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

DEFAULTS = {
    "notes_dir": None,  # required
    "provider": "openai",  # "openai" (any OpenAI-compatible API) or "anthropic"
    "base_url": None,  # provider default if unset
    "model": None,  # required
    "api_key_env": None,  # env var holding the key; provider default if unset
    "work_dir": str(Path.home() / "notes_tagger_data"),  # backups/, logs/, lock
    "tagging": True,  # add LLM tags to notes that have no tags yet
    "organize_mode": "off",  # off | manual (#llmtomove only) | auto (also new unfiled notes)
    "rewrite_mode": "off",  # off | manual (#llmtoprocess only) | auto (also new notes)
    "new_note_days": 3,  # "recently created" window for auto organize/rewrite
    "min_idle_minutes": 60,  # never touch a note edited more recently than this (may be open in the app)
    "archive_folder": "Archived",  # created under notes_dir when needed
    "max_age_hours": None,  # tagging: only notes modified within this window (null = all untagged notes)
    "max_notes_per_run": 0,  # per feature, 0 = unlimited; leftover notes are picked up next run
    "max_consecutive_failures": 3,  # abort the run after this many notes in a row fail (LLM likely down)
    "log_retention_days": 30,  # delete run logs older than this (0 = keep forever)
    "backup_retention_days": 30,  # delete note backups older than this (0 = keep forever)
    "max_retries": 2,  # extra attempts after the first (3 total)
    "retry_wait_seconds": 3600,
    "request_timeout_seconds": 600,
    "max_note_chars": 12000,  # tagging/organize: truncate very long notes before sending
    "rewrite_max_chars": 20000,  # rewrite: notes longer than this are skipped (never truncated)
    "max_output_tokens": 1024,  # cap on the reply for tagging / organizing / --check (a runaway can't block the LLM for long)
    "rewrite_max_tokens": 8192,  # cap on the reply when rewriting a note
    "max_tokens_field": "max_tokens",  # OpenAI-style request field for the cap; "" = don't send one (e.g. "max_completion_tokens" for newer OpenAI models)
    "extra_body": {},  # extra JSON fields merged into OpenAI-style requests, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    "min_tags": 3,
    "max_tags": 6,
    "skip_model_check": False,  # don't require the model in the provider's /models list
    "dry_run": False,
}

PROVIDER_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com", "api_key_env": "ANTHROPIC_API_KEY"},
}

MODES = ("off", "manual", "auto")
TRIGGER_MOVE = "llmtomove"
TRIGGER_PROCESS = "llmtoprocess"
TAG_AI = "AI"
TAG_ARCHIVED = "archived"
SYSTEM_TAGS = frozenset({TRIGGER_MOVE, TRIGGER_PROCESS, TAG_AI.lower(), TAG_ARCHIVED})  # not "real" tags

TAG_TOKEN = re.compile(r"^#[\w.\-]+$")
WIKILINK = re.compile(r"\[\[[^\]\n]+\]\]")
URL = re.compile(r"https?://[^\s)>\]]+")
FOOTER = re.compile(r"\n---[ \t]*\r?\n+Original note \(archived\): \[\[[^\]\n]+\]\][ \t]*\s*$")
LOCK_STALE_SECONDS = 24 * 3600
FAILED = object()
log = logging.getLogger("notes_tagger")


class LLMError(Exception):
    pass


class AbortRun(Exception):
    pass


class Note(NamedTuple):
    path: Path
    stat: os.stat_result
    tags: frozenset  # lowercase tags from the tag line
    archived: bool


# ---------------------------------------------------------------- config / logging
def load_config(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    p.add_argument("--notes-dir")
    p.add_argument("--provider", choices=sorted(PROVIDER_DEFAULTS))
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--api-key-env")
    p.add_argument("--work-dir")
    p.add_argument("--organize-mode", choices=MODES)
    p.add_argument("--rewrite-mode", choices=MODES)
    p.add_argument("--no-tagging", action="store_true", help="disable the tagging step")
    p.add_argument("--archive-folder")
    p.add_argument("--new-note-days", type=float)
    p.add_argument("--min-idle-minutes", type=float)
    p.add_argument("--max-age-hours", type=float)
    p.add_argument("--max-notes-per-run", type=int)
    p.add_argument("--check", action="store_true", help="test config + LLM connection, then exit")
    p.add_argument("--max-retries", type=int)
    p.add_argument("--retry-wait-seconds", type=int)
    p.add_argument("--request-timeout-seconds", type=int)
    p.add_argument("--max-note-chars", type=int)
    p.add_argument("--dry-run", action="store_true", default=None)
    args = p.parse_args(argv)

    cfg = dict(DEFAULTS)
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            file_cfg = json.load(f)
        unknown = set(file_cfg) - set(DEFAULTS)
        if unknown:
            raise SystemExit(f"Unknown config keys in {cfg_path}: {sorted(unknown)}")
        cfg.update(file_cfg)
    elif "--config" in argv:
        raise SystemExit(f"Config file not found: {cfg_path}")

    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.no_tagging:
        cfg["tagging"] = False
    cfg["check"] = args.check

    for key in ("organize_mode", "rewrite_mode"):
        if cfg[key] not in MODES:
            raise SystemExit(f"`{key}` must be one of {MODES}")
    if not isinstance(cfg["extra_body"], dict):
        raise SystemExit("`extra_body` must be a JSON object, e.g. {\"chat_template_kwargs\": {\"enable_thinking\": false}}")
    pd = PROVIDER_DEFAULTS[cfg["provider"]]
    cfg["base_url"] = (cfg["base_url"] or pd["base_url"]).rstrip("/")
    cfg["api_key_env"] = cfg["api_key_env"] or pd["api_key_env"]
    if not cfg["notes_dir"] or not cfg["model"]:
        raise SystemExit("`notes_dir` and `model` are required (config.json or --notes-dir/--model).")
    cfg["notes_dir"] = str(Path(cfg["notes_dir"]).expanduser().resolve())
    cfg["work_dir"] = str(Path(cfg["work_dir"]).expanduser().resolve())
    return cfg


def setup_logging(work_dir):
    logs = Path(work_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    log.setLevel(logging.INFO)
    handlers = [
        logging.FileHandler(logs / f"run-{dt.datetime.now():%Y%m%d-%H%M%S}.log", encoding="utf-8"),
        logging.FileHandler(logs / "latest.log", mode="w", encoding="utf-8"),
    ]
    if sys.stderr and sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)


def pid_alive(pid):
    """True if a process with this id is running (used to detect a lock left by a killed run)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied => it exists, just not ours
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class SingleInstanceLock:
    def __init__(self, path):
        self.path = Path(path)
        self.held = False

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(str(os.getpid()))
                self.held = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    pid = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    age, pid = None, 0
                if age is None and not self.path.exists():
                    continue  # vanished between our attempts; try again
                if not pid_alive(pid):
                    log.warning("Removing stale lock file (process %s is no longer running)", pid or "unknown")
                elif age > LOCK_STALE_SECONDS:
                    log.warning("Removing stale lock file (%.0f h old)", age / 3600)
                else:
                    log.info("Another run is active (process %s).", pid)
                    return self
                self.path.unlink(missing_ok=True)
        return self

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------- LLM client
class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.provider = cfg["provider"]
        self.base = cfg["base_url"]
        self.consecutive_failures = 0
        self.key = os.environ.get(cfg["api_key_env"], "")
        if not self.key and self.base in (d["base_url"] for d in PROVIDER_DEFAULTS.values()):
            raise SystemExit(f"Environment variable {cfg['api_key_env']} is not set (API key required).")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            h["x-api-key"] = self.key
            h["anthropic-version"] = "2023-06-01"
        elif self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def _url(self, path):
        # Accept base URLs with or without a trailing /v1
        base = self.base[:-3] if self.base.endswith("/v1") else self.base
        return f"{base}/v1/{path}"

    def _request(self, method, path, body=None, timeout=30):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            finally:
                e.close()
            raise LLMError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise LLMError(f"{type(e).__name__}: {e}")

    def check_available(self):
        """Return None if OK, else a reason string."""
        try:
            resp = self._request("GET", "models", timeout=30)
        except LLMError as e:
            return str(e)
        if self.cfg["skip_model_check"]:
            return None
        ids = [m.get("id") for m in resp.get("data", []) if isinstance(m, dict)]
        if self.cfg["model"] not in ids:
            return f"model '{self.cfg['model']}' not listed by endpoint (use skip_model_check to bypass)"
        return None

    def complete(self, system, user, max_tokens=None):
        """Return the model's text. Raises LLMError if the output was cut off by a token limit."""
        timeout = self.cfg["request_timeout_seconds"]
        if self.provider == "anthropic":
            resp = self._request("POST", "messages", {
                "model": self.cfg["model"], "max_tokens": max_tokens or 1024, "system": system,
                "messages": [{"role": "user", "content": user}],
            }, timeout)
            if resp.get("stop_reason") == "max_tokens":
                raise LLMError("response truncated (max_tokens reached)")
            return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
        body = {"model": self.cfg["model"],
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.cfg["max_tokens_field"] and max_tokens:
            body[self.cfg["max_tokens_field"]] = max_tokens
        body.update(self.cfg["extra_body"])
        resp = self._request("POST", "chat/completions", body, timeout)
        try:
            choice = resp["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError("unexpected response shape")
        if choice.get("finish_reason") == "length":
            raise LLMError("response truncated (token limit reached)")
        return text


def strip_think(raw):
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I)


def ask(client, cfg, system, user, parser, max_tokens=None):
    """Call the LLM and parse the reply, retrying per config. Returns parsed value or FAILED.

    Raises AbortRun after `max_consecutive_failures` notes in a row fail (LLM is probably down),
    so a dead endpoint doesn't stall the job for hours."""
    attempts = cfg["max_retries"] + 1
    for n in range(1, attempts + 1):
        try:
            result = parser(client.complete(system, user, max_tokens))
            client.consecutive_failures = 0
            return result
        except LLMError as e:
            log.warning("  attempt %d/%d failed: %s", n, attempts, e)
            if n < attempts:
                time.sleep(cfg["retry_wait_seconds"])
    client.consecutive_failures += 1
    limit = cfg["max_consecutive_failures"]
    if limit and client.consecutive_failures >= limit:
        raise AbortRun(f"{client.consecutive_failures} notes in a row failed")
    return FAILED


# ------------------------------------------------------------------- note handling
def nl_of(text):
    return "\r\n" if "\r\n" in text else "\n"


def split_tag_line(text):
    """Return (span, tags, body). span=(start,end) of the tag line in text, or None."""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            offset += len(line)
            continue
        tokens = stripped.split()
        if all(TAG_TOKEN.match(t) for t in tokens):
            return (offset, offset + len(line.rstrip("\r\n"))), [t[1:] for t in tokens], text[offset + len(line):]
        break
    return None, [], text


def render_tag_line(tags):
    return " ".join("#" + t for t in tags)


def set_tags(text, tags):
    """Replace the tag line with `tags` (inserting or removing the line as needed). Rest untouched."""
    span, _, _ = split_tag_line(text)
    if span:
        start, end = span
        if tags:
            return text[:start] + render_tag_line(tags) + text[end:]
        return text[:start] + text[end:].lstrip("\r\n")  # drop the line and the blank lines after it
    if tags:
        nl = nl_of(text)
        return render_tag_line(tags) + nl + nl + text
    return text


def append_line(text, line):
    nl = nl_of(text)
    if not text.endswith("\n"):
        text += nl
    return text + nl + line + nl


def without(tags, *drop):
    drop = {d.lower() for d in drop}
    return [t for t in tags if t.lower() not in drop]


def with_tag(tags, tag):
    return tags if tag.lower() in {t.lower() for t in tags} else tags + [tag]


def created_time(st):
    bt = getattr(st, "st_birthtime", None)
    if bt:
        return bt
    return st.st_ctime if os.name == "nt" else st.st_mtime  # Linux has no creation time: use mtime


def unique_path(directory, name):
    p = Path(directory) / name
    stem, suffix, i = p.stem, p.suffix, 2
    while p.exists():
        p = Path(directory) / f"{stem} ({i}){suffix}"
        i += 1
    return p


def read_note(path):
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    return raw[3:].decode("utf-8") if bom else raw.decode("utf-8"), bom


def write_atomic(path, text, bom):
    data = (b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tagger-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if path.exists():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def backup(path, work):
    bdir = work / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, unique_path(bdir, f"{dt.datetime.now():%Y%m%d-%H%M%S}__{path.name}"))


def iter_notes(cfg):
    """Yield every non-hidden .md note with its tag line peeked (first 4 KB only)."""
    root = Path(cfg["notes_dir"])
    archive = cfg["archive_folder"].lower()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith(".") or not name.lower().endswith(".md"):
                continue
            path = Path(dirpath) / name
            try:
                st = path.stat()
                with open(path, "rb") as f:
                    head = f.read(4096)
            except OSError as e:
                log.warning("Cannot read %s: %s", path, e)
                continue
            _, tags, _ = split_tag_line(head.decode("utf-8", errors="ignore").lstrip("﻿"))
            lower = frozenset(t.lower() for t in tags)
            rel = path.relative_to(root).parts
            in_archive = len(rel) > 1 and rel[0].lower() == archive
            yield Note(path, st, lower, in_archive or TAG_ARCHIVED.lower() in lower)


def is_recent(n, cfg):
    return created_time(n.stat) >= time.time() - cfg["new_note_days"] * 86400


def is_settled(n, cfg):
    return time.time() - n.stat.st_mtime >= cfg["min_idle_minutes"] * 60


def settle(cands, cfg):
    """Drop notes edited too recently (they may be open in the notes app); log how many."""
    ready = [n for n in cands if is_settled(n, cfg)]
    if len(ready) < len(cands):
        log.info("Skipping %d note(s) edited in the last %g min (may be open in the app); "
                 "they will be picked up by a later run", len(cands) - len(ready), cfg["min_idle_minutes"])
    return ready


def cap(cands, cfg):
    limit = cfg["max_notes_per_run"]
    if limit and len(cands) > limit:
        log.info("Limiting to %d of %d note(s) this run (max_notes_per_run); the rest wait for the next run",
                 limit, len(cands))
        return cands[:limit]
    return cands


def tally(counts, result):
    counts[result] = counts.get(result, 0) + 1


def cleanup_old_files(work, cfg):
    """Delete old run logs and note backups. Age comes from the timestamp in the file name
    (backups keep the note's own mtime, so file times are not reliable)."""
    for sub, pattern, days in (("logs", "run-*.log", cfg["log_retention_days"]),
                               ("backups", "*__*", cfg["backup_retention_days"])):
        if not days:
            continue
        cutoff, removed = dt.datetime.now() - dt.timedelta(days=days), 0
        for p in (work / sub).glob(pattern):
            m = re.search(r"(\d{8}-\d{6})", p.name)
            if not m or not p.is_file():
                continue
            try:
                if dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S") < cutoff:
                    p.unlink()
                    removed += 1
            except (ValueError, OSError):
                pass
        if removed:
            log.info("Cleanup: removed %d old file(s) from %s (older than %d days)", removed, sub, days)


# ----------------------------------------------------------------------- 1) tagging
TAG_SYSTEM = (
    "You label personal notes with tags. Reply with ONLY a JSON array of {lo}-{hi} short, "
    "lowercase, single-word tags (no '#', no explanation, no markdown). Example: [\"travel\",\"budget\"]"
)


def parse_tags(raw, max_tags):
    m = re.search(r"\[.*?\]", strip_think(raw), flags=re.S)
    if not m:
        raise LLMError("no JSON array in response")
    try:
        items = json.loads(m.group(0))
    except ValueError:
        raise LLMError("unparseable JSON array")
    tags, seen = [], set()
    for it in items:
        if not isinstance(it, str):
            continue
        t = re.sub(r"[^\w.\-]", "", it.strip().lstrip("#").lower())
        if t and t not in seen:
            seen.add(t)
            tags.append(t)
    if not tags:
        raise LLMError("empty tag list")
    return tags[:max_tags]


def tag_note(path, client, cfg, work):
    try:
        text, bom = read_note(path)
        mtime = path.stat().st_mtime
    except (OSError, UnicodeDecodeError) as e:
        log.warning("SKIP (unreadable) %s: %s", path, e)
        return "skipped"

    _, existing, body = split_tag_line(text)
    if not body.strip():
        log.info("SKIP (empty) %s", path)
        return "skipped"

    log.info("Tagging %s", path)
    system = TAG_SYSTEM.format(lo=cfg["min_tags"], hi=cfg["max_tags"])
    suggested = ask(client, cfg, system, "Note:\n\n" + body[: cfg["max_note_chars"]],
                    lambda raw: parse_tags(raw, cfg["max_tags"]), max_tokens=cfg["max_output_tokens"])
    if suggested is FAILED:
        log.error("  giving up on %s (all attempts failed)", path)
        return "failed"

    have = {t.lower() for t in existing} | SYSTEM_TAGS
    new = [t for t in suggested if t not in have]
    if not new:
        log.info("  no usable tags suggested")
        return "unchanged"
    add = render_tag_line(new)
    if cfg["dry_run"]:
        log.info("  [dry-run] would add: %s", add)
        return "dry-run"

    try:
        if path.stat().st_mtime != mtime:
            log.warning("  note changed during processing, skipping: %s", path)
            return "skipped"
        backup(path, work)
        write_atomic(path, set_tags(text, existing + new), bom)
    except OSError as e:
        log.warning("  write failed for %s: %s", path, e)
        return "skipped"
    log.info("  added: %s", add)
    return "tagged"


def run_tagging(client, cfg, work):
    """Tag notes that have no tags yet. Once a note has tags it is never tagged again,
    so nothing accumulates from run to run (system tags like #AI don't count as tags)."""
    counts = {}
    notes = [n for n in iter_notes(cfg) if not n.archived]
    untagged = [n for n in notes if not (n.tags - SYSTEM_TAGS)]
    if cfg["max_age_hours"]:
        cutoff = time.time() - cfg["max_age_hours"] * 3600
        untagged = [n for n in untagged if n.stat.st_mtime >= cutoff]
    log.info("Found %d note(s); %d without tags", len(notes), len(untagged))
    for n in cap(settle(untagged, cfg), cfg):
        tally(counts, tag_note(n.path, client, cfg, work))
    return counts


# --------------------------------------------------------------------- 2) organizing
FOLDER_SYSTEM = (
    "You file personal notes into folders. You are given a list of existing folders and a note. "
    "Reply with ONLY JSON like {\"folder\": \"<exact name from the list>\"}, or {\"folder\": null} "
    "if no folder clearly fits. Never invent or modify folder names."
)


def list_folders(cfg):
    root, archive = Path(cfg["notes_dir"]), cfg["archive_folder"].lower()
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")
                             and not (Path(dirpath) == root and d.lower() == archive))
        if Path(dirpath) != root:
            out.append(Path(dirpath).relative_to(root).as_posix())
    return out


def parse_folder(raw, folders):
    raw = strip_think(raw)
    val, m = None, re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            val = obj.get("folder") if isinstance(obj, dict) else None
        except ValueError:
            raise LLMError("unparseable JSON object")
    else:
        val = raw.strip().strip("`\"' ")
    if val is None or str(val).strip().lower() in ("", "none", "null"):
        return None
    lookup = {f.lower(): f for f in folders}
    key = str(val).strip().strip("/").lower()
    if key not in lookup:
        raise LLMError(f"unknown folder {val!r}")
    return lookup[key]


def organize_note(n, folders, client, cfg, work):
    path, root = n.path, Path(cfg["notes_dir"])
    manual = TRIGGER_MOVE in n.tags
    try:
        text, bom = read_note(path)
        mtime = path.stat().st_mtime
    except (OSError, UnicodeDecodeError) as e:
        log.warning("SKIP (unreadable) %s: %s", path, e)
        return "skipped"
    _, tags, body = split_tag_line(text)
    if not body.strip():
        log.info("SKIP (empty) %s", path)
        return "skipped"

    log.info("Organizing %s%s", path, " (manual)" if manual else "")
    user = "Folders:\n" + "\n".join(f"- {f}" for f in folders) + "\n\nNote:\n\n" + body[: cfg["max_note_chars"]]
    target = ask(client, cfg, FOLDER_SYSTEM, user, lambda raw: parse_folder(raw, folders),
                 max_tokens=cfg["max_output_tokens"])
    if target is FAILED:
        log.error("  giving up on %s (all attempts failed)", path)
        return "failed"
    if target is None:
        log.info("  no suitable folder%s", "; trigger tag kept" if manual else "")
        return "no-fit"

    current = path.parent.relative_to(root).as_posix()
    dest = path if target == current else unique_path(root / target, path.name)
    if cfg["dry_run"]:
        log.info("  [dry-run] would move to %s%s", target, " and remove #" + TRIGGER_MOVE if manual else "")
        return "dry-run"

    try:
        if path.stat().st_mtime != mtime:
            log.warning("  note changed during processing, skipping: %s", path)
            return "skipped"
        backup(path, work)
        if dest != path:
            shutil.move(str(path), str(dest))
    except OSError as e:
        log.warning("  move failed for %s: %s", path, e)
        return "skipped"
    if manual:
        try:
            write_atomic(dest, set_tags(text, without(tags, TRIGGER_MOVE)), bom)
        except OSError as e:
            log.warning("  moved, but could not remove #%s from %s: %s", TRIGGER_MOVE, dest, e)
    log.info("  moved to %s", target if dest != path else f"{target} (already there)")
    return "moved" if dest != path else "unchanged"


def run_organize(client, cfg, work):
    counts, root = {}, Path(cfg["notes_dir"])
    folders = list_folders(cfg)
    if not folders:
        log.info("No folders exist under notes_dir; nothing to organize into.")
        return counts
    cands = []
    for n in iter_notes(cfg):
        manual = TRIGGER_MOVE in n.tags
        auto = cfg["organize_mode"] == "auto" and n.path.parent == root and is_recent(n, cfg)
        if not n.archived and (manual or auto):
            cands.append(n)
    for n in cap(settle(cands, cfg), cfg):
        tally(counts, organize_note(n, folders, client, cfg, work))
    return counts


# ---------------------------------------------------------------------- 3) rewriting
REWRITE_SYSTEM = (
    "You are a careful copy editor for personal notes. Rewrite the note as clean, well-structured "
    "Markdown: fix spelling and grammar, and use headings, lists, tables or emphasis where they help "
    "readability. Rules: keep the original language and meaning; do not add, remove or invent facts; "
    "keep all names, numbers, dates, URLs, code and [[wikilinks]] exactly as written; keep inline "
    "#hashtags; do not add a top-level title unless the note already has one. "
    "Reply with ONLY the rewritten note in Markdown - no preamble, no explanation, no code fence."
)


def parse_rewrite(raw, original):
    raw = strip_think(raw).strip()
    m = re.match(r"^```(?:markdown|md)?[ \t]*\n(.*)\n```$", raw, re.S)
    out = (m.group(1) if m else raw).strip()
    if not out:
        raise LLMError("empty rewrite")
    orig = original.strip()
    if len(orig) >= 200 and len(out) < 0.5 * len(orig):
        raise LLMError("rewrite is less than half the original length (content likely dropped)")
    needed = set(WIKILINK.findall(orig)) | {u.rstrip(".,;:!?") for u in URL.findall(orig)}
    missing = sorted(x for x in needed if x not in out)
    if missing:
        raise LLMError(f"rewrite dropped links/URLs: {missing[:3]}")
    return out


def rewrite_note(n, client, cfg, work):
    path = n.path
    try:
        text, bom = read_note(path)
        mtime = path.stat().st_mtime
    except (OSError, UnicodeDecodeError) as e:
        log.warning("SKIP (unreadable) %s: %s", path, e)
        return "skipped"
    nl = nl_of(text)
    _, tags, body = split_tag_line(text)
    body = FOOTER.sub("", body).strip("\r\n")  # drop a previous "Original note" footer; re-added below
    if not body.strip():
        log.info("SKIP (empty) %s", path)
        return "skipped"
    if len(body) > cfg["rewrite_max_chars"]:
        log.warning("SKIP (longer than rewrite_max_chars, not truncating) %s", path)
        return "skipped"

    log.info("Rewriting %s%s", path, " (manual)" if TRIGGER_PROCESS in n.tags else "")
    new_body = ask(client, cfg, REWRITE_SYSTEM, "Note:\n\n" + body, lambda raw: parse_rewrite(raw, body),
                   max_tokens=cfg["rewrite_max_tokens"])
    if new_body is FAILED:
        log.error("  giving up on %s (all attempts failed)", path)
        return "failed"

    stem = path.stem
    archive_dir = Path(cfg["notes_dir"]) / cfg["archive_folder"]
    arch_path = unique_path(archive_dir, f"{stem} (archived){path.suffix}")
    arch_tags = with_tag(without(tags, TRIGGER_PROCESS, TRIGGER_MOVE), TAG_ARCHIVED)
    arch_text = append_line(set_tags(text, arch_tags), f"Rewritten version: [[{stem}]]")
    new_tags = with_tag(without(tags, TRIGGER_PROCESS, TAG_ARCHIVED), TAG_AI)
    new_text = (render_tag_line(new_tags) + nl + nl + new_body.replace("\r\n", "\n").replace("\n", nl)
                + nl + nl + "---" + nl + nl + f"Original note (archived): [[{arch_path.stem}]]" + nl)

    if cfg["dry_run"]:
        preview = work / "dry_run" / f"{path.stem}.md"  # so you can review the proposed rewrite
        try:
            preview.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(preview, new_text, bom)
            shown = f"; proposed rewrite saved to {preview}"
        except OSError:
            shown = ""
        log.info("  [dry-run] would archive original as %s/%s and write rewritten note%s",
                 cfg["archive_folder"], arch_path.name, shown)
        return "dry-run"

    try:
        if path.stat().st_mtime != mtime:
            log.warning("  note changed during processing, skipping: %s", path)
            return "skipped"
        backup(path, work)
        archive_dir.mkdir(parents=True, exist_ok=True)
        write_atomic(arch_path, arch_text, bom)
        try:
            write_atomic(path, new_text, bom)
        except OSError:
            arch_path.unlink(missing_ok=True)  # roll back so the original stays the only copy
            raise
    except OSError as e:
        log.warning("  write failed for %s: %s", path, e)
        return "skipped"
    log.info("  rewritten; original archived as %s/%s", cfg["archive_folder"], arch_path.name)
    return "rewritten"


def run_rewrite(client, cfg, work):
    counts, cands = {}, []
    for n in iter_notes(cfg):
        manual = TRIGGER_PROCESS in n.tags
        auto = (cfg["rewrite_mode"] == "auto" and TAG_AI.lower() not in n.tags and is_recent(n, cfg))
        if not n.archived and (manual or auto):
            cands.append(n)
    for n in cap(settle(cands, cfg), cfg):
        tally(counts, rewrite_note(n, client, cfg, work))
    return counts


# -------------------------------------------------------------------------- main
def main(argv=None):
    cfg = load_config(sys.argv[1:] if argv is None else argv)
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    setup_logging(work)
    log.info("Start: provider=%s model=%s url=%s notes=%s tagging=%s organize=%s rewrite=%s dry_run=%s",
             cfg["provider"], cfg["model"], cfg["base_url"], cfg["notes_dir"], cfg["tagging"],
             cfg["organize_mode"], cfg["rewrite_mode"], cfg["dry_run"])

    if cfg["check"]:
        return run_check(cfg)
    if not Path(cfg["notes_dir"]).is_dir():
        log.error("Notes directory does not exist: %s", cfg["notes_dir"])
        return 2

    # Order matters: rewrite first (keeps path), then file the result, then tag it.
    phases = []
    if cfg["rewrite_mode"] != "off":
        phases.append(("rewrite", run_rewrite))
    if cfg["organize_mode"] != "off":
        phases.append(("organize", run_organize))
    if cfg["tagging"]:
        phases.append(("tagging", run_tagging))
    if not phases:
        log.info("Nothing enabled (tagging off, organize_mode/rewrite_mode off); exiting.")
        return 0

    with SingleInstanceLock(work / "notes_tagger.lock") as lock:
        if not lock.held:
            log.info("Another run is in progress; exiting.")
            return 0
        if not cfg["dry_run"]:
            cleanup_old_files(work, cfg)
        client = LLMClient(cfg)
        problem = client.check_available()
        if problem:
            log.error("LLM not available, aborting without touching files: %s", problem)
            return 0
        try:
            for name, fn in phases:
                log.info("== %s", name)
                log.info("%s done: %s", name, fn(client, cfg, work) or "nothing to do")
        except AbortRun as e:
            log.error("Aborting run: %s. The LLM looks unavailable; remaining notes will be retried next run.", e)
    return 0


def run_check(cfg):
    """--check: verify notes folder and LLM connection with one tiny request. Returns exit code."""
    ok = Path(cfg["notes_dir"]).is_dir()
    log.info("Notes folder %s: %s", cfg["notes_dir"], "OK" if ok else "NOT FOUND")
    client = LLMClient(cfg)
    problem = client.check_available()
    if problem:
        log.error("LLM check failed: %s", problem)
        return 1
    try:
        reply = client.complete("Reply with the single word OK.", "ping", cfg["max_output_tokens"])
    except LLMError as e:
        log.error("LLM test request failed: %s", e)
        return 1
    log.info("LLM reachable, model '%s' answered: %r", cfg["model"], strip_think(reply).strip()[:60])
    if ok:
        notes = [n for n in iter_notes(cfg) if not n.archived]
        log.info("Found %d note(s), %d without tags", len(notes), sum(1 for n in notes if not (n.tags - SYSTEM_TAGS)))
    log.info("Check %s", "passed" if ok else "FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)
