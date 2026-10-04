# Notes Auto-Tagger / Organizer / Rewriter

A small, dependency-free Python script that looks after a folder of Markdown notes using an LLM.
Works with a local LLM (MLX, Ollama, LM Studio, llama.cpp, ...) or a hosted one (OpenAI, Anthropic,
OpenRouter, Groq, ...). Designed to run unattended on a schedule.

Python does all the work - finding notes, deciding which qualify, moving/archiving files, editing tag
lines, creating links and validating what comes back. The LLM is only asked for **tags**, a
**folder choice**, or **rewritten text**.

| Feature | What it does | Setting (default) |
|---|---|---|
| **Tagging** | Adds tags to notes that have **no tags yet** (never re-tags a tagged note) | `tagging` (`true`) |
| **Organize** | Moves notes into the best *existing* folder | `organize_mode` (`off`) |
| **Rewrite** | Re-formats a note with clean Markdown, grammar and spelling; archives the original and links both | `rewrite_mode` (`off`) |

Modes for organize/rewrite: `off` - `manual` (only notes you flag with a trigger tag) - `auto` (flagged notes
**plus** newly created ones).

## Requirements

Python 3.9+. Standard library only - nothing to `pip install`.

## Quick start

```bash
cp config.example.json config.json   # then edit it
python notes_tagger.py --check       # verifies the notes folder and LLM connection (one tiny test request)
python notes_tagger.py --dry-run     # logs what WOULD happen, changes nothing
python notes_tagger.py               # real run
```

Logs: `<work_dir>/logs/` (`latest.log` + one file per run; console output only when run from a terminal).
Backups of every note before it is changed: `<work_dir>/backups/`. Logs and backups older than 30 days are
deleted automatically (`log_retention_days`, `backup_retention_days`; `0` keeps them forever).
In `--dry-run`, proposed rewrites are saved to `<work_dir>/dry_run/` so you can review them.

Run the tests (standard library only, no network): `python -m unittest discover -s tests -v`

## Note format and tags

Tags live on a line of `#tag` tokens at the very top of a `.md` file:

```
#travel #dates

Trip notes...
```

A `# Heading` is not a tag line. Notes are linked with wikilinks, `[[Note title]]`, where the title is the
note's file name without `.md`. Tags described below are matched case-insensitively.

| Tag | Who adds it | Meaning |
|---|---|---|
| `#llmtomove` | you | "File this note into the right folder." Removed once moved |
| `#llmtoprocess` | you | "Rewrite this note." Removed from the new note; original is archived |
| `#AI` | script | The note's text was written by the LLM (rewritten notes) |
| `#archived` | script | Archived original. Archived notes are never tagged, moved or rewritten again |

Trigger tags work on **any** note in any folder (not just new ones) as long as the matching mode is not `off`.
They must be on the tag line (first line), which is where the notes app puts tags.

## Feature details

### Order of a run
1. **Rewrite** (the note keeps its name and location) 2. **Organize** 3. **Tagging**. So a new note can be
cleaned up, filed and tagged in one run.

### Organize
- Candidates: notes with `#llmtomove`, and (in `auto`) notes that sit directly in the notes root, were created in the
  last `new_note_days` days.
- Python sends the LLM the list of existing folders (nested ones as `Work/Projects`; hidden and the archive folder
  excluded) plus the note. The LLM must answer with a folder from that list or `null`. Any other answer is treated
  as a failed attempt. **No folders are ever created.**
- If the LLM says nothing fits, the note stays where it is. A `#llmtomove` tag is **kept** in that case (so you can
  see it wasn't filed), which means it is retried on each run.
- A name clash in the destination gets a ` (2)` suffix.

### Rewrite
- Candidates: notes with `#llmtoprocess`, and (in `auto`) notes created in the last `new_note_days` days that aren't
  already `#AI`.
- The rewritten note **keeps the original file name and folder**, so existing `[[links]]` to it stay valid. Its tag line
  keeps the old tags, minus the trigger, plus `#AI`. At the bottom it gets:
  ```
  ---

  Original note (archived): [[My note (archived)]]
  ```
- The original is saved to `<notes_dir>/<archive_folder>/My note (archived).md` (the folder is created if missing) with
  its tags plus `#archived`, its text untouched, and a link back at the bottom: `Rewritten version: [[My note]]`.
- Python rejects and retries a rewrite if it is under half the original's length, or drops any `[[wikilink]]` or URL
  from the original. Notes longer than `rewrite_max_chars` are skipped rather than truncated, and a response cut off
  by a token limit is treated as failed.
- Re-running `#llmtoprocess` on an already-rewritten note works: the old footer is stripped and a new archive copy is
  made; older archived versions stay in the archive folder.
- If writing fails midway, the archive copy is removed so the original remains the only copy.

### Tagging
Only notes with **no tags** are tagged; they get 3-6 tags. Once a note has tags (yours or ours) it is never
tagged again, so tags can't pile up from run to run. System tags (`#AI`, `#archived`, `#llmtomove`,
`#llmtoprocess`) don't count as tags, so a rewritten `#AI` note still gets topic tags. Archived notes are skipped.
Set `max_age_hours` to limit tagging to notes modified recently (default: all untagged notes).

## Configuration

Precedence: **command-line flag > `config.json` > default**. `config.json` sits next to the script (or pass
`--config path`). Relative/`~` paths are expanded.

| Key / flag | Default | Description |
|---|---|---|
| `notes_dir` / `--notes-dir` | *required* | Folder scanned recursively for `*.md` (hidden files/dirs skipped) |
| `provider` / `--provider` | `openai` | `openai` (any OpenAI-compatible API) or `anthropic` |
| `base_url` / `--base-url` | provider default | e.g. `http://localhost:8080` (a trailing `/v1` is fine) |
| `model` / `--model` | *required* | Model name as the provider knows it |
| `api_key_env` / `--api-key-env` | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Name of the **environment variable** holding the key |
| `work_dir` / `--work-dir` | `~/notes_tagger_data` | Where `backups/`, `logs/` and the lock file live |
| `tagging` / `--no-tagging` | `true` | Enable the tagging step |
| `organize_mode` / `--organize-mode` | `off` | `off`, `manual` or `auto` |
| `rewrite_mode` / `--rewrite-mode` | `off` | `off`, `manual` or `auto` |
| `new_note_days` | `3` | "Recently created" window for `auto` modes |
| `min_idle_minutes` | `60` | **Never touch a note edited more recently than this** (it may be open in the app). Applies to every feature, including trigger tags |
| `max_notes_per_run` | `0` | Cap on notes sent to the LLM per feature per run (`0` = unlimited); the rest wait for the next run |
| `max_consecutive_failures` | `3` | Abort the run after this many notes in a row fail (LLM probably down) instead of waiting hours |
| `log_retention_days` | `30` | Delete run logs older than this (`0` = keep forever) |
| `backup_retention_days` | `30` | Delete note backups older than this (`0` = keep forever) |
| `archive_folder` / `--archive-folder` | `Archived` | Folder (under `notes_dir`) for archived originals |
| `max_age_hours` | `null` | Tagging: only consider untagged notes modified within this window (`null` = all) |
| `max_retries` | `2` | Extra attempts per LLM call after the first fails |
| `retry_wait_seconds` | `3600` | Wait between attempts (local LLM may be busy); use e.g. `10` for hosted APIs |
| `request_timeout_seconds` | `600` | Per-request timeout |
| `max_note_chars` | `12000` | Tagging/organize: long notes are truncated before being sent |
| `rewrite_max_chars` | `20000` | Rewrite: longer notes are skipped |
| `max_output_tokens` | `1024` | Cap on the reply for tagging, organizing and `--check`. Protects you from a runaway model reply that would otherwise block a local LLM for many minutes |
| `rewrite_max_tokens` | `8192` | Cap on the reply when rewriting a note (also required by the Anthropic API) |
| `max_tokens_field` | `max_tokens` | Name of the request field that carries the cap. Use `max_completion_tokens` for newer OpenAI models, or `""` to send no cap |
| `extra_body` | `{}` | Extra JSON fields added to every OpenAI-style request, for provider-specific switches (see below) |
| `min_tags` / `max_tags` | `3` / `6` | Tag count requested / kept |
| `skip_model_check` | `false` | Don't require the model to be listed by `GET /v1/models` |
| `dry_run` / `--dry-run` | `false` | Log only, change nothing |

API keys are **never** read from `config.json` - set them as environment variables.

Note: "created" uses the file's creation time on Windows and macOS. Linux has no creation time, so the script falls
back to the modified time there. Notes synced from another device count as created when they arrived on this machine.

### Provider examples

**Local LLM (OpenAI-compatible, no key):**
```json
{ "provider": "openai", "base_url": "http://localhost:8080", "model": "mlx-community/Qwen3.8-27B-4bit" }
```

**Local "thinking" models (Qwen3 and similar): turn thinking off.** These models reason at length before answering
unless told not to, which is wasted effort for tagging and filing, and on a greedy-decoding server it can turn into an
endless loop that blocks the LLM for a long time. For servers that accept chat-template options (`mlx_lm.server`,
vLLM, llama.cpp with `--jinja`) add this to `config.json`:
```json
"extra_body": { "chat_template_kwargs": { "enable_thinking": false } }
```
Together with the default `max_output_tokens` cap, a stuck reply can cost at most a minute or two.

**OpenAI:**
```json
{ "provider": "openai", "model": "gpt-4o-mini", "retry_wait_seconds": 10 }
```
```
set OPENAI_API_KEY=sk-...        (Windows cmd)    export OPENAI_API_KEY=sk-...   (macOS/Linux)
```

**Anthropic:**
```json
{ "provider": "anthropic", "model": "claude-haiku-4-5-20251001", "retry_wait_seconds": 10 }
```
Set `ANTHROPIC_API_KEY`.

**Other OpenAI-compatible providers** (OpenRouter, Groq, Together, ...): use `provider: openai`, set `base_url`
to the provider's URL (e.g. `https://openrouter.ai/api`) and `api_key_env` to the variable holding your key.

> Privacy: with a hosted provider, your note text is sent to that provider. Use a local LLM for private notes.

## Scheduling

### Windows (Task Scheduler)

Give the task the **environment variable** and **absolute paths**. A tiny wrapper `run_tagger.bat` is easiest
(keep it private if it contains a key):

```bat
@echo off
set OPENAI_API_KEY=sk-...
"C:\Python39\pythonw.exe" "C:\tools\notes_organizer\notes_tagger.py" --config "C:\tools\notes_organizer\config.json"
```

Register it to run daily at 03:00:

```bat
schtasks /Create /TN "NotesAutoTagger" /TR "C:\tools\notes_organizer\run_tagger.bat" /SC DAILY /ST 03:00
schtasks /Run /TN "NotesAutoTagger"     :: test it now
```

With a local LLM (no key) you can skip the wrapper and point `/TR` at
`"C:\Python39\pythonw.exe" "C:\tools\notes_organizer\notes_tagger.py"`. In Task Scheduler you may also enable
*Run task as soon as possible after a scheduled start is missed*.

### macOS / Linux (cron)

```cron
0 3 * * * OPENAI_API_KEY=sk-... /usr/bin/python3 /opt/notes_organizer/notes_tagger.py --config /opt/notes_organizer/config.json
```

## Safety and behavior

- **Backups:** every note is copied to `<work_dir>/backups/<timestamp>__<name>.md` before it is changed. Copy it
  back to undo. Writes are atomic (temp file + replace). UTF-8, BOM and line endings are preserved.
- **Single instance:** a lock file (`notes_tagger.lock`) in `work_dir` stops overlapping runs. It records the process
  ID, so a lock left behind by a killed run (closed terminal, shutdown, crash) is detected and ignored automatically.
  If you ever need to clear it by hand, just delete that file.
- **Availability check:** before touching anything the script calls `GET /v1/models`; if the endpoint is down or
  the model isn't listed, it logs and exits 0 without modifying files.
- **Retries:** failed/unparseable/invalid LLM replies are retried `max_retries` times, waiting `retry_wait_seconds`
  between attempts; if all fail the note is skipped and the run continues.
- **Locked/unreadable notes** are logged and skipped. If a note changes while the LLM works on it, it is skipped.
- **Reasoning models:** `<think>...</think>` blocks in responses are ignored.
- **Exit codes:** `0` on success (even if some notes were skipped), `2` if `notes_dir` is missing, `1` on an unexpected error.
- **Notes you might have open:** a note edited within the last `min_idle_minutes` (default 60) is skipped by every
  feature, including notes you just flagged with a trigger tag; they are handled by the first run after the hour is up.
  (The app doesn't tell us which notes are open, so recent modification is the signal used.)
- **Stops when the LLM is down:** after `max_consecutive_failures` failed notes in a row the run aborts (exit 0) and the
  remaining notes are retried next run.
- **One-shot by design:** tagged notes aren't re-tagged, `#AI` notes aren't auto-rewritten again, and moved notes
  leave the notes root.
- **Recommendation:** run `--check` and `"dry_run": true` first, and start `rewrite_mode` at `manual`; `auto` rewrites
  every new note.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). You're free to use, modify and share this script; derivative works must
stay under the same license.
