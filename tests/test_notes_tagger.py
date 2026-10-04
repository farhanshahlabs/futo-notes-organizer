# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for notes_tagger. Run from the project folder:  python -m unittest discover -s tests -v

Uses only the standard library and a local mock LLM server (no network, no real model)."""
import datetime as dt
import json
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import notes_tagger as nt  # noqa: E402


class MockLLM(BaseHTTPRequestHandler):
    """Answers based on which prompt it sees. Class attributes let tests tweak behaviour."""
    folder = "Cooking"
    rewrite = "## Clean\n\nFixed text.\n"
    tags = '["food","planning"]'
    broken = False
    calls = 0
    last_body = None

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "m1"}]})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockLLM.calls += 1
        MockLLM.last_body = data
        if MockLLM.broken:
            return self._send({"error": "boom"}, 500)
        system = data["messages"][0]["content"]
        if "file personal notes" in system:
            out = json.dumps({"folder": MockLLM.folder})
        elif "copy editor" in system:
            out = MockLLM.rewrite
        else:
            out = MockLLM.tags
        self._send({"choices": [{"message": {"content": out}, "finish_reason": "stop"}]})

    def log_message(self, *a):
        pass


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), MockLLM)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.notes = self.root / "notes"
        self.notes.mkdir()
        self.work = self.root / "work"
        MockLLM.folder, MockLLM.rewrite, MockLLM.tags = "Cooking", "## Clean\n\nFixed text.\n", '["food","planning"]'
        MockLLM.broken, MockLLM.calls = False, 0

    def tearDown(self):
        for h in list(nt.log.handlers):
            h.close()
            nt.log.removeHandler(h)
        self._tmp.cleanup()

    def note(self, rel, text, age_hours=2, raw=None):
        p = self.notes / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            p.write_bytes(raw)
        else:
            with open(p, "w", encoding="utf-8", newline="") as f:  # (Path.write_text(newline=) needs Python 3.10)
                f.write(text)
        t = time.time() - age_hours * 3600
        os.utime(p, (t, t))
        return p

    def run_tool(self, **overrides):
        check = overrides.pop("check", False)
        cfg = {"notes_dir": str(self.notes), "work_dir": str(self.work), "model": "m1",
               "base_url": f"http://127.0.0.1:{self.port}", "retry_wait_seconds": 0, "max_retries": 0}
        cfg.update(overrides)
        cfg_file = self.root / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        for h in list(nt.log.handlers):
            h.close()
            nt.log.removeHandler(h)
        return nt.main(["--config", str(cfg_file)] + (["--check"] if check else []))

    def read(self, rel):
        return (self.notes / rel).read_bytes().decode("utf-8")


class TestParsing(unittest.TestCase):
    def test_heading_is_not_a_tag_line(self):
        self.assertEqual(nt.split_tag_line("# Title\n\ntext\n")[1], [])

    def test_tag_line_detected(self):
        span, tags, body = nt.split_tag_line("#a #b-c\n\nbody\n")
        self.assertEqual(tags, ["a", "b-c"])
        self.assertEqual(body, "\nbody\n")

    def test_set_tags_insert_replace_remove(self):
        self.assertEqual(nt.set_tags("text\n", ["x"]), "#x\n\ntext\n")
        self.assertEqual(nt.set_tags("#a\n\ntext\n", ["a", "b"]), "#a #b\n\ntext\n")
        self.assertEqual(nt.set_tags("#a\n\ntext\n", []), "text\n")

    def test_set_tags_keeps_crlf(self):
        self.assertEqual(nt.set_tags("text\r\n", ["x"]), "#x\r\n\r\ntext\r\n")

    def test_parse_tags_cleans_and_dedupes(self):
        self.assertEqual(nt.parse_tags('<think>x</think>["Food","#food","a b"]', 6), ["food", "ab"])

    def test_parse_folder_validates_against_real_list(self):
        folders = ["Cooking", "Work/Projects"]
        self.assertEqual(nt.parse_folder('{"folder": "cooking"}', folders), "Cooking")
        self.assertIsNone(nt.parse_folder('{"folder": null}', folders))
        with self.assertRaises(nt.LLMError):
            nt.parse_folder('{"folder": "Invented"}', folders)

    def test_parse_rewrite_rejects_dropped_links_and_short_output(self):
        orig = "see [[Budget]] and https://x.ee/a. " + "word " * 60
        with self.assertRaises(nt.LLMError):
            nt.parse_rewrite("see the budget " + "word " * 60, orig)
        with self.assertRaises(nt.LLMError):
            nt.parse_rewrite("short", orig)
        ok = nt.parse_rewrite("```markdown\nsee [[Budget]] and https://x.ee/a. " + "word " * 60 + "\n```", orig)
        self.assertNotIn("```", ok)


class TestTagging(Base):
    def test_tags_only_untagged_notes(self):
        a = self.note("a.md", "cook pasta\n")
        b = self.note("b.md", "#mine\n\nkeep my tags\n")
        self.assertEqual(self.run_tool(), 0)
        self.assertEqual(self.read("a.md"), "#food #planning\n\ncook pasta\n")
        self.assertEqual(self.read("b.md"), "#mine\n\nkeep my tags\n")
        self.assertTrue(list((self.work / "backups").glob("*__a.md")))

    def test_second_run_makes_no_llm_calls(self):
        self.note("a.md", "cook pasta\n")
        self.run_tool()
        calls = MockLLM.calls
        MockLLM.tags = '["different","words"]'
        self.run_tool()
        self.assertEqual(self.read("a.md"), "#food #planning\n\ncook pasta\n")
        self.assertEqual(MockLLM.calls, calls)  # no new completion requests

    def test_recently_edited_note_is_not_touched(self):
        self.note("open.md", "being edited\n", age_hours=0.1)
        self.run_tool()
        self.assertEqual(self.read("open.md"), "being edited\n")
        self.assertEqual(MockLLM.calls, 0)
        self.note("open.md", "being edited\n", age_hours=1.5)
        self.run_tool()
        self.assertTrue(self.read("open.md").startswith("#food"))

    def test_crlf_and_bom_preserved(self):
        self.note("c.md", "", raw=b"\xef\xbb\xbfline1\r\nline2\r\n")
        self.run_tool()
        self.assertEqual((self.notes / "c.md").read_bytes(), b"\xef\xbb\xbf#food #planning\r\n\r\nline1\r\nline2\r\n")

    def test_hidden_and_archived_skipped(self):
        self.note(".hidden/x.md", "secret\n")
        self.note("Archived/old.md", "old\n")
        self.note("tagged-archived.md", "#archived\n\nold\n")
        self.run_tool()
        self.assertEqual(MockLLM.calls, 0)

    def test_only_system_tags_still_counts_as_untagged(self):
        self.note("ai.md", "#AI\n\nwritten by llm\n")
        self.run_tool()
        self.assertEqual(self.read("ai.md"), "#AI #food #planning\n\nwritten by llm\n")

    def test_dry_run_changes_nothing(self):
        self.note("a.md", "cook\n")
        self.run_tool(dry_run=True)
        self.assertEqual(self.read("a.md"), "cook\n")

    def test_llm_down_touches_nothing(self):
        self.note("a.md", "cook\n")
        self.assertEqual(self.run_tool(base_url="http://127.0.0.1:9"), 0)
        self.assertEqual(self.read("a.md"), "cook\n")

    def test_missing_notes_dir_exits_2(self):
        self.assertEqual(self.run_tool(notes_dir=str(self.root / "nope")), 2)

    def test_circuit_breaker_aborts_after_consecutive_failures(self):
        for i in range(6):
            self.note(f"n{i}.md", f"text {i}\n")
        MockLLM.broken = True  # /models works but every completion fails
        self.assertEqual(self.run_tool(max_consecutive_failures=3), 0)
        self.assertEqual(MockLLM.calls, 3)  # stopped after 3 failed notes, not all 6


class TestRequestShape(Base):
    def test_output_is_capped_and_extra_body_is_sent(self):
        self.note("a.md", "cook pasta\n")
        self.run_tool(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        body = MockLLM.last_body
        self.assertEqual(body["max_tokens"], 1024)  # default cap: a runaway reply can't hold the LLM for long
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_cap_can_be_renamed_or_disabled(self):
        self.note("a.md", "cook pasta\n")
        self.run_tool(max_tokens_field="max_completion_tokens", max_output_tokens=300)
        self.assertEqual(MockLLM.last_body["max_completion_tokens"], 300)
        self.assertNotIn("max_tokens", MockLLM.last_body)
        self.note("b.md", "buy milk\n")
        self.run_tool(max_tokens_field="")
        self.assertFalse(any(k in MockLLM.last_body for k in ("max_tokens", "max_completion_tokens")))

    def test_rewrite_uses_its_own_larger_cap(self):
        self.note("a.md", "#llmtoprocess\n\nsome text here\n")
        self.run_tool(rewrite_mode="manual", tagging=False, rewrite_max_tokens=4321)
        self.assertEqual(MockLLM.last_body["max_tokens"], 4321)

    def test_extra_body_must_be_an_object(self):
        with self.assertRaises(SystemExit):
            self.run_tool(extra_body="enable_thinking=false")


class TestOrganize(Base):
    def test_manual_move_removes_trigger_tag(self):
        self.note("Cooking/x.md", "x\n")
        self.note("Work/pasta.md", "#llmtomove #food\n\npasta tips\n")
        self.run_tool(organize_mode="manual", tagging=False)
        self.assertFalse((self.notes / "Work/pasta.md").exists())
        self.assertEqual(self.read("Cooking/pasta.md"), "#food\n\npasta tips\n")

    def test_trigger_only_tag_line_is_removed_entirely(self):
        self.note("Cooking/x.md", "x\n")
        self.note("pasta.md", "#llmtomove\n\npasta tips\n")
        self.run_tool(organize_mode="manual", tagging=False)
        self.assertEqual(self.read("Cooking/pasta.md"), "pasta tips\n")

    def test_auto_moves_new_root_notes_only(self):
        self.note("Cooking/x.md", "x\n")
        self.note("new.md", "a new note\n")
        self.note("Work/deep.md", "already filed\n")
        self.run_tool(organize_mode="auto", tagging=False)
        self.assertTrue((self.notes / "Cooking/new.md").exists())
        self.assertTrue((self.notes / "Work/deep.md").exists())

    def test_invented_folder_leaves_note_in_place(self):
        MockLLM.folder = "Nonexistent"
        self.note("Cooking/x.md", "x\n")
        self.note("a.md", "#llmtomove\n\ntext\n")
        self.run_tool(organize_mode="manual", tagging=False)
        self.assertEqual(self.read("a.md"), "#llmtomove\n\ntext\n")

    def test_off_mode_ignores_trigger(self):
        self.note("Cooking/x.md", "x\n")
        self.note("a.md", "#llmtomove\n\ntext\n")
        self.run_tool(organize_mode="off", tagging=False)
        self.assertTrue((self.notes / "a.md").exists())


class TestRewrite(Base):
    def test_manual_rewrite_archives_and_links(self):
        MockLLM.rewrite = "## Trip\n\nTrip to Tallinn. [[Budget]]\n"
        self.note("trip.md", "#llmtoprocess #travel\n\ntrip to tallin [[Budget]]\n")
        self.run_tool(rewrite_mode="manual", tagging=False)
        new = self.read("trip.md")
        self.assertTrue(new.startswith("#travel #AI\n\n## Trip"))
        self.assertIn("Original note (archived): [[trip (archived)]]", new)
        self.assertNotIn("llmtoprocess", new)
        arch = self.read("Archived/trip (archived).md")
        self.assertTrue(arch.startswith("#travel #archived\n\ntrip to tallin [[Budget]]"))
        self.assertTrue(arch.rstrip().endswith("Rewritten version: [[trip]]"))

    def test_bad_rewrite_leaves_note_untouched(self):
        MockLLM.rewrite = "no links here"
        original = "#llmtoprocess\n\nsee [[Keep Me]] please\n"
        self.note("a.md", original)
        self.run_tool(rewrite_mode="manual", tagging=False)
        self.assertEqual(self.read("a.md"), original)
        self.assertFalse((self.notes / "Archived").exists())

    def test_auto_skips_ai_notes_and_rewrites_new_ones(self):
        self.note("new.md", "new note text\n")
        self.note("done.md", "#AI\n\nalready written\n")
        self.run_tool(rewrite_mode="auto", tagging=False)
        self.assertTrue(self.read("new.md").startswith("#AI"))
        self.assertEqual(self.read("done.md"), "#AI\n\nalready written\n")

    def test_reprocessing_replaces_footer_and_keeps_old_archive(self):
        self.note("n.md", "#llmtoprocess\n\nfirst version\n")
        self.run_tool(rewrite_mode="manual", tagging=False)
        p = self.notes / "n.md"
        content = self.read("n.md").replace("#AI", "#AI #llmtoprocess")  # read BEFORE opening for write (truncates)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        t = time.time() - 7200
        os.utime(p, (t, t))
        MockLLM.rewrite = "## Second\n\nsecond version\n"
        self.run_tool(rewrite_mode="manual", tagging=False)
        new = self.read("n.md")
        self.assertEqual(new.count("Original note (archived)"), 1)
        self.assertIn("[[n (archived) (2)]]", new)
        self.assertEqual(len(list((self.notes / "Archived").glob("*.md"))), 2)

    def test_dry_run_saves_preview_only(self):
        self.note("a.md", "#llmtoprocess\n\ntext\n")
        self.run_tool(rewrite_mode="manual", tagging=False, dry_run=True)
        self.assertEqual(self.read("a.md"), "#llmtoprocess\n\ntext\n")
        self.assertTrue((self.work / "dry_run" / "a.md").exists())


class TestLock(Base):
    def test_lock_from_dead_process_is_ignored(self):
        import subprocess
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()  # a pid that definitely no longer runs
        self.work.mkdir(parents=True)
        (self.work / "notes_tagger.lock").write_text(str(proc.pid))
        self.note("a.md", "cook\n")
        self.run_tool()
        self.assertTrue(self.read("a.md").startswith("#food"))
        self.assertFalse((self.work / "notes_tagger.lock").exists())

    def test_lock_from_live_process_blocks_run(self):
        self.work.mkdir(parents=True)
        (self.work / "notes_tagger.lock").write_text(str(os.getpid()))  # this test process is alive
        self.note("a.md", "cook\n")
        self.run_tool()
        self.assertEqual(self.read("a.md"), "cook\n")
        self.assertTrue((self.work / "notes_tagger.lock").exists())

    def test_garbage_lock_is_treated_as_stale(self):
        self.work.mkdir(parents=True)
        (self.work / "notes_tagger.lock").write_text("")
        self.note("a.md", "cook\n")
        self.run_tool()
        self.assertTrue(self.read("a.md").startswith("#food"))


class TestCleanupAndCheck(Base):
    def test_cleanup_uses_name_timestamp_not_file_mtime(self):
        (self.work / "backups").mkdir(parents=True)
        (self.work / "logs").mkdir(parents=True)
        old = (dt.datetime.now() - dt.timedelta(days=40)).strftime("%Y%m%d-%H%M%S")
        new = (dt.datetime.now() - dt.timedelta(days=2)).strftime("%Y%m%d-%H%M%S")
        for p in (f"backups/{old}__a.md", f"backups/{new}__b.md", f"logs/run-{old}.log", f"logs/run-{new}.log"):
            (self.work / p).write_text("x")
        ancient = time.time() - 400 * 86400  # backups keep the note's own (old) mtime; must not matter
        os.utime(self.work / f"backups/{new}__b.md", (ancient, ancient))
        self.run_tool(tagging=False, rewrite_mode="manual")
        self.assertFalse((self.work / f"backups/{old}__a.md").exists())
        self.assertFalse((self.work / f"logs/run-{old}.log").exists())
        self.assertTrue((self.work / f"backups/{new}__b.md").exists())
        self.assertTrue((self.work / f"logs/run-{new}.log").exists())

    def test_check_command(self):
        self.assertEqual(self.run_tool(check=True), 0)
        self.assertEqual(self.run_tool(check=True, base_url="http://127.0.0.1:9"), 1)


if __name__ == "__main__":
    unittest.main()
