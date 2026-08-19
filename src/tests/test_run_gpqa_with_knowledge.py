import csv
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_gpqa_with_knowledge.py"
SPEC = importlib.util.spec_from_file_location("gpqa_knowledge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PromptAndDataTests(unittest.TestCase):
    def test_system_prompt_contains_complete_knowledge(self):
        knowledge = "first line\r\nsecond line\n<xml-like>value</xml-like>"
        prompt = MODULE.build_system_prompt(
            knowledge, "Instructions\n<knowledge_text>\n{{knowledge}}\n</knowledge_text>"
        )
        self.assertIn(knowledge, prompt)
        self.assertNotIn("{{knowledge}}", prompt)

    def test_system_prompt_is_empty_without_template(self):
        self.assertEqual(MODULE.build_system_prompt("complete knowledge"), "")

    def test_system_prompt_template_requires_knowledge_placeholder(self):
        with self.assertRaisesRegex(ValueError, "\\{\\{knowledge\\}\\}"):
            MODULE.build_system_prompt("complete knowledge", "Instructions only")

    def test_bundled_system_prompt_preserves_previous_instructions(self):
        template_path = MODULE_PATH.parent / "sample" / "system-prompt.txt"
        template = MODULE.read_text_exact(template_path)
        prompt = MODULE.build_system_prompt("sentinel knowledge", template)
        self.assertIn("質問回答のために利用可能な知識", prompt)
        self.assertIn("sentinel knowledge", prompt)
        self.assertNotIn("{{knowledge}}", prompt)

    def test_parse_answer_prefers_explicit_final_line(self):
        response = "At first I considered (A).\nThe evidence favors C.\nFINAL_ANSWER: C"
        self.assertEqual(MODULE.parse_answer(response), "C")
        self.assertEqual(MODULE.parse_answer("The correct answer is (B)"), "B")
        self.assertIsNone(MODULE.parse_answer("I cannot determine it."))

    def test_parse_json_answer_is_strict(self):
        self.assertEqual(MODULE.parse_json_answer('{"answer":"C"}'), ("C", {"answer": "C"}))
        self.assertEqual(MODULE.parse_json_answer('```json\n{"answer":"B"}\n```')[0], "B")
        self.assertIsNone(MODULE.parse_json_answer('{"answer":"A","reason":"x"}')[0])
        self.assertIsNone(MODULE.parse_json_answer("FINAL_ANSWER: A")[0])

    def test_jsonl_record_escapes_unicode_line_separators(self):
        serialized = MODULE.jsonl_record({"text": "before\u0085middle\u2028after\u2029end"})
        self.assertEqual(len(serialized.splitlines()), 1)
        self.assertEqual(
            json.loads(serialized), {"text": "before\u0085middle\u2028after\u2029end"}
        )

    def test_default_output_dir_is_under_current_run_directory(self):
        output_dir = MODULE.default_output_dir("diamond")
        self.assertEqual(output_dir.parent, Path.cwd() / "run")
        self.assertTrue(output_dir.name.startswith("diamond-"))

    def test_csv_loading_is_seeded_and_tracks_label_not_duplicate_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpqa.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(MODULE.REQUIRED_COLUMNS) + ["Record ID"])
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "Q?",
                        "Correct Answer": "duplicate",
                        "Incorrect Answer 1": "duplicate",
                        "Incorrect Answer 2": "wrong 2",
                        "Incorrect Answer 3": "wrong 3",
                        "Record ID": "record-1",
                    }
                )
            first, _ = MODULE.load_examples(path, "main", "unused", seed=7)
            second, _ = MODULE.load_examples(path, "main", "unused", seed=7)
            self.assertEqual(first, second)
            self.assertEqual(first[0].choices[first[0].correct_index], "duplicate")
            self.assertEqual(first[0].question_id, "record-1")

    def test_official_style_zip_is_read_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.zip"
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=list(MODULE.REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerow(
                {
                    "Question": "Q?",
                    "Correct Answer": "yes",
                    "Incorrect Answer 1": "no1",
                    "Incorrect Answer 2": "no2",
                    "Incorrect Answer 3": "no3",
                }
            )
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("dataset/gpqa_main.csv", buffer.getvalue())
            examples, source = MODULE.load_examples(path, "main", "unused", seed=0)
            self.assertEqual(len(examples), 1)
            self.assertTrue(source.endswith("!/dataset/gpqa_main.csv"))


class _FakeResponse:
    def __init__(self):
        self.headers = {"x-request-id": "req-test"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"content": "Reasoning.\nFINAL_ANSWER: D"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }
        ).encode("utf-8")


class ApiTests(unittest.TestCase):
    def test_openai_compatible_request_uses_system_message(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse()

        with mock.patch.object(MODULE.urllib.request, "urlopen", side_effect=fake_urlopen):
            text, usage, request_id = MODULE.call_openai_compatible(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
                system_prompt="SYSTEM WITH ALL KNOWLEDGE",
                user_prompt="QUESTION",
                timeout=2,
                max_tokens=32,
                temperature=None,
                retries=0,
            )
        received = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(text, "Reasoning.\nFINAL_ANSWER: D")
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(request_id, "req-test")
        self.assertEqual(received["messages"][0]["role"], "system")
        self.assertEqual(
            received["messages"][0]["content"], "SYSTEM WITH ALL KNOWLEDGE"
        )
        self.assertEqual(received["messages"][1]["role"], "user")
        self.assertEqual(captured["request"].get_header("Authorization"), "Bearer test-key")

    def test_ollama_request_and_response_shape(self):
        captured = {}

        class FakeOllamaResponse(_FakeResponse):
            def read(self):
                return json.dumps(
                    {
                        "message": {"role": "assistant", "content": '{"answer":"A"}'},
                        "prompt_eval_count": 20,
                        "eval_count": 5,
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["request"] = request
            return FakeOllamaResponse()

        with mock.patch.object(MODULE.urllib.request, "urlopen", side_effect=fake_urlopen):
            text, usage, _ = MODULE.call_ollama(
                base_url="https://ollama.com",
                api_key="test-key",
                model="gpt-oss:120b",
                system_prompt="KNOWLEDGE",
                user_prompt="QUESTION",
                timeout=2,
                temperature=0.0,
                seed=0,
                retries=0,
            )
        received = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(captured["request"].full_url, "https://ollama.com/api/chat")
        self.assertEqual(received["model"], "gpt-oss:120b")
        self.assertNotIn("format", received)
        self.assertNotIn("max_tokens", received)
        self.assertNotIn("options", received)
        self.assertEqual(text, '{"answer":"A"}')
        self.assertEqual(usage, {"prompt_eval_count": 20, "eval_count": 5})


class EvaluationTests(unittest.TestCase):
    def test_run_scores_and_writes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_path = root / "knowledge.txt"
            knowledge_path.write_text("complete knowledge", encoding="utf-8")
            env_path = root / ".env"
            env_path.write_text("API_KEY=test-key\n", encoding="utf-8")
            prompt_path = root / "system_prompt.txt"
            prompt_path.write_text(
                "Use this knowledge:\n{{knowledge}}", encoding="utf-8"
            )
            data_path = root / "gpqa.csv"
            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(MODULE.REQUIRED_COLUMNS) + ["Record ID"])
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "Q?",
                        "Correct Answer": "yes",
                        "Incorrect Answer 1": "no1",
                        "Incorrect Answer 2": "no2",
                        "Incorrect Answer 3": "no3",
                        "Record ID": "record-1",
                    }
                )
            examples, _ = MODULE.load_examples(data_path, "main", "unused", seed=0)
            correct_letter = MODULE.CHOICE_LETTERS[examples[0].correct_index]
            output_dir = root / "output"
            args = MODULE.build_parser().parse_args(
                [
                    str(knowledge_path),
                    "--system-prompt-file",
                    str(prompt_path),
                    "--data-file",
                    str(data_path),
                    "--model",
                    "test-model",
                    "--base-url",
                    "http://localhost:8000/v1",
                    "--env-file",
                    str(env_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            fake_result = (f'{{"answer":"{correct_letter}"}}', {"eval_count": 12}, "req-1")
            with mock.patch.object(MODULE, "call_ollama", return_value=fake_result):
                self.assertEqual(MODULE.run(args), 0)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            result = json.loads((output_dir / "results.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["usage_totals"], {"eval_count": 12})
            self.assertEqual(summary["system_prompt_file"], str(prompt_path.resolve()))
            self.assertTrue(summary["knowledge_is_complete_substring"])
            self.assertTrue(result["is_correct"])
            self.assertTrue(result["response_format_valid"])
            self.assertEqual(result["question_id"], "record-1")


if __name__ == "__main__":
    unittest.main()
