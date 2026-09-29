import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from streamlit.testing.v1 import AppTest


class _FakeModels:
    def embed_content(self, *, model, contents, config):
        vector = [0.0] * 768
        vector[0] = 1.0
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector)])

    def generate_content(self, *, model, contents):
        if "force-generation-error" in contents:
            raise RuntimeError("simulated generation failure")
        if "What is the project name?" in contents:
            text = "The project name is Atlas."
        else:
            text = "I could not find that detail in the uploaded documents."
        return SimpleNamespace(text=text)


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


class StreamlitStateTests(unittest.TestCase):
    def test_upload_ask_reset_and_upload_again(self) -> None:
        app_test = AppTest.from_file("app.py").run()
        self.assertEqual(len(app_test.exception), 0)

        first_pdf = ("one.pdf", _pdf_bytes("Project name: Atlas."), "application/pdf")
        pdfs = [
            first_pdf,
            ("two.pdf", _pdf_bytes("Project owner: Ada."), "application/pdf"),
            ("three.pdf", _pdf_bytes("Launch year: 2024."), "application/pdf"),
        ]
        with patch("google.genai.Client", return_value=_FakeClient()):
            app_test.file_uploader[0].set_value(first_pdf).run()
            self.assertTrue(any("1 PDF uploaded" in item.value for item in app_test.markdown))
            app_test.button[0].click().run()
            self.assertEqual(app_test.session_state["processed_files"], ["one.pdf"])
            self.assertEqual(len(app_test.session_state["document_chunks"]), 1)

            app_test.file_uploader[0].set_value(pdfs).run()
            self.assertTrue(any("3 PDFs uploaded" in item.value for item in app_test.markdown))
            markdown_before_processing = "\n".join(item.value for item in app_test.markdown)
            self.assertNotIn("one.pdf", markdown_before_processing)
            self.assertNotIn("two.pdf", markdown_before_processing)
            self.assertNotIn("three.pdf", markdown_before_processing)
            app_test.button[0].click().run()
            self.assertEqual(app_test.session_state["processed_files"], ["one.pdf", "two.pdf", "three.pdf"])
            self.assertEqual(len(app_test.session_state["document_chunks"]), 3)
            self.assertTrue(any("Processed: 3 PDFs" in item.value and "3 searchable chunks" in item.value for item in app_test.markdown))
            markdown_after_processing = "\n".join(item.value for item in app_test.markdown)
            self.assertNotIn("one.pdf", markdown_after_processing)
            self.assertNotIn("two.pdf", markdown_after_processing)
            self.assertNotIn("three.pdf", markdown_after_processing)

            app_test.text_input[0].set_value("What is the project name?").run()
            app_test.button[-1].click().run()
            self.assertEqual(app_test.session_state["chat_history"][-1]["question"], "What is the project name?")
            self.assertEqual(app_test.text_input[0].value, "")
            self.assertEqual(len(app_test.get("expander")), 1)

            app_test.text_input[0].set_value("What is the project budget?").run()
            app_test.button[-1].click().run()
            self.assertEqual(app_test.session_state["chat_history"][-1]["question"], "What is the project budget?")
            self.assertEqual(
                app_test.session_state["chat_history"][-1]["answer"],
                "The answer to this question was not found in the uploaded documents.",
            )
            self.assertEqual(app_test.text_input[0].value, "")
            self.assertEqual(len(app_test.get("expander")), 1)

            app_test.text_input[0].set_value("force-generation-error").run()
            app_test.button[-1].click().run()
            self.assertEqual(app_test.session_state["chat_history"][-1]["question"], "force-generation-error")
            self.assertIn("couldn't generate", app_test.session_state["chat_history"][-1]["answer"])
            self.assertEqual(app_test.session_state["chat_history"][-1]["sources"], [])
            self.assertEqual(app_test.text_input[0].value, "")

            app_test.button[1].click().run()
            self.assertEqual(app_test.session_state["processed_files"], [])
            self.assertEqual(app_test.session_state["document_chunks"], [])
            self.assertIsNone(app_test.session_state["faiss_index"])
            self.assertEqual(app_test.session_state["chat_history"], [])
            self.assertEqual(app_test.file_uploader[0].value, [])
            self.assertEqual(app_test.text_input[0].value, "")

            app_test.file_uploader[0].set_value(
                ("again.pdf", _pdf_bytes("Fresh document."), "application/pdf")
            ).run()
            app_test.button[0].click().run()
            self.assertEqual(app_test.session_state["processed_files"], ["again.pdf"])
            self.assertEqual(len(app_test.session_state["document_chunks"]), 1)
            self.assertEqual(len(app_test.exception), 0)


if __name__ == "__main__":
    unittest.main()
