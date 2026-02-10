import base64
import os

from frontend.main import Api


class TestSaveFileChunk:
    def test_creates_file_on_first_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frontend.main.here", str(tmp_path))
        monkeypatch.setattr("frontend.main.subprocess.run", lambda *a, **k: None)

        api = Api.__new__(Api)

        data = base64.b64encode(b"hello").decode()
        result = api.save_file_chunk(
            filename="test.txt",
            chunk_data=data,
            chunk_index=0,
            is_last=False,
        )

        assert result == "success"

        file_path = os.path.join(tmp_path, "../data/raw/test.txt")
        with open(file_path, "rb") as f:
            assert f.read() == b"hello"

    def test_appends_on_subsequent_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frontend.main.here", str(tmp_path))
        monkeypatch.setattr("frontend.main.subprocess.run", lambda *a, **k: None)

        api = Api.__new__(Api)

        chunk1 = base64.b64encode(b"hello ").decode()
        chunk2 = base64.b64encode(b"world").decode()

        api.save_file_chunk("test.txt", chunk1, 0, False)
        api.save_file_chunk("test.txt", chunk2, 1, False)

        file_path = os.path.join(tmp_path, "../data/raw/test.txt")
        with open(file_path, "rb") as f:
            assert f.read() == b"hello world"

    def test_runs_subprocess_on_last_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("frontend.main.here", str(tmp_path))
        monkeypatch.setattr("frontend.main.subprocess.run", lambda *a, **k: None)

        api = Api.__new__(Api)

        called = {"ran": False}

        def fake_run(*args, **kwargs):
            called["ran"] = True

        monkeypatch.setattr("frontend.main.subprocess.run", fake_run)

        data = base64.b64encode(b"data").decode()
        result = api.save_file_chunk("final.txt", data, 0, True)

        assert result == "success"
        assert called["ran"] is True

    def test_returns_error_on_exception(self, monkeypatch):
        monkeypatch.setattr("frontend.main.subprocess.run", lambda *a, **k: None)

        api = Api.__new__(Api)

        def fake_open(*args, **kwargs):
            raise IOError("disk failure")

        monkeypatch.setattr("builtins.open", fake_open)

        data = base64.b64encode(b"x").decode()
        result = api.save_file_chunk("fail.txt", data, 0, False)

        assert result.startswith("error:")
