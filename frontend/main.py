import base64
import json
import os
import subprocess
import sys
from datetime import datetime

import webview

# === Add project root to sys.path ===
here = os.path.dirname(os.path.abspath(__file__))  # frontend/
project_root = os.path.abspath(os.path.join(here, ".."))  # ai4org/
sys.path.append(project_root)

# === Import your inference logic ===
from hallucination_reduction.inference import (  # noqa: E402
    build_embeddings,
    generate_answer,
    load_corpus,
    load_model,
    retrieve_relevant_chunks,
)

# User login history file
USER_HISTORY_FILE = os.path.join(here, "user_history.json")
CORPUS_FILE = os.path.join(project_root, "data", "processed", "corpus.txt")


class Api:
    def __init__(self):
        # Load once on startup to save time
        self.model, self.tokenizer, self.device = load_model()
        self.docs = load_corpus(CORPUS_FILE)
        self.embedder, self.corpus_embeddings = build_embeddings(
            self.docs, device=self.device
        )

        # Initialize user history
        self.user_history = self.load_user_history()

    def get_ai_response(self, question):
        try:
            retrieved_docs = retrieve_relevant_chunks(
                question, self.embedder, self.corpus_embeddings, self.docs
            )
            answer = generate_answer(
                self.model, self.tokenizer, self.device, question, retrieved_docs
            )
            return {"status": "ok", "answer": answer}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load_user_history(self):
        """Load user login history from file"""
        try:
            if os.path.exists(USER_HISTORY_FILE):
                with open(USER_HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading user history: {e}")
            return {}

    def save_user_history(self):
        """Save user login history to file"""
        try:
            with open(USER_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.user_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving user history: {e}")

    def log_user_login(self, user_name, user_email):
        """Log user login with timestamp"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if user_name not in self.user_history:
            self.user_history[user_name] = {"email": user_email, "login_history": []}

        self.user_history[user_name]["login_history"].append(
            {
                "timestamp": current_time,
                "date": current_time.split()[0],
                "time": current_time.split()[1],
            }
        )

        # Keep only last 50 logins per user
        if len(self.user_history[user_name]["login_history"]) > 50:
            self.user_history[user_name]["login_history"] = self.user_history[
                user_name
            ]["login_history"][-50:]

        self.save_user_history()
        return {"status": "success", "message": "Login logged successfully"}

    def get_user_history(self, admin_pin):
        """Get user login history (admin only)"""
        if admin_pin != "9999":
            return {"status": "error", "message": "Unauthorized access"}

        return {
            "status": "success",
            "data": self.user_history,
            "total_users": len(self.user_history),
        }

    def get_user_stats(self, admin_pin):
        """Get user statistics (admin only)"""
        if admin_pin != "9999":
            return {"status": "error", "message": "Unauthorized access"}

        stats = {
            "total_users": len(self.user_history),
            "total_logins": sum(
                len(user["login_history"]) for user in self.user_history.values()
            ),
            "users": [],
        }

        for user_name, user_data in self.user_history.items():
            user_stats = {
                "name": user_name,
                "email": user_data["email"],
                "login_count": len(user_data["login_history"]),
                "last_login": (
                    user_data["login_history"][-1]["timestamp"]
                    if user_data["login_history"]
                    else "Never"
                ),
            }
            stats["users"].append(user_stats)

        return {"status": "success", "data": stats}

    def save_file(self, filename, file_data_base64):
        try:
            file_data = base64.b64decode(file_data_base64)
            folder = os.path.join(project_root, "data", "raw")
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)
            with open(filepath, "wb") as f:
                f.write(file_data)

            # Force CPU to avoid CUDA OOM during training
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["TOKENIZERS_PARALLELISM"] = "false"
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:true")
            subprocess.run(
                ["python", "-m", "hallucination_reduction.main"],
                cwd=project_root,
                check=True,
                env=env,
            )

            return "success"
        except Exception as e:
            return f"error: {str(e)}"

    def save_file_chunk(self, filename, chunk_data, chunk_index, is_last):
        try:
            folder = os.path.join(here, "../data/raw")
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)

            mode = "ab" if chunk_index > 0 else "wb"

            with open(filepath, mode) as f:
                f.write(base64.b64decode(chunk_data))

            if is_last:
                project_root = os.path.abspath(os.path.join(here, ".."))
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["TOKENIZERS_PARALLELISM"] = "false"
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:true")
                subprocess.run(
                    ["python", "-m", "hallucination_reduction.main"],
                    cwd=project_root,
                    check=True,
                    env=env,
                )

            return "success"
        except Exception as e:
            return f"error: {str(e)}"


def try_backends():
    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import http.server
    import socket
    import socketserver
    import threading

    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    PORT = get_free_port()

    def start_server():
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=here, **kwargs)

            def log_message(self, format, *args):
                pass

        try:
            with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
                httpd.serve_forever()
        except OSError as e:
            print(f"Port {PORT} is in use or could not be bound: {e}")

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    backends = ["qt", "gtk", "edgechromium", "cef"]
    for backend in backends:
        try:
            print(f"\nAttempting {backend} backend...")
            api = Api()
            url = f"http://127.0.0.1:{PORT}/html/index.html"
            print(f"Loading URL: {url}")

            webview.create_window(
                title="AI4ORG - AI For Organization",
                url=url,
                width=1400,
                height=900,
                resizable=True,
                js_api=api,
                min_size=(800, 600),
            )
            webview.start(gui=backend, debug=False)
            return True
        except Exception as e:
            print(f"{backend} backend failed: {e}")
            continue
    return False


if __name__ == "__main__":
    success = try_backends()
    if not success:
        print("All GUI backends failed.")
        sys.exit(1)
