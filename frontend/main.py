import webview
import os
import base64
import sys
import subprocess
import torch
import json
import datetime
from datetime import datetime

# === Import your inference logic ===
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hallucination_reduction.inference import (
    load_model, load_corpus, build_embeddings, retrieve_relevant_chunks, generate_answer
)

here = os.path.dirname(os.path.abspath(__file__))

# User login history file
USER_HISTORY_FILE = os.path.join(here, 'user_history.json')


class Api:
    def __init__(self):
        # Load once on startup to save time
        self.model, self.tokenizer, self.device = load_model()
        self.docs = load_corpus()
        self.embedder, self.corpus_embeddings = build_embeddings(self.docs, device=self.device)
        
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
                with open(USER_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading user history: {e}")
            return {}

    def save_user_history(self):
        """Save user login history to file"""
        try:
            with open(USER_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.user_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving user history: {e}")

    def log_user_login(self, user_name, user_email):
        """Log user login with timestamp"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if user_name not in self.user_history:
            self.user_history[user_name] = {
                "email": user_email,
                "login_history": []
            }
        
        self.user_history[user_name]["login_history"].append({
            "timestamp": current_time,
            "date": current_time.split()[0],
            "time": current_time.split()[1]
        })
        
        # Keep only last 50 logins per user
        if len(self.user_history[user_name]["login_history"]) > 50:
            self.user_history[user_name]["login_history"] = self.user_history[user_name]["login_history"][-50:]
        
        self.save_user_history()
        return {"status": "success", "message": "Login logged successfully"}

    def get_user_history(self, admin_pin):
        """Get user login history (admin only)"""
        # Admin PIN: 9999
        if admin_pin != "9999":
            return {"status": "error", "message": "Unauthorized access"}
        
        return {
            "status": "success", 
            "data": self.user_history,
            "total_users": len(self.user_history)
        }

    def get_user_stats(self, admin_pin):
        """Get user statistics (admin only)"""
        if admin_pin != "9999":
            return {"status": "error", "message": "Unauthorized access"}
        
        stats = {
            "total_users": len(self.user_history),
            "total_logins": sum(len(user["login_history"]) for user in self.user_history.values()),
            "users": []
        }
        
        for user_name, user_data in self.user_history.items():
            user_stats = {
                "name": user_name,
                "email": user_data["email"],
                "login_count": len(user_data["login_history"]),
                "last_login": user_data["login_history"][-1]["timestamp"] if user_data["login_history"] else "Never"
            }
            stats["users"].append(user_stats)
        
        return {"status": "success", "data": stats}

    def save_file(self, filename, file_data_base64):
        try:
            file_data = base64.b64decode(file_data_base64)
            folder = os.path.join(here, "../data/raw")
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)
            with open(filepath, 'wb') as f:
                f.write(file_data)

            project_root = os.path.abspath(os.path.join(here, ".."))
            # Force CPU to avoid CUDA OOM during training
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["TOKENIZERS_PARALLELISM"] = "false"
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:true")
            subprocess.run(["python", "-m", "hallucination_reduction.main"], cwd=project_root, check=True, env=env)

            return "success"
        except Exception as e:
            return f"error: {str(e)}"

def try_backends():
    backends = ['edgechromium', 'qt', 'gtk', 'cef']
    for backend in backends:
        try:
            print(f"\nAttempting {backend} backend...")
            api = Api()
            # Create window with proper HTML file path
            html_path = os.path.abspath(os.path.join(here, 'html', 'index.html'))
            print(f"Loading HTML from: {html_path}")
            
            window = webview.create_window(
                title="AI4ORG - AI For Organization",
                url=f"file://{html_path}",
                width=1400,
                height=900,
                resizable=True,
                js_api=api,
                min_size=(800, 600)
            )
            webview.start(http_server=True, gui=backend)
            return True
        except Exception as e:
            print(f"{backend} backend failed: {e}")
            continue
    return False


if __name__ == "__main__":
    success = try_backends()
    if not success:
        print("All GUI backends failed.")
# /home/shehroz/Desktop/New Folder/ai4org/data_cleaning_pipeline/dataset_corpus_generation.py
        sys.exit(1)