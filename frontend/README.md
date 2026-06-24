# AI4Org Desktop Frontend

A cross-platform desktop application (built with **pywebview**) that provides an
interactive, RAG-enhanced chat interface to the AI4Org hallucination-reduction
model, plus user login, document upload, and an admin dashboard.

## Run

1. Install Python 3.10+.
2. From the **project root**, install the core dependencies (the frontend imports
   `hallucination_reduction.inference`):
   ```bash
   pip install -r requirements.txt
   ```
3. Install the frontend dependency and launch the app:
   ```bash
   cd frontend
   pip install -r requirements.txt
   python main.py
   ```

> On first launch the base model and embeddings are downloaded/loaded, which can
> take a few minutes. If a fine-tuned checkpoint exists in
> `saved_models_improved/`, it is loaded automatically; otherwise the base model
> is used.

## Features

- 🔐 User login and registration (history stored in `user_history.json`)
- 💬 Interactive chat with RAG-enhanced responses
- 📤 Document upload for training data
- 👨‍💼 Admin dashboard (PIN configurable via the `ADMIN_PIN` environment variable)
- 📊 User statistics and login history

## Structure

| Path | Purpose |
|------|---------|
| `main.py` | App entry point — wires the `Api` bridge to `hallucination_reduction.inference` |
| `html/` | Application pages (login, chat, admin, upload, …) |
| `css/`, `script/`, `assets/` | Styles, client-side JS, and static assets |

## Build a Windows EXE

```bash
pip install pyinstaller
pyinstaller -F --add-data "html;html" --add-data "css;css" \
    --add-data "script;script" --add-data "assets;assets" main.py
```

The built executable will be in `dist/main.exe`.
