# Frontend Application

The AI4Org frontend is a modern, cross-platform desktop application that allows users to interact with the trained hallucination reduction model. It is built using `pywebview`, which bridges Python logic with a web-based UI (HTML/CSS/JavaScript).

## 🖥️ Features

*   **Interactive Chat**: A familiar chat interface for asking questions and receiving RAG-enhanced answers.
*   **Source Citations**: Answers include references to the specific policy documents and sections used.
*   **User Management**: Secure login and registration system.
*   **Admin Dashboard**: A restricted area for administrators to view usage statistics and manage users.
*   **Document Upload**: Interface for uploading new policy documents to the system.

## 📦 Installation & Setup

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    python main.py
    ```

## 📖 User Guide

### Login / Register
*   **New Users**: Click "Register" to create an account. Your data is stored locally in `user_history.json`.
*   **Existing Users**: Enter your username and password to log in.

### Chat Interface
1.  Type your question in the input box at the bottom.
2.  Press Enter or click the Send button.
3.  The system will retrieve relevant context and generate an answer.
4.  **Citations**: Click on the citation numbers (e.g., `[1]`) to see the source text.

### Admin Dashboard
*   **Access**: Click the "Admin" link in the navigation menu.
*   **PIN Code**: The default admin PIN is `9999`.
*   **Capabilities**:
    *   View total number of users and queries.
    *   See a list of recent queries and system responses.
    *   Manage uploaded files.

## 🔧 Technical Details

### Architecture
The frontend uses a Client-Server architecture, but running locally within a single process.

*   **Python Backend (`main.py`)**:
    *   Initializes the `pywebview` window.
    *   Exposes a Python API class to the JavaScript frontend.
    *   Handles calls to the `hallucination_reduction` inference engine.
*   **Web Frontend (`html/`, `css/`, `script/`)**:
    *   Standard web technologies.
    *   Calls Python functions via `pywebview.api.function_name()`.

### API Bridge
The `Api` class in `main.py` defines the methods available to the frontend:

*   `login(username, password)`
*   `register(username, password)`
*   `send_message(message)`: The core function that calls the ML model.
*   `get_admin_stats()`
*   `upload_file(file_content)`

## 🎨 Customization

*   **Styling**: Edit `css/style.css` to change the look and feel.
*   **Layout**: Edit the HTML files in `html/`.
*   **Logic**: Edit `script/app.js` for frontend logic or `main.py` for backend logic.
