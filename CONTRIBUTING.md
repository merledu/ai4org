# Contributing to AI4Org

Thank you for your interest in contributing to AI4Org! We welcome contributions from the community to help improve the project.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/ai4org.git
    cd ai4org
    ```
3.  **Create a new branch** for your feature or bugfix:
    ```bash
    git checkout -b feature/my-new-feature
    ```

## Development Environment

1.  Set up a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Code Style

*   We follow **PEP 8** guidelines for Python code.
*   Please ensure your code is well-documented with docstrings.
*   Use meaningful variable and function names.

## Testing

Before submitting a pull request, please run the existing tests to ensure no regressions:

```bash
pytest tests/
```

If you are adding a new feature, please include appropriate tests.

## Submitting a Pull Request

1.  Push your branch to your fork:
    ```bash
    git push origin feature/my-new-feature
    ```
2.  Open a **Pull Request** on the main repository.
3.  Provide a clear description of your changes and the problem they solve.
4.  Link to any relevant issues.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Provide as much detail as possible, including steps to reproduce the issue.
