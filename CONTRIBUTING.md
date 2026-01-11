# Contributing to AI4Org

Thank you for your interest in contributing to **AI4Org**! We welcome contributions from the community to help advance hallucination reduction in Large Language Models. Whether you're fixing bugs, adding features, improving documentation, or proposing new ideas, your contributions are valued.

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Getting Started](#-getting-started)
- [Development Workflow](#-development-workflow)
- [Code Style Guidelines](#-code-style-guidelines)
- [Testing Requirements](#-testing-requirements)
- [Pull Request Process](#-pull-request-process)
- [Issue Guidelines](#-issue-guidelines)
- [Community](#-community)

---

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful, inclusive, and collaborative environment. We are committed to providing a welcoming experience for everyone, regardless of background or experience level.

### Our Standards

- ✅ Be respectful and constructive in discussions
- ✅ Welcome newcomers and help them get started
- ✅ Focus on what is best for the community
- ✅ Show empathy towards other community members
- ❌ No harassment, trolling, or discriminatory language
- ❌ No spam or off-topic discussions

---

## 💡 How Can I Contribute?

There are many ways to contribute to AI4Org:

### 1. 🐛 Report Bugs

Found a bug? Help us fix it by:
- Checking if the issue already exists in [GitHub Issues](https://github.com/merledu/ai4org/issues)
- Creating a detailed bug report with:
  - Clear description of the problem
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version, GPU)
  - Error messages and stack traces

### 2. 💡 Suggest Features

Have an idea for improvement?
- Open a feature request issue
- Describe the problem it solves
- Explain your proposed solution
- Discuss alternatives you've considered

### 3. 📝 Improve Documentation

Documentation is crucial! You can:
- Fix typos or clarify existing docs
- Add examples and tutorials
- Improve code comments and docstrings
- Create guides for specific use cases
- Translate documentation

### 4. 🧪 Add Tests

Help improve code coverage:
- Write unit tests for untested modules
- Add integration tests for workflows
- Create regression tests for fixed bugs
- Improve test documentation

### 5. 🚀 Implement Features

Ready to code? Areas where we need help:

#### Hallucination Reduction Module
- Improve discriminator architectures
- Experiment with different RL algorithms
- Optimize training efficiency
- Add support for larger models

#### Data Generation Pipeline
- Support for more document formats
- Improve Q&A quality validation
- Add multilingual support
- Enhance deduplication algorithms

#### Frontend Application
- UI/UX improvements
- New visualization features
- Performance optimizations
- Cross-platform compatibility fixes

#### Infrastructure
- CI/CD pipeline improvements
- Docker containerization
- Deployment automation
- Monitoring and logging

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/ai4org.git
cd ai4org

# Add upstream remote
git remote add upstream https://github.com/merledu/ai4org.git
```

### 2. Set Up Development Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 3. Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/

# Check code style
flake8 hallucination_reduction/ --max-line-length=120
```

### 4. Create a Branch

```bash
# Always create a new branch for your work
git checkout -b feature/your-feature-name

# Branch naming conventions:
# - feature/add-new-discriminator
# - fix/training-memory-leak
# - docs/improve-readme
# - test/add-rl-tests
```

---

## 🔄 Development Workflow

### 1. Keep Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream changes into your main branch
git checkout main
git merge upstream/main

# Rebase your feature branch
git checkout feature/your-feature-name
git rebase main
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed
- Commit frequently with clear messages

### 3. Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
# Format: <type>(<scope>): <description>

# Examples:
git commit -m "feat(discriminator): add attention mechanism to factuality classifier"
git commit -m "fix(rl): resolve NaN loss during REINFORCE training"
git commit -m "docs(readme): add installation troubleshooting section"
git commit -m "test(generator): add unit tests for SFT fine-tuning"
git commit -m "refactor(retriever): optimize embedding computation"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes (formatting)
- `chore`: Maintenance tasks

---

## 📐 Code Style Guidelines

### Python Code Standards

We follow **PEP 8** with some project-specific conventions:

#### 1. Formatting

```python
# ✅ Good: Clear, readable formatting
def train_discriminator(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    labels: List[int],
    device: str = "cuda",
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 2e-5
) -> nn.Module:
    """
    Train a discriminator model on text classification task.

    Args:
        model: The discriminator model to train
        tokenizer: Tokenizer for text encoding
        texts: List of training texts
        labels: List of binary labels (0 or 1)
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate

    Returns:
        Trained model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # ... implementation
    return model


# ❌ Bad: No type hints, no docstring, unclear parameters
def train(m, t, x, y, d="cuda"):
    opt = torch.optim.AdamW(m.parameters(), lr=0.00002)
    # ... implementation
    return m
```

#### 2. Naming Conventions

```python
# ✅ Good: Descriptive names
hallucination_rate = 0.123
factuality_discriminator = load_discriminator("distilbert-base-uncased")
MAX_GENERATION_TOKENS = 64

# ❌ Bad: Unclear abbreviations
hr = 0.123
fd = load_discriminator("distilbert-base-uncased")
mgt = 64
```

#### 3. Docstrings

Use Google-style docstrings:

```python
def retrieve_relevant_chunks(
    query: str,
    embedder: SentenceTransformer,
    corpus_embeddings: np.ndarray,
    docs: List[str],
    top_k: int = 3
) -> List[str]:
    """
    Retrieve the most relevant document chunks for a query.

    Uses cosine similarity between query and corpus embeddings to find
    the top-k most relevant documents.

    Args:
        query: The user's question or search query
        embedder: Sentence transformer model for encoding
        corpus_embeddings: Pre-computed embeddings for all documents
        docs: List of document strings
        top_k: Number of documents to retrieve

    Returns:
        List of the top-k most relevant document strings

    Example:
        >>> embedder = SentenceTransformer("all-MiniLM-L6-v2")
        >>> docs = ["Policy document 1", "Policy document 2"]
        >>> embeddings = embedder.encode(docs)
        >>> results = retrieve_relevant_chunks(
        ...     "What is the policy?", embedder, embeddings, docs
        ... )
    """
    query_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, corpus_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [docs[i] for i in top_indices]
```

#### 4. Imports

```python
# ✅ Good: Organized imports
# Standard library
import os
import sys
from typing import List, Dict, Tuple, Optional

# Third-party
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local
from .config import DEVICE, GEN_MODEL
from .retriever import SimpleRetriever

# ❌ Bad: Unorganized, wildcard imports
from transformers import *
import torch, numpy, os, sys
from .config import *
```

### Code Formatting Tools

```bash
# Format code with Black
black hallucination_reduction/ --line-length 120

# Check style with flake8
flake8 hallucination_reduction/ --max-line-length=120 --ignore=E203,W503

# Type checking with mypy
mypy hallucination_reduction/ --ignore-missing-imports
```

---

## 🧪 Testing Requirements

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_discriminator.py

# Run with coverage
pytest --cov=hallucination_reduction --cov-report=html tests/

# Run tests matching a pattern
pytest -k "test_discriminator" tests/
```

### Writing Tests

#### Unit Tests

```python
# tests/unit/test_retriever.py
import pytest
from hallucination_reduction.retriever import SimpleRetriever


def test_retriever_initialization():
    """Test that retriever initializes with documents."""
    docs = ["Document 1", "Document 2", "Document 3"]
    retriever = SimpleRetriever(docs)
    assert retriever.docs == docs
    assert len(retriever.docs) == 3


def test_retriever_empty_docs():
    """Test that retriever handles empty document list."""
    with pytest.raises(ValueError):
        SimpleRetriever([])


@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_retriever_top_k(top_k):
    """Test retrieval with different top_k values."""
    docs = [f"Document {i}" for i in range(10)]
    retriever = SimpleRetriever(docs)
    results = retriever.retrieve("test query", top_k=top_k)
    assert len(results) == top_k
```

#### Integration Tests

```python
# tests/integration/test_training_pipeline.py
import torch
from hallucination_reduction.main import main


def test_full_training_pipeline():
    """Test complete training pipeline runs without errors."""
    # This test may take several minutes
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"Training pipeline failed: {e}")
```

### Test Coverage Requirements

- **New features**: Must include tests covering main functionality
- **Bug fixes**: Should include regression tests
- **Minimum coverage**: Aim for 70%+ coverage on new code
- **Critical paths**: 90%+ coverage for core training/inference logic

---

## 🔍 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### Submitting Your PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open Pull Request on GitHub**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues (e.g., "Fixes #123")
   - Add screenshots/videos for UI changes

3. **PR Description Template**
   ```markdown
   ## Description
   Brief description of what this PR does.

   ## Type of Change
   - [ ] Bug fix (non-breaking change fixing an issue)
   - [ ] New feature (non-breaking change adding functionality)
   - [ ] Breaking change (fix or feature causing existing functionality to break)
   - [ ] Documentation update

   ## Changes Made
   - Change 1
   - Change 2
   - Change 3

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed

   ## Screenshots (if applicable)

   ## Related Issues
   Fixes #(issue number)
   ```

### Review Process

1. **Automated Checks**: CI/CD will run tests and linting
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

### After Merge

- Delete your feature branch
- Update your local main branch
- Celebrate! 🎉

---

## 📋 Issue Guidelines

### Creating Issues

Use the appropriate issue template:

#### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With configuration '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 11.8]
- GPU: [e.g., RTX 3090]

**Error messages**
```
Paste error messages here
```

**Additional context**
Any other relevant information.
```

#### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other approaches you've thought about.

**Additional context**
Any other relevant information, mockups, or examples.
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested
- `wontfix`: This will not be worked on

---

## 🌟 Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the [README](README.md) and code comments

### Recognition

Contributors will be:
- Listed in the project's contributors page
- Mentioned in release notes for significant contributions
- Invited to join the core team for sustained contributions

### Stay Updated

- Watch the repository for updates
- Star the project to show support
- Follow [MeRL-EDU](https://github.com/merledu) for related projects

---

## 📚 Additional Resources

- [README.md](README.md) - Project overview and setup
- [Python PEP 8 Style Guide](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

---

## 🙏 Thank You!

Your contributions make AI4Org better for everyone. Whether you're fixing a typo or implementing a major feature, every contribution matters. We appreciate your time and effort!

**Happy Contributing! 🚀**

---

<div align="center">

**Questions?** Open a [GitHub Discussion](https://github.com/merledu/ai4org/discussions)

**Found a bug?** Create an [Issue](https://github.com/merledu/ai4org/issues)

</div>
