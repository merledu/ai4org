from cleaner import clean_text

def test_cleaner_basic():
    raw = "PREFACE\nsomething\nSECTION 1\nThis is text.\n\n\nA m a n a h\n"
    cleaned = clean_text(raw)
    assert "PREFACE" not in cleaned or "SECTION 1" in cleaned
    assert "A m a n a h" not in cleaned

