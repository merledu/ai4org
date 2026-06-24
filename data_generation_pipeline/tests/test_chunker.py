from chunker import chunk_text


def test_chunk_small():
    text = "one two three"
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert len(chunks) == 1
