from qa_parser import parse_qa_block

def test_parse_basic():
    block = "Q1: What is Section 1?\nA1: It is the initial section.\nQ2: What is Section 2?\nA2: Another section."
    pairs = parse_qa_block(block)
    assert len(pairs) == 2
    assert pairs[0][0].startswith("What is")

