from .qa_generation import build_corpus_and_qa
from .utils import save_corpus, save_qa

if __name__ == "__main__":
    folder = "data/raw"
    passages, qa_dataset = build_corpus_and_qa(folder, num_qs=2)

    print("Sample passage:", passages[0][:200])
    print("Sample QA:", qa_dataset[0])

    # Save to disk (optional)
    save_corpus(passages)
    save_qa(qa_dataset)
