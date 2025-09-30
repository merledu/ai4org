from dataclasses import dataclass
from typing import List

@dataclass
class QAPair:
    question: str
    answer: str
    supporting_passages: List[str]
