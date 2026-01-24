from datasets import Dataset
import os 
from ragas import evaluate
from ragas.metrics import _faithfulness, _answer_correctness