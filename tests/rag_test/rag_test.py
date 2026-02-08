from src.config.settings import get_synthesis_llm
from dotenv import load_dotenv
from langsmith import evaluate
from langsmith.schemas import Run, Example 
load_dotenv()

judge_llm = get_synthesis_llm()
