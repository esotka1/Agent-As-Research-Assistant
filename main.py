import os
import re
import time
import threading
from pathlib import Path
from utils import read_paper_content

import openai
from agents import Agent, Runner

# -------------------------------------------
# CONFIG
# -------------------------------------------
DATA_DIR = Path("resources")
PAPER_ANALYZER_INPUT = DATA_DIR / "input/The_impact_of_exposure_to_air_pollution_on_cognitive_performance_PNAS.pdf"
PAPER_ANALYZER_OUTPUT = DATA_DIR / "output/analyzed_paper.txt"

MODEL = "gpt-4o" # Default Model

# Models are subject to change for different tasks
PAPER_ANALYZER_MODEL = "gpt-4o"
DATA_FINDER_MODEL = "gpt-4o"
CODE_IMPLEMENTER_MODEL = "gpt-4o"
RESULT_VALIDATOR_MODEL = "gpt-4o"

# LLM specific configs
MAX_WORKERS = 10 # See if there is potential for parallelism
BATCH_SIZE = 500
TEMPERATURE = .3
MAX_RETRIES = 10

SYSTEM_ROLE = """
        Act as an academic research expert. Read and digest the content of the research paper. Produce a concise and clear summary that encapsulates the main findings, methodology, results, and implications of the study. Ensure that the summary is written in a manner that is accessible to a general audience while retaining the core insights and nuances of the original paper. Include key terms and concepts, and provide any necessary context or background information. The summary should serve as a standalone piece that gives readers a comprehensive understanding of the paper's significance without needing to read the entire document.
        """

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found")

openai.api_key = API_KEY
LLM_INTERFACE = openai.OpenAI()


# Token Tracking => Made to work w/ threading
total_tokens_used = 0
total_prompt_tokens = 0
total_completion_tokens = 0
token_lock = threading.Lock() # Allows potential threading

token_cost = 0
input_cost = .0025 / 1000
output_cost = .01 / 1000

def run_paper_analyzer():
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, token_cost

    paperContent = read_paper_content(str(PAPER_ANALYZER_INPUT))
    PAPER_ANALYZER_PROMPT = f"""
            {SYSTEM_ROLE}

            Please analyze the following paper content:
            {paperContent[:400000]}  # Limit length to avoid token overflow
    Please act as an experienced PhD student in artificial intelligence, carefully read this paper and provide an in-depth analysis of about 3000 words. Be very detailed and thoughtfully answer the following questions:

    # Prompt: In-depth Paper Analysis

    ## Task Description
    As an experienced PhD student in artificial intelligence, please carefully read the specified paper and provide an in-depth analysis of about 3000 words. Be very detailed and thoughtfully answer the following questions:

    ### 1. Abstract
    - What problem does the paper attempt to solve? What kind of model needs to be constructed?

    ### 2. Introduction
    - Explain the overall process of the paper in detail.
    - If you want to reproduce the paper, what steps are needed? How would you specifically operate?

    ### 3. Method
    - What is the solution?
    - Carefully sort out each step, formula, and strategy in the paper.
    - Analyze in detail what each part of the variables in the formula represents.
    - What data should be collected?

    ### 4. Experiment
    - (1) Describe the methodology of the paper in detail.
    - (2)Recognize each figure and answer the questions of each figure as follows.
    How is figure drawn, you must contain Task Description,Variable Description,Data Fields, and write the corresponding prompt so that code can be generated directly from your prompt，note that it's the prompt for generating code, not the code itself. (including variable parameters, model usage, control variable names, independent and dependent variable types). I need to read my data and then plot it.

    ## Answer Requirements
    - The answer should use Markdown format, with appropriate use of lists, bold, and other formatting elements.
    - Use secondary headings (##) corresponding to the above questions to clearly divide different sections.


    ## Output Format
    - Clear structure and well-defined sections.
    - Academic language and rigorous expression.
    - Suitable for direct use in automated paper analysis tasks. 
            """

    response = LLM_INTERFACE.responses.create(
        model=PAPER_ANALYZER_MODEL,
        input=PAPER_ANALYZER_PROMPT,
        temperature=TEMPERATURE
    )

    tokens_used = 0
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        tokens_used = usage.total_tokens

        total_prompt_tokens += usage.input_tokens
        total_completion_tokens += usage.output_tokens
        total_tokens_used += usage.total_tokens

        token_cost = usage.input_tokens * input_cost + usage.output_tokens * output_cost

    PAPER_ANALYZER_OUTPUT.write_text(response.output[0].content[0].text)
    print(f"✅ Paper Analyzer ran successfully w/ {tokens_used} tokens")

    return True

def run_code_implementor():

    coding_agent = Agent(
        name="ResearchAssistant",
        instructions="You are a helpful research assistant specializing in data analysis.",
        model=CODE_IMPLEMENTER_MODEL,
    )

    result = Runner.run(coding_agent, "What is the square of 3?")

    print(result)


    return True

def run_result_validator():
    return True


def run_full_agent_flow():
    print("\n" + "="*100)
    print("🚀 STARTING RESEARCH REPRODUCTION PIPELINE 🚀") # Starting with reproduction
    print("=" * 100 + "\n")

    pipelineStart = time.time()

    if not run_paper_analyzer():
        print("❌ Pipeline failed at Paper Analyzer step")
        return False

    if not run_code_implementor():
        print("❌ Pipeline failed at Code Implementor step")
        return False

    print(f"\n🏁 Total Pipeline Time: {time.time() - pipelineStart:.2f}")
    print(f"💰Estimated Cost: ${token_cost:.4f}\n")

    return True


if __name__ == "__main__":
    if run_full_agent_flow():
        print("✅ All stages ran successfully")
    else:
        print("❌ There was an error in the pipeline")