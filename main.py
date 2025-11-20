import os
import re
import time
import threading
from pathlib import Path
from unittest import result
from utils import read_paper_content, extract_images, summarize_csv, convert_pngs_to_pdfs, clear_non_folder_items
import contextlib
import io
import textwrap
import matplotlib
import matplotlib.pyplot as plt
import builtins
import shutil
import warnings
import json
from PIL import Image

import openai
from agents import Agent,Runner
import asyncio

# TODO: Add format checker agent
# TODO: Allow use to add how ever many figures as they want
# TODO: Create MVP for tables

# -------------------------------------------
# CONFIG
# -------------------------------------------
DATA_DIR = Path("resources")
PAPER_ANALYZER_INPUT = DATA_DIR / "input/The_impact_of_exposure_to_air_pollution_on_cognitive_performance_PNAS.pdf"
PAPER_ANALYZER_OUTPUT = DATA_DIR / "output/analyzed_paper.txt"

CODE_IMPLEMENTER_DATA = DATA_DIR / "input/data/data_for_reproduce.csv"
CODE_IMPLEMENTER_OUTPUT = DATA_DIR / "output/extracted_code.py"

CODE_IMPLEMENTER_GRAPH_PDF = DATA_DIR / "input/graphs"
CODE_IMPLEMENTER_GRAPH_PNG = DATA_DIR / "output/graphs"

EXTRACTED_FIGURES_PDF = DATA_DIR / "input/figures"
EXTRACTED_FIGURES_PNG = DATA_DIR / "output/figures"

RESULT_VALIDATOR_OUTPUT = DATA_DIR / "output/validation_results.json"

FIGURES_ARE_GIVEN = True # Set to False if you want to find all figures (if false will delete existing ones)

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
client = openai.OpenAI()


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

    response = client.responses.create(
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


'''
async def run_figure_describer(max_retries=3):
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, token_cost

    # === Upload PDF ===
    print("📄 Uploading PDF...")
    file = client.files.create(
        file=open(PAPER_ANALYZER_INPUT, "rb"),
        purpose="assistants"
    )

    # === Create assistant agent ===
    describing_agent = Agent(
        name="FigureDescriber",
        instructions="You are an expert research assistant that identifies and describes figures in academic PDFs.",
        model=PAPER_ANALYZER_MODEL,
    )

    # === User prompt ===
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file.id},
                {
                    "type": "input_text",
                    "text": (
                        "Read the uploaded research paper and locate each figure.\n"
                        "For each figure, provide a concise description of what it visually shows — "
                        "including the type of plot, trends, variables, or comparisons depicted.\n\n"
                        "Output only one description per line, in order of appearance.\n"
                        "Do not include captions or any extra commentary — just the pure descriptions."
                    ),
                },
            ],
        },
    ]

    # === Retry loop ===
    for attempt in range(1, max_retries + 1):
        print(f"\n🔁 Attempt {attempt} of {max_retries}")

        try:
            result = await Runner.run(starting_agent=describing_agent, input=messages)
            response_text = str(result)

            # === Token accounting ===
            if hasattr(result, "raw_responses") and result.raw_responses:
                resp = result.raw_responses[0]
                if hasattr(resp, "usage") and resp.usage:
                    usage = resp.usage
                    total_prompt_tokens += usage.input_tokens
                    total_completion_tokens += usage.output_tokens
                    total_tokens_used += usage.total_tokens
                    token_cost += (
                        (usage.input_tokens * input_cost)
                        + (usage.output_tokens * output_cost)
                    )

            # === Clean and save descriptions ===
            clean_text = textwrap.dedent(response_text).strip()
            clean_text = re.sub(r"```.*?```", "", clean_text, flags=re.DOTALL).strip()

            with open(PAPER_ANALYZER_OUTPUT, "w") as f:
                f.write(clean_text)

            print(f"\n✅ Figure descriptions written to {PAPER_ANALYZER_OUTPUT}")
            print(f"💰 Tokens used: {total_tokens_used} | Cost: ${token_cost:.4f}")
            print("\n=== OUTPUT ===\n")
            print(clean_text)
            return True

        except Exception as e:
            print(f"❌ Error on attempt {attempt}: {e}")
            if attempt < max_retries:
                messages.append({
                    "role": "user",
                    "content": f"The last request failed with this error:\n{e}\nPlease retry extracting figure descriptions."
                })
                await asyncio.sleep(2)
            else:
                print("🚫 Maximum retries reached.")
                return False
'''

'''
async def run_code_implementor(max_retries=3):
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, token_cost

    figure_files = sorted(EXTRACTED_FIGURES_PDF.glob("*.*"))
    if not figure_files:
        print("⚠️ No figure files found in resources/output/figures.")
        return False

    # Ensure graph directory exists but do NOT create subfolders
    if CODE_IMPLEMENTER_GRAPH_PNG.exists():
        shutil.rmtree(CODE_IMPLEMENTER_GRAPH_PNG)
    os.makedirs(CODE_IMPLEMENTER_GRAPH_PNG, exist_ok=True)

    # Upload study for global context
    study_file = client.files.create(
        file=open(PAPER_ANALYZER_INPUT, "rb"),
        purpose="assistants"
    )

    coding_agent = Agent(
        name="ResearchAssistant",
        instructions=(
            "You are a helpful research assistant specializing in creating Python code "
            "to replicate figures from research studies. You have access to the full study "
            "as context. For each uploaded figure, write clean and reproducible Python code "
            "that recreates the figure as closely as possible using matplotlib and pandas."
        ),
        model=CODE_IMPLEMENTER_MODEL,
    )

    all_successful = True  # track overall success

    # === Loop through each extracted figure ===
    for fig_path in figure_files:
        print(f"\n===== Processing figure: {fig_path.name} =====")

        try:
            fig_file = client.files.create(
                file=open(fig_path, "rb"),
                purpose="assistants"
            )
        except Exception as e:
            print(f"❌ Failed to upload {fig_path.name}: {e}")
            all_successful = False
            continue

        with open(CODE_IMPLEMENTER_DATA, "r", newline="") as infile:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": study_file.id},
                        {"type": "input_file", "file_id": fig_file.id},
                        {
                            "type": "input_text",
                            "text": (
                                f"This figure is from the study you have access to. "
                                f"Please generate Python code that replicates this figure "
                                f"as accurately as possible. Use the following data  "
                                f"from the path {CODE_IMPLEMENTER_DATA} as your dataset context.\n"
                                f"Here is a summary of the data's structure: {summarize_csv(CODE_IMPLEMENTER_DATA)}"
                            ),
                        },
                    ],
                },
            ]

        success_for_figure = False  # track per-figure success

        for attempt in range(1, max_retries + 1):
            print(f"\n🔁 Attempt {attempt} of {max_retries} for {fig_path.name}")

            result = await Runner.run(starting_agent=coding_agent, input=messages)

            # === Track token usage ===
            tokens_used = 0
            if hasattr(result, "raw_responses") and result.raw_responses:
                resp = result.raw_responses[0]
                if hasattr(resp, "usage") and resp.usage:
                    usage = resp.usage
                    tokens_used = usage.total_tokens
                    total_prompt_tokens += usage.input_tokens
                    total_completion_tokens += usage.output_tokens
                    total_tokens_used += usage.total_tokens
                    token_cost += (
                        (usage.input_tokens * input_cost) +
                        (usage.output_tokens * output_cost)
                    )

            # === Extract and save code ===
            code_blocks = re.findall(r"```python(.*?)```", str(result), re.DOTALL)
            combined_code = "\n\n".join(textwrap.dedent(block).strip() for block in code_blocks)
            fig_code_output = Path(CODE_IMPLEMENTER_OUTPUT).with_name(f"{fig_path.stem}_code.py")

            with open(fig_code_output, "w") as f:
                f.write(combined_code)

            print(f"\n===== Running Extracted Code for {fig_path.name} =====\n")

            try:
                matplotlib.use("Agg")
                warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
                builtins.plt = plt

                # Create a single image (no folders)
                fig_save_path = Path(CODE_IMPLEMENTER_GRAPH_PNG) / f"{fig_path.stem}_replica.png"

                # Run code in isolated namespace and capture figures
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec_namespace = {"plt": plt}
                    exec(combined_code, exec_namespace)

                figs = plt.get_fignums()
                if figs:
                    # Always overwrite the previous saved figure
                    for i, num in enumerate(figs, 1):
                        fig = plt.figure(num)
                        fig.savefig(fig_save_path, bbox_inches="tight")
                        print(f"📊 Saved (and replaced if existed): {fig_save_path}")
                    plt.close("all")
                else:
                    print("⚠️ No figures were generated by the code.")

                print(output.getvalue())
                print(f"\n✅ Successfully replicated {fig_path.name} with {tokens_used} tokens\n")
                success_for_figure = True
                break

            except Exception as e:
                print(f"❌ Error running code for {fig_path.name} on attempt {attempt}: {e}")
                if attempt < max_retries:
                    messages.append({
                        "role": "user",
                        "content": f"The code failed with this error:\n{e}\n\n"
                                   "Please fix the code and try again."
                    })
                    print("🧠 Sending error back to agent for correction...")
                    await asyncio.sleep(2)
                else:
                    print(f"🚫 Maximum retries reached for {fig_path.name}. Moving on.\n")

        if not success_for_figure:
            all_successful = False  # mark overall failure if one figure fails

    return all_successful
'''
    

async def run_result_validator():
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, token_cost

    # Step 1: Create agent
    validator_agent = Agent(
        name="ResultValidator",
        instructions=(
            "You are a precise visual data comparison agent. "
            "You will be given two figures: one from a research paper, and one generated by an LLM. "
            "Your goal is to determine if they convey the same underlying data, even if styles differ. "
            "Be strict: focus on whether the axes, data trends, labels, and general meaning align. "
            "If they differ significantly in data, trends, or labeling, they are NOT equivalent. "
            "Output a JSON object like:\n"
            "{'same_data': true/false, 'explanation': 'brief reason'}"
        ),
        model=RESULT_VALIDATOR_MODEL,
    )

    results = {}  # store validation outcomes

    # Step 2: Iterate through all extracted figures
    for figure_path in sorted(EXTRACTED_FIGURES_PDF.glob("*.*")):
        replica_path = CODE_IMPLEMENTER_GRAPH_PDF / f"{figure_path.stem}_replica.pdf"

        print(f"\n===== Comparing {figure_path.name} to {replica_path.name} =====")

        if not replica_path.exists():
            print(f"⚠️ Skipping {figure_path.name}: replica not found at {replica_path}")
            results[figure_path.name] = {
                "same_data": False,
                "explanation": "Replica figure not found.",
            }
            continue

        try:
            original_file = client.files.create(
                file=open(figure_path, "rb"),
                purpose="assistants"
            )
            generated_file = client.files.create(
                file=open(replica_path, "rb"),
                purpose="assistants"
            )
        except Exception as e:
            print(f"❌ Failed to upload files for {figure_path.name}: {e}")
            results[figure_path.name] = {
                "same_data": False,
                "explanation": f"File upload failed: {e}",
            }
            continue

        # Step 3: Prepare message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": original_file.id},
                    {"type": "input_file", "file_id": generated_file.id},
                    {
                        "type": "input_text",
                        "text": (
                            "Compare the two images and determine if they convey the same conclusion. "
                            "Focus on structure, scale, and data relationships, not artistic style. "
                            "Respond in strict JSON format as described above."
                        ),
                    },
                ],
            },
        ]

        # Step 4: Run validation
        result = await Runner.run(starting_agent=validator_agent, input=messages)

        # === Track token usage ===
        tokens_used = 0
        if hasattr(result, "raw_responses") and result.raw_responses:
            resp = result.raw_responses[0]
            if hasattr(resp, "usage") and resp.usage:
                usage = resp.usage
                tokens_used = usage.total_tokens
                total_prompt_tokens += usage.input_tokens
                total_completion_tokens += usage.output_tokens
                total_tokens_used += usage.total_tokens
                token_cost += (
                    (usage.input_tokens * input_cost) +
                    (usage.output_tokens * output_cost)
                )

        # === Parse and save validation result ===
        validation_text = str(result).strip()
        json_match = re.search(r"\{.*\}", validation_text, re.DOTALL)
        if not json_match:
            print(f"⚠️ No JSON object found for {figure_path.name}")
            results[figure_path.name] = {
                "same_data": False,
                "explanation": "No JSON output from validator.",
            }
            continue

        try:
            validation_json = json.loads(json_match.group(0))
            same_data = validation_json.get("same_data", False)
            explanation = validation_json.get("explanation", "")
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error for {figure_path.name}: {e}")
            results[figure_path.name] = {
                "same_data": False,
                "explanation": f"Invalid JSON format: {e}",
            }
            continue

        print("\n===== Validation Result =====")
        print(f"Same data: {same_data}")
        print(f"Explanation: {explanation}")
        print(f"✅ Validator succeeded w/ {tokens_used} tokens")

        results[figure_path.name] = {
            "same_data": same_data,
            "explanation": explanation,
            "tokens_used": tokens_used,
        }

    # Step 5: Save all results
    with open(RESULT_VALIDATOR_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 All validation results saved to {RESULT_VALIDATOR_OUTPUT}\n")
    return True


'''
async def run_code_implementor(max_retries=5):
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, token_cost

    figure_files = sorted(EXTRACTED_FIGURES_PDF.glob("*.*"))
    if not figure_files:
        print("⚠️ No figure files found in resources/output/figures.")
        return False

    # Ensure graph directory exists but do NOT create subfolders
    if CODE_IMPLEMENTER_GRAPH_PNG.exists():
        shutil.rmtree(CODE_IMPLEMENTER_GRAPH_PNG)
    os.makedirs(CODE_IMPLEMENTER_GRAPH_PNG, exist_ok=True)

    # Upload study for global context
    study_file = client.files.create(
        file=open(PAPER_ANALYZER_INPUT, "rb"),
        purpose="assistants"
    )

    # === Coding agent (unchanged) ===
    coding_agent = Agent(
        name="ResearchAssistant",
        instructions=(
            "You are a helpful research assistant specializing in creating Python code "
            "to replicate figures from research studies. You have access to the full study "
            "as context. For each uploaded figure, write clean and reproducible Python code "
            "that recreates the figure as closely as possible using matplotlib and pandas."
        ),
        model=CODE_IMPLEMENTER_MODEL,
    )

    # === NEW: Formatting check agent ===
    format_checker_agent = Agent(
        name="FormatChecker",
        instructions=(
            "You compare two graphs (original and reproduced). "
            "Evaluate ONLY formatting similarity: axes, titles, labels, legends, colors, layout. "
            "Respond EXACTLY with either 'PASS' or 'FAIL'. "
            "If FAIL, also provide a short explanation of what is wrong."
        ),
        model=CODE_IMPLEMENTER_MODEL,
    )

    all_successful = True

    # === Loop through each figure ===
    for fig_path in figure_files:
        print(f"\n===== Processing figure: {fig_path.name} =====")

        try:
            fig_file = client.files.create(
                file=open(fig_path, "rb"),
                purpose="assistants"
            )
        except Exception as e:
            print(f"❌ Failed to upload {fig_path.name}: {e}")
            all_successful = False
            continue

        with open(CODE_IMPLEMENTER_DATA, "r", newline="") as infile:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": study_file.id},
                        {"type": "input_file", "file_id": fig_file.id},
                        {
                            "type": "input_text",
                            "text": (
                                f"This figure is from the study. "
                                f"Please generate Python code that replicates the figure "
                                f"as accurately as possible using the dataset at {CODE_IMPLEMENTER_DATA}.\n"
                                f"Data summary: {summarize_csv(CODE_IMPLEMENTER_DATA)}"
                            ),
                        },
                    ],
                },
            ]

        success_for_figure = False

        for attempt in range(1, max_retries + 1):
            print(f"\n🔁 Attempt {attempt} of {max_retries} for {fig_path.name}")

            # === Run the coding agent ===
            result = await Runner.run(starting_agent=coding_agent, input=messages)

            # Track tokens
            tokens_used = 0
            if hasattr(result, "raw_responses") and result.raw_responses:
                resp = result.raw_responses[0]
                if hasattr(resp, "usage") and resp.usage:
                    usage = resp.usage
                    tokens_used = usage.total_tokens
                    total_prompt_tokens += usage.input_tokens
                    total_completion_tokens += usage.output_tokens
                    total_tokens_used += usage.total_tokens
                    token_cost += (
                        (usage.input_tokens * input_cost) +
                        (usage.output_tokens * output_cost)
                    )

            # Extract code
            code_blocks = re.findall(r"```python(.*?)```", str(result), re.DOTALL)
            combined_code = "\n\n".join(textwrap.dedent(block).strip() for block in code_blocks)
            fig_code_output = Path(CODE_IMPLEMENTER_OUTPUT).with_name(f"{fig_path.stem}_code.py")

            with open(fig_code_output, "w") as f:
                f.write(combined_code)

            print(f"\n===== Running Extracted Code for {fig_path.name} =====\n")

            try:
                matplotlib.use("Agg")
                warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
                builtins.plt = plt

                # Save path for reproduced figure
                fig_save_path = Path(CODE_IMPLEMENTER_GRAPH_PNG) / f"{fig_path.stem}_replica.png"

                # Execute figure-producing code
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec_namespace = {"plt": plt}
                    exec(combined_code, exec_namespace)

                figs = plt.get_fignums()
                if figs:
                    for num in figs:
                        fig = plt.figure(num)
                        fig.savefig(fig_save_path, bbox_inches="tight")
                    plt.close("all")
                else:
                    print("⚠️ No figures were generated.")
                    raise RuntimeError("No figure produced")

                print(f"📊 Saved: {fig_save_path}")

                # === NEW: LLM Format Validation Step ===
                original_uploaded = client.files.create(
                    file=open(fig_path, "rb"),
                    purpose="assistants"
                )

                replica_uploaded = client.files.create(
                    file=open(fig_save_path, "rb"),
                    purpose="assistants"
                )

                check_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": original_uploaded.id},
                            {"type": "input_file", "file_id": replica_uploaded.id},
                            {"type": "input_text", 
                             "text": (
                                 "Compare these two graphs. "
                                 "Respond with PASS if the formatting matches closely. "
                                 "Respond with FAIL otherwise."
                             )}
                        ]
                    }
                ]

                check_result = await Runner.run(
                    starting_agent=format_checker_agent,
                    input=check_messages
                )

                check_text = str(check_result).strip()
                print(f"🔍 Format Check Result: {check_text}")

                if "PASS" in check_text.upper():
                    print(f"\n✅ Formatting accepted for {fig_path.name}\n")
                    success_for_figure = True
                    break

                # === If FAIL: send feedback and retry ===
                else:
                    print(f"❌ Format mismatch on attempt {attempt}")
                    messages.append({
                        "role": "user",
                        "content": (
                            "The formatting does not match the original figure. "
                            f"Here is the evaluator's response:\n{check_text}\n\n"
                            "Please fix the formatting and try again."
                        )
                    })

            except Exception as e:
                print(f"❌ Error running code for {fig_path.name} on attempt {attempt}: {e}")
                messages.append({
                    "role": "user",
                    "content": f"The code crashed with error:\n{e}\nFix it and retry."
                })

        if not success_for_figure:
            print(f"🚫 Failed to reproduce {fig_path.name} after {max_retries} attempts.")
            all_successful = False

    return all_successful
'''

async def run_code_implementor(max_retries=5):
    global total_prompt_tokens, total_completion_tokens, total_tokens_used, token_cost

    # Gather extracted figure files (assumed to be in EXTRACTED_FIGURES_PDF)
    figure_files = sorted(EXTRACTED_FIGURES_PDF.glob("*.*"))
    if not figure_files:
        print("⚠️ No figure files found in resources/output/figures.")
        return False

    # Ensure graph directories exist but do NOT create subfolders.
    # Clean and recreate both PDF (input) and PNG (output) directories.
    for d in (CODE_IMPLEMENTER_GRAPH_PDF, CODE_IMPLEMENTER_GRAPH_PNG):
        if d.exists():
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Upload study for global context
    with open(PAPER_ANALYZER_INPUT, "rb") as study_f:
        study_file = client.files.create(
            file=study_f,
            purpose="assistants"
        )

    # === Coding agent (unchanged) ===
    coding_agent = Agent(
        name="ResearchAssistant",
        instructions=(
            "You are a helpful research assistant specializing in creating Python code "
            "to replicate figures from research studies. You have access to the full study "
            "as context. For each uploaded figure, write clean and reproducible Python code "
            "that recreates the figure as closely as possible using matplotlib and pandas."
        ),
        model=CODE_IMPLEMENTER_MODEL,
    )

    # === NEW: Formatting check agent ===
    format_checker_agent = Agent(
        name="FormatChecker",
        instructions=(
            "You compare two graphs (original and reproduced). "
            "Evaluate ONLY formatting similarity: axes, titles, labels, legends, colors, layout. "
            "Respond EXACTLY with either 'PASS' or 'FAIL'. "
            "If FAIL, also provide a short explanation of what is wrong."
        ),
        model=CODE_IMPLEMENTER_MODEL,
    )

    all_successful = True

    # === Loop through each figure ===
    for fig_path in figure_files:
        print(f"\n===== Processing figure: {fig_path.name} =====")

        # --- Ensure we upload an original PDF to the LLM ---
        # If original is already PDF, copy it into CODE_IMPLEMENTER_GRAPH_PDF for record and upload that.
        if fig_path.suffix.lower() == ".pdf":
            original_pdf_path = CODE_IMPLEMENTER_GRAPH_PDF / fig_path.name
            shutil.copy2(fig_path, original_pdf_path)
        else:
            # Convert image (png/jpg/etc.) to PDF and place in input/graphs
            original_pdf_path = CODE_IMPLEMENTER_GRAPH_PDF / f"{fig_path.stem}.pdf"
            try:
                with Image.open(fig_path) as im:
                    # Convert to RGB for multi-mode formats
                    im_rgb = im.convert("RGB")
                    im_rgb.save(original_pdf_path, "PDF", resolution=300.0)
            except Exception as e:
                print(f"❌ Failed to convert original {fig_path.name} to PDF: {e}")
                all_successful = False
                continue

        # Upload the original PDF to the LLM
        try:
            with open(original_pdf_path, "rb") as f:
                fig_file = client.files.create(
                    file=f,
                    purpose="assistants"
                )
        except Exception as e:
            print(f"❌ Failed to upload original PDF for {fig_path.name}: {e}")
            all_successful = False
            continue

        # Prepare messages (study_file and fig_file provided as input files)
        with open(CODE_IMPLEMENTER_DATA, "r", newline="") as infile:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": study_file.id},
                        {"type": "input_file", "file_id": fig_file.id},
                        {
                            "type": "input_text",
                            "text": (
                                f"This figure is from the study. "
                                f"Please generate Python code that replicates the figure "
                                f"as accurately as possible using the dataset at {CODE_IMPLEMENTER_DATA}.\n"
                                f"Data summary: {summarize_csv(CODE_IMPLEMENTER_DATA)}"
                            ),
                        },
                    ],
                },
            ]

        success_for_figure = False

        for attempt in range(1, max_retries + 1):
            print(f"\n🔁 Attempt {attempt} of {max_retries} for {fig_path.name}")

            # === Run the coding agent ===
            result = await Runner.run(starting_agent=coding_agent, input=messages)

            # Track tokens (if the response includes usage info)
            tokens_used = 0
            if hasattr(result, "raw_responses") and result.raw_responses:
                resp = result.raw_responses[0]
                if hasattr(resp, "usage") and resp.usage:
                    usage = resp.usage
                    tokens_used = usage.total_tokens
                    total_prompt_tokens += usage.input_tokens
                    total_completion_tokens += usage.output_tokens
                    total_tokens_used += usage.total_tokens
                    token_cost += (
                        (usage.input_tokens * input_cost) +
                        (usage.output_tokens * output_cost)
                    )

            # Extract code blocks from the agent output
            code_blocks = re.findall(r"```python(.*?)```", str(result), re.DOTALL)
            combined_code = "\n\n".join(textwrap.dedent(block).strip() for block in code_blocks)
            fig_code_output = Path(CODE_IMPLEMENTER_OUTPUT).with_name(f"{fig_path.stem}_code.py")

            with open(fig_code_output, "w") as f:
                f.write(combined_code)

            print(f"\n===== Running Extracted Code for {fig_path.name} =====\n")

            try:
                matplotlib.use("Agg")
                warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")
                builtins.plt = plt

                # Ensure the output PNG directory exists (already created above)
                # We'll save both PNG and PDF replicas into the corresponding directories
                replica_png_path = Path(CODE_IMPLEMENTER_GRAPH_PNG) / f"{fig_path.stem}.png"
                replica_pdf_path = Path(CODE_IMPLEMENTER_GRAPH_PDF) / f"{fig_path.stem}.pdf"

                # Execute figure-producing code (provide plt in namespace)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    exec_namespace = {"plt": plt}
                    exec(combined_code, exec_namespace)

                figs = plt.get_fignums()
                if figs:
                    # Save each open figure; if multiple figs, append index to filename
                    if len(figs) == 1:
                        # Save single figure as PNG and PDF
                        fig = plt.figure(figs[0])
                        fig.savefig(replica_png_path, bbox_inches="tight")
                        fig.savefig(replica_pdf_path, bbox_inches="tight", format="pdf")
                        plt.close(fig)
                    else:
                        for idx, num in enumerate(figs, start=1):
                            fig = plt.figure(num)
                            png_path = Path(CODE_IMPLEMENTER_GRAPH_PNG) / f"{fig_path.stem}_{idx}.png"
                            pdf_path = Path(CODE_IMPLEMENTER_GRAPH_PDF) / f"{fig_path.stem}_{idx}.pdf"
                            fig.savefig(png_path, bbox_inches="tight")
                            fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
                            plt.close(fig)
                    plt.close("all")
                else:
                    print("⚠️ No figures were generated.")
                    raise RuntimeError("No figure produced")

                print(f"📊 Saved replica PNG(s) in {CODE_IMPLEMENTER_GRAPH_PNG} and PDF(s) in {CODE_IMPLEMENTER_GRAPH_PDF}")

                # === NEW: LLM Format Validation Step ===
                # Upload original (already uploaded above as fig_file) and replica PDF(s).
                # For simplicity, upload the primary replica PDF for the checker.
                try:
                    with open(replica_pdf_path, "rb") as rf:
                        replica_uploaded = client.files.create(
                            file=rf,
                            purpose="assistants"
                        )
                except Exception as e:
                    print(f"❌ Failed to upload replica PDF for {fig_path.name}: {e}")
                    messages.append({
                        "role": "user",
                        "content": f"Failed to upload replica PDF: {e}\nFix it and retry."
                    })
                    continue

                check_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": fig_file.id},        # original PDF
                            {"type": "input_file", "file_id": replica_uploaded.id}, # replica PDF
                            {"type": "input_text",
                             "text": (
                                 "Compare these two graphs. "
                                 "Respond with PASS if the formatting matches closely. "
                                 "Respond with FAIL otherwise."
                             )}
                        ]
                    }
                ]

                check_result = await Runner.run(
                    starting_agent=format_checker_agent,
                    input=check_messages
                )

                check_text = str(check_result).strip()
                #print(f"🔍 Format Check Result: {check_text}")

                if "PASS" in check_text.upper():
                    print(f"\n✅ Formatting accepted for {fig_path.name}\n")
                    success_for_figure = True
                    break

                # === If FAIL: send feedback and retry ===
                else:
                    print(f"❌ Format mismatch on attempt {attempt}")
                    messages.append({
                        "role": "user",
                        "content": (
                            "The formatting does not match the original figure. "
                            f"Here is the evaluator's response:\n{check_text}\n\n"
                            "Please fix the formatting and try again."
                        )
                    })

            except Exception as e:
                print(f"❌ Error running code for {fig_path.name} on attempt {attempt}: {e}")
                messages.append({
                    "role": "user",
                    "content": f"The code crashed with error:\n{e}\nFix it and retry."
                })

        if not success_for_figure:
            print(f"🚫 Failed to reproduce {fig_path.name} after {max_retries} attempts.")
            all_successful = False

    return all_successful

def run_full_agent_flow():
    print("\n" + "="*100)
    print("🚀 STARTING RESEARCH REPRODUCTION PIPELINE 🚀") # Starting with reproduction
    print("=" * 100 + "\n")

    pipelineStart = time.time()

    if not clear_non_folder_items("resources/output"):
        print("❌ Failed to clear output directory")
        return False

    '''
    if not run_paper_analyzer():
        print("❌ Pipeline failed at Paper Analyzer step")
        return False
    '''

    if not FIGURES_ARE_GIVEN:
        print("\n📊 Starting Figure Extraction Step...\n")
        try:
            extract_images(str(PAPER_ANALYZER_INPUT))
            print("✅ Figure Extraction step completed")
        except Exception as e:
            print(f"📝 Image Extraction step failed: {e}")

    
    convert_pngs_to_pdfs(EXTRACTED_FIGURES_PNG, EXTRACTED_FIGURES_PDF)


    if not asyncio.run(run_code_implementor()):
        print("❌ Pipeline failed at Code Implementor step")
        return False

    convert_pngs_to_pdfs(CODE_IMPLEMENTER_GRAPH_PNG, CODE_IMPLEMENTER_GRAPH_PDF)

    if not asyncio.run(run_result_validator()):
        print("❌ Pipeline failed at Result Validator step")
        return False
    


    print(f"\n🏁 Total Pipeline Time: {time.time() - pipelineStart:.2f}")
    try:
        print(f"💰Estimated Cost: ${token_cost:.4f}\n")

    except Exception as e:
        print(f"❌ Error calculating token cost: {e}")

    return True


if __name__ == "__main__":
    if run_full_agent_flow():
        print("✅ All stages ran successfully")
    else:
        print("❌ There was an error in the pipeline")