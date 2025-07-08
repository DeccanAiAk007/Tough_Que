import os
import csv
import json
import re
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
# import google.generativeai as genai
import together
from pydantic import BaseModel, Field

# ——— CONFIGURATION ———
load_dotenv()
# OpenAI (o3)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)
# Gemini (2.5)
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# GEMINI_MODEL = "gemini-2.5-turbo"
# Together (deepseekr1)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# TOGETHER_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
# TOGETHER_MODEL = "deepseek-ai/DeepSeek-R1"

INPUT_CSV = "input.csv"
OUTPUT_CSV = "solutions_verifying.csv"
TIMEOUT_SECONDS = 400  # 5 minutes per question

# ——— Define function schema for OpenAI function-calling ———
FUNCTIONS = [
    {
        "name": "record_solution",
        "description": "Record the solution and confidence score for a STEM question",
        "parameters": {
            "type": "object",
            "properties": {
                "solution": {"type": "string"},
                "confidence_score": {"type": "number"}
            },
            "required": ["solution", "confidence_score"]
        }
    }
]

class TogetherResponse(BaseModel):
    solution: str = Field(description="The solution to the question")
    confidence_score: float = Field(description="Confidence score of the solution between 0 and 1")

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are an advanced AI assistant trained to solve complex, interdisciplinary STEM problems. "
        "Your task is to carefully analyze each user's question, execute all necessary reasoning steps (symbolic, numerical, conceptual), and return a final answer by making a JSON function call to `record_solution`.\n\n"

        "You must respond with the function call `record_solution`, passing exactly two fields:\n"
        "  - `solution`: A well-structured explanation of the answer, optionally including formulas, step-by-step logic, or boxed final expressions.\n"
        "  - `confidence_score`: A floating-point number between 0.0 and 1.0 that reflects your overall confidence in the validity of your answer.\n\n"

        "**Definition of `confidence_score`:**\n"
        "The confidence score quantifies your belief in the **overall correctness and executability** of your response, based on:\n"
        "1. How clearly the question is posed (well-defined vs ambiguous or underspecified).\n"
        "2. How complete and error-free your solution is (logical consistency, math correctness, correct interpretation).\n"
        "3. Whether the problem has a standard resolution path or requires speculative or novel reasoning.\n"
        "4. How likely it is that the user could verify and reproduce the solution from your steps.\n\n"

        "**Scoring scale:**\n"
        "- `0.95 – 1.0` → Extremely confident: The question is well-defined, and your solution is complete, mathematically sound, and verified.\n"
        "- `0.75 – 0.94` → High confidence: Your reasoning is mostly solid, with minor ambiguities or assumptions, but no major flaws.\n"
        "- `0.5 – 0.74` → Moderate confidence: Partial uncertainty in the question or some steps; your logic is plausible but may require validation.\n"
        "- `0.25 – 0.49` → Low confidence: Substantial gaps or potential errors in logic, or the question is poorly defined.\n"
        "- `< 0.25` → Very low confidence: Question likely malformed or unsolvable with available information; speculative or placeholder answer.\n\n"

        "**Instructions:**\n"
        "- Do NOT add any commentary, headers, markdown, or explanation outside the JSON function call.\n"
        "- Respond ONLY with a single valid JSON object suitable for calling `record_solution`.\n"
        "- Format all LaTeX equations as plain strings (escaped properly).\n\n"

        "**Example return format:**\n"
        "{\n"
        '  "solution": "Using energy-momentum relation: v = c * sqrt(1 - (m^2 c^4 / E^2))",\n'
        '  "confidence_score": 0.96\n'
        "}"
    )
}




def solve_question_o3(question: str) -> dict:
    messages = [SYSTEM_MSG, {"role": "user", "content": question}]
    resp = openai_client.chat.completions.create(
        model="o3",
        messages=messages,
        functions=FUNCTIONS,
        function_call={"name": "record_solution"},
    )
    msg = resp.choices[0].message
    func = getattr(msg, "function_call", None)
    if func and func.name == "record_solution":
        args = json.loads(func.arguments)
        print(f"OpenAI O3 confidence: {args['confidence_score']}")
        return {"solution": args["solution"], "confidence_score": args["confidence_score"]}
    print("OpenAI O3 0")
    return {"solution": None, "confidence_score": 0.0}

def solve_question_gpt4_1(question: str) -> dict:
    messages = [SYSTEM_MSG, {"role": "user", "content": question}]
    resp = openai_client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
        functions=FUNCTIONS,
        function_call={"name": "record_solution"},
    )
    msg = resp.choices[0].message
    func = getattr(msg, "function_call", None)
    if func and func.name == "record_solution":
        args = json.loads(func.arguments)
        print(f"OpenAI gpt_4.1 confidence: {args['confidence_score']}")
        return {"solution": args["solution"], "confidence_score": args["confidence_score"]}
    print("OpenAI gpt_4.1 0")
    return {"solution": None, "confidence_score": 0.0}



# def solve_question_gemini(question: str) -> dict:
#     chat = genai.chat.create(
#         model=GEMINI_MODEL,
#         prompt=question,
#         structured_output=True,
#         output_schema={
#             "solution": {"type": "string"},
#             "confidence_score": {"type": "number"}
#         }
#     )
#     return {
#         "solution": chat.response.get("solution"),
#         "confidence_score": chat.response.get("confidence_score", 0.0)
#     }


def safe_parse_response(content: str) -> dict:
    """
    Try to parse a JSON object from `content`. If direct loads fail,
    extract the first {...} block via regex and retry. On failure, return fallback.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Attempt to extract JSON block
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:

                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    # Fallback
    return {"solution": "could not answer", "confidence_score": 0.0}


# def solve_question_together_qwen(question: str) -> dict:
#     client = together.Together(api_key=TOGETHER_API_KEY)
#     extract = client.chat.completions.create(
#         model="Qwen/Qwen3-235B-A22B-fp8-tput",
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an advanced AI that solves complex, interdisciplinary STEM questions. "
#                     "Respond **only** with a JSON object containing 'solution' and 'confidence_score' (0–1)."
#                 )
#             },
#             {"role": "user", "content": question}
#         ],
#         response_format={
#             "type": "json_object",
#             "schema": TogetherResponse.model_json_schema(),
#         },
#     )
#     raw = extract.choices[0].message.content
#     data = safe_parse_response(raw)
#     print(f"Together Qwen confidence: {data.get('confidence_score', 0.0)}")
#     return {
#         "solution": data.get("solution"),
#         "confidence_score": data.get("confidence_score", 0.0)
#     }
    # return {"solution": None, "confidence_score": 0.0}


# def solve_question_together(question: str) -> dict:
#     client = together.Together(api_key=TOGETHER_API_KEY)
#     extract = client.chat.completions.create(
#         model=TOGETHER_MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an advanced AI that solves complex, interdisciplinary STEM questions. "
#                     "Respond **only** with a JSON object containing 'solution' and 'confidence_score' (0–1)."
#                 )
#             },
#             {"role": "user", "content": question}
#         ],
#         response_format={
#             "type": "json_object",
#             "schema": TogetherResponse.model_json_schema(),
#         },
#     )
#     raw = extract.choices[0].message.content
#     data = safe_parse_response(raw)
#     print(f"Together Qwen confidence: {data.get('confidence_score', 0.0)}")

#     return {
#         "solution": data.get("solution"),
#         "confidence_score": data.get("confidence_score", 0.0)
#     }
    # return {"solution": None, "confidence_score": 0.0}



if __name__ == "__main__":
    # Read questions
    for i in range(3):
        questions = []
        OUTPUT_CSV = f"turn_{i+1}"+ OUTPUT_CSV
        # with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        try:
            with open(INPUT_CSV, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except UnicodeDecodeError:
            print("[Warning] UTF-8 decode failed. Retrying with cp1252 encoding...")
            with open(INPUT_CSV, newline="", encoding="cp1252") as f:
                rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("question"):
                questions.append(row["question"])

        # Prepare CSV
        write_header = not os.path.exists(OUTPUT_CSV)
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "question",
                "solution_o3", "confidence_o3",
                "solution_gpt4.1", "confidence_gpt4.1",
                # "solution_qwen", "confidence_qwen",
                # "solution_deepseekr1", "confidence_deepseekr1"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                for idx, q in enumerate(questions, 1):
                    print(f"Processing {idx}/{len(questions)}: {q[:50]}...")
                    future_o3 = executor.submit(solve_question_o3, q)
                    future_gpt41 = executor.submit(solve_question_gpt4_1, q)
                    # future_gen = executor.submit(solve_question_gemini, q)
                    # future_tog = executor.submit(solve_question_together, q)
                    # future_togqwen = executor.submit(solve_question_together_qwen, q)

                    try:
                        res_o3 = future_o3.result(timeout=TIMEOUT_SECONDS)
                    except concurrent.futures.TimeoutError:
                        res_o3 = {"solution": "could not answer", "confidence_score": 0.0}
                        
                    try:
                        res_gpt41 = future_gpt41.result(timeout=TIMEOUT_SECONDS)
                    except concurrent.futures.TimeoutError:
                        res_gpt41 = {"solution": "could not answer", "confidence_score": 0.0}

                    # try:
                    #     res_gen = future_gen.result(timeout=TIMEOUT_SECONDS)
                    # except concurrent.futures.TimeoutError:
                    #     res_gen = {"solution": "could not answer", "confidence_score": 0.0}

                    # try:
                    #     res_deepseek = future_tog.result(timeout=TIMEOUT_SECONDS)
                    # except concurrent.futures.TimeoutError:
                    #     res_deepseek = {"solution": "could not answer", "confidence_score": 0.0}

                    # try:
                    #     res_qwen = future_togqwen.result(timeout=TIMEOUT_SECONDS)
                    # except concurrent.futures.TimeoutError:
                    #     res_qwen = {"solution": "could not answer", "confidence_score": 0.0}

                    writer.writerow({
                        "question": q,
                        "solution_o3": res_o3["solution"],
                        "confidence_o3": res_o3["confidence_score"],
                        "solution_gpt4.1": res_gpt41["solution"],
                        "confidence_gpt4.1": res_gpt41["confidence_score"],
                        # "solution_gemini_2.5": res_gen["solution"],
                        # "confidence_gemini_2.5": res_gen["confidence_score"],
                        # "solution_qwen": res_qwen["solution"],
                        # "confidence_qwen": res_qwen["confidence_score"],
                        # "solution_deepseekr1": res_deepseek["solution"],
                        # "confidence_deepseekr1": res_deepseek["confidence_score"],
                    })


        print(f"Written solutions to {OUTPUT_CSV}")