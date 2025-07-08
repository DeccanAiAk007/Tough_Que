import os
import csv
import json
import re
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI
from generativeai import google as genai
import together

# Load environment variables
load_dotenv()

# API configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
together_client = together.Together(api_key=TOGETHER_API_KEY)

# File paths
INPUT_CSV = "input.csv"
OUTPUT_CSV = "model_solutions.csv"
TIMEOUT_SECONDS = 400

# Function definition for OpenAI
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

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are an advanced AI assistant trained to solve complex, interdisciplinary STEM problems. "
        "Return ONLY a valid JSON object with 'solution' and 'confidence_score' as described."
    )
}

# Helper to parse JSON

def safe_parse_response(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"solution": "could not answer", "confidence_score": 0.0}

# Model Solvers

def solve_with_openai(model: str, question: str):
    messages = [SYSTEM_MSG, {"role": "user", "content": question}]
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            functions=FUNCTIONS,
            function_call={"name": "record_solution"},
        )
        func = resp.choices[0].message.function_call
        args = json.loads(func.arguments)
        return args
    except Exception:
        return {"solution": "could not answer", "confidence_score": 0.0}

def solve_with_gemini(question: str):
    try:
        chat = genai.chat.create(model="gemini-1.5-pro", messages=[{"role": "user", "content": question}])
        response = chat.last.text
        return safe_parse_response(response)
    except Exception:
        return {"solution": "could not answer", "confidence_score": 0.0}

def solve_with_deepseek(question: str):
    try:
        extract = together_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "system", "content": "Respond with a JSON object containing 'solution' and 'confidence_score'"},
                {"role": "user", "content": question},
            ]
        )
        return safe_parse_response(extract.choices[0].message.content)
    except Exception:
        return {"solution": "could not answer", "confidence_score": 0.0}

# Pipeline Runner

def process_questions():
    try:
        with open(INPUT_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(INPUT_CSV, newline="", encoding="cp1252") as f:
            rows = list(csv.DictReader(f))

    questions = [row["question"] for row in rows if row.get("question")]
    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question",
            "solution_o3", "answer_o3", "confidence_o3",
            "solution_gpt4.1", "answer_gpt4.1", "confidence_gpt4.1",
            "solution_gemini", "answer_gemini", "confidence_gemini",
            "solution_deepseekr1", "answer_deepseekr1", "confidence_deepseekr1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, q in enumerate(questions, 1):
            print(f"Processing {idx}/{len(questions)}: {q[:60]}...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    "o3": executor.submit(solve_with_openai, "o3", q),
                    "gpt4.1": executor.submit(solve_with_openai, "gpt-4.1-2025-04-14", q),
                    "gemini": executor.submit(solve_with_gemini, q),
                    "deepseek": executor.submit(solve_with_deepseek, q),
                }

                results = {}
                for name, future in futures.items():
                    try:
                        result = future.result(timeout=TIMEOUT_SECONDS)
                    except concurrent.futures.TimeoutError:
                        result = {"solution": "could not answer", "confidence_score": 0.0}
                    results[name] = result

            writer.writerow({
                "question": q,
                "solution_o3": results["o3"]["solution"],
                "answer_o3": extract_final_answer(results["o3"]["solution"]),
                "confidence_o3": results["o3"]["confidence_score"],
                "solution_gpt4.1": results["gpt4.1"]["solution"],
                "answer_gpt4.1": extract_final_answer(results["gpt4.1"]["solution"]),
                "confidence_gpt4.1": results["gpt4.1"]["confidence_score"],
                "solution_gemini": results["gemini"]["solution"],
                "answer_gemini": extract_final_answer(results["gemini"]["solution"]),
                "confidence_gemini": results["gemini"]["confidence_score"],
                "solution_deepseekr1": results["deepseek"]["solution"],
                "answer_deepseekr1": extract_final_answer(results["deepseek"]["solution"]),
                "confidence_deepseekr1": results["deepseek"]["confidence_score"],
            })
    print(f"All results written to {OUTPUT_CSV}")

# Extract final boxed answer

def extract_final_answer(solution_text):
    match = re.search(r"\\boxed\{([^}]*)\}|Answer: ([^\n]*)", solution_text)
    if match:
        return match.group(1) or match.group(2)
    return "not found"

if __name__ == "__main__":
    process_questions()
