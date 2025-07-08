import os
import json
import time
from openai import OpenAI
import together
import random
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic
from concurrent.futures import ThreadPoolExecutor

# === Load Keys ===
load_dotenv()
together.api_key = os.getenv("TOGETHER_API_KEY")
client = together.Together()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL_mx = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_gpt = "gpt-4.1-2025-04-14"
MODEL_o3 = "o3"
MODEL_cld = "claude-3-5-sonnet-20241022"
MODEL_cld4 = "claude-sonnet-4-20250514"
MODEL_gemini = "gemini-2.5-pro"

MAX_CYCLES = 4

PHYSICS_BASE_TOPICS = [
    "a variable-mass system like a chain falling off a table or rocket expelling mass in space",
    "a complex pulley system involving rotational inertia and moving masses",
    "a projectile launched from a moving, accelerating platform (non-inertial frame)",
    "collision between a rotating disc and a point mass using conservation of angular and linear momentum",
    "a double pendulum system with damping or forced oscillation components",
    "a particle sliding inside a rotating hemispherical bowl or cone",
    "a block on an inclined plane in a lift undergoing variable acceleration (pseudo forces)",
    "a bead on a rotating wire loop with friction and Coriolis forces considered",
    "center of mass motion involving disconnection or explosion of system components mid-flight",
    "fluid flow through a non-uniform pipe under gravity with Bernoulli’s theorem and viscous losses",
    "torque on a submerged object due to pressure gradient and buoyancy force",
    "a rotating cylindrical container partially filled with liquid (fluid surface profile and pressure distribution)",
    "Doppler effect with both source and observer accelerating non-uniformly",
    "a tuning fork vibrating over a moving open pipe (resonance conditions with relative motion)",
    "interference pattern of two coherent sources with a moving detector on a rotating frame",
    "diffraction and interference in a thin wedge-shaped film under varying pressure",
    "adiabatic expansion of an ideal gas in a moving piston inside a variable-gravity elevator",
    "entropy change in mixing two ideal gases with different temperatures and volumes",
    "heat engine working with real gases involving Van der Waals corrections",
    "cooling of a gas due to adiabatic throttling (Joule-Thomson effect) and internal energy considerations",
    "capacitor network with variable dielectric slab insertion and battery connection changes",
    "potential distribution in a semi-infinite grounded conducting plane with a point charge",
    "electrostatic force on a dielectric partially pulled out of a capacitor connected to a variable voltage source",
    "a Wheatstone bridge with a sliding contact and varying temperature-dependent resistance",
    "power dissipation in a circuit with time-varying EMF and multiple branches",
    "transient analysis of an LC circuit with initial charge on capacitor and mutual inductance",
    "a conducting rod falling in a non-uniform magnetic field under gravity (eddy current forces)",
    "induced current in a complex rotating coil with changing angular velocity and non-uniform B-field",
    "a charged particle in a region with magnetic field varying in both space and time",
    "light beam passing through a multi-layered dielectric slab with increasing refractive index (total internal reflection case)",
    "lens system with spherical and chromatic aberrations affecting image formation",
    "a non-paraxial ray incident on a curved mirror system with conic sections (e.g., paraboloid or ellipsoid)",
    "time dilation and length contraction of a decaying muon observed from a train moving at relativistic speed",
    "photoelectric effect with variable frequency light and potential applied across detector plates",
    "a particle accelerated in a cyclotron approaching relativistic limits",
    "decay of a moving particle in a lab frame vs its own rest frame—relativistic transformation needed",
    "analyzing a force vs displacement graph to deduce potential energy curve and equilibrium conditions",
    "determining acceleration from non-linear velocity-time graph with instantaneous slope changes",
    "error propagation in a compound experiment involving thermocouple and potentiometer readings",
    "a thermally expanding rod in a magnetic field with induced EMF and mechanical stress buildup",
    "a rotating ring with embedded charges exposed to a radial electric field and analyzed from rotating frame",
    "motion of a charged particle through crossed E and B fields inside a varying gravitational field (combined Lorentz + pseudo + gravity)",
    "multi-stage rocket ejecting mass while moving through a resistive medium under Earth's gravity",
]


PHYSICS_COMPLEXITY_TEMPLATES = [
    "Integrate concepts from two or more domains, such as rotational dynamics with electrostatics, or thermodynamics with mechanics.",
    "Design a situation with multiple correct approaches, but only one is physically valid under real-world constraints.",
    "Include variable dependencies like position-dependent forces, velocity-dependent drag, or time-varying mass or acceleration, requiring calculus-based setup and resolution.",
    "Construct the problem with at least three tightly chained reasoning steps such as Newton’s laws → energy transformation → symbolic integration → final numeric evaluation.",
    "Require graphical reasoning, such as interpreting slopes, areas under curves, or analyzing inflection points on a force-position graph.",
    "Introduce hidden or implicit quantities (e.g., derive effective mass, pressure, or geometric quantity) that are not given directly but required for solution.",
    "Use edge-case analysis or limiting conditions such as relativistic speeds, extremely low temperatures, or non-ideal gas behaviors.",
    "Force the setup and reasoning to occur within a non-inertial frame where pseudo-forces (e.g., centrifugal, Coriolis) must be explicitly introduced and evaluated.",
    "Ensure that all variables are symbolic and defined clearly, and that each assumption (e.g., massless pulley, ideal gas) is stated explicitly to avoid ambiguity.",
]

# === Utility: Call LLM ===

def call_model_openai(prompt: str, model: str) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def call_model_gemini(prompt: str, model: str) -> str:
    model = genai.GenerativeModel(model_name=model)
    response = model.generate_content(prompt)
    return response.text.strip()

def call_model_tog(prompt: str, model: str) -> str:
    prompt = prompt
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0
    )
    return response.choices[0].message.content.strip()
 
def call_model_claude(prompt: str, model: str) -> str:
    try:
        response = claude_client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print("[Claude Error]", e)
        return "[Error]" 
    

# === Step 1: Generate Initial Question (Refined) ===
def generate_seed_question() -> str:
    def generate_initial_prompt() -> str:
        topic_k = random.randint(2, min(4, len(PHYSICS_BASE_TOPICS)))
        complexity_k = random.randint(4, min(6, len(PHYSICS_COMPLEXITY_TEMPLATES)))
        
        topic = random.sample(PHYSICS_BASE_TOPICS, k=topic_k)
        complexity = random.sample(PHYSICS_COMPLEXITY_TEMPLATES, k=complexity_k)
        return f"""
You are a domain-expert AI in physics problem generation. Your task is to generate a **rigorous, self-contained graduate-level physics question** that:

🔹 Combines multiple physics domains  
🔹 Involves real-world physical dependencies  
🔹 Requires chained symbolic reasoning  
🔹 Is fully defined, reproducible, and yields **one unique final numeric answer**
🔹 **Expressed using proper LaTeX notation** for all mathematical symbols and equations.

---

###  Problem Requirements:

1. **Interdisciplinary Concept Integration**  
   Integrate at least **two physics domains**, e.g., rotational mechanics + electromagnetism, or thermodynamics + fluid dynamics.

2. **Structured Symbolic Reasoning**  
   Problem must require **minimum four-five tightly linked symbolic steps** before computing final numeric result.

3. **Fully Defined Initial and Boundary Conditions**  
   Explicitly state:  
   - System setup with no ambiguity  
   - All assumptions required for the problem and the values of the parameters to be used in the problem
   - Units and dimensions of all given quantities

4. **Question Construction Constraints**  
   - Construct the problem so that it admits only **one unique, boxed numeric result** with **correct SI units**.  
   - Do **not** ask for intermediate results — present only one **explicit, final question**.  
   - The final answer must be **fully determined and logically derived** from the information and assumptions provided in the problem statement, with no ambiguity or missing data.

5. **Completely unambiguous**
   - with all assumptions, parameters, and given data explicitly specified.
   
6. **Expressed using proper LaTeX notation** for all mathematical symbols and equations. 
   
7. **No ambiguity** — do not assume unspecified values; all numbers must be given.

### Problem Construction Constraints:

- The question must present one clear physical setup with all parameters, variables, and conditions specified.
- Do **not** ask for intermediate results — present only one **explicit, final question**.
- The **final question** must require one numeric answer with correct SI units.
- The final answer must be a **boxed numeric value with correct SI units**.
- Clearly state **what to find** as the last part of the question.
- Avoid qualitative, vague, or open-ended prompts.
- Present the final answer in one explicit box at the end of the solution.
   
---

###  Use These Concepts (Topics):  
{topic}

###  Follow These Complexity Constraints:  
{complexity}

---

###  Output Format (Strict JSON Only):

Return **only** the following JSON object (no markdown, no comments — with):

{{
  "question": "<Complete and rigorous LaTeX-formatted physics problem, with all assumptions, given parameters, boundary conditions, units, and one final numeric answer requested>",
  "topic_tags": ["<tag1>", "<tag2>", ...]
}}

---

### Important Rules:
- **No ambiguous terms** (e.g. "assume reasonable values").
- **All assumptions must be explicitly stated.**
- **All required constants must be provided numerically with units.**
- The solution must require symbolic derivations with intermediate sub-questions, all leading to one final numeric answer.
- The final numeric answer must have a **correct SI unit**.

Produce **only** this strict JSON — do not include any other text.
""".strip()

    prompt = generate_initial_prompt()
    return call_model_openai(prompt, MODEL_gpt)


# === Step 2: Extract Parameters, Assumptions, Ambiguities ===
def extract_parts_from_question(seed_json: str) -> str:
    prompt = f"""
You are a physics reasoning assistant.

Your task is to extract **all** key physical components from the provided physics problem JSON. Be thorough, accurate, and follow these guidelines:

1. **Parameters**  
   - List all symbolic variables used in the problem  
   - For each variable, give:
     - `symbol` (e.g., "v₀")
     - `meaning` (e.g., "initial velocity of the projectile")
     - `unit` strict SI unit (e.g. "m/s^2", "kg"), or "dimensionless" if none
   - Only include variables relevant to the physical formulation or reasoning steps.
   - Include all numeric constants too (e.g. gravitational acceleration g, elementary charge e) as parameters if they appear.


2. **Assumptions**  
   - Extract all explicitly stated assumptions (e.g., "neglect air resistance", "ideal gas", "non-inertial frame")  
   - Also infer any common implicit assumptions if they impact reasoning (e.g., "constant gravitational field").

3. **Concepts Used**  
   - Identify all **physics concepts, laws, or principles** explicitly or implicitly required to solve the problem.
   - For each concept, briefly state its role or relevance in the context of the problem.
   - Include both fundamental and advanced concepts as appropriate.

### Output JSON Format (strict):
{{
  "parameters": [
    {{"symbol": "...", "meaning": "...", "unit": "..."}},
    ...
  ],
  "assumptions": ["...", "..."],
  "concepts": ["...", "..."]
}}

### Input JSON:
{seed_json}
"""
    return call_model_gemini(prompt, MODEL_gemini)




# === Step 3: Toughen Each Part and Reconstruct ===
def rewrite_parts_to_make_tougher(extracted_json: str) -> str:
    prompt = f"""
You are an expert physics reasoning engine.


Your task is to:
1. Take the extracted components of a physics problem.
2. Rewrite and enhance them to make the question **significantly 1000 times challenging** — at an advanced graduate level — using complex, multi-domain reasoning.
3. Ensure the final question is:
   - **Entirely in LaTeX math notation** where appropriate.
   - Do **not** ask for intermediate results — present only one **explicit, final question**.
   - The **final question** must require one numeric answer with correct SI units.
   - Fully self-contained with all assumptions, given values, and conditions specified explicitly.
   - Requires **multiple, chained symbolic and numeric reasoning steps**.
   - Leads to **one unique numeric final answer with correct SI unit** that can be boxed at the end.
   - Uses realistic parameters and **no ambiguous terms** like “reasonable assumptions” or “arbitrary values.”
---

### OUTPUT FORMAT (JSON only):

Return a fully rewritten **toughened problem** in the following strict JSON format:

{{
  "question": "<Complete, rigorous problem in LaTeX notation>",
  "topic_tags": ["<relevant physics domains>"]
}}

---

### INPUT (Original Extracted JSON):
{extracted_json}

---
Only return the final JSON — no explanation, no commentary.
"""
    return call_model_openai(prompt, MODEL_gpt)


# === Step 4: Get Feedback and Improve Question ===
def get_feedback(question_json: str) -> str:
    prompt = f"""
You are a critical physics evaluator AI trained to review complex physics problems and solutions with an expert lens.

Analyze the following JSON which contains a physics question.

### Your tasks:
1. **Identify and point out ambiguities** in the question. Are any variables, conditions, or assumptions undefined or unclear?
2. **Check for consistency**:
   - Are all the symbols used in the question properly introduced in the question?
   - Are there any steps or concepts introduced in the question that are **not justified** or **not stated** in the question?
   - Are there terms in the question that are never used or resolved in the question?
3. **Assess logical flow and physics reasoning**:
   - Does the question follow a valid chain of physical principles?
   - Are there reasoning chains missing or skipped in the question?
   - Are there physically invalid reasoning, false conclusions, or unjustified shortcuts?
4. **Check assumptions and constraints**:
   - Are all assumptions stated **explicitly**?
   - Are any assumptions **unnecessary**, **irrelevant**, or **physically unrealistic**?
   - Are there any hidden assumptions in the question not declared in the question?
5. **Ensure No Intermediate Sub-Questions**:
   Finally, ensure:
   - The question does not explicitly ask for sub-answers or intermediate values.
   - The final query is one clear question asking for a single final numeric answer with SI units.
   - Avoid phrasing like:
      - “Find the intermediate concentration...”
      - “First, calculate X then use it to find Y...”
      - “Derive the expression for Z and then find the numeric value.”
   - The question must state a final single, numeric goal — all intermediate derivations are implicit.   

### Input JSON:
{json.dumps(question_json, indent=2)}

### Output:
Return only a paragraph of structured, critical feedback highlighting any flaws, gaps, or improvements as per the task. Avoid generic praise. Be extremely precise, provide examples, be thorough, and technically sound.
""".strip()

    return call_model_openai(prompt, MODEL_o3)

# === Step 5: IMproved QUestion
def improve_question_based_on_feedback(feedback: str, original_json: str) -> str:
    prompt = f"""
You previously generated a high-level physics problem, but it has been reviewed and received the following detailed feedback:
### Original question:
{original_json}

### Feedback:
\"\"\"{feedback}\"\"\"

Your new task:
- Revise and improve the original question so that it fully addresses **every point** in the feedback.
- The resulting question must be:
  1. **Entirely in LaTeX math notation** where appropriate (for all variables, symbols, equations, etc.).
  2. **Fully self-contained** with all assumptions, parameters, boundary conditions, constants, and SI units explicitly specified.
  3. **Physically rigorous and interdisciplinary**, integrating at least two domains of physics.
  4. **Challenging and solvable with multi-step symbolic and numeric reasoning.**
  5. Do **not** ask for intermediate results — present only one **explicit, final question**.
  6. The **final question** must require one numeric answer with correct SI units.
  5. Designed so that there is **exactly one final numeric answer** requested at the end — with correct SI unit — that is uniquely determined by the given information.

### Important:
- Do NOT include any commentary or markdown formatting.
- Output must be strictly JSON and fully parsable.
- The regenerated problem must **fully reflect and satisfy the feedback** above.

Strictly return ONLY a valid, properly formatted JSON object with the following format:
{{
  "question": "<Improved and fully self-contained problem statement in strict LaTeX notation>",
  "topic_tags": ["<tag1>", "<tag2>", ...]
}}
Produce ONLY the JSON as specified.
""".strip()

    return call_model_openai(prompt, MODEL_gpt)



# === Step 5: Get Final Answer from 2 Models ===
def get_final_answer(question_json: str) -> dict:
    prompt = f"""
You are a high-precision physics solver.

Solve the following physics problem and return **only the final boxed numeric result with SI units**.

---

### STRICT OUTPUT RULES:
- Do NOT show any working, steps or explanations.
- Do NOT return anything other than the final answer.
- Use boxed format: Example → `"F = 3.14 N"` or `"ΔS = 12.5 J/K"`.
- Include proper SI units and symbols (e.g., `m/s`, `kg`, `J/K`).
- If symbolic constants (e.g., `ln(2)`, `π`) are involved, compute numerically to **at least 3 significant digits**.

---

### PHYSICS PROBLEM INPUT:
{question_json}

---

### FINAL OUTPUT:
Strictly Return just a single string with the boxed numeric result. No markdown, no JSON.
"""
    # return call_model_openai(prompt, model_name)
    def run_openai():
        return call_model_openai(prompt, MODEL_o3)  # o3 or any GPT variant

    def run_gemini():
        return call_model_gemini(prompt, MODEL_gemini)  # Claude variant

    with ThreadPoolExecutor() as executor:
        future_openai = executor.submit(run_openai)
        future_gemini = executor.submit(run_gemini)

        openai_result = future_openai.result()
        gemini_result = future_gemini.result()

    return {
        "gpt_o3": openai_result,
        "gemini2.5pro": gemini_result
    }


# === Step 6: Compare Answers ===
def compare_answers(ans1: str, ans2: str) -> dict:
    prompt = f"""
You are a physics answer verification engine.

Your task is to compare two final answers from physics models and determine whether they represent the **same physical result**. Use technical judgment to evaluate unit compatibility, numeric equivalence, symbolic form, and physical interpretation.

---

###  Evaluation Criteria and Scoring

Each of the following aspects contributes to a **similarity score** between 0.0 and 1.0. The total score is calculated by adding weighted partial scores:

| Criterion                    | Weight | Description |
|-----------------------------|--------|-------------|
| 1. Unit Compatibility       | 0.25   | Full score if units are same or dimensionally equivalent (e.g., N·m vs J). Zero if incompatible (e.g., N vs m/s). |
| 2. Numerical Closeness      | 0.30   | Compare using relative error. Score full if relative error < 1%, partial for 1–2%. Zero if >2%. |
| 3. Symbolic/Decimal Match   | 0.15   | Accept cases like `1/√2` ≈ `0.707`, or `π` ≈ `3.14`. Partial if unclear equivalence. |
| 4. Rounding/Notation Format | 0.10   | Full score if formats differ but values are effectively same (e.g., 3.14 vs 3.1416). |
| 5. Expression Equivalence   | 0.20   | Score based on structural or algebraic similarity (e.g., `mv²/2` vs `0.5mv²`). |

---

###  Final Decision Rule

- If total score **≥ 0.80**, return: `"decision": "similar"`
- If total score **< 0.80**, return: `"decision": "different"`

---

###  Example:

**Answer 1:** `1/√2 m/s`  
**Answer 2:** `0.707 m/s`

Evaluation:

- Units: m/s = m/s → 0.25
- Numeric: 1/√2 ≈ 0.707 → 0.30
- Symbolic/Decimal: equivalent → 0.15
- Rounding: acceptable → 0.10
- Expression: scalar match → 0.20

**Total score = 1.00 → "similar"**

---

### Your Task

Compare the following two answers and return a **valid JSON object** in this format:

{{
  "similarity_score": <float between 0.0 and 1.0>,
  "decision": "similar" or "different",
  "comment": "<brief technical explanation>"
}}

Only return the JSON object. Do NOT include Markdown, LaTeX formatting, or extra commentary.
### Input:
Answer 1: {ans1}  
Answer 2: {ans2}
"""
    response = call_model_openai(prompt, MODEL_gpt)
    return json.loads(response)



# === Main Pipeline Loop ===
def pipeline_loop():
    seed_json = None  # Initialize seed_json
    improved_question = None  # Initialize improved_question
    cycle_logs = []

    for cycle in range(MAX_CYCLES):
        print(f"\n [Cycle {cycle + 1}]")

        # Step 1: Generate new seed only if first cycle or models disagreed last time
        if seed_json is None:
            seed_json = generate_seed_question()

        # Step 2
        extracted = extract_parts_from_question(seed_json)

        # Step 3
        tougher_question = rewrite_parts_to_make_tougher(extracted)

        # Step 4
        feedback = get_feedback(tougher_question)
        improved_question = improve_question_based_on_feedback(feedback, tougher_question)

        # Step 5
        # ans_gpt = get_final_answer(improved_question, MODEL_gpt)
        # ans_o3 = get_final_answer(improved_question, MODEL_o3)
        answer_dict = get_final_answer(improved_question)
        ans_o3 = answer_dict["gpt_o3"]
        ans_gemini = answer_dict["gemini2.5pro"]

        # Step 6
        similarity_result = compare_answers(ans_gemini, ans_o3)

        # If using enhanced version with JSON output:
        if isinstance(similarity_result, dict):
            decision = similarity_result.get("decision", "different")
            score = similarity_result.get("similarity_score", 0.0)
        else:
            decision = similarity_result
            score = None

        print(f"\n O3 Answer: {ans_o3}\n Gemini2.5pro Answer: {ans_gemini}\n Similarity Decision: {decision}  (Score: {score})")
        
        # Save this cycle's outputs
        cycle_logs.append({
            "cycle": cycle + 1,
            "seed_json": seed_json,
            "extracted": extracted,
            "tougher_question": tougher_question,
            "feedback": feedback,
            "improved_question": improved_question,
            "Gemini2.5pro": ans_gemini,
            "ans_O3": ans_o3,
            "similarity_result": similarity_result,
        })

        if decision == "different":
            print("\n Final Refined Question (as models disagree):")
            print(improved_question)
            return improved_question, cycle_logs

        print(" Answers similar — reusing same question as new seed...\n")
        # Reuse the improved question as next seed
        seed_json = improved_question
        time.sleep(2)
    
    print("\nReturning last question.")
    print(improved_question)
    return improved_question, cycle_logs

# # === Run ===
# if __name__ == "__main__":
#     final_question = pipeline_loop()
#     with open("final_physics_question.json", "w") as f:
#         json.dump(final_question, f, indent=2)

# === Run ===
if __name__ == "__main__":
    all_results = []
    iter = 10
    all_logs = []
    
    for i in range(iter):
        print(f"\n=== Running pipeline_loop #{i+1}/{iter} ===")
        try: 
            final_question, cycle_logs = pipeline_loop()
            all_results.append(final_question)
            all_logs.append({
                "iteration": i + 1,
                "logs": cycle_logs
            })
        except Exception as e:
            print(f"[ERROR] Iteration {i+1} failed: {e}")
            all_results.append({"error": str(e)})
            all_logs.append({
                "iteration": i + 1,
                "logs": [],
                "error": str(e)
            })
    
                
    all_results_path = "all_results5.json"            
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    # Save all cycle logs
    all_cycle_logs_path = "all_logs5.json"
    with open(all_cycle_logs_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2)
            
    print("\n Completed all iterations and saved results to all_results.json")        
