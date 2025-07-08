import json, re, random, pathlib
from ollama import Client

client = Client(host="http://localhost:11434")

MODEL = "llama3.1:8b-instruct-q4_K_M"

SOLVE_PROMPT = """
You are a careful tutor. Think step-by-step, but finish with a line
that reads exactly:
ANSWER: <integer>
Problem: {{question}}
"""

CRITIQUE_TEMPLATE = """
You are a math teacher checking your own work.

Question:
{{question}}

Previous answer:
{{answer}}

If the previous answer is already correct, reply with:
CRITIQUE=OK || ANSWER={{answer}}

Otherwise:
1  In ≤40 chars say what is wrong.
2  Re-solve step-by-step.
3  Finish with  ➜  CRITIQUE=<…> || ANSWER=<integer>

Return **exactly one line**; no markdown, no extra text.
"""

REFINE_PROMPT = """
You previously solved the problem and got:

ANSWER₀ = {{old}}

The teacher pointed out:

{{critique}}

Think through the problem again, fix the mistake, and
end with:
ANSWER: <integer>
Problem: {{question}}
"""

def call_llm(prompt: str, model: str = MODEL, temperature: float = 0.2) -> str:
    print("\n🔸 Prompt (truncated):")
    print(prompt[:200].replace("\n", " ") + "…", flush=True)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    reply =  resp["message"]["content"].strip()
    print("🔹 Reply (truncated):", reply[:200].replace("\n", " ") + "…\n", flush=True)
    return reply

def solve_once (q):
    return call_llm(SOLVE_PROMPT.replace("{{question}}", q))

def parse_crit_line(line: str, prev_answer: str):
    if "||" not in line:
        raise ValueError
    crit_part, ans_part = [s.strip() for s in line.split("||", 1)]
    crit = crit_part.replace("CRITIQUE=", "", 1).strip()
    ans  = ans_part .replace("ANSWER="  , "", 1).strip() or prev_answer
    return crit, ans

def critique_once(question, prev_answer):
    """Return only the CRITIQUE string (no answer). Retry once on bad format."""
    def ask(prompt_txt):
        return call_llm(prompt_txt, temperature=0.0)

    prompt = CRITIQUE_TEMPLATE.replace("{{question}}", question)\
                              .replace("{{answer}}" , prev_answer)

    raw = ask(prompt)
    try:                                   # first try to parse
        crit, _ = parse_crit_line(raw, prev_answer)
        return crit
    except Exception:
        # ------------- retry -------------
        retry_prompt = (
            "ONLY one line: CRITIQUE=<…> || ANSWER=<number>\n" + prompt
        )
        raw2 = ask(retry_prompt)
        try:
            crit, _ = parse_crit_line(raw2, prev_answer)
            return crit
        except Exception:
            return "parse_fail"

def refine_once(question, old_answer, critique):
    prompt = REFINE_PROMPT.replace("{{question}}", question)\
                          .replace("{{old}}", old_answer)\
                          .replace("{{critique}}", critique)
    
    return call_llm(prompt, temperature=0.2)


def extract_int(text):
    nums = re.findall(r"-?\d+", str(text).replace(",", ""))
    return nums[-1].lstrip("0") or "0" if nums else None


def load_gsm8k(n=30, seed=42):
    file = pathlib.Path("grade-school-math/grade_school_math/data/test.jsonl")
    items = [json.loads(line)
             for line in file.read_text().splitlines()
             if line.strip()]
    random.seed(seed)
    random.shuffle(items)
    return items[:n]

def evaluate():
    questions = load_gsm8k()
    base_hits = final_hits = 0
    loop_fixes = 0

    for idx, ex in enumerate(questions, 1):
        q = ex["question"]
        gold_int = extract_int(ex["answer"])

        print(f"\n=== Problem {idx}/{len(questions)} ===")
        print(q)

        a0       = solve_once(q)
        a0_int   = extract_int(a0)
        print("A0 ➜", a0_int, "| gold:", gold_int)
        if a0_int == gold_int:
            base_hits += 1

        if a0_int == gold_int:            
            final_int = a0_int
            crit = "baseline correct"
        else:
            crit          = critique_once(q, a0)
            if crit.strip().upper() == "OK" and a0_int != gold_int:
                crit = "No, the numerical answer is wrong. Check your arithmetic."
            a1            = refine_once(q, a0, crit)
            a1_int        = extract_int(a1)
            if a1_int == gold_int:
                loop_fixes += 1
                final_int = a1_int
            else:
                final_int = a0_int           

        if final_int == gold_int:          
            final_hits += 1

        print("Critique:", crit)
        print("Final ➜", final_int, "| improved?:",
              final_int == gold_int and a0_int != gold_int)

    total = len(questions)
    print(f"Baseline accuracy : {base_hits}/{total} = {base_hits/total:.1%}")
    print(f"Final   accuracy  : {final_hits}/{total} = {final_hits/total:.1%}")
    print(f"Problems fixed by loop: {loop_fixes}")


if __name__ == "__main__":
    evaluate()