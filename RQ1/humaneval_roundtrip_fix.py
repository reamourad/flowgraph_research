"""
HumanEval Round-Trip Pipeline
==============================
Code -> Mermaid diagram -> Regenerated code

Install:
    pip install groq datasets radon nltk code-bleu

Usage:
    export GROQ_API_KEY=gsk_...
    python humaneval_roundtrip_fix.py --n 20 --approach all --model llama3-70b
    python humaneval_roundtrip_fix.py --n 5  --approach mermaid --save-diagrams
"""

import os, re, ast, json, time, signal, argparse, traceback
import typing, math, collections, itertools, functools, string
import re as _re
from dataclasses import dataclass, asdict, field
from typing import Optional

from groq import Groq
from datasets import load_dataset
import radon.complexity as radon_cc
import radon.metrics   as radon_metrics
import radon.raw       as radon_raw
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

MODELS: dict[str, str] = {
    "llama3-70b":    "llama-3.3-70b-versatile",
    "llama3-8b":     "llama-3.1-8b-instant",
    "qwen3-32b":     "qwen/qwen3-32b",
    "llama4-scout":  "meta-llama/llama-4-scout-17b-16e-instruct",
    "gpt-oss-120b":  "openai/gpt-oss-120b",
    "gpt-oss-20b":   "openai/gpt-oss-20b",
}

DEFAULT_MODEL = "gpt-oss-120b"
MAX_RETRIES   = 3
EXEC_TIMEOUT  = 10


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    passed:              bool  = False
    syntax_ok:           bool  = False
    token_bleu:          float = 0.0
    code_bleu:           float = 0.0
    ast_edit_dist:       int   = -1
    cc_delta:            float = 0.0
    halstead_vol_delta:  float = 0.0
    halstead_diff_delta: float = 0.0
    loc_delta:           int   = 0
    mermaid_valid:       bool  = False
    mermaid_nodes:       int   = 0
    token_usage:         dict  = field(default_factory=dict)
    error:               Optional[str] = None


@dataclass
class Result:
    task_id:         str
    model:           str
    approach:        str
    generated_code:  Optional[str]
    mermaid_diagram: Optional[str]
    metrics:         Metrics = field(default_factory=Metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_humaneval(n: int = 20) -> list[dict]:
    print(f"  Downloading HumanEval...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [dict(ds[i]) for i in range(min(n, len(ds)))]


# ─────────────────────────────────────────────────────────────────────────────
# HumanEval anatomy helper
# ─────────────────────────────────────────────────────────────────────────────
# HumanEval fields:
#   prompt            -- "def foo(x):\n    '''docstring'''\n"  (ends with \n)
#   canonical_solution-- "    return x + 1\n"  (indented body, NO def line)
#   test              -- "def check(candidate):\n    assert ..."
#   entry_point       -- "foo"

def full_reference(instance: dict) -> str:
    """Return the complete canonical function (def line + body)."""
    return instance["prompt"] + instance["canonical_solution"]


# ─────────────────────────────────────────────────────────────────────────────
# Strip <think> blocks from reasoning models (Qwen3, DeepSeek-R1)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
    return cleaned if cleaned else text


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = "You are an expert Python developer.",
    temperature: float = 0.1,
) -> tuple[str, dict]:
    model_id = MODELS.get(model, model)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
            )
            text  = _strip_thinking(resp.choices[0].message.content or "")
            usage = {
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens":      resp.usage.total_tokens,
            }
            return text, usage
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"LLM failed: {exc}") from exc
            wait = 2 ** attempt
            print(f"      Retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
            time.sleep(wait)


def accum(total: dict, new: dict) -> dict:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[k] = total.get(k, 0) + new.get(k, 0)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 -- Code to Mermaid
# ─────────────────────────────────────────────────────────────────────────────

_S1_SYSTEM = """\
You are an expert at converting Python functions into Mermaid flowcharts.
Return ONLY a ```mermaid block -- no prose, no explanation.
Use flowchart TD syntax."""

_S1_PROMPT = """\
Convert this Python function into a Mermaid flowchart (flowchart TD).
Capture: inputs, all branches, loops, and return values.
Focus on logic flow, not variable names.

```python
{code}
```

Return only the ```mermaid block."""


def code_to_mermaid(full_code: str, model: str) -> tuple[str, dict]:
    text, usage = call_llm(_S1_PROMPT.format(code=full_code), model=model, system=_S1_SYSTEM)
    return _extract_mermaid(text), usage


def _extract_mermaid(text: str) -> str:
    m = re.search(r"```mermaid\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if stripped.startswith(("flowchart", "graph ", "sequenceDiagram")):
        return stripped
    return stripped


def check_mermaid(diagram: str) -> tuple[bool, int]:
    if not diagram or len(diagram.strip()) < 10:
        return False, 0

    valid = diagram.strip().startswith(
        ("flowchart", "graph ", "sequenceDiagram", "stateDiagram")
    )
    has_arrows = bool(re.search(r"-->|->|===|--", diagram))

    node_count = len(re.findall(r"^\s*\w[\w\s]*\s*[\[\(\{]", diagram, re.MULTILINE))

    # Fall back to counting --> arrows when no labeled nodes are found
    if node_count == 0:
        node_count = len(re.findall(r"-->", diagram))

    return (valid and has_arrows), node_count


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 -- Mermaid + signature to Code
# ─────────────────────────────────────────────────────────────────────────────

_S2_SYSTEM = """\
You are an expert Python developer.
Implement the Python function exactly as specified.
Return ONLY the complete function, starting with `def`.
No explanation, no markdown fences, no imports unless required."""

_S2_PROMPT = """\
Implement this Python function.

Signature and docstring (DO NOT change the function name or signature):
```python
{prompt}
```

Logic to implement (Mermaid flowchart):
```mermaid
{diagram}
```

Return the complete function starting with `def {entry_point}(`."""


def mermaid_to_code(instance: dict, diagram: str, model: str) -> tuple[str, dict]:
    prompt = _S2_PROMPT.format(
        prompt=instance["prompt"],
        diagram=diagram,
        entry_point=instance["entry_point"],
    )
    text, usage = call_llm(prompt, model=model, system=_S2_SYSTEM)
    return _extract_code(text, instance), usage


# ─────────────────────────────────────────────────────────────────────────────
# Baseline -- Docstring to Code (no diagram)
# ─────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM = """\
You are an expert Python developer.
Implement the function as described.
Return ONLY the complete function starting with `def`.
No explanation, no markdown fences."""

_BASE_PROMPT = """\
Implement this Python function:

```python
{prompt}
```

Return the complete function starting with `def {entry_point}(`."""


def direct_code(instance: dict, model: str) -> tuple[str, dict]:
    prompt = _BASE_PROMPT.format(
        prompt=instance["prompt"],
        entry_point=instance["entry_point"],
    )
    text, usage = call_llm(prompt, model=model, system=_BASE_SYSTEM)
    return _extract_code(text, instance), usage


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_code(text: str, instance: dict) -> str:
    """
    Extract the generated function from LLM output.

    Priority order:
      1. Fenced ```python block
      2. Any fenced ``` block containing def
      3. First occurrence of `def <entry_point>` in raw text
      4. First occurrence of any `def ` line in raw text
      5. Raw text (last resort)
    """
    entry = instance["entry_point"]

    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return _clean_code(m.group(1).strip(), instance)

    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m and "def " in m.group(1):
        return _clean_code(m.group(1).strip(), instance)

    m = re.search(rf"(def {re.escape(entry)}\s*\(.*)", text, re.DOTALL)
    if m:
        return _clean_code(m.group(1).strip(), instance)

    m = re.search(r"(def \w+\s*\(.*)", text, re.DOTALL)
    if m:
        return _clean_code(m.group(1).strip(), instance)

    return text.strip()


def _clean_code(code: str, instance: dict) -> str:
    """
    If the LLM returned only the function body (no def line), prepend the
    signature from instance["prompt"]. Also truncates trailing reasoning text
    that thinking models sometimes append after the code.
    """
    stripped = code.strip()
    lines = stripped.split("\n")
    func_end = len(lines)
    in_func = False
    for i, line in enumerate(lines):
        if line.startswith("def "):
            in_func = True
            continue
        if in_func and line and not line[0].isspace():
            if not re.match(r"^(def |class |import |from |@)", line):
                func_end = i
                break
    stripped = "\n".join(lines[:func_end]).strip()

    if stripped.startswith("def "):
        return stripped
    if stripped and (stripped[0] in (" ", "\t") or stripped.startswith("return")):
        return instance["prompt"] + stripped
    return stripped


# ─────────────────────────────────────────────────────────────────────────────
# Test runner with pre-seeded namespace
# ─────────────────────────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("test timed out")


# Pre-seed with typing + stdlib so generated code using List, Dict, Optional,
# math, etc. never raises NameError for missing imports.
_BASE_NAMESPACE: dict = {
    **{k: v for k, v in vars(typing).items() if not k.startswith("_")},
    "math":        math,
    "collections": collections,
    "itertools":   itertools,
    "functools":   functools,
    "string":      string,
    "re":          _re,
}


def run_tests(gen_code: str, instance: dict) -> tuple[bool, str]:
    """
    Execute gen_code + HumanEval test suite.

    If gen_code is body-only (no def line), prepends the prompt so the
    function is always defined with the correct name and signature.

    WARNING: executes LLM-generated code — use a VM/container for real security.
    """
    entry  = instance["entry_point"]
    prompt = instance["prompt"]
    test   = instance["test"]

    if gen_code.strip().startswith("def "):
        exec_source = gen_code
    else:
        exec_source = prompt + gen_code

    namespace: dict = dict(_BASE_NAMESPACE)

    try:
        exec(compile(exec_source, "<generated>", "exec"), namespace)   # noqa: S102
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Load error: {e}"

    if entry not in namespace:
        defined = [k for k in namespace if not k.startswith("_")]
        return False, f"Entry '{entry}' not found. Defined: {defined}"

    full_test = f"{test}\ncheck({entry})"
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EXEC_TIMEOUT)
        exec(compile(full_test, "<tests>", "exec"), namespace)          # noqa: S102
        signal.alarm(0)
        return True, ""
    except TimeoutError:
        return False, "TimeoutError"
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Similarity metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_bleu(ref: str, hyp: str) -> float:
    ref_t = ref.split()
    hyp_t = hyp.split()
    if not hyp_t or not ref_t:
        return 0.0
    return sentence_bleu([ref_t], hyp_t, smoothing_function=SmoothingFunction().method1)


def compute_code_bleu(ref: str, hyp: str) -> float:
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu([ref], [hyp], lang="python")
        return float(result["codebleu"])
    except Exception:
        return compute_token_bleu(ref, hyp)


def ast_edit_distance(ref: str, hyp: str) -> int:
    from collections import Counter
    try:
        ref_nodes = [type(n).__name__ for n in ast.walk(ast.parse(ref))]
        hyp_nodes = [type(n).__name__ for n in ast.walk(ast.parse(hyp))]
    except SyntaxError:
        return -1
    rc, hc = Counter(ref_nodes), Counter(hyp_nodes)
    return sum(abs(rc[k] - hc[k]) for k in set(rc) | set(hc))


# ─────────────────────────────────────────────────────────────────────────────
# Code quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_cc(code: str) -> float:
    try:
        r = radon_cc.cc_visit(code)
        return sum(x.complexity for x in r) / len(r) if r else 1.0
    except Exception:
        return -1.0


def get_halstead(code: str) -> dict:
    try:
        h = radon_metrics.h_visit(code)
        if not h:
            return {}
        first = h[0] if isinstance(h, list) else h
        return {"volume": first.volume, "difficulty": first.difficulty}
    except Exception:
        return {}


def get_loc(code: str) -> int:
    try:
        return radon_raw.analyze(code).sloc
    except Exception:
        return len([l for l in code.splitlines() if l.strip()])


# ─────────────────────────────────────────────────────────────────────────────
# Fill metrics
# ─────────────────────────────────────────────────────────────────────────────

def fill_metrics(m: Metrics, gen_code: Optional[str], instance: dict) -> None:
    ref_code = full_reference(instance)

    if not gen_code or not gen_code.strip():
        m.error = "empty generated code"
        return

    m.syntax_ok = is_syntax_valid(gen_code)
    if not m.syntax_ok:
        m.error = "syntax error"
        return

    m.passed, err = run_tests(gen_code, instance)
    if err and not m.passed:
        m.error = err

    m.token_bleu    = round(compute_token_bleu(ref_code, gen_code), 4)
    m.code_bleu     = round(compute_code_bleu(ref_code, gen_code), 4)
    m.ast_edit_dist = ast_edit_distance(ref_code, gen_code)

    ref_cc, gen_cc = get_cc(ref_code), get_cc(gen_code)
    if ref_cc > 0 and gen_cc > 0:
        m.cc_delta = round(gen_cc - ref_cc, 2)

    ref_h, gen_h = get_halstead(ref_code), get_halstead(gen_code)
    if ref_h and gen_h:
        m.halstead_vol_delta  = round(gen_h.get("volume", 0) - ref_h.get("volume", 0), 2)
        m.halstead_diff_delta = round(gen_h.get("difficulty", 0) - ref_h.get("difficulty", 0), 2)

    m.loc_delta = get_loc(gen_code) - get_loc(ref_code)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

def run_mermaid(instance: dict, model: str) -> Result:
    tid     = instance["task_id"]
    metrics = Metrics()
    tokens: dict = {}
    diagram  = None
    gen_code = None

    try:
        print("      [1/2] Code -> Mermaid...")
        diagram, u1 = code_to_mermaid(full_reference(instance), model)
        accum(tokens, u1)
        metrics.mermaid_valid, metrics.mermaid_nodes = check_mermaid(diagram)

        print("      [2/2] Mermaid -> Code...")
        gen_code, u2 = mermaid_to_code(instance, diagram, model)
        accum(tokens, u2)

        print("      [eval] Tests + metrics...")
        fill_metrics(metrics, gen_code, instance)

    except Exception:
        metrics.error = traceback.format_exc(limit=3)

    metrics.token_usage = tokens
    return Result(tid, model, "mermaid", gen_code, diagram, metrics)


def run_direct(instance: dict, model: str) -> Result:
    tid      = instance["task_id"]
    metrics  = Metrics()
    gen_code = None

    try:
        print("      [direct] Generating code...")
        gen_code, usage = direct_code(instance, model)
        metrics.token_usage = usage
        print("      [eval] Tests + metrics...")
        fill_metrics(metrics, gen_code, instance)
    except Exception:
        metrics.error = traceback.format_exc(limit=3)

    return Result(tid, model, "direct", gen_code, None, metrics)


def run_identity(instance: dict) -> Result:
    """Return canonical solution as-is -- ceiling for all metrics."""
    tid      = instance["task_id"]
    metrics  = Metrics()
    gen_code = full_reference(instance)
    fill_metrics(metrics, gen_code, instance)
    return Result(tid, "identity", "identity", gen_code, None, metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Pass@k
# ─────────────────────────────────────────────────────────────────────────────

def pass_at_k(results: list[dict], k: int = 1) -> dict[str, float]:
    import math
    from collections import defaultdict
    by_ap: dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_ap[r["approach"]][r["task_id"]].append(r["metrics"]["passed"])
    scores: dict[str, float] = {}
    for ap, tasks in by_ap.items():
        rates = []
        for passes in tasks.values():
            n, c = len(passes), sum(passes)
            if n < k:
                rates.append(float(c > 0))
            elif n - c < k:
                rates.append(1.0)
            else:
                rates.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
        scores[ap] = round(sum(rates) / len(rates), 4)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    approaches = sorted(set(r["approach"] for r in results))
    p1 = pass_at_k(results, k=1)

    print(f"\n{'='*74}")
    print(f"  {'APPROACH':<12} {'PASS@1':>7} {'SYNTAX':>7} {'CODEBLEU':>10} "
          f"{'AST delta':>9} {'CC delta':>8} {'LOC delta':>9} {'TOKENS':>8}")
    print(f"{'-'*74}")

    for ap in approaches:
        sub    = [r for r in results if r["approach"] == ap]
        n      = len(sub)
        syntax = sum(1 for r in sub if r["metrics"]["syntax_ok"]) / n
        cbleu  = sum(r["metrics"]["code_bleu"] for r in sub) / n
        ast_d  = [r["metrics"]["ast_edit_dist"] for r in sub if r["metrics"]["ast_edit_dist"] >= 0]
        ast_avg = sum(ast_d) / len(ast_d) if ast_d else float("nan")
        cc_d   = sum(r["metrics"]["cc_delta"] for r in sub) / n
        loc_d  = sum(r["metrics"]["loc_delta"] for r in sub) / n
        toks   = sum(r["metrics"]["token_usage"].get("total_tokens", 0) for r in sub) / n

        print(
            f"  {ap:<12} {p1.get(ap,0):>7.3f} {syntax:>7.3f} {cbleu:>10.4f} "
            f"{ast_avg:>9.1f} {cc_d:>+8.2f} {loc_d:>+9.1f} {toks:>8.0f}"
        )

    print(f"{'-'*74}")
    print("  CC delta < 0 = simpler   |   AST delta = 0 = identical structure")
    print(f"{'='*74}\n")

    mmx = [r for r in results if r["approach"] == "mermaid"]
    if mmx:
        v = sum(1 for r in mmx if r["metrics"]["mermaid_valid"])
        avg_n = sum(r["metrics"]["mermaid_nodes"] for r in mmx) / len(mmx)
        print(f"  Mermaid validity: {v}/{len(mmx)}  ({100*v/len(mmx):.0f}%)   "
              f"avg nodes: {avg_n:.1f}\n")

    print(f"{'-'*74}")
    print(f"  {'TASK':<22} {'APPROACH':<10} {'PASS':<5} {'BLEU':>8} {'CC delta':>9}  ERROR")
    print(f"{'-'*74}")
    for r in results:
        m   = r["metrics"]
        err = (m.get("error") or "")[:40]
        print(
            f"  {r['task_id']:<22} {r['approach']:<10} "
            f"{'YES' if m['passed'] else 'no':<5} "
            f"{m['code_bleu']:>8.4f} {m['cc_delta']:>+9.2f}  {err}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model",    default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--n",        type=int, default=10)
    parser.add_argument("--approach", default="all",
                        choices=["mermaid", "direct", "identity", "all"])
    parser.add_argument("--output",   default="results_multimodel/results_gpt-oss-120b.jsonl")
    parser.add_argument("--save-diagrams", action="store_true")
    args = parser.parse_args()

    approaches = (
        ["mermaid", "direct", "identity"] if args.approach == "all" else [args.approach]
    )

    print(f"\nHumanEval Round-Trip Pipeline")
    print(f"  model      : {args.model} ({MODELS.get(args.model, args.model)})")
    print(f"  approaches : {', '.join(approaches)}")
    print(f"  n tasks    : {args.n}")
    print(f"  output     : {args.output}\n")

    instances   = load_humaneval(n=args.n)
    all_results: list[dict] = []

    for idx, inst in enumerate(instances):
        tid = inst["task_id"]
        print(f"[{idx+1}/{args.n}] {tid}")

        for ap in approaches:
            if ap == "mermaid":
                result = run_mermaid(inst, model=args.model)
            elif ap == "direct":
                result = run_direct(inst, model=args.model)
            else:
                result = run_identity(inst)

            if args.save_diagrams and result.mermaid_diagram:
                safe = tid.replace("/", "_")
                with open(f"{safe}_diagram.md", "w") as f:
                    f.write(f"# {tid}\n\n```mermaid\n{result.mermaid_diagram}\n```\n")
                    f.write(f"\nValid: {result.metrics.mermaid_valid}  "
                            f"Nodes: {result.metrics.mermaid_nodes}\n")

            m = result.metrics
            status = "PASS" if m.passed else ("SYNX" if m.syntax_ok else "FAIL")
            diag   = f"| diag:{m.mermaid_nodes}nodes" if ap == "mermaid" else ""
            print(f"      {ap:<10} [{status}]  bleu={m.code_bleu:.3f}  "
                  f"cc_delta={m.cc_delta:+.1f}  loc_delta={m.loc_delta:+d}  {diag}")

            d = asdict(result)
            all_results.append(d)
            with open(args.output, "a") as f:
                f.write(json.dumps(d) + "\n")

    print_report(all_results)


if __name__ == "__main__":
    main()
