"""LLM faithfulness eval: does RAG grounding actually reduce hallucination
and improve context-sensitivity versus the same model with no retrieval?

For each (mudra, context) test case we run the interpreter WITH retrieval
(what the app ships) and WITHOUT retrieval (mudra name passed raw, model
answers from parametric knowledge only), then score each transcript against
the ground-truth KB entries for that mudra:

  (a) grounded     — does the output substantively match at least one
                      canonical viniyoga meaning for this mudra?
  (b) hallucinated  — does the output make a specific interpretive claim
                      that does NOT match any canonical viniyoga meaning?
  (c) contextual    — when context is given, is the FIRST canonical
                      viniyoga mentioned the one whose KB context matches
                      (ground truth, since eval contexts are copied
                      verbatim from real KB context strings)?

Scoring method: lexical/word-overlap heuristics (see meaning_matches,
extract_claim_sentences below), not semantic entailment. This is a
directional check suited to comparing RAG vs no-RAG on the same rubric —
it is not a substitute for human judgment on individual transcripts, and
the JSON output includes full transcripts so failures can be spot-checked
by hand.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm.interpreter import ViniyogaInterpreter

KB_PATH = "data/kb/viniyoga.json"
OUT_PATH = Path("eval/llm_faithfulness.json")

TEST_MUDRAS = [
    "Pataka", "Alapadma", "Mushti", "Tripataka", "Suchi", "Mayura",
    "Anjali", "Kapotha", "Garuda", "Shanka",
]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "held", "near",
    "above", "toward", "into", "over", "used", "when", "while", "such",
    "like", "some", "than", "then", "also",
}


def word_overlap(a: str, b_words: set) -> float:
    words = {w for w in re.findall(r"[a-z]+", a.lower()) if len(w) > 3 and w not in STOPWORDS}
    if not words:
        return 0.0
    return len(words & b_words) / len(words)


def meaning_matches(output_lower: str, meaning: str) -> bool:
    m = meaning.lower()
    if m in output_lower:
        return True
    words = {w for w in re.findall(r"[a-z]+", m) if len(w) > 3 and w not in STOPWORDS}
    if not words:
        return False
    hits = sum(1 for w in words if w in output_lower)
    return (hits / len(words)) >= 0.6


def first_mentioned_meaning(output_lower: str, meanings: list[str]) -> str | None:
    best, best_idx = None, len(output_lower) + 1
    for m in meanings:
        ml = m.lower()
        idx = output_lower.find(ml)
        if idx == -1:
            words = [w for w in re.findall(r"[a-z]+", ml) if len(w) > 3 and w not in STOPWORDS]
            if not words:
                continue
            positions = [output_lower.find(w) for w in words if output_lower.find(w) != -1]
            if len(positions) / max(len(words), 1) < 0.6:
                continue
            idx = min(positions) if positions else -1
        if idx != -1 and idx < best_idx:
            best, best_idx = m, idx
    return best


CLAIM_SPAN_RE = re.compile(r"\*\*(.+?)\*\*")


def extract_claim_spans(output: str, mudra: str, context: str | None) -> list[str]:
    """The system prompt asks the model to 'reference specific viniyogas by
    name', which in practice it does consistently via **bold** text (e.g.
    '**firmness or steadfastness**', '**the number one**'). Bold spans are
    the model's actual namable claims. We deliberately don't also treat
    quoted text as a claim — the model quotes all sorts of things for
    emphasis (single words from the input context, etc.) that aren't
    assertions of a viniyoga meaning, which made the check too trigger-happy.
    Spans that just restate the mudra name or the given context aren't new
    claims either, so they're filtered out.
    """
    spans = []
    context_lower = (context or "").lower()
    for raw in CLAIM_SPAN_RE.findall(output):
        span = raw.strip()
        if len(span) <= 3:
            continue
        cleaned = re.sub(rf"^{re.escape(mudra.lower())}[\s,:–-]*", "", span.lower()).strip()
        if not cleaned or cleaned == mudra.lower():
            continue
        if context_lower and cleaned in context_lower:
            continue
        spans.append(cleaned)
    return spans


def span_matches_meaning(span: str, meaning: str) -> bool:
    """Looser, direction-flipped check for short bolded/quoted spans (often
    a 1-3 word paraphrase of a longer canonical meaning, e.g. '**longing**'
    for the KB meaning 'longing or wistful glance'). meaning_matches checks
    whether the *meaning's* words mostly appear in a block of text, which
    is right for the whole-output grounded check but too strict for a
    short span that can't reproduce every word of the meaning it's citing —
    so here we check the *span's* words against the meaning instead, with a
    lower overlap bar."""
    span_lower, meaning_lower = span.lower(), meaning.lower()
    if span_lower in meaning_lower or meaning_lower in span_lower:
        return True
    words = {w for w in re.findall(r"[a-z]+", span_lower) if len(w) > 3 and w not in STOPWORDS}
    if not words:
        return False
    hits = sum(1 for w in words if w in meaning_lower)
    return (hits / len(words)) >= 0.5


def is_hallucinated(output: str, mudra: str, context: str | None, meanings: list[str]) -> bool:
    spans = extract_claim_spans(output, mudra, context)
    if not spans:
        # No distinctly named claim to check against the KB — nothing to flag.
        return False
    return any(not any(span_matches_meaning(span, m) for m in meanings) for span in spans)


async def run_case(interp, kb, mudra, context, expected_meaning):
    entry = kb["mudras"][mudra]
    meanings = [v["meaning"] for v in entry["viniyogas"]]

    rag_chunks, no_rag_chunks = [], []
    async for c in interp.interpret_stream([mudra], context=context):
        rag_chunks.append(c)
    async for c in interp.interpret_stream_no_rag([mudra], context=context):
        no_rag_chunks.append(c)

    rag_text = "".join(rag_chunks)
    no_rag_text = "".join(no_rag_chunks)

    def score(text):
        text_lower = text.lower()
        grounded = any(meaning_matches(text_lower, m) for m in meanings)
        hallucinated = is_hallucinated(text, mudra, context, meanings)
        contextual = None
        if expected_meaning is not None:
            top = first_mentioned_meaning(text_lower, meanings)
            contextual = (top == expected_meaning)
        return {
            "text": text,
            "grounded": grounded,
            "hallucinated": hallucinated,
            "contextual_correct": contextual,
        }

    return {
        "mudra": mudra,
        "context": context,
        "expected_meaning": expected_meaning,
        "rag": score(rag_text),
        "no_rag": score(no_rag_text),
    }


def summarize(cases, arm):
    n = len(cases)
    grounded = sum(c[arm]["grounded"] for c in cases) / n
    hallucinated = sum(c[arm]["hallucinated"] for c in cases) / n
    ctx_cases = [c for c in cases if c[arm]["contextual_correct"] is not None]
    contextual = (
        sum(c[arm]["contextual_correct"] for c in ctx_cases) / len(ctx_cases)
        if ctx_cases else None
    )
    return {
        "grounded_rate": grounded,
        "hallucination_rate": hallucinated,
        "contextual_accuracy": contextual,
        "n_cases": n,
        "n_contextual_cases": len(ctx_cases),
    }


async def main():
    with open(KB_PATH) as f:
        kb = json.load(f)

    interp = ViniyogaInterpreter()
    cases = []
    for mudra in TEST_MUDRAS:
        entry = kb["mudras"][mudra]
        first_viniyoga = entry["viniyogas"][0]
        # Context case: exact KB context string, ground truth = its meaning.
        cases.append((mudra, first_viniyoga["context"], first_viniyoga["meaning"]))
        # No-context case: ambiguous, no single ground-truth meaning.
        cases.append((mudra, None, None))

    results = []
    for i, (mudra, context, expected) in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {mudra} | context={context!r}")
        result = await run_case(interp, kb, mudra, context, expected)
        results.append(result)
        await asyncio.sleep(0.3)

    output = {
        "methodology": (
            "Lexical word-overlap heuristics against canonical KB viniyoga "
            "meanings, not semantic entailment. Directional signal for "
            "comparing RAG vs no-RAG under an identical rubric; inspect "
            "'cases' transcripts for ground truth on individual examples."
        ),
        "model": interp.__class__.__module__,
        "n_cases": len(results),
        "summary": {
            "rag": summarize(results, "rag"),
            "no_rag": summarize(results, "no_rag"),
        },
        "cases": results,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Summary ===")
    print(f"{'metric':<24}{'RAG':>10}{'No-RAG':>10}")
    for key in ("grounded_rate", "hallucination_rate", "contextual_accuracy"):
        rag_v = output["summary"]["rag"][key]
        norag_v = output["summary"]["no_rag"][key]
        fmt = lambda v: f"{v:.2%}" if v is not None else "n/a"
        print(f"{key:<24}{fmt(rag_v):>10}{fmt(norag_v):>10}")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
