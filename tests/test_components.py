"""Unit tests for the deterministic CPU components (tokenizer, normalizer, pruner, cache, router, verifier)."""
import json

from src.cache import SemanticCache, context_hash
from src.normalize import extract_entities, extract_numbers, is_arabic, normalize_text
from src.pruner import drop_boilerplate, prune, relaxed
from src.router import WEIGHTS, dispersion, route
from src.tokenizer import count_tokens
from src.verifier import answers_agree, check_degenerate, check_grounding, check_language, check_schema, verify


# ---------------------------------------------------------------- tokenizer / normalization
def test_tiktoken_counts_both_languages():
    assert count_tokens("hello world") == 2
    assert count_tokens("مرحبا بالعالم") > 0


def test_arabic_normalization_and_clitics():
    assert normalize_text("الإجازة") == normalize_text("الاجازه")
    assert normalize_text("علی") == normalize_text("على") or normalize_text("علي") == normalize_text("على")
    assert normalize_text("بالمادة") == normalize_text("المادة")  # proclitic stripped
    assert normalize_text("مُدَّة") == normalize_text("مده")  # tashkeel stripped
    assert normalize_text("في") == "في"  # short tokens untouched


def test_numbers_and_entities():
    assert extract_numbers("٢٤٠٬٠٠٠ ريال و٢٥ ريالاً ٠٫٥٪") == [240000.0, 25.0, 0.5]
    assert extract_numbers("SAR 12,400 due 2025-09-15 for PO-5520") == [12400.0]
    e = extract_entities("Amount of #INV-8841 at 15%")
    assert "id:INV-8841" in e and "pct:15" in e
    assert extract_entities("invoice INV-8841") != extract_entities("invoice INV-8842")
    assert extract_entities("١٥٪") == extract_entities("15%")
    assert is_arabic("ما قيمة العقد PO-7310؟") and not is_arabic("What is PO-7310?")


# ---------------------------------------------------------------- pruner
def test_boilerplate_dropped():
    chunks = ["Confidential. All rights reserved.", "Fee is SAR 10.", "Footer X", "Footer X"]
    keep, n = drop_boilerplate(chunks)
    assert keep == [1] and n == 3


def test_pruner_keeps_evidence_and_saves_tokens(held_out):
    for tid in ("std_en_01", "std_ar_01", "mh_en_01", "aro_ar_03"):
        t = held_out[tid]
        p = prune(t.query, t.context)
        ctx = "\n".join(p.kept)
        assert all(e in ctx for e in t.evidence), tid
        assert p.tokens_after < p.tokens_before
        assert len(p.kept) >= 2
        assert p.kept_indices == sorted(p.kept_indices)  # original order


def test_relaxed_is_superset_with_top4(held_out):
    t = held_out["llo_en_01"]
    p = prune(t.query, t.context)
    r = relaxed(t.query, t.context, p, k=4)
    assert set(p.kept_indices) <= set(r.kept_indices) and len(r.kept) >= 4


# ---------------------------------------------------------------- cache
def test_cache_hit_paraphrase_and_block_trap(held_out):
    c = SemanticCache(threshold=0.88, entity_guard=True)
    base = held_out["std_en_04"]
    h = context_hash(base.context[:2])
    c.insert(base.task_id, base.query, h, "12,400")
    assert c.lookup(held_out["cache_en_hit"].query, h).event == "hit"
    assert c.lookup(held_out["cache_en_trap"].query, h).event == "blocked_trap"
    assert c.lookup(held_out["cache_en_hit"].query, "other-context").event == "miss"


def test_cache_without_guard_returns_false_hit(held_out):
    c = SemanticCache(threshold=0.88, entity_guard=False)
    base = held_out["mh_ar_01"]
    c.insert(base.task_id, base.query, "h", "7200")
    look = c.lookup(held_out["cache_ar_trap"].query, "h")
    assert look.event == "hit" and look.entry.answer == "7200"


def test_cache_no_hits_between_distinct_tasks(held_out):
    c = SemanticCache(entity_guard=False)
    for tid, t in held_out.items():
        if t.category == "cache_near_duplicates":
            continue
        assert c.lookup(t.query, "same").event == "miss", tid
        c.insert(tid, t.query, "same", "x")


# ---------------------------------------------------------------- router
def test_router_signals_and_bounds(held_out):
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    easy = held_out["std_en_07"]
    d = route(easy.query, prune(easy.query, easy.context).scores, easy.schema, 0.55)
    assert d.tier == 1 and 0 <= d.complexity < 0.55
    hard = held_out["mh_en_03"]
    d = route(hard.query, prune(hard.query, hard.context).scores, hard.schema, 0.55)
    assert d.signals["multi_hop_markers"] == 1 and d.signals["quantitative_reasoning"] == 1 and d.tier == 2
    js = held_out["mh_en_02"]
    assert route(js.query, [0.5, 0.5], js.schema, 0.55).signals["structural_strictness"] == 1.0
    assert route("قارن واحسب الفرق بين البندين", [0.5], None, 0.55).signals["multi_hop_markers"] == 1


def test_lower_tau_routes_more_to_tier2():
    d_hi = route("Calculate the fee", [0.3, 0.2], None, 0.55)
    d_lo = route("Calculate the fee", [0.3, 0.2], None, 0.25)
    assert d_hi.tier == 1 and d_lo.tier == 2


def test_dispersion():
    assert dispersion([1.0, 0, 0, 0]) == 0.0
    assert abs(dispersion([0.5, 0.5, 0.5, 0.5]) - 1.0) < 1e-9


# ---------------------------------------------------------------- verifier
def test_verifier_degenerate():
    assert check_degenerate("")
    assert check_degenerate("I don't know based on the context.")
    assert check_degenerate("غير مذكور في السياق")
    assert check_degenerate("the the the the the the the the the the the the the the")
    assert check_degenerate("The context does not provide information on how fast the vendor must respond.")
    assert check_degenerate("The policy doesn't specify any details about this.")
    assert check_degenerate("There is no specific information about refunds.")
    assert check_degenerate("السياق لا يذكر مدة الضمان")
    assert not check_degenerate("The monthly fee is SAR 48,000.")
    assert not check_degenerate("No, restricted data may not be stored on removable media.")
    assert not check_degenerate("Priority-1 incidents are acknowledged within 15 minutes; this does not include weekends.")


def test_verifier_schema():
    schema = {"invoice_id": "string", "amount_payable": "number"}
    assert check_schema('{"invoice_id": "INV-9120", "amount_payable": 4508}', schema)
    assert check_schema('```json\n{"invoice_id": "INV-9120", "amount_payable": "4,508 SAR"}\n```', schema)
    assert not check_schema('{"invoice_id": "INV-9120"}', schema)
    assert not check_schema("amount is 4508", schema)


def test_verifier_grounding_allows_arithmetic():
    ctx = "Monthly fee SAR 48,000. Credit of 25% of the monthly fee."
    assert check_grounding("The credit is SAR 12,000.", "Calculate the credit.", ctx)  # 48000 * 25%
    assert check_grounding("Fee is 48,000", "fee?", ctx)
    assert not check_grounding("The credit is SAR 987,654.", "Calculate the credit.", ctx)


def test_verifier_lookup_requires_verbatim_numbers():
    ctx = "Contract value SAR 240,000. Payment within 45 days; delivery within 14 days."
    assert check_grounding("The value is SAR 240,000.", "What is the contract value?", ctx)
    assert not check_grounding("The value is SAR 24,000.", "What is the contract value?", ctx)  # 240000/10
    assert not check_grounding("31 days", "How many days to pay?", ctx)  # 45-14: derivable, not stated
    assert check_grounding("31 days", "Calculate the difference between the two periods.", ctx)
    ar_ctx = "غرامة قدرها ٠٫٥٪ من قيمة ٢٤٠٬٠٠٠ ريال"
    assert check_grounding("الغرامة ٧٬٢٠٠ ريال", "احسب الغرامة إذا تأخر ٦ أيام", ar_ctx)  # 240000*0.5%*6


def test_verifier_language():
    assert check_language("المبلغ هو ٨٠٠ ريال", "كم بدل النقل؟", None)
    assert not check_language("The transport allowance is 800 SAR per month.", "كم بدل النقل؟", None)
    assert check_language("800", "كم بدل النقل؟", None)
    assert check_language("anything", "English question?", None)


def test_verifier_end_to_end_is_fast():
    r = verify("What is the fee?", "Fee is SAR 48,000 per month, 30 days.", "SAR 48,000")
    assert r.passed and r.elapsed_ms < 50
    r = verify("What is the fee?", "Fee is SAR 48,000.", "I cannot determine. 777,777")
    assert not r.passed and set(r.failed_checks) == {"degenerate_or_refusal", "numeric_grounding"}


def test_verifier_has_no_ground_truth_access():
    import src.verifier as v
    src = open(v.__file__, encoding="utf-8").read()
    assert "ground_truth" not in src and "expected_answer" not in src


def test_answers_agree():
    assert answers_agree("The fee is SAR 48,000.", "Monthly fee: 48,000 SAR")
    assert not answers_agree("The value is ٢٤٠٠٠ ريال", "القيمة ٢٤٠٬٠٠٠ ريال")
    assert answers_agree("Economy class.", "You should book economy class for this flight.")
    assert not answers_agree("Business class.", "Economy class is required for flights under 6 hours.")
