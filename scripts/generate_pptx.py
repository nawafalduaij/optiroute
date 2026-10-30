"""Build deliverables/OptiRoute_Presentation.pptx (9 slides) from measured numbers in results/metrics.json.

  .venv/bin/python scripts/generate_pptx.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
M = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
assert M["mode"] == "replay", "deck must be built from recorded (replay) results, not sim"

BG, PANEL, PANEL2, LINE = "07110E", "0D1A16", "112520", "1D3A32"
TEXT, MUTED, EM, EM2, AMBER, RED = "E6F2EE", "8AA69D", "10B981", "34D399", "F59E0B", "F87171"
FONT = "Segoe UI"

C, L, V = M["cumulative"], M["loo"], M["verifier"]
FULL, T2, T1 = C["5_OptiRoute_Full"], C["1_Always_Tier2_Raw"], C["2_Always_Tier1_Raw"]
VP = next(v for v in V if v["variant"] == "pruned")
WAVE = M["wave"]["adaptive"]["trace"]
NOISE = M["noise_scaling"]


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def rgb(h):
    return RGBColor.from_string(h)


COMPUTE_CUT = 1 - FULL["compute_units"] / T2["compute_units"]
TOKEN_CUT = 1 - FULL["total_tokens"] / T2["total_tokens"]
CPS_X = T2["cps"] / FULL["cps"]
CPU_MS = sum(FULL["overhead_ms"].values())
N_TESTS = sum(1 for line in subprocess.run(
    [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")], capture_output=True, text=True,
    cwd=ROOT, env={**os.environ, "OPTIROUTE_MODE": "sim"}).stdout.splitlines() if "::" in line)
MIN_RATE = min(x["pass_rate"] for x in WAVE if x["pass_rate"] is not None)
MAX_P95 = max(x["p95_ms"] for x in WAVE)
AUD = M["wave"]["adaptive_audit"]
AUD_ACTS = [x for x in AUD["trace"] if x["action"] != "hold"]
AUD_TIGHTEN = next(x["step"] for x in AUD_ACTS if x["action"] == "tighten")
AUD_RECOVER = [x["step"] for x in AUD_ACTS if x["action"] == "recover"]
AUD_TAU_MIN = min(x["tau"] for x in AUD["trace"])
AUD_COST = AUD["summary"]["compute_units"] / M["wave"]["adaptive"]["summary"]["compute_units"] - 1
XM = M.get("cross_model") or []
GITHUB = "github.com/nawafalduaij/optiroute"
SHOTS = ROOT / "deliverables" / "screenshots"
TOTAL = 10

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- drawing helpers
def text(slide, x, y, w, h, content, size=14, color=TEXT, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         font=FONT):
    """content: str or list of (str | (str, dict)) paragraphs; dict may set size/color/bold/bullet."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.02)
    paras = content if isinstance(content, list) else [content]
    for i, p in enumerate(paras):
        s, opt = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(opt.get("space", 4))
        run = para.add_run()
        run.text = ("•  " if opt.get("bullet") else "") + s
        run.font.size = Pt(opt.get("size", size))
        run.font.bold = opt.get("bold", bold)
        run.font.color.rgb = rgb(opt.get("color", color))
        run.font.name = font
    return tb


def rect(slide, x, y, w, h, fill=PANEL, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    return s


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle"})
    ln.append(tail)
    return c


def new_slide(num, kicker, title):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    rect(s, 0, 0, 13.333, 0.08, fill=EM, line=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, 0.6, 0.35, 9, 0.35, kicker.upper(), size=12, color=EM2, bold=True)
    text(s, 0.6, 0.65, 12.2, 0.8, title, size=28, bold=True)
    text(s, 11.6, 7.0, 1.2, 0.3, f"{num} / {TOTAL}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    text(s, 0.6, 7.0, 8, 0.3, "OptiRoute · HUMAIN Hackathon · Challenge 2: Deliver AI with Less Compute",
         size=10, color=MUTED)
    return s


def kpi(slide, x, y, w, value, label, color=EM2, h=1.25):
    rect(slide, x, y, w, h)
    text(slide, x + 0.2, y + 0.12, w - 0.4, 0.6, value, size=28, bold=True, color=color)
    text(slide, x + 0.2, y + 0.72, w - 0.4, h - 0.75, label, size=11, color=MUTED)


def table(slide, x, y, w, rows, col_w, highlight=(), size=10.5, row_h=0.32):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for child in list(tblPr):  # drop default theme style banding
        tblPr.remove(child)
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            hl = i in highlight
            cell.fill.fore_color.rgb = rgb(PANEL2 if i == 0 else ("0B2A20" if hl else PANEL))
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = FONT
            r.font.bold = i == 0 or hl
            r.font.color.rgb = rgb(MUTED if i == 0 else (EM2 if hl else TEXT))
    return tbl


def metric_rows(configs: dict, names: list[str]) -> list[list[str]]:
    rows = [["Configuration", "Accuracy", "Tokens", "Compute kTok·B", "Δ compute", "CPS", "P50 ms", "Tier 0/1/2",
             "False cache hits"]]
    for n in names:
        m = configs[n]
        rows.append([n.replace("_", " "), f"{pct(m['accuracy'])} ({m['correct']}/{m['n']})", f"{m['total_tokens']:,}",
                     f"{m['compute_units']:.1f}", f"{m['compute_units'] / T2['compute_units'] - 1:+.1%}",
                     f"{m['cps']:.2f}", f"{m['p50_ms']:.0f}", f"{m['tier0']}/{m['tier1']}/{m['tier2']}",
                     f"{m['false_cache_hits']}/{m['n_traps']}"])
    return rows


# ================================================================ 1. Title
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = rgb(BG)
rect(s, 0, 0, 0.18, 7.5, fill=EM, line=None, shape=MSO_SHAPE.RECTANGLE)
text(s, 0.9, 0.9, 11, 0.4, "HUMAIN HACKATHON · CHALLENGE 2: DELIVER AI WITH LESS COMPUTE", size=13, color=EM2, bold=True)
text(s, 0.9, 1.45, 11, 1.2, "OptiRoute", size=60, bold=True)
text(s, 0.9, 2.55, 11.5, 1.0, "A bilingual (Arabic / English) compute cascade that sends each task to the cheapest "
     "tier able to answer it correctly, with every number measured on recorded real model runs.", size=20, color=MUTED)
kpi(s, 0.9, 4.0, 2.75, f"−{pct(COMPUTE_CUT)}", "compute vs always-large-model (kTok × active params)")
kpi(s, 3.85, 4.0, 2.75, f"{CPS_X:.1f}×", "cheaper per successful task (CPS)")
kpi(s, 6.8, 4.0, 2.75, f"{pct(FULL['accuracy'])}", f"accuracy on 36 held-out cases (always-14B: {pct(T2['accuracy'])})",
    color=TEXT)
kpi(s, 9.75, 4.0, 2.75, f"{CPU_MS:.1f} ms", "total CPU overhead per task for all 4 guard components", color=TEXT)
text(s, 0.9, 6.1, 11.5, 0.8, [("Nawaf Alduaij", {"size": 16, "bold": True}),
     ("qwen2.5:3b → qwen2.5:14b, both co-resident on one AMD RX 7900 XTX (24 GB) · 180 recorded calls · "
      "deterministic replay in under 2 s", {"size": 12, "color": MUTED})])

# ================================================================ 2. The issue
s = new_slide(2, "The issue", "Most enterprise requests overpay for compute")
text(s, 0.6, 1.55, 6.0, 4.8, [
    ("The default setup sends every request to the largest model with the whole document attached.",
     {"size": 16, "space": 14}),
    ("Measured on our 36 held-out bilingual tasks:", {"size": 14, "color": MUTED, "space": 8}),
    (f"Always-14B on raw context: {T2['compute_units']:.0f} kTok·B, {T2['total_tokens']:,} tokens, "
     f"{pct(T2['accuracy'])} accuracy", {"bullet": True, "size": 14}),
    (f"Always-3B: {pct(1 - T1['compute_units'] / T2['compute_units'])} cheaper and only "
     f"{(T2['accuracy'] - T1['accuracy']) * 100:.1f} pts less accurate, so most tasks never needed the large model",
     {"bullet": True, "size": 14}),
    (f"Pruning alone removes {pct(NOISE[0]['reduction'])} of context tokens; with 4× distractor noise, "
     f"{pct(NOISE[-1]['reduction'])}", {"bullet": True, "size": 14}),
    ("A naive semantic cache returns the answer for #INV-8841 when asked about #INV-8842: "
     "a cheap but wrong result", {"bullet": True, "size": 14}),
])
items = [("1", "Model choice per task", "route easy work to the 3B model"),
         ("2", "Context that earns its tokens", "pruning + boilerplate removal"),
         ("3", "Verification where it pays", "< 1 ms rule checks, escalate only on failure"),
         ("4", "Cost per successful task", "CPS reported for every configuration"),
         ("5", "Accuracy & latency floors", "graded by a separate ground-truth scorer")]
text(s, 7.2, 1.55, 5.5, 0.4, "WHAT THE CHALLENGE ASKS, AND WHERE OPTIROUTE ANSWERS IT", size=11, color=EM2, bold=True)
for i, (n, a, b) in enumerate(items):
    y = 2.0 + i * 0.92
    rect(s, 7.2, y, 5.5, 0.8)
    text(s, 7.35, y + 0.1, 0.5, 0.6, n, size=22, bold=True, color=EM)
    text(s, 7.9, y + 0.08, 4.7, 0.7, [(a, {"bold": True, "size": 14, "space": 1}), (b, {"size": 11.5, "color": MUTED})])

# ================================================================ 3. Our solution
s = new_slide(3, "Our solution", "A 3-tier cascade with deterministic CPU guards between the tiers")
stages = [("Pruner", "char n-gram + word TF-IDF\n+ shared-number bonus\nAR clitic normalizer", EM),
          ("Cache Guard", "Tier-0 · 0 LLM tokens\ncosine ≥ 0.88 AND\nnumbers/IDs exactly equal", EM),
          ("Router", "complexity C from 4\ninspectable signals\nC < τ → Tier-1", "60A5FA"),
          ("Tier-1 + Verifier", "qwen2.5:3b\nrefusal · schema ·\ngrounding · language", EM),
          ("Tier-2", "qwen2.5:14b\ndirect if C ≥ τ, or on\nverifier escalation", AMBER)]
w, gap, y0 = 2.2, 0.32, 1.9
for i, (name, desc, col) in enumerate(stages):
    x = 0.6 + i * (w + gap)
    rect(s, x, y0, w, 2.1, line=col)
    text(s, x + 0.15, y0 + 0.12, w - 0.3, 0.4, name, size=16, bold=True, color=col)
    text(s, x + 0.15, y0 + 0.6, w - 0.3, 1.4, desc, size=11.5, color=MUTED)
    if i < len(stages) - 1:
        arrow(s, x + w + 0.03, y0 + 1.05, x + w + gap - 0.03, y0 + 1.05)
rect(s, 0.6 + 3 * (w + gap), 4.15, 2 * w + gap, 0.8, fill="1A1405", line=AMBER)
text(s, 0.75 + 3 * (w + gap), 4.2, 2 * w + gap - 0.3, 0.7,
     "Verifier fails → escalate. A refusal means the pruner starved the context, so retry with the full context; "
     "otherwise retry with the top-4 chunks.", size=11, color=AMBER)
rect(s, 0.6, 5.1, 12.1, 1.55, fill=PANEL2)
text(s, 0.8, 5.2, 11.8, 1.4, [
    ("Adaptation loop: a W = 20 window of Tier-1 verifier outcomes and P95 latency moves τ ∈ [0.25, 0.75]. "
     "τ tightens by 0.08 when the pass rate drops below 70% or P95 exceeds the SLO, and recovers by 0.05 per step "
     "after 5 healthy steps (hysteresis).", {"size": 13}),
    ("Every component is deterministic, CPU-only and individually timed: mean " +
     " · ".join(f"{k} {v:.2f} ms" for k, v in FULL["overhead_ms"].items()) + ".", {"size": 12, "color": MUTED}),
])

# ================================================================ 4. How it works
s = new_slide(4, "How it works", "Four steps, each one measurable and switchable for ablation")
steps = [
    ("1", "Context pruning & numeric-guarded cache",
     ["Paragraphs scored: 0.50·cos(char 3–5g) + 0.35·cos(word TF-IDF) + 0.15·shared numbers/IDs",
      "Arabic: Alef/Teh-Marbuta/Alef-Maksura unified, tashkeel stripped, proclitics (وال، بال، لل…) removed",
      "Cache key = query wording (digits masked) + hash of top-2 chunks; hit only if entity sets are identical"]),
    ("2", "Complexity routing & Tier-1 execution",
     ["C = 0.30 multi-hop + 0.30 arithmetic + 0.25 strict JSON/table + 0.15 relevance dispersion",
      "C < τ → qwen2.5:3b on the pruned context; C ≥ τ → qwen2.5:14b directly",
      f"Measured: {FULL['tier1']} of 36 tasks finished on Tier-1, {FULL['tier0']} on Tier-0 cache"]),
    ("3", "Verification & full-context escalation",
     ["Refusal/degeneracy · JSON schema · numeric grounding (each output number must appear in, or follow by + − × ÷ % "
      "from, the context) · Arabic-in/Arabic-out",
      f"Precision {pct(VP['precision'], 0)} and recall {pct(VP['recall'], 0)} on recorded Tier-1 outputs; never sees ground truth",
      "Refusal → Tier-2 with full context (recovers pruner misses); other failures → top-4 relaxed context"]),
    ("4", "Bidirectional τ adaptation + shadow audit",
     ["Estimated Tier-1 health = verifier pass rate × agreement on sampled Tier-2 audits (every 4th verified answer)",
      "Tightens when health < 70% or P95 > SLO; recovers +0.05/step after 5 healthy steps (hysteresis)",
      f"Recorded wave: τ 0.55 → {AUD_TAU_MIN:.2f} in the hard burst (request {AUD_TIGHTEN}) → back to 0.55 "
      f"(requests {', '.join(map(str, AUD_RECOVER))})"]),
]
for i, (n, title, bullets) in enumerate(steps):
    x, y = 0.6 + (i % 2) * 6.15, 1.6 + (i // 2) * 2.7
    rect(s, x, y, 5.95, 2.5)
    text(s, x + 0.2, y + 0.12, 0.6, 0.6, n, size=26, bold=True, color=EM)
    text(s, x + 0.8, y + 0.2, 5.0, 0.5, title, size=15, bold=True)
    text(s, x + 0.8, y + 0.7, 5.0, 1.8, [(b, {"bullet": True, "size": 11.5, "color": MUTED, "space": 5})
                                          for b in bullets])

# ================================================================ 5. Implementation & live demo
s = new_slide(5, "Implementation & live demo", "A working prototype you can click through")
img = SHOTS / "crop_live_trap.png"
if img.exists():
    pic = s.shapes.add_picture(str(img), Inches(0.6), Inches(1.55), height=Inches(5.15))
    pic.line.color.rgb = rgb(LINE)
    text(s, 0.6, 6.72, pic.width / 914400, 0.3, "Live UI: the #INV-8842 trap is blocked by the entity guard "
         "(wording cosine 1.000, IDs differ)", size=10, color=MUTED)
x0 = 7.45
text(s, x0, 1.5, 5.3, 3.4, [
    ("WHAT WAS BUILT", {"bold": True, "size": 12, "color": EM2, "space": 6}),
    ("Cascade engine (src/): pruner, entity-guarded cache, router, verifier, adaptive τ, orchestrator; "
     "every component can be switched off for ablation", {"bullet": True, "size": 12, "space": 5}),
    ("Interactive UI: 8 bilingual presets replay recorded runs; custom questions run on the live models; every "
     "stage visualised with its cost vs always-14B", {"bullet": True, "size": 12, "space": 5}),
    ("evaluate.py: --record / --replay / --sim, 2 ablation tables, 5 robustness experiments", {"bullet": True, "size": 12, "space": 5}),
    (f"{N_TESTS} unit tests; slides and reports generated from measured results", {"bullet": True, "size": 12}),
])
rect(s, x0, 4.95, 5.25, 1.75, fill=PANEL2)
text(s, x0 + 0.15, 5.02, 5.0, 1.65, [
    ("TECH STACK", {"bold": True, "size": 11, "color": EM2, "space": 4}),
    ("Python · Ollama on ROCm (AMD RX 7900 XTX, 24 GB) · qwen2.5 3B / 14B (+ llama3.2 3B, qwen2.5 0.5B) · "
     "OpenAI-compatible client · tiktoken · FastAPI + Chart.js · pytest · python-pptx", {"size": 11.5, "space": 8}),
    (GITHUB, {"bold": True, "size": 13, "color": EM2}),
])

# ================================================================ 6. Methodology
s = new_slide(6, "Evaluation methodology", "Held-out, recorded once, replayed, graded separately")
cats = {}
held = json.loads((ROOT / "data" / "held_out_test.json").read_text(encoding="utf-8"))
for c in held:
    cats[c["category"]] = cats.get(c["category"], 0) + 1
n_ar = sum(c["language"] == "ar" for c in held)
rows = [["Category", "Cases", "What it stresses"]]
desc = {"standard": "single-chunk policy / invoice lookups",
        "low_lexical_overlap": "paraphrased query, almost no shared keywords",
        "multi_hop": "facts from 2 chunks + arithmetic (SLA credit, penalties)",
        "arabic_orthographic_variants": "missing hamza, ه/ة, ي/ى, clitics, mixed numerals",
        "cache_near_duplicates": "2 paraphrase hits + 2 traps (#INV-8841→8842, 6→8 days)"}
for k in ["standard", "low_lexical_overlap", "multi_hop", "arabic_orthographic_variants", "cache_near_duplicates"]:
    rows.append([k.replace("_", " "), cats.get(k, 0), desc[k]])
rows.append(["total", len(held), f"{len(held) - n_ar} English / {n_ar} Arabic · separate 10-case dev set"])
table(s, 0.6, 1.6, 7.3, rows, [2.4, 0.7, 4.2], highlight=(len(rows) - 1,), size=11, row_h=0.42)
text(s, 8.3, 1.55, 4.4, 5.2, [
    ("Record & replay", {"bold": True, "size": 15, "color": EM2, "space": 4}),
    ("180 real calls (36 tasks × raw/pruned/relaxed × tier), temperature 0, models pre-warmed so load time is "
     "excluded, prompt hash stored to detect drift", {"size": 12, "color": MUTED, "space": 12}),
    ("No circular metrics", {"bold": True, "size": 15, "color": EM2, "space": 4}),
    ("Accuracy comes from a ground-truth scorer (numeric tolerance, required facts, JSON fields). The runtime "
     "verifier is graded against it, never used for it.", {"size": 12, "color": MUTED, "space": 12}),
    ("Measured units", {"bold": True, "size": 15, "color": EM2, "space": 4}),
    ("tiktoken o200k token counts · compute = Σ tokens/1000 × active params (3B / 14B, cache = 0) · "
     "latency = recorded LLM time + perf_counter CPU time", {"size": 12, "color": MUTED, "space": 12}),
    ("Reproducible", {"bold": True, "size": 15, "color": EM2, "space": 4}),
    (f"Full evaluation replays in {M['eval_seconds']:.1f} s without a GPU or API keys; "
     f"{N_TESTS} unit tests in sim mode", {"size": 12, "color": MUTED}),
])

# ================================================================ 7. Results (cumulative)
s = new_slide(7, "Results: held-out, recorded real runs", "Cumulative build-up: what each component adds")
names = list(C)
table(s, 0.6, 1.55, 12.1, metric_rows(C, names), [2.6, 1.45, 0.95, 1.35, 1.05, 0.8, 0.9, 1.15, 1.85],
      highlight=(len(names),), size=11, row_h=0.4)
cd = CategoryChartData()
cd.categories = [n.split("_", 1)[1].replace("_", " ") for n in names]
cd.add_series("Compute (kTok·B)", [round(C[n]["compute_units"], 1) for n in names])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(4.15), Inches(7.4), Inches(2.75), cd)
ch = gf.chart
ch.has_legend = False
ch.font.size = Pt(10)
ch.font.color.rgb = rgb(MUTED)
ch.font.name = FONT
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
plot.data_labels.font.size = Pt(10)
plot.data_labels.font.color.rgb = rgb(TEXT)
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ser = plot.series[0]
for i, n in enumerate(names):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(EM if "Full" in n else "1F4D40")
ch.value_axis.has_major_gridlines = False
ch.value_axis.visible = False
ch.category_axis.format.line.color.rgb = rgb(LINE)
ch.has_title = True
ch.chart_title.text_frame.text = "Compute per configuration (kTok·B, lower is better)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(MUTED)
text(s, 8.3, 4.2, 4.4, 2.7, [
    (f"Full cascade vs always-14B", {"bold": True, "size": 15, "color": EM2}),
    (f"−{pct(COMPUTE_CUT)} compute, −{pct(TOKEN_CUT)} tokens", {"bullet": True, "size": 13}),
    (f"CPS {FULL['cps']:.2f} vs {T2['cps']:.2f} kTok·B per success ({CPS_X:.1f}× cheaper)", {"bullet": True, "size": 13}),
    (f"P50 latency {FULL['p50_ms']:.0f} ms vs {T2['p50_ms']:.0f} ms", {"bullet": True, "size": 13}),
    (f"Accuracy {pct(FULL['accuracy'])} vs {pct(T2['accuracy'])}: {round((T2['accuracy'] - FULL['accuracy']) * FULL['n'])} "
     f"fewer correct answers out of 36", {"bullet": True, "size": 13, "color": AMBER}),
])

# ================================================================ 8. LOO + cross-model + adaptation
s = new_slide(8, "Ablations & robustness", "Leave-one-out, other models, and adaptation on real runs")
names = list(L)
table(s, 0.6, 1.5, 12.1, metric_rows(L, names), [2.6, 1.45, 0.95, 1.35, 1.05, 0.8, 0.9, 1.15, 1.85],
      highlight=(1,), size=10.5, row_h=0.32)
ng = L["LOO_No_Cache_Entity_Guard"]
nv = L["LOO_No_Verifier"]
npr = L["LOO_No_Pruner"]
xrows = [["Tier-1 model (Tier-2: qwen2.5:14b)", "Tier-1 alone", "OptiRoute", "Compute vs 14B", "CPS 14B → OptiRoute",
          "Verifier P / R"]]
for r in XM:
    c = r["configs"]
    xrows.append([r["tier1"], pct(c["2_Always_Tier1_Raw"]["accuracy"]), pct(c["5_OptiRoute_Full"]["accuracy"]),
                  f"{c['5_OptiRoute_Full']['compute_units'] / c['1_Always_Tier2_Raw']['compute_units'] - 1:+.1%}",
                  f"{c['1_Always_Tier2_Raw']['cps']:.2f} → {c['5_OptiRoute_Full']['cps']:.2f}",
                  f"{pct(r['verifier']['precision'], 0)} / {pct(r['verifier']['recall'], 0)}"])
text(s, 0.6, 3.95, 8.3, 0.3, "SAME CASCADE, DIFFERENT RECORDED MODELS (only the recording changes)", size=11,
     bold=True, color=EM2)
table(s, 0.6, 4.28, 8.4, xrows, [2.55, 1.05, 1.05, 1.2, 1.45, 1.1], highlight=(1,), size=10.5, row_h=0.36)
weak = min(XM, key=lambda r: r["configs"]["2_Always_Tier1_Raw"]["accuracy"]) if XM else None
text(s, 0.6, 5.85, 8.4, 0.9, [
    (f"Entity guard: {L['Full_OptiRoute']['false_cache_hits']}/{L['Full_OptiRoute']['n_traps']} false cache hits vs "
     f"{ng['false_cache_hits']}/{ng['n_traps']} without it · Verifier: +{(FULL['accuracy'] - nv['accuracy']) * 100:.1f} "
     f"pts accuracy · Pruner: −{pct(1 - FULL['total_tokens'] / npr['total_tokens'], 0)} tokens vs no pruner",
     {"size": 11, "color": MUTED, "space": 3}),
    (f"With a weak {weak['tier1']} Tier-1 the cascade adds "
     f"{(weak['configs']['5_OptiRoute_Full']['accuracy'] - weak['configs']['2_Always_Tier1_Raw']['accuracy']) * 100:.1f} "
     "pts over using that model alone." if weak else "", {"size": 11, "color": MUTED}),
])
tau_img = SHOTS / "crop_tau.png"
if tau_img.exists():
    s.shapes.add_picture(str(tau_img), Inches(9.3), Inches(4.28), width=Inches(3.3))
    text(s, 9.3, 3.95, 3.4, 0.3, "REAL τ(t), WAVE WORKLOAD", size=11, bold=True, color=EM2)
    text(s, 9.3, 6.6, 3.4, 0.3, f"Tightens in the burst, then recovers · +{pct(AUD_COST, 0)} compute",
         size=9, color=MUTED)

# ================================================================ 9. Limitations + roadmap
s = new_slide(9, "Honest limitations & roadmap", "What this prototype does not solve yet")
mh = {k: v["per_category"].get("multi_hop", 0) for k, v in C.items()}
llo_full = FULL["per_category"].get("low_lexical_overlap", 0)
lims = [
    ("Grounded-but-wrong arithmetic slips past the verifier",
     f"Recall is {pct(VP['recall'], 0)}: rule checks cannot tell 10% of the fee from 25% when both numbers are in the "
     f"context. Multi-hop accuracy is {pct(mh['5_OptiRoute_Full'], 0)} for the cascade, and even the 14B gets only "
     f"{pct(mh['1_Always_Tier2_Raw'], 0)} (it wrote 900×4 + 1200×3 = 4,500, and switched to Chinese mid-answer in Arabic)."),
    ("A lexical pruner is blind to zero-overlap paraphrases",
     f"Low-overlap accuracy is {pct(llo_full, 0)} with the pruner vs "
     f"{pct(npr['per_category'].get('low_lexical_overlap', 0), 0)} without it. Recovery only works when the 3B admits "
     "it doesn't know; a confident wrong answer (\"USB copy forbidden by the encryption policy\") passes."),
    ("Adaptation needs a costly signal; small benchmark",
     f"Rule checks miss confident errors: verifier-only health never fell below {pct(MIN_RATE, 0)}, so τ stayed flat. "
     f"Shadow audits make τ react on real runs but cost +{pct(AUD_COST, 0)} compute, and since complexity scores "
     "cluster, a 0.08 τ shift barely re-routes tasks. n = 36 synthetic docs: ±1 task = 2.8 pts."),
]
for i, (h, d) in enumerate(lims):
    y = 1.55 + i * 1.72
    rect(s, 0.6, y, 7.2, 1.58, line=AMBER)
    text(s, 0.8, y + 0.1, 6.9, 0.4, f"{i + 1}. {h}", size=14, bold=True, color=AMBER)
    text(s, 0.8, y + 0.52, 6.9, 1.05, d, size=11.5, color=MUTED)
rect(s, 8.1, 1.55, 4.6, 5.0, fill=PANEL2, line=EM)
text(s, 8.3, 1.65, 4.3, 4.8, [
    ("PRODUCTION ROADMAP", {"bold": True, "size": 12, "color": EM2, "space": 10}),
    ("Calculator tool + program-of-thought for multi-hop math, with the verifier recomputing the expression",
     {"bullet": True, "size": 12, "space": 8}),
    ("Small bilingual embedding re-ranker as a fallback when the lexical scores are flat",
     {"bullet": True, "size": 12, "space": 8}),
    ("Token-logprob confidence as a 5th verifier signal for confident-but-wrong answers",
     {"bullet": True, "size": 12, "space": 8}),
    ("Output language guard on Tier-2 (catches code-switching)", {"bullet": True, "size": 12, "space": 8}),
    ("Risk-weighted audit sampling + a larger dev set to calibrate τ, the SLO and the verifier",
     {"bullet": True, "size": 12, "space": 8}),
    ("Serve with vLLM + batching; swap in HUMAIN/ALLaM models through the same OpenAI-compatible client",
     {"bullet": True, "size": 12}),
])

# ================================================================ 10. Close
s = new_slide(10, "Takeaways", "Cheap only counts if it is right, and we measured both")
kpi(s, 0.6, 1.6, 2.9, f"−{pct(COMPUTE_CUT)}", "compute vs always-14B")
kpi(s, 3.65, 1.6, 2.9, f"{CPS_X:.1f}×", "cheaper per successful task")
kpi(s, 6.7, 1.6, 2.9, f"{L['Full_OptiRoute']['false_cache_hits']}/{L['Full_OptiRoute']['n_traps']}",
    "false cache hits with the entity guard")
kpi(s, 9.75, 1.6, 2.95, f"{pct(VP['precision'], 0)}", "verifier precision (no wasted escalations)")
text(s, 0.6, 3.2, 6.2, 3.5, [
    ("Why this design", {"bold": True, "size": 16, "color": EM2, "space": 8}),
    ("Evaluation first: a held-out set, recorded real runs, and leave-one-out ablations, instead of a "
     "prompt wrapper with hand-picked numbers", {"bullet": True, "size": 13, "space": 6}),
    ("Bilingual by construction: Arabic normalization in the pruner, cache and verifier", {"bullet": True, "size": 13, "space": 6}),
    ("Runs fully local on one 24 GB GPU: no rate limits, no API cost, data stays on-premise",
     {"bullet": True, "size": 13, "space": 6}),
    (f"Model-agnostic: any OpenAI-compatible endpoint; verified with {len(XM)} recorded model pairs",
     {"bullet": True, "size": 13, "space": 6}),
    (f"Every guard is inspectable and costs about {CPU_MS:.1f} ms of CPU time", {"bullet": True, "size": 13}),
])
rect(s, 7.1, 3.2, 5.6, 3.4, fill=PANEL2)
text(s, 7.3, 3.3, 5.3, 3.2, [
    (GITHUB, {"bold": True, "size": 14, "color": EM2, "space": 8}),
    ("pip install -r requirements.txt", {"size": 12, "space": 4}),
    ("python evaluate.py --replay data/recorded_runs.jsonl", {"size": 12, "space": 4}),
    ("uvicorn app:app --port 8000   # live demo UI", {"size": 12, "space": 12}),
    ("Re-record on your own hardware:", {"size": 12, "color": MUTED, "space": 4}),
    ("python evaluate.py --record --fresh", {"size": 12, "space": 12}),
    ("Nawaf Alduaij · thank you", {"size": 14, "bold": True}),
], font="Consolas")

out = ROOT / "deliverables" / "OptiRoute_Presentation.pptx"
out.parent.mkdir(exist_ok=True)
prs.save(out)
print(f"wrote {out} ({len(prs.slides)} slides)")
