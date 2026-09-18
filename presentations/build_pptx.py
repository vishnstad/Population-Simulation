import sys
import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

src_path = r'd:\Documents\College_purpose\Final Year\Docs\presentations\First_Review\First_Review_Sample_Presentation.pptx'
dst_path = r'd:\Documents\College_purpose\Final Year\Docs\presentations\First_Review\First_Review_Presentation.pptx'

prs = Presentation(src_path)

# Palette
NAVY = RGBColor(19, 78, 124)       # #134E7C
HEADER_NAVY = RGBColor(19, 49, 92) # #13315C
ORANGE = RGBColor(224, 126, 25)    # #E07E19
GREEN = RGBColor(46, 139, 87)      # #2E8B57
RED = RGBColor(192, 57, 43)        # #C0392B
DARK = RGBColor(34, 34, 34)        # #222222
GRAY = RGBColor(100, 100, 100)     # #646464
LIGHT_BLUE = RGBColor(233, 242, 250) # #E9F2FA
LIGHT_ROW = RGBColor(238, 243, 248)  # #EEF3F8
WHITE = RGBColor(255, 255, 255)

def style_text(p, text, size=Pt(13), bold=False, color=DARK, font_name='Calibri', align=PP_ALIGN.LEFT):
    p.text = text
    p.font.name = font_name
    p.font.size = size
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align

def clear_body_shapes(slide):
    # Keep title (TextBox 1 or Title 1) and footer (TextBox 3 or TextBox 8)
    to_remove = []
    for s in slide.shapes:
        if s.name not in ['TextBox 1', 'Title 1', 'TextBox 3', 'TextBox 8']:
            to_remove.append(s)
    for s in to_remove:
        sp = s._element
        sp.getparent().remove(sp)

# ----------------------------------------------------
# SLIDE 1: Title Slide
# ----------------------------------------------------
slide1 = prs.slides[0]
for s in slide1.shapes:
    if s.name == 'TextBox 4':
        tf = s.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        style_text(p, 'First review - Final Year project', size=Pt(20), bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    elif s.name == 'Title 1':
        tf = s.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        style_text(p, 'Hierarchical Multi-Agent Population Simulation', size=Pt(24), bold=True, color=HEADER_NAVY, align=PP_ALIGN.CENTER)
        p2 = tf.add_paragraph()
        style_text(p2, 'Calibrated Distribution-Valued Cluster Agents for Held-Out Survey Prediction', size=Pt(14), bold=False, color=NAVY, align=PP_ALIGN.CENTER)
    elif s.name == 'TextBox 6':
        tf = s.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        style_text(p, 'Team No.: B17', size=Pt(14), bold=True, color=HEADER_NAVY, align=PP_ALIGN.CENTER)
        p1 = tf.add_paragraph()
        style_text(p1, 'Nitin Krishna V  (CB.SC.U4AIE23156)   |   Vishal S  (CB.SC.U4AIE23160)', size=Pt(12), bold=False, color=DARK, align=PP_ALIGN.CENTER)
        p2 = tf.add_paragraph()
        style_text(p2, 'Manogna Challa  (CB.SC.U4AIE23175)   |   Gaurav Mahesh  (CB.SC.U4AIE23176)', size=Pt(12), bold=False, color=DARK, align=PP_ALIGN.CENTER)
    elif s.name == 'TextBox 7':
        tf = s.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        style_text(p, 'Mentor: Dr. Snigdhatanu Acharya   |   Co-Mentor: Dr. Premjith B.', size=Pt(13), bold=True, color=HEADER_NAVY, align=PP_ALIGN.CENTER)
        p2 = tf.add_paragraph()
        style_text(p2, 'School of Artificial Intelligence, Amrita Vishwa Vidyapeetham', size=Pt(12), bold=False, color=GRAY, align=PP_ALIGN.CENTER)
    elif s.name == 'TextBox 8':
        tf = s.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        style_text(p, 'First review - Final Year project • Team B17 • Slide 1', size=Pt(10), color=GRAY, align=PP_ALIGN.LEFT)
    elif s.name == 'Subtitle 2':
        # Remove empty placeholder
        sp = s._element
        sp.getparent().remove(sp)

# Set footers for slides 2 to 15
for idx in range(1, 15):
    slide = prs.slides[idx]
    for s in slide.shapes:
        if s.name == 'TextBox 3':
            tf = s.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            style_text(p, f'First review - Final Year project • Team B17 • Slide {idx+1}', size=Pt(10), color=GRAY, align=PP_ALIGN.LEFT)

# Helper function to create content boxes
def add_card(slide, left, top, width, height, title, bg_color=WHITE, border_color=NAVY):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.2)
    
    # Title text
    tf = shape.text_frame
    tf.margin_top = Inches(0.1)
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_bottom = Inches(0.1)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    style_text(p, title, size=Pt(13), bold=True, color=HEADER_NAVY)
    return shape, tf

# ----------------------------------------------------
# SLIDE 2: 01 | Project Progress Since Zeroth Review
# ----------------------------------------------------
clear_body_shapes(prs.slides[1])
s2 = prs.slides[1]

# Subtitle banner
sub_box = s2.shapes.add_textbox(Inches(0.55), Inches(0.85), Inches(8.9), Inches(0.4))
p = sub_box.text_frame.paragraphs[0]
style_text(p, 'Core focus: What has been built, tested, and validated on real microdata since Zeroth Review.', size=Pt(13), bold=True, color=DARK)

# Left card: Completed work
_, tf_left = add_card(s2, Inches(0.55), Inches(1.35), Inches(4.3), Inches(5.45), '1. What We Built & Validated', bg_color=LIGHT_BLUE)
items_left = [
    ('End-to-End Pipeline:', ' Built and frozen modules M1 through M10 (174 automated tests passing, 0 lint warnings).'),
    ('Real Survey Data:', ' Ingested and harmonized pooled GSS 2010–2022 (n = 18,772 respondents).'),
    ('Pre-Registered Verdict:', ' Tested across 3 open model tiers (Ministral 3B, 8B, 14B).'),
    ('Headline 14B Result:', ' -25.3% W1 error reduction over the national baseline on held-out questions (beating our pre-registered -20% target).'),
    ('Variance Recovered:', ' Population variance ratio = 1.009 (target [0.8, 1.2]), overcoming persona variance collapse.')
]
for title, desc in items_left:
    p = tf_left.add_paragraph()
    p.space_before = Pt(8)
    run1 = p.add_run()
    run1.text = '• ' + title
    run1.font.bold = True
    run1.font.size = Pt(11.5)
    run1.font.color.rgb = HEADER_NAVY
    run2 = p.add_run()
    run2.text = desc
    run2.font.size = Pt(11)
    run2.font.color.rgb = DARK

# Right table: Initial Spec vs Measured Reality
shape_r, tf_r = add_card(s2, Inches(5.05), Inches(1.35), Inches(4.4), Inches(5.45), '2. Where Real Data Corrected Our Spec')
p = tf_r.add_paragraph()
style_text(p, 'Original proposal had reasoned estimates. Testing on real data caught 5 critical parameters:', size=Pt(11), color=GRAY)

# Add table inside right card area
table_shape = s2.shapes.add_table(6, 3, Inches(5.15), Inches(2.1), Inches(4.2), Inches(4.5))
table = table_shape.table
table.columns[0].width = Inches(1.3)
table.columns[1].width = Inches(1.1)
table.columns[2].width = Inches(1.8)

headers = ['Parameter', 'Initial Spec', 'Measured Reality']
for col_idx, h in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = HEADER_NAVY
    p = cell.text_frame.paragraphs[0]
    style_text(p, h, size=Pt(10.5), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

rows = [
    ('Cluster count (K)', 'K = 150', 'K = 56 (min cell >= 60)'),
    ('Survey bed', 'GSS 2024', 'Pooled 2010–2022 (2024 lacks region)'),
    ('Partition axes', 'Age x Edu x Geo', 'Age x Degree x Sex (Geo in card)'),
    ('Usable items', '148 items', '74 clear SNR >= 1.5 noise gate'),
    ('API spend', '$1,200 budget', '$0 (Free tiers + caching router)')
]

for row_idx, r in enumerate(rows):
    bg = LIGHT_ROW if row_idx % 2 == 0 else WHITE
    for col_idx, val in enumerate(r):
        cell = table.cell(row_idx + 1, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
        p = cell.text_frame.paragraphs[0]
        is_bold = (col_idx == 2)
        c_color = GREEN if (col_idx == 2) else DARK
        style_text(p, val, size=Pt(9.5), bold=is_bold, color=c_color, align=PP_ALIGN.LEFT if col_idx > 0 else PP_ALIGN.CENTER)

# ----------------------------------------------------
# SLIDE 3: 02 | Motivation & Problem Statement
# ----------------------------------------------------
clear_body_shapes(prs.slides[2])
s3 = prs.slides[2]

_, tf_prob = add_card(s3, Inches(0.55), Inches(0.95), Inches(8.9), Inches(2.3), 'The Core Scientific Problem', bg_color=LIGHT_BLUE)
p = tf_prob.add_paragraph()
style_text(p, 'Given a demographic subgroup described only by population shares and answers to other survey questions, what is the full response distribution (histogram) to an unasked question?', size=Pt(13), bold=True, color=HEADER_NAVY)
p.space_before = Pt(4)

p2 = tf_prob.add_paragraph()
p2.space_before = Pt(6)
p2.text = '• Why full distributions? A single average score hides polarization. Bimodal public opinions look identical to moderate consensus when only means are calculated.\n• Why persona prompting fails: Prompting LLMs as single personas suffers from within-group variance collapse (Bisbee et al.) and regression to the mean between groups.\n• Why cross-tabs cannot solve it: Standard cross-tabulations only report questions already asked in historical surveys.'
p2.font.size = Pt(11)
p2.font.color.rgb = DARK

_, tf_obj = add_card(s3, Inches(0.55), Inches(3.45), Inches(8.9), Inches(3.4), 'Pre-Registered Objectives (Fixed Before Any Run)')
objs = [
    ('Objective 1 (Accuracy):', ' Beat the national-marginal baseline (B0a) by at least 20% in Wasserstein-1 (W1) distance on held-out survey items where demographic subgroups genuinely differ.'),
    ('Objective 2 (Distributional Fidelity):', ' Recover true population variance within a ratio band of [0.8, 1.2], proving the method prevents persona variance collapse.'),
    ('Objective 3 (Computational Efficiency):', ' Match or exceed per-individual persona sampling accuracy at equal or lower query cost ($0 budget using free-tier APIs and prompt caching).'),
    ('Scope & Honesty Boundary:', ' Evaluated on US General Social Survey (pooled 2010–2022). India Global Flourishing Study (n = 12,765) staged for replication. Synthetic policy scenarios are accompanied by an explicit Honesty Box.')
]
for title, text in objs:
    p = tf_obj.add_paragraph()
    p.space_before = Pt(6)
    r1 = p.add_run()
    r1.text = '• ' + title
    r1.font.bold = True
    r1.font.size = Pt(11.5)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = text
    r2.font.size = Pt(11)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 4: 03 | Consolidated Literature Review & Research Gap
# ----------------------------------------------------
clear_body_shapes(prs.slides[3])
s4 = prs.slides[3]

# Literature table
t_lit = s4.shapes.add_table(6, 3, Inches(0.55), Inches(0.95), Inches(8.9), Inches(3.4)).table
t_lit.columns[0].width = Inches(2.2)
t_lit.columns[1].width = Inches(3.6)
t_lit.columns[2].width = Inches(3.1)

headers_lit = ['Key Literature', 'Finding / Approach', 'Identified Research Gap']
for col_idx, h in enumerate(headers_lit):
    cell = t_lit.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = HEADER_NAVY
    p = cell.text_frame.paragraphs[0]
    style_text(p, h, size=Pt(11), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

rows_lit = [
    ('Argyle et al. (2023)\nPolitical Analysis', 'GPT-3 personas reproduce demographic voting and survey trends.', 'Static personas; lacks distributional calibration.'),
    ('Bisbee et al. (2024)\nPolitical Analysis', 'Evaluates LLMs against human surveys; means track data.', 'Severe within-group variance collapse (F1); run instability.'),
    ('Santurkar et al. (2023)\nICML (OpinionQA)', 'Benchmarks LLM opinions against Pew survey distributions.', 'Evaluates steering towards national averages, no subgroup calibration.'),
    ('Park et al. (2023, 2024)\nACM UIST / arXiv', 'Generative agents / 1,000 interview-grounded agents.', 'Linear cost scaling (1 agent per person); flat population structure.'),
    ('Li et al. (2024) / Pi (2025)\nACL / arXiv', 'EconAgent & AgentSociety: large-scale multi-agent sims.', 'Homogeneous personas; heavy compute; lacks statistical hierarchy.')
]

for row_idx, r in enumerate(rows_lit):
    bg = LIGHT_ROW if row_idx % 2 == 0 else WHITE
    for col_idx, val in enumerate(r):
        cell = t_lit.cell(row_idx + 1, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
        p = cell.text_frame.paragraphs[0]
        style_text(p, val, size=Pt(10), color=DARK, align=PP_ALIGN.LEFT)

# Gaps addressed cards below
gaps = [
    ('Gap 1: Compute Explosion', 'Simulating thousands of individuals is cost-prohibitive.\nOur Fix: 1 query per demographic cluster (K = 56).'),
    ('Gap 2: Distributional Collapse', 'Prompting personas collapses within-group variance.\nOur Fix: Elicit full histograms + cross-fitted calibration.'),
    ('Gap 3: Flat Hierarchy', 'Flat agent pools lack structured statistical shrinkage.\nOur Fix: Supervised tree pooling + subspace projection.')
]
for idx, (g_title, g_desc) in enumerate(gaps):
    _, tf_g = add_card(s4, Inches(0.55 + idx * 3.05), Inches(4.55), Inches(2.85), Inches(2.25), g_title, bg_color=LIGHT_BLUE)
    p = tf_g.add_paragraph()
    p.space_before = Pt(4)
    style_text(p, g_desc, size=Pt(10), color=DARK)

# ----------------------------------------------------
# SLIDE 5: 04 | Finalized System Architecture & Methodology
# ----------------------------------------------------
clear_body_shapes(prs.slides[4])
s5 = prs.slides[4]

_, tf_arch = add_card(s5, Inches(0.55), Inches(0.95), Inches(8.9), Inches(3.2), 'System Architecture: The 10-Module Pipeline (M1–M10)', bg_color=WHITE)
steps = [
    ('Data Foundation:', ' M1 Ingestion (GSS pooled 2010–2022) -> M2 Cluster Tree (K=56 leaves on age x degree x sex) -> M3 Stat Cards (12 observed anchor survey items + marginals).'),
    ('Oracle Router (M7):', ' Incoming questions are checked against the GSS codebook. If already asked -> served from the weighted cross-tab ($0 cost, 0 error). The LLM is called ONLY for unasked questions.'),
    ('Elicitation & Calibration:', ' M4 Distributional Elicitation (14B model, 3 repeats) -> M5 Anchor Calibration (Level/Deviation Decomposition).'),
    ('Rollup & Reporting:', ' M6 ACS 2024 Post-Stratification Raking -> M9 Evaluation Harness (W1 vs B0a) -> M10 Honesty Box Report.')
]
for title, desc in steps:
    p = tf_arch.add_paragraph()
    p.space_before = Pt(4)
    r1 = p.add_run()
    r1.text = '• ' + title
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = DARK

_, tf_math = add_card(s5, Inches(0.55), Inches(4.35), Inches(8.9), Inches(2.55), 'Core Mathematical Formulation: Level/Structure Decomposition', bg_color=LIGHT_BLUE)
p_math = tf_math.add_paragraph()
p_math.space_before = Pt(4)
style_text(p_math, 'pred_cdf(c) = level_cdf  +  s · P_r ( raw_cdf(c) - Σ w_c · raw_cdf(c) )', size=Pt(14), bold=True, color=HEADER_NAVY, align=PP_ALIGN.CENTER)

math_bullets = [
    ('level_cdf:', ' National marginal (observed topline or population LLM call). Handed identically to baseline B0a.'),
    ('raw_cdf(c) - Σ w_c · raw_cdf(c):', ' The model\'s raw subgroup deviation, isolating its between-cluster opinion gradient.'),
    ('P_r (Subspace Projection):', ' Projects deviation onto top-r principal directions of anchor variance (removes isotropic noise).'),
    ('s (Scaling Factor):', ' Fitted strictly on 34 anchors via F=3 cross-fitting. No target truth is ever touched.')
]
for term, exp in math_bullets:
    p = tf_math.add_paragraph()
    p.space_before = Pt(2)
    r1 = p.add_run()
    r1.text = '• ' + term + ' '
    r1.font.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = exp
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 6: 05 | Implementation Status – Prototype / Proof-of-Concept
# ----------------------------------------------------
clear_body_shapes(prs.slides[5])
s6 = prs.slides[5]

# Module Status Table
t_mod = s6.shapes.add_table(11, 2, Inches(0.55), Inches(0.95), Inches(4.4), Inches(5.8)).table
t_mod.columns[0].width = Inches(3.2)
t_mod.columns[1].width = Inches(1.2)

headers_m = ['Pipeline Module', 'Status']
for col_idx, h in enumerate(headers_m):
    cell = t_mod.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = HEADER_NAVY
    p = cell.text_frame.paragraphs[0]
    style_text(p, h, size=Pt(10.5), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

modules = [
    ('M1: Ingestion & Harmonization', 'Completed'),
    ('M2: Cluster Tree & Sampling Noise', 'Completed'),
    ('M3: Stat Card Generator', 'Completed'),
    ('M4: LLM Distributional Elicitation', 'Completed'),
    ('M5: Cross-Fitted Calibration Layer', 'Completed'),
    ('M6: Post-Stratification Raking', 'Completed'),
    ('M7: Oracle Router (TF-IDF Cosine)', 'Completed'),
    ('M8: Opinion Dynamics Simulation', 'Stage B'),
    ('M9: Evaluation Benchmark Harness', 'Completed'),
    ('M10: Streamlit App + Honesty Box', 'Completed')
]

for row_idx, (m_name, m_stat) in enumerate(modules):
    bg = LIGHT_ROW if row_idx % 2 == 0 else WHITE
    c1 = t_mod.cell(row_idx + 1, 0)
    c1.fill.solid()
    c1.fill.fore_color.rgb = bg
    p1 = c1.text_frame.paragraphs[0]
    style_text(p1, m_name, size=Pt(9.5), color=DARK)

    c2 = t_mod.cell(row_idx + 1, 1)
    c2.fill.solid()
    c2.fill.fore_color.rgb = bg
    p2 = c2.text_frame.paragraphs[0]
    col = GREEN if m_stat == 'Completed' else ORANGE
    style_text(p2, m_stat, size=Pt(9.5), bold=True, color=col, align=PP_ALIGN.CENTER)

# Right card: Rigor & Infrastructure
_, tf_qual = add_card(s6, Inches(5.15), Inches(0.95), Inches(4.3), Inches(5.8), 'Engineering Rigor & Verification', bg_color=LIGHT_BLUE)
qual_points = [
    ('Automated Test Suite:', ' 174 automated tests passing via pytest; 0 lint warnings. Contract tests verify that cards never leak target answers.'),
    ('Free-Tier API Router:', ' Built custom provider router supporting Mistral, Groq, and Gemini. Daily call counters and token ceiling prevent rate-limit crashes.'),
    ('Atomic SQLite Caching:', ' Every single prompt-response pair is cached. An interrupted batch run resumes instantly with zero redundant spend.'),
    ('Two Independent Verifications:', ' (1) verify_headline.py re-calculated macro W1 from scratch, agreeing to 0.00e+00.\n(2) verify_toplines.py re-parsed published GSS codebook PDF; 1,147 of 1,148 tables matched exactly.'),
    ('Zero Dollar Compute:', ' Evaluated entire benchmark across 37k+ calls at $0 budget.')
]
for title, desc in qual_points:
    p = tf_qual.add_paragraph()
    p.space_before = Pt(8)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 7: 06 | Experimental Setup & Test Cases
# ----------------------------------------------------
clear_body_shapes(prs.slides[6])
s7 = prs.slides[6]

_, tf_setup = add_card(s7, Inches(0.55), Inches(0.95), Inches(4.3), Inches(5.8), '1. Experimental Setup', bg_color=LIGHT_BLUE)
setup_items = [
    ('Survey Dataset Bed:', ' GSS pooled 2010–2022 (18,772 respondents). Provides high statistical power across decades.'),
    ('Demographic Partition:', ' age_band x degree x sex yielding K = 56 clusters (enforcing min cell size >= 60 to suppress noise).'),
    ('Item Pool Filtering:', ' 148 survey items evaluated -> 74 cleared the split-half noise gate (SNR >= 1.5, noise floor 0.0273).'),
    ('Frozen Item Split:', ' 34 anchors (context for cards) / 40 targets (held-out and scored). Stratified by topic.'),
    ('Census Raking Margins:', ' ACS 2024 PUMS (60 demographic cells, 267.2M US adults), validated against GSS within 2.9 pp.'),
    ('Model Tiers Tested:', ' Ministral 3B, 8B (development arm), and 14B (headline confirmatory arm).')
]
for title, desc in setup_items:
    p = tf_setup.add_paragraph()
    p.space_before = Pt(6)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

# Right table: Six verification gates
t_gates = s7.shapes.add_table(7, 3, Inches(5.05), Inches(0.95), Inches(4.4), Inches(5.8)).table
t_gates.columns[0].width = Inches(0.8)
t_gates.columns[1].width = Inches(2.5)
t_gates.columns[2].width = Inches(1.1)

headers_g = ['Gate', 'Verification Layer Objective', 'Verdict']
for col_idx, h in enumerate(headers_g):
    cell = t_gates.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = HEADER_NAVY
    p = cell.text_frame.paragraphs[0]
    style_text(p, h, size=Pt(10), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

gates_data = [
    ('L0', 'Match GSS codebook toplines within 0.5 pp', 'PASS (142/142)'),
    ('L1', 'Estimate sampling noise floor via split-half', 'PASS (0.0273)'),
    ('L2', 'Permutation test: stat-card reading check', 'PASS (+0.0689)'),
    ('L3', 'Calibration arithmetic oracle pass-through', 'PASS (1.1e-16)'),
    ('L4', 'Beat national baseline on held-out targets', 'PASS (-25.3%)'),
    ('L5', 'Direct memorization & leakage audit', 'PASS (2/39 flag)')
]
for row_idx, (g_id, g_desc, g_res) in enumerate(gates_data):
    bg = LIGHT_ROW if row_idx % 2 == 0 else WHITE
    c1 = t_gates.cell(row_idx + 1, 0)
    c1.fill.solid()
    c1.fill.fore_color.rgb = bg
    p1 = c1.text_frame.paragraphs[0]
    style_text(p1, g_id, size=Pt(10), bold=True, color=HEADER_NAVY, align=PP_ALIGN.CENTER)

    c2 = t_gates.cell(row_idx + 1, 1)
    c2.fill.solid()
    c2.fill.fore_color.rgb = bg
    p2 = c2.text_frame.paragraphs[0]
    style_text(p2, g_desc, size=Pt(9.5), color=DARK)

    c3 = t_gates.cell(row_idx + 1, 2)
    c3.fill.solid()
    c3.fill.fore_color.rgb = bg
    p3 = c3.text_frame.paragraphs[0]
    style_text(p3, g_res, size=Pt(9.5), bold=True, color=GREEN, align=PP_ALIGN.CENTER)

# ----------------------------------------------------
# SLIDE 8: 07 | Results – Quantitative Performance
# ----------------------------------------------------
clear_body_shapes(prs.slides[7])
s8 = prs.slides[7]

# Top table: Headline quantitative results
t_res = s8.shapes.add_table(5, 5, Inches(0.55), Inches(0.95), Inches(8.9), Inches(2.8)).table
t_res.columns[0].width = Inches(2.9)
t_res.columns[1].width = Inches(1.3)
t_res.columns[2].width = Inches(1.3)
t_res.columns[3].width = Inches(1.3)
t_res.columns[4].width = Inches(2.1)

headers_r = ['Evaluation Subset (14B Model)', 'Achieved W1', 'Baseline B0a', 'Noise Floor', 'Relative Error Gain']
for col_idx, h in enumerate(headers_r):
    cell = t_res.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = HEADER_NAVY
    p = cell.text_frame.paragraphs[0]
    style_text(p, h, size=Pt(10), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

results_data = [
    ('All 39 Scored Held-Out Targets', '0.0602', '0.0807', '0.0273', '-25.3% (PASS >= 20%)'),
    ('Top-Quartile Heterogeneity Items', '0.0787', '0.1139', '0.0402', '-30.9% (PASS both)'),
    ('Leakage-Resistant Subset (32 items)', '0.0565', '0.0757', '0.0324', '-25.4% (PASS both)'),
    ('Uncalibrated Baseline (B4)', '0.1926', '0.0807', '0.0273', 'Calibration cuts error 69%')
]
for row_idx, (r_name, r_w1, r_b0a, r_fl, r_gain) in enumerate(results_data):
    bg = LIGHT_ROW if row_idx % 2 == 0 else WHITE
    row_vals = [r_name, r_w1, r_b0a, r_fl, r_gain]
    for col_idx, val in enumerate(row_vals):
        cell = t_res.cell(row_idx + 1, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
        p = cell.text_frame.paragraphs[0]
        is_bold = (col_idx in [1, 4])
        col_c = GREEN if col_idx == 4 and 'PASS' in val else DARK
        style_text(p, val, size=Pt(10), bold=is_bold, color=col_c, align=PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER)

# Bottom cards: Distributional fidelity & Model Capability Curve
_, tf_dist = add_card(s8, Inches(0.55), Inches(3.95), Inches(4.3), Inches(2.8), 'Distributional Shape Metrics', bg_color=LIGHT_BLUE)
dist_items = [
    ('Population Variance Ratio = 1.009:', ' Pre-registered pass band was [0.8, 1.2]. Objective 2 is passed.'),
    ('90% Interval Coverage = 0.899:', ' Matches nominal 0.90 expectation almost exactly.'),
    ('Tolerance-Battery Ordering ρ = +0.879:', ' Model accurately captures cross-demographic ordering across 5 target groups (atheist, communist, gay man, etc.).'),
    ('Calibration ECE = 0.0227:', ' Expected Calibration Error indicates strong probability alignment.')
]
for title, desc in dist_items:
    p = tf_dist.add_paragraph()
    p.space_before = Pt(3)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = DARK

_, tf_cap = add_card(s8, Inches(5.05), Inches(3.95), Inches(4.4), Inches(2.8), 'Capability Scaling Across Model Tiers')
cap_items = [
    ('Ministral 3B (Open Weights):', ' Achieved W1 = 0.0686 (-15.0% vs B0a). Fast and lightweight for local prototyping.'),
    ('Ministral 8B (Development Arm):', ' Achieved W1 = 0.0634 (-21.4% vs B0a). Used to pre-register and freeze all calibration hyperparameters.'),
    ('Ministral 14B (Confirmatory Arm):', ' Achieved W1 = 0.0602 (-25.3% vs B0a). Significant jump in capability.'),
    ('Key Takeaway:', ' Monotonic improvement in W1 error as model parameter size increases. Each arm\'s strength was predictable from anchor fit alone.')
]
for title, desc in cap_items:
    p = tf_cap.add_paragraph()
    p.space_before = Pt(3)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 9: 08 | Results Analysis & Interpretation
# ----------------------------------------------------
clear_body_shapes(prs.slides[8])
s9 = prs.slides[8]

_, tf_err = add_card(s9, Inches(0.55), Inches(0.95), Inches(4.3), Inches(5.8), '1. The Discovery: Error Decomposition', bg_color=LIGHT_BLUE)
err_points = [
    ('Where LLMs Go Wrong:', ' Raw LLM error (~0.20 W1) is NOT caused by collapsed subgroup differences. It is dominated by a constant per-item level offset (e.g. predicting 72% for communist speech allowance when true value is 29%).'),
    ('Subgroup Ordering Is Present:', ' The between-cluster Spearman correlation is high even in raw output (+0.597 macro on 14B, 8 of 10 items significant). The LLM understands which demographics are more/less supportive.'),
    ('Gate 3 Diagnosis:', ' The raw permutation test was red (+0.0011) because raw W1 was overwhelmed by the level offset. Once level-matched, real vs permuted gap was +0.0689 (green).'),
    ('Why M5 Works:', ' M5 decouples national level from subgroup deviation, rescuing the LLM\'s valid subgroup signal.')
]
for title, desc in err_points:
    p = tf_err.add_paragraph()
    p.space_before = Pt(6)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

_, tf_abl = add_card(s9, Inches(5.05), Inches(0.95), Inches(4.4), Inches(5.8), '2. Ablations: What Is Load-Bearing?')
abl_points = [
    ('Subspace Projection (P_r):', ' Restricting deviations to top anchor directions cuts error from 0.0688 to 0.0638 (-7% gain by discarding isotropic noise).'),
    ('Cross-Topic Transfer:', ' Standard calibration (0.0638 W1) vs adversarial topic-disjoint calibration (0.0641 W1) costs only 0.0003. Proves anchors transfer across topics!'),
    ('Anchor Contribution:', ' Stat cards with anchors improve over pure demographics by -6.8%.'),
    ('Item Win-Rate:', ' The calibrated system beats the national baseline on 36 of 39 held-out items (92.3% win rate).'),
    ('Where It Loses (3 items):', ' Flat national spending items (natroad, natarms, natmass) where demographic groups barely differ.')
]
for title, desc in abl_points:
    p = tf_abl.add_paragraph()
    p.space_before = Pt(6)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 10: 09 | Prototype / Demonstration
# ----------------------------------------------------
clear_body_shapes(prs.slides[9])
s10 = prs.slides[9]

_, tf_proto = add_card(s10, Inches(0.55), Inches(0.95), Inches(8.9), Inches(2.6), 'Working Prototype & Query Execution', bg_color=LIGHT_BLUE)
p = tf_proto.add_paragraph()
style_text(p, 'Executable CLI and Streamlit web dashboard: simulate_region.py & M10_app', size=Pt(12), bold=True, color=HEADER_NAVY)

p_flow = tf_proto.add_paragraph()
p_flow.space_before = Pt(6)
p_flow.text = '• Path A (Observed Survey Question): If the query matches a GSS item (e.g., satfin: financial satisfaction), the router serves the empirical weighted cross-tab instantly (0 API calls, $0 cost, exact historical ground truth).\n• Path B (Unasked Policy Scenario): Elicits distributions across the 56 demographic cluster agents, applies cross-fitted anchor calibration, and rakes to ACS 2024 census weights to yield regional/national distributions with 90% confidence intervals.'
p_flow.font.size = Pt(11)
p_flow.font.color.rgb = DARK

_, tf_box = add_card(s10, Inches(0.55), Inches(3.75), Inches(8.9), Inches(3.1), 'The Lead Demo Item & Mandatory Honesty Box')
demo_items = [
    ('Lead Demo Item (natroad - Highway Spending):', ' Selected because it has a clean demographic age gradient (SNR 3.61) and has not been over-debated in national media (unlike pray or homosex). Reviewers cannot dismiss it as memorized training data.'),
    ('Mandatory Honesty Box Architecture:', ' Every rendered prediction must display: (1) The nearest validated survey item in semantic space, (2) Its measured W1 score, and (3) Its distance to the national baseline.'),
    ('Why This Matters:', ' Product and policy scenarios have no ground truth and never will. The Honesty Box ensures users know exactly how much confidence the underlying benchmark warrants.')
]
for title, desc in demo_items:
    p = tf_box.add_paragraph()
    p.space_before = Pt(6)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11.5)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(11)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 11: 10 | Current Limitations & Technical Challenges
# ----------------------------------------------------
clear_body_shapes(prs.slides[10])
s11 = prs.slides[10]

_, tf_chal = add_card(s11, Inches(0.55), Inches(0.95), Inches(4.3), Inches(5.8), 'Engineering Challenges Overcome', bg_color=LIGHT_BLUE)
chal_items = [
    ('Free-Tier Rate Limits:', ' Ministral 14B was capped at 30 req/min (vs 750 for 3B). Solved by building multi-provider failover and pacing queues.'),
    ('Shell Timeouts & Freezes:', ' Long batch runs froze cloud shells. Solved by chunking runs into atomic sub-tasks with SQLite prompt caching.'),
    ('GSS 2024 Data Defect:', ' GSS 2024 release left region_7222 0% populated, causing complete demographic case count n = 0. Solved by detecting the issue and pooling 2010–2022.'),
    ('Cloudflare Blocking Groq:', ' Groq API was blocked in our sandbox. Solved by dynamic failover to Mistral and Gemini.')
]
for title, desc in chal_items:
    p = tf_chal.add_paragraph()
    p.space_before = Pt(6)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

_, tf_lim = add_card(s11, Inches(5.05), Inches(0.95), Inches(4.4), Inches(5.8), 'Current Scientific Limitations')
lim_items = [
    ('Under-Dispersed Magnitude:', ' The system captures cluster ordering accurately (ρ = +0.651), but between-cluster SD is ~2x conservative (predicted 0.0550 vs true 0.0979) due to W1-minimizing shrinkage (s < 1).'),
    ('Single-Country Validation:', ' US General Social Survey is completed. Replication on India (Global Flourishing Study) is implemented but not yet executed.'),
    ('Prompt Paraphrasing:', ' Current headline results use 1 prompt paraphrase x 3 repeats. Robustness across 3 diverse paraphrases is scheduled for Stage B.'),
    ('Coarse Cluster Overfitting (K=18):', ' At very coarse partitions, out-of-fold selection overfits to small anchor counts. Resolved by partial pooling.')
]
for title, desc in lim_items:
    p = tf_lim.add_paragraph()
    p.space_before = Pt(6)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 12: 11 | Remaining Work & Completion Plan
# ----------------------------------------------------
clear_body_shapes(prs.slides[11])
s12 = prs.slides[11]

_, tf_stage_b = add_card(s12, Inches(0.55), Inches(0.95), Inches(4.3), Inches(5.8), 'Stage B Execution Tasks', bg_color=LIGHT_BLUE)
stage_b_items = [
    ('India GFS Replication:', ' Run pipeline on Global Flourishing Study (India n = 12,765, COUNTRY=6) to evaluate cross-cultural generalizability.'),
    ('Cross-Instrument Validation:', ' Benchmark against Pew OpinionQA (15 ATP waves, individual-level microdata).'),
    ('Paraphrase-Axis Ensemble:', ' Evaluate 3 hand-written question paraphrases x 3 repeats to test phrasing invariance.'),
    ('Opinion Dynamics (M8):', ' Model temporal attitude shifts across 35 GSS historical waves with time horizon as an explicit variable.')
]
for title, desc in stage_b_items:
    p = tf_stage_b.add_paragraph()
    p.space_before = Pt(8)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = DARK

_, tf_time = add_card(s12, Inches(5.05), Inches(0.95), Inches(4.4), Inches(5.8), 'Target Deliverables & Timeline')
timeline = [
    ('Weeks 1–4 (India & Cross-Instrument):', ' Complete GFS India replication run and Pew OpinionQA evaluation.'),
    ('Weeks 5–7 (Dynamics & Paraphrases):', ' Implement M8 opinion dynamics on repeated cross-sections; run paraphrase-axis ensemble.'),
    ('Weeks 8–10 (Deployment & Scenarios):', ' Finalize interactive Streamlit app with US and India policy scenario demonstrations.'),
    ('Weeks 11–12 (Documentation & Defense):', ' Complete final project report and prepare research manuscript based on paper/RESULTS.md.')
]
for title, desc in timeline:
    p = tf_time.add_paragraph()
    p.space_before = Pt(8)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 13: 12 | Project Contribution & Current Achievement
# ----------------------------------------------------
clear_body_shapes(prs.slides[12])
s13 = prs.slides[12]

_, tf_contr = add_card(s13, Inches(0.55), Inches(0.95), Inches(8.9), Inches(2.6), 'Defensible Project Contributions', bg_color=LIGHT_BLUE)
contr_items = [
    ('Distributional Cluster-Agent Architecture:', ' Replaces unscalable individual personas with 56 interpretable demographic clusters, recovering true population variance (1.009 ratio).'),
    ('Level/Deviation Calibration Mechanism:', ' Proves that LLM survey error is primarily a constant item level offset, and introduces an anchor-fitted subspace projection that cuts error by 69%.'),
    ('Pre-Registered 6-Gate Falsification Framework:', ' Pre-registered pass marks (-20% W1) with 3-number honest reporting (achieved, baseline, floor) and independent re-derivation.')
]
for desc_idx, (title, desc) in enumerate(contr_items):
    p = tf_contr.add_paragraph()
    p.space_before = Pt(4)
    r1 = p.add_run()
    r1.text = f'{desc_idx+1}. ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = DARK

_, tf_claims = add_card(s13, Inches(0.55), Inches(3.75), Inches(8.9), Inches(3.1), 'Scientific Honesty: What We Claim vs What We Do Not')
claims_items = [
    ('Six Spec Corrections Discovered by Real Data:', ' K=150 -> 56; GSS 2024 -> pooled 2010–2022; $1,200 budget -> $0 router; tuned router threshold 0.85 -> 0.25 (restoring recall from 0 to 81%); restricted subspace projection; dropped counter-productive variance expansion.'),
    ('Explicit Non-Claims:', ' We do NOT claim to "simulate complete human societies", "predict real-world elections", or "replace human surveys".'),
    ('What We Defensibly Claim:', ' Given demographic marginals and anchor survey data, calibrated cluster agents accurately estimate held-out subgroup opinion distributions where demographic groups genuinely differ.')
]
for title, desc in claims_items:
    p = tf_claims.add_paragraph()
    p.space_before = Pt(5)
    r1 = p.add_run()
    r1.text = '• ' + title + ' '
    r1.font.bold = True
    r1.font.size = Pt(11)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = desc
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 14: 13 | Team Contribution & Responsibility
# ----------------------------------------------------
clear_body_shapes(prs.slides[13])
s14 = prs.slides[13]

students = [
    ('Vishal S (CB.SC.U4AIE23160)', 'Data Ingestion & Clustering Lead', [
        'M1 Data Ingestion & Harmonization (GSS adapter, recodes)',
        'M2 Cluster Tree (K = 56) & Kish Effective Sample Size',
        'Layer 0 Verification: Published codebook matching (142/142 toplines)'
    ]),
    ('Nitin Krishna V (CB.SC.U4AIE23156)', 'Agent Design Lead', [
        'M3 Stat Card Builder & Fold Isolation Contract',
        'M4 LLM Distributional Elicitation Runner & Schemas',
        'Multi-provider API router, token guards, and SQLite caching'
    ]),
    ('Gaurav Mahesh (CB.SC.U4AIE23176)', 'Simulation & Integration Lead', [
        'M5 Cross-Fitted Calibration Layer (Level/Subspace)',
        'M6 Post-Stratification Raking to ACS 2024 Census Margins',
        'Layer 2 Permutation Test & Gate 3 Diagnostic Analysis'
    ]),
    ('Manogna Challa (CB.SC.U4AIE23175)', 'Analytics & Visualization Lead', [
        'M9 Evaluation Benchmark Harness (W1, JS, ECE, Coverage)',
        'M10 Streamlit Application & Mandatory Honesty Box',
        'Layer 5 Memorization & Direct Recall Leakage Probing'
    ])
]

for idx, (name, role, mods) in enumerate(students):
    left = Inches(0.55 + (idx % 2) * 4.5)
    top = Inches(0.95 + (idx // 2) * 2.95)
    _, tf_s = add_card(s14, left, top, Inches(4.3), Inches(2.8), name, bg_color=LIGHT_BLUE)
    p_role = tf_s.add_paragraph()
    p_role.space_before = Pt(2)
    style_text(p_role, f'Role: {role}', size=Pt(11), bold=True, color=ORANGE)
    for m in mods:
        p = tf_s.add_paragraph()
        p.space_before = Pt(2)
        r = p.add_run()
        r.text = '• ' + m
        r.font.size = Pt(9.5)
        r.font.color.rgb = DARK

# ----------------------------------------------------
# SLIDE 15: 14 | References
# ----------------------------------------------------
clear_body_shapes(prs.slides[14])
s15 = prs.slides[14]

_, tf_ref = add_card(s15, Inches(0.55), Inches(0.95), Inches(8.9), Inches(5.8), 'Key Academic References', bg_color=WHITE)
refs = [
    ('L. P. Argyle et al.,', ' "Out of One, Many: Using Language Models to Simulate Human Samples," Political Analysis, vol. 31, no. 3, pp. 337–351, 2023.'),
    ('J. Bisbee et al.,', ' "The Perils of Using Large Language Models to Replace Human Participants," Political Analysis, pp. 1–17, 2024.'),
    ('S. Santurkar et al.,', ' "Whose Opinions Do Language Models Reflect?," in International Conference on Machine Learning (ICML - OpinionQA), 2023.'),
    ('J. S. Park et al.,', ' "Generative Agents: Interactive Simulacra of Human Behavior," in Proc. ACM UIST, 2023.'),
    ('J. S. Park et al.,', ' "Generative Agent Simulations of 1,000 People," arXiv preprint arXiv:2411.10109, 2024.'),
    ('Y. Li et al.,', ' "EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities," in Proc. ACL, 2024.'),
    ('X. Pi et al.,', ' "AgentSociety: Large-Scale Simulation of Agent Societies," arXiv preprint arXiv:2502.08601, 2025.'),
    ('S. Chopra et al.,', ' "Large-Scale LLM-Archetype Agent-Based Modeling," arXiv preprint, 2025.')
]
for authors, cite in refs:
    p = tf_ref.add_paragraph()
    p.space_before = Pt(8)
    r1 = p.add_run()
    r1.text = authors
    r1.font.bold = True
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = HEADER_NAVY
    r2 = p.add_run()
    r2.text = cite
    r2.font.size = Pt(10)
    r2.font.color.rgb = DARK

prs.save(dst_path)
print('Successfully saved First_Review_Presentation.pptx!')
