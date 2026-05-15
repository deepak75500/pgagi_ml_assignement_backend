"""
PDF Export Module — Generate a rich, professional interview summary report.
Includes: session metadata, full resume profile, performance metrics,
per-question Q&A with scores, key concepts, and missed concepts.

FIX: Guarded every None-unsafe format call:
  - created_at: falls back to datetime.now() if None/0/invalid
  - _fmt_one_decimal: already safe, but callers now never pass raw None to f-strings
  - score, t_sec, follow_up, feedback: all guarded before use in f-strings
  - diff / q_type / topic: str() coerced so .title() never hits None
"""
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE_DARK    = "#1a3a5c"
BLUE_MID     = "#2c5aa0"
BLUE_LIGHT   = "#e8f0f7"
GREEN_DARK   = "#1e5e2a"
GREEN_LIGHT  = "#e6f4ea"
AMBER_DARK   = "#7a5c00"
AMBER_LIGHT  = "#fff8e1"
RED_DARK     = "#7d1f1f"
RED_LIGHT    = "#fdecea"
GREY_TEXT    = "#555555"
GREY_BORDER  = "#cccccc"
WHITE        = "#ffffff"
PAGE_BG      = "#f9fafb"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(value: Any, default=None):
    """Best-effort numeric coercion for optional export fields."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_one_decimal(value: Any, suffix: str = "", missing: str = "N/A") -> str:
    number = _to_float(value)
    if number is None:
        return missing
    return f"{number:.1f}{suffix}"


def _safe_str(value: Any, fallback: str = "") -> str:
    """Return str(value).strip() or fallback if value is None/empty."""
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _safe_title(value: Any, fallback: str = "") -> str:
    return _safe_str(value, fallback).title()


def _score_palette(score: float):
    """Return (bg_hex, fg_hex) based on score out of 10."""
    if score >= 7.5:
        return GREEN_LIGHT, GREEN_DARK
    if score >= 5.0:
        return AMBER_LIGHT, AMBER_DARK
    return RED_LIGHT, RED_DARK


def _parse_timestamp(value: Any) -> str:
    """
    Convert a Unix timestamp (int/float/str) to a formatted date string.
    Returns a formatted string, or falls back to current time if value is
    None, 0, or unparseable — preventing NoneType.__format__ crashes.
    """
    ts = _to_float(value)
    if not ts:                          # None, 0, or negative → use now
        return datetime.now().strftime("%d %b %Y  %H:%M")
    try:
        return datetime.fromtimestamp(ts).strftime("%d %b %Y  %H:%M")
    except (OSError, OverflowError, ValueError):
        return datetime.now().strftime("%d %b %Y  %H:%M")


# ── Main export function ──────────────────────────────────────────────────────

def generate_summary_pdf(data: Dict[str, Any]) -> bytes:
    """
    Generate a professional PDF report from session export data.
    Requires: pip install reportlab
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, PageBreak, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        raise RuntimeError("reportlab not installed. Run: pip install reportlab")

    # ── Document setup ───────────────────────────────────────────────────────
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    PAGE_W = A4[0] - 1.5 * inch   # usable width

    # ── Styles ───────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    def _style(name, **kwargs):
        return ParagraphStyle(name, parent=styles["Normal"], **kwargs)

    title_style  = _style("Title",  fontSize=22, textColor=colors.HexColor(BLUE_DARK),
                           alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold")
    sub_style    = _style("Sub",    fontSize=11, textColor=colors.HexColor(GREY_TEXT),
                           alignment=TA_CENTER, spaceAfter=14)
    h2_style     = _style("H2",     fontSize=13, textColor=colors.HexColor(BLUE_MID),
                           fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6)
    h3_style     = _style("H3",     fontSize=11, textColor=colors.HexColor(BLUE_DARK),
                           fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    body_style   = _style("Body",   fontSize=9,  textColor=colors.black, leading=13)
    small_style  = _style("Small",  fontSize=8,  textColor=colors.HexColor(GREY_TEXT), leading=11)
    label_style  = _style("Label",  fontSize=8,  textColor=colors.HexColor(BLUE_DARK),
                           fontName="Helvetica-Bold")
    tag_style    = _style("Tag",    fontSize=8,  textColor=colors.HexColor(GREEN_DARK))
    miss_style   = _style("Miss",   fontSize=8,  textColor=colors.HexColor(RED_DARK))
    score_good   = _style("SGood",  fontSize=13, textColor=colors.HexColor(GREEN_DARK),
                           fontName="Helvetica-Bold", alignment=TA_CENTER)
    score_mid    = _style("SMid",   fontSize=13, textColor=colors.HexColor(AMBER_DARK),
                           fontName="Helvetica-Bold", alignment=TA_CENTER)
    score_bad    = _style("SBad",   fontSize=13, textColor=colors.HexColor(RED_DARK),
                           fontName="Helvetica-Bold", alignment=TA_CENTER)
    footer_style = _style("Footer", fontSize=7,  textColor=colors.grey, alignment=TA_CENTER)

    # ── Helper: info table ───────────────────────────────────────────────────
    def _info_table(rows: List, col_widths=None):
        col_widths = col_widths or [1.6 * inch, PAGE_W - 1.6 * inch]
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1),  colors.HexColor(BLUE_LIGHT)),
            ("TEXTCOLOR",     (0, 0), (-1, -1), colors.black),
            ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor(GREY_BORDER)),
            ("ROWBACKGROUNDS",(1, 0), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
        ]))
        return t

    # ── Data extraction ──────────────────────────────────────────────────────
    session_data = data.get("session") or {}
    resume_data  = data.get("resume") or {}
    performance  = data.get("performance") or {}
    qa_records   = data.get("qa_records") or []

    candidate   = _safe_str(session_data.get("candidate"), "Unknown")
    role        = _safe_str(session_data.get("role"), "Unknown")
    status      = _safe_str(session_data.get("status"), "N/A").upper()
    session_id  = _safe_str(session_data.get("id"), "N/A")
    created_str = _parse_timestamp(session_data.get("created_at"))   # ← FIX

    avg_score   = _to_float(performance.get("average_score"), 0.0)
    answered    = int(performance.get("answered") or 0)
    total_q     = int(performance.get("total_questions") or 0)
    avg_time    = _to_float(performance.get("average_time_seconds"), 0.0)

    # ── Build story ──────────────────────────────────────────────────────────
    story = []

    # ── Cover block ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("AI Interview Summary Report", title_style))
    story.append(Paragraph(f"{candidate}  ·  {role}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor(BLUE_MID), spaceAfter=10))

    info_rows = [
        [Paragraph("Session ID", label_style), Paragraph(session_id, body_style)],
        [Paragraph("Date",       label_style), Paragraph(created_str, body_style)],
        [Paragraph("Role",       label_style), Paragraph(role, body_style)],
        [Paragraph("Status",     label_style), Paragraph(status, body_style)],
    ]
    story.append(_info_table(info_rows))
    story.append(Spacer(1, 0.25 * inch))

    # ── Performance Scorecard ─────────────────────────────────────────────────
    story.append(Paragraph("Performance Scorecard", h2_style))

    bg, fg = _score_palette(avg_score or 0.0)
    score_st = (score_good if (avg_score or 0) >= 7.5 else
                score_mid  if (avg_score or 0) >= 5.0 else score_bad)

    perf_table_data = [
        ["Metric", "Value"],
        ["Questions Answered", f"{answered} / {total_q}"],
        ["Average Score",      _fmt_one_decimal(avg_score, " / 10", "N/A")],
        ["Avg Response Time",  _fmt_one_decimal(avg_time, " s", "N/A")],
    ]
    pt = Table(perf_table_data, colWidths=[PAGE_W * 0.55, PAGE_W * 0.45])
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor(BLUE_MID)),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor(GREY_BORDER)),
        ("ROWBACKGROUNDS",(1, 0), (-1, -1), [colors.white, colors.HexColor(BLUE_LIGHT)]),
        ("BACKGROUND",    (1, 2), (1, 2),   colors.HexColor(bg)),
        ("TEXTCOLOR",     (1, 2), (1, 2),   colors.HexColor(fg)),
        ("FONTNAME",      (1, 2), (1, 2),   "Helvetica-Bold"),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.25 * inch))

    # ── Resume Profile ────────────────────────────────────────────────────────
    if resume_data:
        story.append(Paragraph("Resume Profile", h2_style))

        skills    = resume_data.get("skills") or []
        techs     = resume_data.get("technologies") or []
        domains   = resume_data.get("domains") or []
        education = resume_data.get("education") or []
        exp_years = _to_float(resume_data.get("experience_years"))
        seniority = _safe_title(resume_data.get("seniority_level"), "Unknown")
        raw_prev  = _safe_str(resume_data.get("raw_text_preview"), "")
        preview   = raw_prev[:400]

        resume_rows = [
            [Paragraph("Experience",  label_style),
             Paragraph(_fmt_one_decimal(exp_years, " years", "N/A"), body_style)],
            [Paragraph("Seniority",   label_style),
             Paragraph(seniority, body_style)],
            [Paragraph("Skills",      label_style),
             Paragraph(", ".join(skills) if skills else "—", body_style)],
            [Paragraph("Technologies",label_style),
             Paragraph(", ".join(techs) if techs else "—", body_style)],
            [Paragraph("Domains",     label_style),
             Paragraph(", ".join(domains) if domains else "—", body_style)],
            [Paragraph("Education",   label_style),
             Paragraph(", ".join(education) if education else "—", body_style)],
        ]
        if preview:
            ellipsis = "…" if len(raw_prev) > 400 else ""
            resume_rows.append([
                Paragraph("Resume Preview", label_style),
                Paragraph(preview + ellipsis, small_style),
            ])

        story.append(_info_table(resume_rows))
        story.append(Spacer(1, 0.25 * inch))

    # ── Detailed Q&A Analysis ─────────────────────────────────────────────────
    if qa_records:
        story.append(PageBreak())
        story.append(Paragraph("Detailed Q&A Analysis", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                 color=colors.HexColor(GREY_BORDER), spaceAfter=8))

        for qa in qa_records:
            # ── Safe field extraction ────────────────────────────────────────
            idx       = _safe_str(qa.get("index"), "?")
            topic     = _safe_str(qa.get("topic"), "Unknown")
            diff      = _safe_title(qa.get("difficulty"), "")          # ← FIX
            q_type    = _safe_title(qa.get("question_type"), "")       # ← FIX
            question  = _safe_str(qa.get("question"), "")
            answer    = _safe_str(qa.get("answer"), "— Not answered —")
            score     = _to_float(qa.get("score"))                     # may be None
            feedback  = _safe_str(qa.get("feedback"), "")
            key_con   = qa.get("key_concepts") or []
            miss_con  = qa.get("missed_concepts") or []
            t_sec     = _to_float(qa.get("time_seconds"))              # may be None
            follow_up = _safe_str(qa.get("follow_up"), "")

            # ── Score display ────────────────────────────────────────────────
            score_str = _fmt_one_decimal(score, "/10", "N/A")          # ← FIX
            score_val = score if score is not None else 0.0
            sb_c, sf_c = _score_palette(score_val)
            sc_style   = (score_good if score_val >= 7.5 else
                          score_mid  if score_val >= 5.0 else score_bad)

            # ── Question header row ──────────────────────────────────────────
            meta_str = "  ·  ".join(filter(None, [diff, q_type]))      # ← FIX: skip blanks
            hdr_data = [[
                Paragraph(f"Q{idx}: {topic}", h3_style),
                Paragraph(meta_str, small_style),
                Paragraph(score_str, sc_style),
            ]]
            hdr_t = Table(hdr_data, colWidths=[PAGE_W * 0.55, PAGE_W * 0.25, PAGE_W * 0.20])
            hdr_t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor(BLUE_LIGHT)),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("BACKGROUND",    (2, 0), (2, 0),   colors.HexColor(sb_c)),
                ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor(GREY_BORDER)),
            ]))
            story.append(hdr_t)

            story.append(Spacer(1, 3))
            story.append(Paragraph(f"<b>Question:</b> {question}", body_style))

            story.append(Spacer(1, 3))
            story.append(Paragraph(f"<b>Answer:</b> {answer}", body_style))

            if t_sec is not None:                                       # ← FIX
                story.append(Paragraph(
                    f"<i>Response time: {t_sec:.1f}s</i>", small_style))

            if feedback:
                story.append(Spacer(1, 3))
                story.append(Paragraph(f"<b>Feedback:</b> {feedback}", body_style))

            if key_con:
                story.append(Spacer(1, 3))
                story.append(Paragraph(
                    "<b>✓ Concepts Covered:</b> " + "  ·  ".join(key_con), tag_style))

            if miss_con:
                story.append(Spacer(1, 3))
                story.append(Paragraph(
                    "<b>✗ Missed Concepts:</b> " + "  ·  ".join(miss_con), miss_style))

            if follow_up:
                story.append(Spacer(1, 3))
                story.append(Paragraph(
                    f"<i>Follow-up hint: {follow_up}</i>", small_style))

            story.append(Spacer(1, 0.15 * inch))
            story.append(HRFlowable(width="100%", thickness=0.4,
                                     color=colors.HexColor(GREY_BORDER), spaceAfter=6))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor(GREY_BORDER), spaceAfter=4))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%d %b %Y %H:%M:%S')}  ·  "
        "PGAGI AI Screening System",
        footer_style,
    ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()