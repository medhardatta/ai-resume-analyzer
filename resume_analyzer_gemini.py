"""
AI Resume Analyzer (Gemini version - new google-genai SDK)
-------------------------------------------------------------
Predicts the best-fit job role for a candidate purely via LLM reasoning
(no keyword/dictionary matching) and generates custom interview questions
tailored to that specific resume — powered by Google's Gemini API.

Setup:
    pip install google-genai PyPDF2 python-docx

    Get a free API key: https://aistudio.google.com/apikey
    $env:GEMINI_API_KEY="your-key-here"      (PowerShell, current session)
    setx GEMINI_API_KEY "your-key-here"      (PowerShell, permanent - reopen terminal after)

Usage:
    python resume_analyzer_gemini.py
    (it will prompt you for the resume file path)
"""

import sys
import os
import json
import re
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Get a free key at https://aistudio.google.com/apikey")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
# MODEL_NAME = "gemini-flash-latest"  # always points to the current recommended flash model
# MODEL_NAME = "gemini-2.5-flash-lite"
MODEL_NAME = "gemini-flash-lite-latest"

# ---------- 1. Resume text extraction ----------

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        import PyPDF2
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        import docx
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------- helper: pull JSON out of a model response safely ----------

def parse_json_response(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("\n--- Could not parse model response as JSON ---")
        print(f"Error: {e}")
        print("--- Raw response was ---")
        print(raw_text)
        print("--- end raw response ---\n")
        raise


def call_gemini(prompt: str, max_tokens: int = 1500, retries: int = 3) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.4,
                    response_mime_type="application/json",  # forces valid JSON output
                ),
            )
            return response.text
        except genai_errors.ServerError as e:
            last_error = e
            if attempt < retries - 1:
                print(f"Gemini servers busy (attempt {attempt + 1}/{retries}), retrying...")
                time.sleep(3 * (attempt + 1))  # wait 3s, then 6s
                continue
            raise last_error


# ---------- 2. AI-based job role prediction (no rules/dictionaries) ----------

def predict_role(resume_text: str) -> dict:
    prompt = f"""
You are an expert technical recruiter and career advisor. Analyze resumes
purely on their substantive content — projects, skills demonstrated,
experience, tools used, and impact — and infer the single job role the
candidate is best suited for right now. Do not rely on fixed categories;
reason freely about what role fits best.

Resume:
\"\"\"{resume_text}\"\"\"

Return JSON in exactly this shape:
{{
  "predicted_role": "string - the specific job title",
  "confidence": "High | Medium | Low",
  "reasoning": "2-4 sentences explaining WHY, citing specific things from the resume",
  "key_strengths": ["list", "of", "3-5 standout skills/projects"],
  "skill_gaps": ["list", "of", "1-3 things missing for this role, if any"],
  "alternative_roles": ["1-2 other roles this candidate could also target"]
}}
"""
    raw = call_gemini(prompt, max_tokens=2000)
    return parse_json_response(raw)


# ---------- 3. AI-generated interview questions (dynamic, per-resume) ----------

def generate_questions(resume_text: str, role_info: dict, num_questions: int = 8) -> list:
    prompt = f"""
You are a senior hiring manager preparing a personalized interview for one
specific candidate. Every question must be grounded in something concrete
from their resume — a project, a tool, a claim they made — not generic
textbook questions.

Candidate resume:
\"\"\"{resume_text}\"\"\"

Predicted role: {role_info['predicted_role']}
Reasoning: {role_info['reasoning']}

Generate {num_questions} interview questions personalized to this exact
candidate for this exact role. Mix question types:
- a few that probe specific projects/claims on their resume
- a few technical/role-specific questions calibrated to their apparent skill level
- 1-2 behavioral questions relevant to gaps or transitions in their background

Return JSON in exactly this shape:
[
  {{
    "question": "string",
    "type": "resume-specific | technical | behavioral",
    "why_this_question": "one sentence on what it's probing for"
  }}
]
"""
    raw = call_gemini(prompt, max_tokens=4000)
    return parse_json_response(raw)


# ---------- PDF report generation ----------

def generate_pdf_report(role_info: dict, questions: list, out_path: str, candidate_label: str = "Candidate"):
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20, spaceAfter=6)
    h2_style = ParagraphStyle("H2Custom", parent=styles["Heading2"], spaceBefore=16, spaceAfter=8,
                               textColor=colors.HexColor("#1a3d7c"))
    body_style = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=6)
    label_style = ParagraphStyle("LabelCustom", parent=styles["Normal"], fontSize=10.5, leading=15,
                                  textColor=colors.HexColor("#444444"))

    story = []

    story.append(Paragraph("AI Resume Analysis Report", title_style))
    story.append(Paragraph(candidate_label, styles["Normal"]))
    story.append(Spacer(1, 14))

    # Role prediction section
    story.append(Paragraph("Predicted Role", h2_style))
    role_table = Table([
        ["Role", role_info["predicted_role"]],
        ["Confidence", role_info["confidence"]],
    ], colWidths=[1.3 * inch, 4.7 * inch])
    role_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
    ]))
    story.append(role_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Reasoning:</b> " + role_info["reasoning"], body_style))

    if role_info.get("key_strengths"):
        story.append(Paragraph("<b>Key Strengths</b>", label_style))
        for s in role_info["key_strengths"]:
            story.append(Paragraph(f"&bull; {s}", body_style))

    if role_info.get("skill_gaps"):
        story.append(Paragraph("<b>Skill Gaps</b>", label_style))
        for g in role_info["skill_gaps"]:
            story.append(Paragraph(f"&bull; {g}", body_style))

    if role_info.get("alternative_roles"):
        story.append(Paragraph(
            "<b>Alternative Roles:</b> " + ", ".join(role_info["alternative_roles"]), body_style))

    # Interview questions section
    story.append(Paragraph("Personalized Interview Questions", h2_style))
    type_labels = {"resume-specific": "Resume-specific", "technical": "Technical", "behavioral": "Behavioral"}

    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>{i}. {q['question']}</b>", body_style))
        story.append(Paragraph(
            f"<i>Type: {type_labels.get(q['type'], q['type'])} &mdash; {q['why_this_question']}</i>",
            label_style))
        story.append(Spacer(1, 6))

    doc.build(story)


# ---------- 4. Orchestration ----------

def analyze_resume(file_path: str):
    print(f"\nReading resume: {file_path}")
    resume_text = extract_text(file_path)

    if not resume_text.strip():
        print("Could not extract any text from this file.")
        return

    print("Predicting job role via AI...")
    role_info = predict_role(resume_text)

    print("\n" + "=" * 60)
    print(f"PREDICTED ROLE: {role_info['predicted_role']}")
    print(f"CONFIDENCE: {role_info['confidence']}")
    print("=" * 60)
    print(f"\nReasoning:\n{role_info['reasoning']}")
    print(f"\nKey strengths:")
    for s in role_info['key_strengths']:
        print(f"  - {s}")
    if role_info.get('skill_gaps'):
        print(f"\nSkill gaps:")
        for g in role_info['skill_gaps']:
            print(f"  - {g}")
    if role_info.get('alternative_roles'):
        print(f"\nAlternative roles to consider: {', '.join(role_info['alternative_roles'])}")

    print("\nGenerating personalized interview questions via AI...")
    questions = generate_questions(resume_text, role_info)

    print("\n" + "=" * 60)
    print("INTERVIEW QUESTIONS")
    print("=" * 60)
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. [{q['type']}] {q['question']}")
        print(f"   -> {q['why_this_question']}")

    output = {"role_analysis": role_info, "interview_questions": questions}
    json_out_path = os.path.splitext(file_path)[0] + "_analysis.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    pdf_out_path = os.path.splitext(file_path)[0] + "_analysis.pdf"
    candidate_label = os.path.splitext(os.path.basename(file_path))[0]
    generate_pdf_report(role_info, questions, pdf_out_path, candidate_label)

    print(f"\nJSON results saved to: {json_out_path}")
    print(f"PDF report saved to: {pdf_out_path}")


if __name__ == "__main__":
    file_path = input("Enter the path to your resume (pdf/docx/txt): ").strip().strip('"')

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    analyze_resume(file_path)