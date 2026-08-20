"""
AI Resume Analyzer - Streamlit UI
-----------------------------------
Upload a resume, get an AI-predicted job role and personalized interview
questions, all in a browser UI. Powered by Google's Gemini API.

Setup:
    pip install streamlit google-genai PyPDF2 python-docx

    Get a free API key: https://aistudio.google.com/apikey

Usage:
    streamlit run app.py
    (opens automatically in your browser at http://localhost:8501)
"""

import os
import json
import re
import io
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ---------- Page setup ----------

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")
st.title("📄 AI Resume Analyzer")
st.caption("Upload a resume → get an AI-predicted job role and personalized interview questions.")

# MODEL_NAME = "gemini-flash-latest"
MODEL_NAME = "gemini-flash-lite-latest"

# ---------- API key handling (sidebar input, no env var needed) ----------

with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    st.markdown("[Get a free API key](https://aistudio.google.com/apikey)")
    num_questions = st.slider("Number of interview questions", 4, 12, 8)

if not api_key_input:
    st.warning("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

client = genai.Client(api_key=api_key_input)


# ---------- Resume text extraction ----------

def extract_text(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext == ".pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(uploaded_file)
        return "".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".txt":
        return uploaded_file.read().decode("utf-8")

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------- JSON parsing helper ----------

def parse_json_response(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def call_gemini(prompt: str, max_tokens: int, retries: int = 3) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.4,
                    response_mime_type="application/json",
                ),
            )
            return response.text
        except genai_errors.ServerError as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))  # wait 2s, then 4s
                continue
            raise last_error


# ---------- AI calls ----------

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
    return parse_json_response(call_gemini(prompt, max_tokens=2000))


def generate_questions(resume_text: str, role_info: dict, n: int) -> list:
    prompt = f"""
You are a senior hiring manager preparing a personalized interview for one
specific candidate. Every question must be grounded in something concrete
from their resume — a project, a tool, a claim they made — not generic
textbook questions.

Candidate resume:
\"\"\"{resume_text}\"\"\"

Predicted role: {role_info['predicted_role']}
Reasoning: {role_info['reasoning']}

Generate {n} interview questions personalized to this exact candidate for
this exact role. Mix question types:
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
    return parse_json_response(call_gemini(prompt, max_tokens=4000))


# ---------- PDF report generation ----------

def generate_pdf_bytes(role_info: dict, questions: list, candidate_label: str = "Candidate") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
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

    story.append(Paragraph("Personalized Interview Questions", h2_style))
    type_labels = {"resume-specific": "Resume-specific", "technical": "Technical", "behavioral": "Behavioral"}

    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>{i}. {q['question']}</b>", body_style))
        story.append(Paragraph(
            f"<i>Type: {type_labels.get(q['type'], q['type'])} &mdash; {q['why_this_question']}</i>",
            label_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------- Main UI flow ----------

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    if st.button("Analyze Resume", type="primary"):
        try:
            with st.spinner("Reading resume..."):
                resume_text = extract_text(uploaded_file)

            if not resume_text.strip():
                st.error("Could not extract any text from this file.")
                st.stop()

            with st.spinner("Predicting job role via AI..."):
                role_info = predict_role(resume_text)

            with st.spinner("Generating personalized interview questions..."):
                questions = generate_questions(resume_text, role_info, num_questions)

            # ---- Display results ----
            st.divider()
            st.subheader("🎯 Predicted Role")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {role_info['predicted_role']}")
            with col2:
                confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                st.markdown(f"**Confidence:** {confidence_color.get(role_info['confidence'], '')} {role_info['confidence']}")

            st.markdown(f"**Reasoning:** {role_info['reasoning']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Key Strengths**")
                for s in role_info.get("key_strengths", []):
                    st.markdown(f"- {s}")
            with col2:
                if role_info.get("skill_gaps"):
                    st.markdown("**⚠️ Skill Gaps**")
                    for g in role_info["skill_gaps"]:
                        st.markdown(f"- {g}")

            if role_info.get("alternative_roles"):
                st.info(f"**Alternative roles to consider:** {', '.join(role_info['alternative_roles'])}")

            st.divider()
            st.subheader("💬 Personalized Interview Questions")

            type_badges = {
                "resume-specific": "📄 Resume-specific",
                "technical": "⚙️ Technical",
                "behavioral": "🧠 Behavioral",
            }

            for i, q in enumerate(questions, 1):
                badge = type_badges.get(q["type"], q["type"])
                with st.expander(f"{i}. {q['question']}"):
                    st.markdown(f"**Type:** {badge}")
                    st.markdown(f"**Why this question:** {q['why_this_question']}")

            # ---- Download buttons ----
            output = {"role_analysis": role_info, "interview_questions": questions}
            candidate_label = os.path.splitext(uploaded_file.name)[0]
            pdf_bytes = generate_pdf_bytes(role_info, questions, candidate_label)

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{candidate_label}_analysis.pdf",
                    mime="application/pdf",
                    type="primary",
                )
            with col2:
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json.dumps(output, indent=2),
                    file_name=f"{candidate_label}_analysis.json",
                    mime="application/json",
                )

        except json.JSONDecodeError:
            st.error("The AI response couldn't be parsed. Try again — this can happen occasionally.")
        except genai_errors.ServerError:
            st.error("Gemini's servers are experiencing high demand right now. Please wait a moment and click 'Analyze Resume' again.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.info("Upload a PDF, DOCX, or TXT resume to get started.")