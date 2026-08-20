# AI Resume Analyzer

An AI-powered resume analysis tool that predicts a candidate's best-fit job role and generates personalized interview questions — using Google's Gemini API instead of rule-based keyword matching.

## Features

- **AI-driven role prediction** — analyzes resume content holistically (projects, skills, experience) to predict the single best-fit job role, with confidence scoring and reasoning
- **Personalized interview questions** — generates questions grounded in the candidate's actual resume (specific projects, tools, claims), mixing technical, behavioral, and resume-specific question types
- **Two interfaces**:
  - `app.py` — Streamlit web UI with file upload and downloadable PDF/JSON reports
  - `resume_analyzer_gemini.py` — command-line version for quick local use
- **PDF report export** via ReportLab
- Supports PDF, DOCX, and TXT resume formats

## Tech Stack

- Python
- Google Gemini API (`google-genai` SDK)
- Streamlit
- ReportLab (PDF generation)
- PyPDF2 / python-docx (resume text extraction)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/medhardatta/ai-resume-analyzer.git
   cd ai-resume-analyzer
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1      # Windows PowerShell
   pip install -r requirements.txt
   ```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

### Web UI (recommended)
```bash
streamlit run app.py
```
Paste your Gemini API key into the sidebar, upload a resume, and click "Analyze Resume."

### Command line
```bash
python resume_analyzer_gemini.py
```
Set your API key first:
```bash
$env:GEMINI_API_KEY="your-key-here"
```
Then enter the resume file path when prompted. Results are saved as JSON and PDF next to the resume file.

## How It Works

1. Resume text is extracted from the uploaded PDF/DOCX/TXT file
2. A structured prompt sends the resume text to Gemini, requesting a JSON response with the predicted role, confidence, reasoning, strengths, and skill gaps
3. A second prompt uses the predicted role and resume content to generate custom interview questions grounded in the candidate's specific experience
4. Results are displayed in the UI and exportable as a formatted PDF report

## Notes

- This project uses Gemini's free tier via API — no cost to run
- Each user supplies their own API key; no keys are stored or shared
- Role prediction and question generation are entirely LLM-driven (prompt engineering), not a custom-trained classification model

   ## License
   
   This project is licensed under the [MIT License](LICENSE).
