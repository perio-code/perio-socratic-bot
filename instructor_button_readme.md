# Periodontal Socratic Faculty App

A Streamlit chat app that role-plays an expert periodontist / dental
faculty instructor, guiding a dental student through a periodontal case
using strict Socratic questioning (2017 World Workshop Classification,
McGuire & Nunn / Kwok & Caton prognosis literature, Segelnick & Weinberg
re-evaluation timelines, AAP referral guidelines). Faculty responses are
read aloud with a natural AI voice.

## Features
- **Socratic chat** with an AI attending faculty member (powered by Claude).
- **Natural voice** — faculty responses are read aloud (OpenAI TTS, with a
  free browser-voice fallback).
- **Case checklist** in the sidebar — tracks which of the 5 workflow phases
  (Case Entry → Diagnosis → Prognosis → Referral Decision → Wrap-up) the
  student is currently on, updated automatically as the conversation
  progresses.
- **Structured intake form** — an optional form (PD, CAL, furcation,
  mobility, smoking, HbA1c, radiographic bone loss) so the student can
  submit case data in a structured format instead of free-typing it.
- **File upload** — students can attach full-mouth periodontal charting
  and/or radiographs (PNG/JPG/WEBP images or PDF). The faculty AI can see
  these directly (vision), but stays Socratic about them — it asks the
  student to read out findings first rather than diagnosing from the
  images itself.
- **Instructor View** — a PIN-gated section in the sidebar (hidden from
  students by default) that gives the supervising instructor a private,
  AI-generated assessment of how the student is doing: ON_TRACK,
  NEEDS_HELP, or OFF_TASK, plus notes on engagement, comprehension,
  strengths, gaps, and a concrete recommendation. Generated on demand,
  not visible to the student.

## Files
- `app.py` — the full app
- `requirements.txt` — dependencies

## Setup

### 1. Get an Anthropic API key (required)
https://console.anthropic.com → API Keys

### 2. (Optional) Get an OpenAI API key for a natural voice
https://platform.openai.com/api-keys
If you skip this, the app falls back to your browser's free built-in
voice (still works, just less natural-sounding).

### 3. Run locally
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."   # optional
streamlit run app.py
```

### 4. Deploy on Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to your GitHub repo.
2. Go to https://share.streamlit.io → "New app" → point at your repo and `app.py`.
3. In the app's **Settings → Secrets** (NOT GitHub repo secrets — these
   are two different systems, and only Streamlit's own Secrets are
   visible to the running app), add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy. The "🔊 Read faculty responses aloud" toggle in the sidebar
   turns voice on/off.

## Notes
- Model used: `claude-sonnet-4-6`. Update the `MODEL` constant in `app.py`
  if you want to switch models later.
- Voice: OpenAI TTS model `tts-1`, voice `onyx` (deep, measured — good fit
  for a senior faculty tone). Change `OPENAI_TTS_VOICE` / `OPENAI_TTS_MODEL`
  in `app.py` to try others (e.g. `fable`, `tts-1-hd`).
- Use the "🔄 Restart case" button in the sidebar to reset the conversation.
- The phase checklist works by having Claude append a hidden `[PHASE:N]`
  tag to the end of every reply; the app parses and strips this before
  displaying or speaking the message, so it's invisible to the student.
- Uploaded images/PDFs are sent to Claude as part of that single chat
  turn (Claude's vision support), not stored anywhere outside the
  conversation.
- **Instructor View PIN**: defaults to `1234` if not configured. Set your
  own PIN via Streamlit Secrets (recommended, so it's not hardcoded in
  your repo):
  ```toml
  INSTRUCTOR_PIN = "your-own-pin-here"
  ```
  The Instructor View lives in a collapsed "🔒 Instructor View" expander
  near the bottom of the sidebar — students would need to know to look
  for it, and can't get past the PIN prompt without it.
