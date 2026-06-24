# Periodontal Socratic Faculty App

A Streamlit chat app that role-plays an expert periodontist / dental
faculty instructor, guiding a dental student through a periodontal case
using strict Socratic questioning (2017 World Workshop Classification,
McGuire & Nunn / Kwok & Caton prognosis literature, Segelnick & Weinberg
re-evaluation timelines, AAP referral guidelines). Faculty responses are
read aloud with a natural AI voice.

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
3. In the app's **Settings → Secrets**, add:
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
