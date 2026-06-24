"""
Perio Socratic Faculty App
---------------------------
A Streamlit chat app that role-plays an expert Periodontist / Dental School
Faculty Instructor, guiding a dental student through a periodontal case
analysis using strict Socratic questioning (per the 2017 World Workshop
Classification, McGuire & Nunn / Kwok & Caton prognosis literature,
Segelnick & Weinberg re-evaluation timelines, and AAP referral guidelines).

The faculty member's questions are read aloud in a natural voice using
OpenAI's text-to-speech API (falls back to the browser's built-in
speech synthesis if no OpenAI key is configured).

ENV VARS REQUIRED (set as Streamlit secrets or environment variables):
    ANTHROPIC_API_KEY   - required, powers the Socratic faculty logic
    OPENAI_API_KEY      - optional, enables natural AI voice (TTS)
                           if absent, app falls back to free browser TTS

Run locally:
    pip install streamlit anthropic openai
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file + requirements.txt to your GitHub repo
    2. On share.streamlit.io, create a new app pointing at app.py
    3. In "Secrets", add:
         ANTHROPIC_API_KEY = "sk-ant-..."
         OPENAI_API_KEY = "sk-..."     # optional
"""

import base64
import os

import streamlit as st
from anthropic import Anthropic

# Optional import — app still runs (with browser TTS fallback) without it
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =========================================================================
# CONFIG
# =========================================================================

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

# Voice for OpenAI TTS. Options include: alloy, ash, ballad, coral, echo,
# fable, onyx, nova, sage, shimmer, verse. "onyx" or "fable" both read
# well as a senior, measured faculty voice.
OPENAI_TTS_VOICE = "onyx"
OPENAI_TTS_MODEL = "tts-1"  # use "tts-1-hd" for higher quality, slower/pricier

SYSTEM_PROMPT = """ROLE AND CONTEXT:
You are an expert Periodontist and Dental School Faculty Instructor. Your goal is to guide a dental student through a periodontal case analysis using strict Socratic questioning. Do not give away the diagnosis, prognosis, or treatment plan immediately. Act as an attending faculty member on the clinic floor—encourage critical thinking by asking guiding questions, highlighting discrepancies in their logic, and prompting them to discover the evidence-based conclusions themselves.

LITERATURE AND EVIDENCE-BASED CRITERIA:
You must hold the student accountable to these specific foundational literature standards:
1. 2017 World Workshop Classification (Tonetti et al. 2018; Papapanou et al. 2018) for Staging and Grading.
2. Prognosis Stability (McGuire & Nunn 1996; Kwok & Caton 2007) regarding tooth-by-tooth predictability.
3. Phase I Re-evaluation Timelines (Segelnick & Weinberg 2006): Re-evaluation must occur 4–6 weeks post-scaling and root planing (SRP) to allow for proper tissue healing, junctional epithelium formation, and pocket reduction.
4. AAP Guidelines for Periodontal Referral:
   - Level 1 (Must Refer Immediately): Stage III/IV, aggressive disease, Class II/III furcations, vertical bone defects, or uncontrolled systemic modifiers (e.g., HbA1c > 8%, smoking > 10 cigarettes/day).
   - Post-Phase I Referral (Must Refer After Re-evaluation): If, at the 4–6 week re-evaluation, the patient presents with non-responding sites characterized by persistent Probing Depths (PD) >= 6mm with active Bleeding on Probing (BOP), indicating the need for advanced Phase II surgical, resective, or regenerative therapy.
   - Level 2/3 (Can treat in student clinic): Stage I/II, gingivitis, probing depths <= 5mm with no complex anatomy, or patients who successfully resolve post-SRP.

SOCRATIC OPERATIONAL WORKFLOW (ONE STEP AT A TIME):
Do not lecture or provide full analyses. Ask one question at a time and wait for the student's response. If they make an error, do not give them the answer; instead, ask a question that directs their attention to their mistake.

Phase 1: Socratic Case Entry
- Greet the student formally: "Welcome to the clinic, Doctor. Let's look at your case."
- Prompt them to present their specific patient's data, ensuring they provide: Chief Complaint/Medical History, Max Probing Depths (PD) & Clinical Attachment Loss (CAL), Furcation/Mobility, and Radiographic Bone Loss (RBL).

Phase 2: Guided Diagnosis (Staging & Grading)
- Ask the student: "Based on the clinical parameters you just gathered, what is your definitive 2017 AAP/EFP diagnosis regarding Stage, Grade, and Extent?"
- Socratic Correction Rule: If a student misdiagnoses, ask a question that forces them to check the parameters. For example: "Look back at your patient's smoking habits. According to the Papapanou et al. 2018 consensus, how does smoking more than 10 cigarettes a day modify our grading?"

Phase 3: Tooth-by-Tooth Prognosis
- Prompt the student to evaluate compromised teeth: "Let's look at the individual teeth, specifically the molars. What is your prognosis for these teeth, and which classic periodontal studies support your predictability assessment?"

Phase 4: The Critical Referral Decision (Initial vs. Post-SRP)
- Ask the student to evaluate the appropriate referral pathway: "Based on the current data, should this patient be referred to a periodontist immediately, or is it appropriate to initiate Phase I (Etiotropic) therapy here in the undergraduate clinic first?"
- CRITICAL ERROR CONTROL (POST-SRP TIMELINE): If the case moves to Phase I therapy, skip forward to the re-evaluation milestone. Ask the student: "We have completed scaling and root planing, and the patient is back for their 4–6 week re-evaluation. You note persistent 6mm pockets with bleeding on probing in the maxillary molars. What does the literature tell us about these non-responding sites, and should this patient now be referred to a periodontist? Defend your decision."
- If the student attempts to perform Phase II periodontal surgery themselves or keep managing a non-responsive >= 6mm pocket in the student clinic, push back Socratically: "Review the anatomical limitations of root instrumentation and the criteria for surgical pocket elimination. Is a student clinic equipped to handle surgical therapy for non-responding 6mm pockets, or does the AAP dictate a specialist referral at this stage? Why?"

Phase 5: Treatment Sequencing & Wrap-up
- Once the referral path is correct, ask them to outline the maintenance or transition plan for the patient.
- Conclude by summarizing their clinical performance and giving faculty sign-off.

TONE AND STYLE:
Maintain a rigorous, academic, encouraging, yet highly precise clinical tone. Speak as a senior colleague guiding a junior colleague. Never give answers away—always make the student do the cognitive work.
"""


# =========================================================================
# API CLIENTS
# =========================================================================

def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key:
        st.error(
            "No ANTHROPIC_API_KEY found. Add it in Streamlit Secrets "
            "(Settings → Secrets) or as an environment variable."
        )
        st.stop()
    return Anthropic(api_key=api_key)


def get_openai_client():
    """Returns an OpenAI client if a key is configured, else None."""
    if not OPENAI_AVAILABLE:
        return None
    api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# =========================================================================
# CLAUDE CALL
# =========================================================================

def get_faculty_response(client, messages):
    """Send the conversation so far to Claude and return the faculty's reply."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# =========================================================================
# TEXT-TO-SPEECH
# =========================================================================

def synthesize_speech_openai(openai_client, text):
    """Calls OpenAI TTS and returns raw MP3 bytes, or None on failure."""
    try:
        result = openai_client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_TTS_VOICE,
            input=text,
        )
        return result.read()
    except Exception as e:
        st.warning(f"Voice synthesis failed, falling back to browser voice: {e}")
        return None


def autoplay_audio_html(mp3_bytes):
    """Returns an HTML <audio> tag that autoplays the given MP3 bytes."""
    b64 = base64.b64encode(mp3_bytes).decode()
    return f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """


def browser_tts_html(text):
    """
    Fallback: uses the browser's built-in SpeechSynthesis API.
    Free, no API key needed, works in virtually every modern browser.
    Picks a deeper/male-leaning voice where available for a 'faculty' feel.
    """
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f"""
        <script>
            (function() {{
                const utter = new SpeechSynthesisUtterance("{safe_text}");
                utter.rate = 0.95;
                utter.pitch = 0.9;
                const voices = window.speechSynthesis.getVoices();
                const preferred = voices.find(v => /male|david|daniel|fred/i.test(v.name));
                if (preferred) utter.voice = preferred;
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utter);
            }})();
        </script>
    """


def speak(text, openai_client):
    """Speaks the given text aloud using the best available TTS method."""
    if openai_client is not None:
        audio_bytes = synthesize_speech_openai(openai_client, text)
        if audio_bytes:
            st.components.v1.html(autoplay_audio_html(audio_bytes), height=0)
            return
    # Fallback to free browser speech synthesis
    st.components.v1.html(browser_tts_html(text), height=0)


# =========================================================================
# STREAMLIT APP
# =========================================================================

def main():
    st.set_page_config(page_title="Perio Clinic Faculty", page_icon="🦷", layout="centered")

    st.title("🦷 Periodontal Case Analysis — Clinic Floor")
    st.caption(
        "Socratic case-based learning with an AI attending faculty member. "
        "Present your patient data and defend your clinical reasoning."
    )

    anthropic_client = get_anthropic_client()
    openai_client = get_openai_client()

    with st.sidebar:
        st.header("Settings")
        voice_enabled = st.toggle("🔊 Read faculty responses aloud", value=True)
        if openai_client is not None:
            st.success("Using natural AI voice (OpenAI TTS).")
        else:
            st.info(
                "No OPENAI_API_KEY found — using your browser's built-in "
                "voice instead. Add an OPENAI_API_KEY secret for a more "
                "natural-sounding voice."
            )
        if st.button("🔄 Restart case"):
            st.session_state.clear()
            st.rerun()

    # Initialize conversation with the faculty's opening greeting
    if "messages" not in st.session_state:
        opening_line = (
            "Welcome to the clinic, Doctor. Let's look at your case. "
            "Please present your patient: chief complaint and medical history, "
            "maximum probing depths (PD) and clinical attachment loss (CAL), "
            "furcation involvement and mobility, and radiographic bone loss (RBL)."
        )
        st.session_state.messages = [{"role": "assistant", "content": opening_line}]
        st.session_state.spoken_count = 0  # tracks which assistant turns have been spoken

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🦷" if msg["role"] == "assistant" else "🧑‍⚕️"):
            st.markdown(msg["content"])

    # Speak any not-yet-spoken assistant messages (handles the opening line + new replies)
    assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if voice_enabled and st.session_state.spoken_count < len(assistant_msgs):
        for m in assistant_msgs[st.session_state.spoken_count:]:
            speak(m["content"], openai_client)
        st.session_state.spoken_count = len(assistant_msgs)

    # Chat input
    user_input = st.chat_input("Present your case findings or respond to the question above...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🦷"):
            with st.spinner("Faculty is reviewing your response..."):
                reply = get_faculty_response(anthropic_client, st.session_state.messages)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

        if voice_enabled:
            speak(reply, openai_client)
        st.session_state.spoken_count = len(
            [m for m in st.session_state.messages if m["role"] == "assistant"]
        )


if __name__ == "__main__":
    main()
