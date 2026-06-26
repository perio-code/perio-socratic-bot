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
import re

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

# PIN to unlock the Instructor View in the sidebar. Change this, or better,
# set it via Streamlit Secrets as INSTRUCTOR_PIN so it isn't hardcoded in
# source control. Falls back to this default only if no secret is set.
DEFAULT_INSTRUCTOR_PIN = "1234"

SYSTEM_PROMPT = """IMPORTANT DISCLAIMER (ALWAYS IN EFFECT):
This is an educational simulation only, designed to help dental students practice diagnostic reasoning. You are not providing real medical or dental advice, diagnosis, or treatment for any actual patient. If the student indicates at any point that the "case" is in fact a real, current patient of theirs, gently remind them that this tool is for educational practice only and that any real patient must be managed by a licensed clinician using their own independent clinical judgment — then continue the Socratic exercise as a hypothetical/practice case if they wish to proceed.

ROLE AND CONTEXT:
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

CLINICAL IMAGES (PERIODONTAL CHARTING / RADIOGRAPHS):
The student may attach images of full-mouth periodontal charting and/or radiographs. When images are present, do NOT diagnose or interpret them for the student. Instead, ask the student to read out the relevant findings themselves first (e.g., "Looking at the chart you've attached, what are the deepest probing depths you see, and where?"). Use the images only to verify or gently challenge the student's own stated interpretation — for example, if their stated PD doesn't match what you can see, ask a guiding question that sends them back to look again, rather than correcting them directly.

PHASE TRACKING (REQUIRED, MACHINE-READABLE TAG):
At the very end of every single response, on its own new line, output a machine-readable tag indicating which phase of the workflow this response belongs to, in the exact format: [PHASE:N] where N is 1, 2, 3, 4, or 5, corresponding to the five phases above. This tag is for the clinic's progress-tracking software and is stripped before the student sees your message — it does not break character and is not visible to the student, so always include it, exactly once, at the very end.
"""

INSTRUCTOR_SUMMARY_PROMPT = """You are a teaching-assistant analytics engine for a dental school's periodontal Socratic-case simulator. You are NOT the faculty persona the student talked to — you are a separate, candid evaluator producing a private report for the supervising instructor only. The student will never see this report.

You will be given the full transcript of a Socratic dialogue between a dental student and an AI faculty member. Your job is to assess the STUDENT's performance and engagement — not to grade the AI faculty member.

Evaluate:
1. **Engagement status** — is the student actively engaging with the clinical reasoning task, or are they off-task (e.g., one-word non-answers, irrelevant tangents, trying to get the AI to just give them the answer, repeated stalling)?
2. **Comprehension status** — are they correctly applying the literature (2017 World Workshop Classification, McGuire & Nunn / Kwok & Caton prognosis criteria, Segelnick & Weinberg re-evaluation timeline, AAP referral guidelines), or do they show repeated/significant misunderstanding even after Socratic correction?
3. **Specific strengths** — what did the student get right or reason through well?
4. **Specific gaps** — what concepts did they struggle with, get wrong, or need to be redirected on? Be concrete (cite the actual exchange if possible).
5. **Overall status** — choose exactly one: "ON_TRACK", "NEEDS_HELP", or "OFF_TASK".
   - ON_TRACK: actively engaged, reasoning soundly, minor or no corrections needed.
   - NEEDS_HELP: engaged and trying, but showing real conceptual gaps or repeated errors on key literature/criteria even after Socratic prompting.
   - OFF_TASK: not meaningfully engaging with the clinical task (non-answers, derailing, refusing to engage, trying to bypass the exercise).

Respond ONLY in the following format, with no preamble or extra commentary:

STATUS: <ON_TRACK | NEEDS_HELP | OFF_TASK>
ENGAGEMENT: <1-2 sentences>
COMPREHENSION: <1-2 sentences>
STRENGTHS: <1-2 sentences, or "None observed yet" if too early to tell>
GAPS: <1-2 sentences, or "None observed yet" if too early to tell>
RECOMMENDATION: <1-2 sentences of concrete next-step advice for the instructor>

If the transcript is too short to assess meaningfully (e.g., only the opening greeting with no student response yet), respond with exactly:
STATUS: INSUFFICIENT_DATA
ENGAGEMENT: Not enough conversation yet to assess.
COMPREHENSION: Not enough conversation yet to assess.
STRENGTHS: None observed yet.
GAPS: None observed yet.
RECOMMENDATION: Check back after the student has responded to at least one or two faculty questions.
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


def get_instructor_pin():
    """Returns the configured instructor PIN (secrets/env override the default)."""
    return st.secrets.get(
        "INSTRUCTOR_PIN", os.environ.get("INSTRUCTOR_PIN", DEFAULT_INSTRUCTOR_PIN)
    )


# =========================================================================
# CLAUDE CALL
# =========================================================================

def get_faculty_response(client, messages):
    """Send the conversation so far to Claude and return the faculty's reply."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        st.error(
            f"The faculty AI couldn't respond — there may be an issue with your "
            f"ANTHROPIC_API_KEY or the API request. Details: {e}"
        )
        st.stop()


def get_instructor_summary(client, messages):
    """
    Sends the full conversation (including any attached images/PDFs, for
    full context) to Claude under a separate evaluator system prompt and
    returns the raw structured assessment text. Strips the trailing
    [PHASE:N] tags from prior assistant turns first, since those are
    irrelevant noise for the evaluator.
    """
    cleaned_messages = []
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], str):
            clean_text, _ = extract_phase_tag(m["content"])
            cleaned_messages.append({"role": "assistant", "content": clean_text})
        else:
            cleaned_messages.append(m)

    # Wrap the transcript as a single evaluation request so it's clearly
    # distinguished from "continue the roleplay."
    eval_request = {
        "role": "user",
        "content": (
            "Here is the full transcript above between the student and the "
            "AI faculty persona. Please produce your private instructor "
            "assessment now, following the required format exactly."
        ),
    }

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=INSTRUCTOR_SUMMARY_PROMPT,
            messages=cleaned_messages + [eval_request],
        )
        return response.content[0].text
    except Exception as e:
        return f"STATUS: ERROR\nCould not generate summary: {e}"


def parse_instructor_summary(raw_text):
    """Parses the structured STATUS/ENGAGEMENT/etc. fields into a dict."""
    fields = ["STATUS", "ENGAGEMENT", "COMPREHENSION", "STRENGTHS", "GAPS", "RECOMMENDATION"]
    result = {f: "" for f in fields}
    pattern = "|".join(fields)
    matches = list(re.finditer(rf"({pattern}):\s*", raw_text))
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        result[key] = raw_text[start:end].strip()
    return result


STATUS_DISPLAY = {
    "ON_TRACK": ("🟢", "On Track"),
    "NEEDS_HELP": ("🟡", "Needs Help"),
    "OFF_TASK": ("🔴", "Off Task"),
    "INSUFFICIENT_DATA": ("⚪", "Not Enough Data Yet"),
    "ERROR": ("⚠️", "Error Generating Summary"),
}


def encode_image_for_claude(uploaded_file):
    """
    Converts a Streamlit UploadedFile (image) into an Anthropic API
    image content block. Returns None for unsupported types (e.g. PDFs
    are handled separately as document blocks).
    """
    mime = uploaded_file.type or "image/png"
    if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        return None
    uploaded_file.seek(0)
    data = base64.b64encode(uploaded_file.read()).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": data},
    }


def encode_pdf_for_claude(uploaded_file):
    """Converts a Streamlit UploadedFile (PDF) into an Anthropic document block."""
    uploaded_file.seek(0)
    data = base64.b64encode(uploaded_file.read()).decode()
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }


def build_user_content(text, uploaded_files):
    """
    Builds the multi-part content list for a user turn: any attached
    images/PDFs first, then the text. Returns a plain string instead of
    a list when there are no attachments, since the API/UI is simpler
    that way and history stays readable.
    """
    if not uploaded_files:
        return text

    content = []
    for f in uploaded_files:
        if f.type == "application/pdf":
            content.append(encode_pdf_for_claude(f))
        else:
            block = encode_image_for_claude(f)
            if block:
                content.append(block)
    content.append({"type": "text", "text": text})
    return content


def render_user_message(text, uploaded_files):
    """What gets shown in the chat bubble for a user turn with attachments."""
    if uploaded_files:
        names = ", ".join(f.name for f in uploaded_files)
        return f"📎 *Attached: {names}*\n\n{text}"
    return text


def extract_phase_tag(text):
    """
    Pulls the trailing [PHASE:N] tag off a faculty reply.
    Returns (clean_text, phase_int_or_None).
    """
    match = re.search(r"\[PHASE:\s*([1-5])\s*\]\s*$", text.strip())
    if not match:
        return text, None
    phase = int(match.group(1))
    clean_text = text[: match.start()].rstrip()
    return clean_text, phase


PHASE_LABELS = {
    1: "Case Entry",
    2: "Diagnosis (Staging & Grading)",
    3: "Tooth-by-Tooth Prognosis",
    4: "Referral Decision",
    5: "Treatment Sequencing & Wrap-up",
}


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
            st.iframe(autoplay_audio_html(audio_bytes), height=1)
            return
    # Fallback to free browser speech synthesis
    st.iframe(browser_tts_html(text), height=1)


# =========================================================================
# STREAMLIT APP
# =========================================================================

def main():
    st.set_page_config(page_title="Perio Clinic Faculty", page_icon="🦷", layout="centered")

    st.title("🦷 Periodontal Case Analysis")

    st.warning(
        "⚠️ **For educational purposes only.** This tool is designed to help "
        "dental students practice diagnostic concepts and clinical reasoning. "
        "It is **not** a substitute for professional medical advice, diagnosis, "
        "or treatment. All real patient care decisions must be made by a "
        "licensed clinician.",
        icon="⚠️",
    )

    st.caption(
        "Socratic case-based learning with an AI attending faculty member. "
        "Present your patient data and defend your clinical reasoning."
    )

    anthropic_client = get_anthropic_client()
    openai_client = get_openai_client()

    # ---------------------------------------------------------------
    # Session state init
    # ---------------------------------------------------------------
    # Each key is initialized independently with setdefault, rather than
    # gated behind a single "if 'messages' not in state" check. This makes
    # the app resilient to stale sessions left over from an older deployed
    # version of this code (e.g. right after you push an update that adds
    # new session keys) — every key gets created if missing, even if some
    # other keys already existed from before.
    if "messages" not in st.session_state:
        opening_line = (
            "Welcome to the clinic. Let's look at your case. "
            "Please present your patient: chief complaint and medical history, "
            "maximum probing depths (PD) and clinical attachment loss (CAL), "
            "furcation involvement and mobility, and radiographic bone loss (RBL)."
        )
        st.session_state.messages = [{"role": "assistant", "content": opening_line}]
        st.session_state.display_messages = [{"role": "assistant", "content": opening_line}]

    st.session_state.setdefault("spoken_count", 0)
    st.session_state.setdefault("current_phase", 1)
    st.session_state.setdefault("intake_data", None)
    st.session_state.setdefault("instructor_unlocked", False)
    st.session_state.setdefault("instructor_summary", None)

    # ---------------------------------------------------------------
    # Sidebar: settings + phase checklist
    # ---------------------------------------------------------------
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

        st.divider()
        st.header("Case Checklist")
        current = st.session_state.current_phase
        for n, label in PHASE_LABELS.items():
            if n < current:
                st.markdown(f"✅ ~~Phase {n}: {label}~~")
            elif n == current:
                st.markdown(f"▶️ **Phase {n}: {label}**")
            else:
                st.markdown(f"⬜ Phase {n}: {label}")

        st.divider()
        with st.expander("🔒 Instructor View"):
            if not st.session_state.instructor_unlocked:
                pin_input = st.text_input("Enter instructor PIN", type="password", key="pin_input")
                if st.button("Unlock"):
                    if pin_input == get_instructor_pin():
                        st.session_state.instructor_unlocked = True
                        st.rerun()
                    else:
                        st.error("Incorrect PIN.")
            else:
                student_turns = [m for m in st.session_state.display_messages if m["role"] == "user"]
                if not student_turns:
                    st.caption("Student hasn't responded yet — check back after they engage with the case.")
                else:
                    if st.button("🔍 Generate progress summary"):
                        with st.spinner("Analyzing transcript..."):
                            raw_summary = get_instructor_summary(anthropic_client, st.session_state.messages)
                        st.session_state.instructor_summary = parse_instructor_summary(raw_summary)

                    if st.session_state.instructor_summary:
                        s = st.session_state.instructor_summary
                        emoji, label = STATUS_DISPLAY.get(s.get("STATUS", ""), ("⚪", "Unknown"))
                        st.markdown(f"### {emoji} {label}")
                        st.markdown(f"**Engagement:** {s.get('ENGAGEMENT', '—')}")
                        st.markdown(f"**Comprehension:** {s.get('COMPREHENSION', '—')}")
                        st.markdown(f"**Strengths:** {s.get('STRENGTHS', '—')}")
                        st.markdown(f"**Gaps:** {s.get('GAPS', '—')}")
                        st.markdown(f"**Recommendation:** {s.get('RECOMMENDATION', '—')}")
                    else:
                        st.caption("Click the button above to generate a private assessment of this student's performance so far.")

                if st.button("Lock instructor view"):
                    st.session_state.instructor_unlocked = False
                    st.session_state.instructor_summary = None
                    st.rerun()

    # ---------------------------------------------------------------
    # Structured intake form (optional — student can use this instead
    # of free-typing the initial case data)
    # ---------------------------------------------------------------
    if st.session_state.current_phase == 1 and st.session_state.intake_data is None:
        with st.expander("📋 Structured Patient Intake Form (optional)", expanded=False):
            with st.form("intake_form"):
                st.markdown("**Chief Complaint / Medical History**")
                chief_complaint = st.text_area("Chief complaint & relevant medical history", height=80)
                col1, col2 = st.columns(2)
                with col1:
                    max_pd = st.text_input("Max Probing Depths (PD)", placeholder="e.g. 7mm, #3 & #14 mesial")
                    furcation = st.text_input("Furcation involvement", placeholder="e.g. Class II, #14 buccal")
                    smoking = st.text_input("Smoking history", placeholder="e.g. 15 cig/day x 10 yrs")
                with col2:
                    max_cal = st.text_input("Max Clinical Attachment Loss (CAL)", placeholder="e.g. 6mm")
                    mobility = st.text_input("Mobility", placeholder="e.g. Grade 2, #14")
                    hba1c = st.text_input("HbA1c / systemic modifiers", placeholder="e.g. HbA1c 7.2%")
                rbl = st.text_area("Radiographic Bone Loss (RBL)", height=60, placeholder="e.g. 30% generalized horizontal, vertical defect #14 distal")

                submitted = st.form_submit_button("Submit intake to faculty")
                if submitted:
                    intake_text = (
                        "Here is my structured patient intake:\n\n"
                        f"- **Chief Complaint / Medical History:** {chief_complaint or 'N/A'}\n"
                        f"- **Max PD:** {max_pd or 'N/A'}\n"
                        f"- **Max CAL:** {max_cal or 'N/A'}\n"
                        f"- **Furcation:** {furcation or 'N/A'}\n"
                        f"- **Mobility:** {mobility or 'N/A'}\n"
                        f"- **Smoking history:** {smoking or 'N/A'}\n"
                        f"- **HbA1c / systemic modifiers:** {hba1c or 'N/A'}\n"
                        f"- **Radiographic Bone Loss (RBL):** {rbl or 'N/A'}"
                    )
                    st.session_state.intake_data = intake_text
                    st.session_state._pending_submit = intake_text
                    st.rerun()

    # ---------------------------------------------------------------
    # File upload: periodontal charting + radiographs
    # ---------------------------------------------------------------
    with st.expander("📎 Attach periodontal charting / x-rays", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload full-mouth periodontal charting and/or radiographs",
            type=["png", "jpg", "jpeg", "webp", "pdf"],
            accept_multiple_files=True,
            help="Images (PNG/JPG/WEBP) are read directly by the faculty AI. "
                 "PDFs (e.g. exported perio charts) are also supported.",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) ready to attach to your next message.")

    # ---------------------------------------------------------------
    # Render chat history (display version — text only, no raw image blocks)
    # ---------------------------------------------------------------
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"], avatar="🦷" if msg["role"] == "assistant" else "🧑‍⚕️"):
            st.markdown(msg["content"])

    # Speak any not-yet-spoken assistant messages
    assistant_msgs = [m for m in st.session_state.display_messages if m["role"] == "assistant"]
    if voice_enabled and st.session_state.spoken_count < len(assistant_msgs):
        for m in assistant_msgs[st.session_state.spoken_count:]:
            speak(m["content"], openai_client)
        st.session_state.spoken_count = len(assistant_msgs)

    # ---------------------------------------------------------------
    # Handle a pending submission from the intake form
    # ---------------------------------------------------------------
    pending_text = st.session_state.pop("_pending_submit", None)

    # Chat input
    user_input = st.chat_input("Present your case findings or respond to the question above...")

    final_input = pending_text or user_input
    if final_input:
        files_for_this_turn = uploaded_files if (user_input and uploaded_files) else None

        api_content = build_user_content(final_input, files_for_this_turn)
        display_text = render_user_message(final_input, files_for_this_turn)

        st.session_state.messages.append({"role": "user", "content": api_content})
        st.session_state.display_messages.append({"role": "user", "content": display_text})

        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(display_text)

        with st.chat_message("assistant", avatar="🦷"):
            with st.spinner("Faculty is reviewing your response..."):
                raw_reply = get_faculty_response(anthropic_client, st.session_state.messages)
            clean_reply, phase = extract_phase_tag(raw_reply)
            st.markdown(clean_reply)

        # Keep the raw (tagged) reply in API history so Claude sees its own
        # prior phase tags for continuity; show the clean version to the user.
        st.session_state.messages.append({"role": "assistant", "content": raw_reply})
        st.session_state.display_messages.append({"role": "assistant", "content": clean_reply})

        if phase:
            st.session_state.current_phase = phase

        if voice_enabled:
            speak(clean_reply, openai_client)
        st.session_state.spoken_count = len(
            [m for m in st.session_state.display_messages if m["role"] == "assistant"]
        )
        st.rerun()


if __name__ == "__main__":
    main()
