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

The instructor receives an email notification (via SendGrid) when a student
starts a session and again when they submit/finish, with an AI-generated
performance summary.

ENV VARS / STREAMLIT SECRETS REQUIRED:
    ANTHROPIC_API_KEY   - required, powers the Socratic faculty logic
    OPENAI_API_KEY      - optional, enables natural AI voice (TTS)
    SENDGRID_API_KEY    - optional, enables email notifications to instructor
    INSTRUCTOR_EMAIL    - required if SendGrid is enabled (your email address)
    INSTRUCTOR_PIN      - optional, overrides default PIN for Instructor View
    FROM_EMAIL          - optional, sender address for notifications
                          (must be verified in your SendGrid account;
                           defaults to INSTRUCTOR_EMAIL if not set)

Run locally:
    pip install streamlit anthropic openai sendgrid
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this file + requirements.txt to your GitHub repo
    2. On share.streamlit.io, create a new app pointing at app.py
    3. In "Secrets", add:
         ANTHROPIC_API_KEY  = "sk-ant-..."
         OPENAI_API_KEY     = "sk-..."          # optional
         SENDGRID_API_KEY   = "SG...."          # optional
         INSTRUCTOR_EMAIL   = "you@dental.edu"  # required for email
         INSTRUCTOR_PIN     = "your-pin"        # optional, default 1234
         FROM_EMAIL         = "perio-app@dental.edu"  # optional
"""

import base64
import os
import re
from datetime import datetime

import streamlit as st
from anthropic import Anthropic

# Optional import — app still runs (with browser TTS fallback) without it
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional import — app still runs (without email notifications) without it
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False


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
5. UFCD Medical Consultation Guidelines (adapted from Firriolo, U of Louisville; Rev. 03/11/19): See Phase 0 below. A medical consultation is required before invasive dental treatment for any patient meeting the listed medical criteria.

UFCD MEDICAL CONSULTATION CRITERIA (Phase 0 Reference):
Before any invasive or physiologically stressful dental treatment, a medical consult is required if the patient meets ANY of the following:

GENERAL TRIGGERS (any of the following mandates a consult):
- Potential allergy to local anesthetic, dental material (mercury, nickel, methylmethacrylate), or sulfite preservatives.
- Any medical problem that could cause complications from invasive/stressful dental treatment (e.g., angina, MI history, CVA/TIA, CHF, hypertension with BP >140/90, arrhythmia, diabetes, COPD, poorly controlled/stress-induced asthma, symptomatic thyroid disease, seizure disorder with >1 seizure/month, hepatitis/cirrhosis, chronic kidney disease/dialysis, adrenal insufficiency).
- Any medical problem or medication that increases risk of adverse drug reaction to antibiotics, local anesthetics, vasoconstrictors, N2O, or NSAID/narcotic analgesics.
- Immunosuppression or impaired wound healing: HIV/AIDS, blood dyscrasias, aplastic anemia, myeloproliferative disease (leukemia, lymphoma), systemic corticosteroids or immunosuppressants (TNF blockers, azathioprine, methotrexate), cytotoxic chemotherapy, prior head/neck radiation, organ/bone marrow/stem cell transplant.
- Impaired hemostasis: hemophilia, von Willebrand's disease, thrombocytopenia, warfarin (INR must be ≤3.0 and tested within 48 hrs prior to invasive tx in student clinic; surgical procedures go to Grad Oral Surgery), direct oral anticoagulants (Pradaxa, Xarelto, Eliquis, Savaysa), LMWH, valproic acid.
- Psychiatric/cognitive problems affecting consent capacity or ability to cooperate with treatment.
- Possibly unresolved active infectious disease posing transmission risk despite universal precautions (e.g., active TB, pulmonary MRSA).

CONDITION-SPECIFIC CONSULT TRIGGERS:

HYPERTENSION (per UFCD/JNC-7):
- BP 140–159 / 90–99 (Stage 1): recheck within 2 weeks; if still elevated, physician evaluation within 2 weeks required before invasive treatment.
- BP ≥160 / ≥100 (Stage 2): physician evaluation immediately or within 1 week required.
- BP ≥180 systolic and/or ≥110 diastolic (uncontrolled, ASA IV): defer elective dental care; local anesthetic with vasoconstrictor not recommended; urgent care only.
- Pre-appointment: always check BP and pulse before administering local anesthetic.

CARDIOVASCULAR DISEASE:
- Arrhythmia (any diagnosed or suspected): consult required; assess if adequately controlled or still symptomatic.
- CHF: consult required for any suspected/undiagnosed CHF, or NYHA Class II, III, or IV.
- Angina: consult required for suspected/undiagnosed angina, vasospastic angina, or CCS Class III–IV angina.
- Myocardial infarction (any history): consult required; assess post-MI status, stability, concurrent CHF/arrhythmia/hypertension.
- Cerebrovascular accident / TIA (any history, not under active physician care): consult required.

DIABETES:
- Any patient with signs/symptoms of undiagnosed diabetes: refer to physician.
- Any patient with diagnosed diabetes: consult to assess glycemic control (HbA1c), wound healing risk, and current status. Both hyperglycemia (>150 mg/dL / HbA1c >6.9%) and hypoglycemia (<80 mg/dL) increase perioperative infection risk.
- Patients ≥45 yr (especially BMI ≥25 kg/m²) without prior diabetes screening: advise physician visit.

ASTHMA / COPD:
- "Not well controlled" or "very poorly controlled" asthma (per ACT score ≤19 or Figure 2 criteria): consult required.
- Very poorly controlled asthma (ASA IV): defer all elective care; urgent care only in hospital dental clinic.
- COPD: consult if symptomatic or limiting daily activity.

ANTICOAGULANTS (specific thresholds for student clinic):
- Warfarin: INR must be ≤3.0, tested within 48 hrs. Surgical procedures (extractions) require Grad Oral Surgery, not student clinic.
- Direct oral anticoagulants (DOAs): physician consult for any surgical procedure; generally do not need to be stopped for simple dental procedures if renal function is normal.
- Aspirin/antiplatelet drugs (≤325 mg): usually no need to discontinue or consult for routine dental procedures. Exception: patient <1 year post drug-eluting coronary stent placement — consult required before discontinuing.
- No patient should be told to stop any anticoagulant or antiplatelet drug without physician approval.

CANCER / LEUKEMIA / LYMPHOMA:
- Pre-treatment: consult oncologist before starting dental care; determine stage, treatment plan, start date, risk of MRONJ (zoledronate, pamidronate, denosumab, bevacizumab).
- Currently receiving cytotoxic chemotherapy or head/neck radiotherapy: NOT candidates for elective dental care; urgent care only in hospital setting.
- Post-treatment: consult to determine current stage, remission status, recurrence history.
- History of head/neck radiotherapy: consult radiation oncologist; determine total dose and field to assess osteoradionecrosis risk.

HIV/AIDS:
- Consult required; obtain CBC with differential, CD4+ count, HIV RNA (viral load), TB status, current meds.
- CD4 count must be >200/mm³ for elective dental care. Neutrophil count <500/mm³ or platelet count <50,000/mm³: prophylactic antibiotics or defer surgery.
- INR >1.7 (suggestive of HIV-related hepatic failure): do not perform invasive dental treatment in student clinic.
- All oral surgery for HIV-positive patients: performed at ACB (unless approved by OS faculty).

HEPATITIS / CIRRHOSIS:
- Any patient with unspecified hepatitis history, jaundice, or scleral icterus: lab testing required (HBsAg, anti-HCV, HCV-RNA).
- Cirrhosis: mandatory consult; obtain CMP/hepatic function panel, CBC with differential and platelets, PT/INR; assess for ascites, encephalopathy, coagulopathy.

RENAL DISEASE:
- Chronic renal failure: consult required; obtain BMP/CMP with GFR, CBC with differential; rule out uncontrolled hypertension or symptomatic hypotension.
- Hemodialysis patients: determine vascular access type and dialysis schedule; treat on non-dialysis days.

INFECTIVE ENDOCARDITIS PROPHYLAXIS (2007 AHA/ADA guidelines):
- Prophylaxis required for: prosthetic cardiac valves, previous IE, unrepaired cyanotic CHD, repaired CHD with residual hemodynamic defect, completely repaired CHD with prosthetic material within 6 months, cardiac transplant recipients with valvulopathy.
- Regimen: amoxicillin 2g PO (adult) 30–60 min before procedure; clindamycin 600mg or azithromycin 500mg if penicillin-allergic.
- Does NOT require prophylaxis: pacemakers, coronary stents, peripheral vascular stents, CSF shunts, AV fistulas for dialysis, central venous catheters, or any device NOT listed in Box 1 above.

PREGNANCY:
- Consult patient's OB/GYN for any invasive dental treatment.

CORTICOSTEROID USE / ADRENAL INSUFFICIENCY:
- Patients on long-term systemic corticosteroids: consult required; assess adrenal suppression risk; may need supplemental dosing for stressful dental procedures.

AUTOIMMUNE DISEASE (SLE, RA, etc.):
- Consult required only if patient is on immunosuppressive therapy (high-dose corticosteroids, methotrexate, cyclosporine, TNF-α inhibitors, JAK inhibitors); obtain CBC with differential and CMP.

BISPHOSPHONATE / MEDICATION-RELATED OSTEONECROSIS OF THE JAWS (MRONJ):
- History of IV bisphosphonates (zoledronate, pamidronate) or RANKL inhibitors (denosumab) or angiogenesis inhibitors (bevacizumab, sunitinib): consult oncologist/physician before any surgical procedure; assess MRONJ risk.
- Oral bisphosphonates (e.g., alendronate for osteoporosis): risk is lower but clinically significant risk begins at approximately 4 years of use; assess with patient and physician.

PROSTHETIC JOINTS:
- Per current AAOS/ADA guidelines: antibiotic prophylaxis is NOT routinely recommended for all patients with prosthetic joints. Clinical judgment should be used for patients at high risk of hematogenous total joint infection (immunocompromised, prior joint infection). Consult orthopedic surgeon or physician if uncertain.

SOCRATIC OPERATIONAL WORKFLOW (ONE STEP AT A TIME):
Do not lecture or provide full analyses. Ask one question at a time and wait for the student's response. If they make an error, do not give them the answer; instead, ask a question that directs their attention to their mistake.

Phase 0: Medical Risk Screening (NEW — occurs BEFORE perio diagnosis)
- After the student presents their patient, ALWAYS ask about the patient's medical history first if they have not already provided it.
- Once the student has provided the medical history, ask ONE Socratic question about the most significant medical finding. The question must always be phrased as: "Based on what you've told me about this patient's [condition/finding], do you think we should refer to their physician for a medical consult before we proceed with any invasive dental treatment? Why or why not?"
- NEVER tell the student whether a consult is or is not required — always make them commit to a yes/no answer and defend it first.
- If the student says YES correctly: ask them "What specific information would you request from the physician, and why?" Then ask what, if anything, needs to happen before dental treatment can proceed.
- If the student says YES but for the wrong reason, or identifies the wrong condition: ask a guiding question that redirects them to the correct finding without giving it away. Example: "Good instinct to think about a consult — but let's look more closely at the blood pressure reading you mentioned. Where does that fall on the JNC-7 classification, and does that change your answer?"
- If the student says NO when a consult IS required: do not correct them directly. Instead ask: "Walk me through your reasoning. The patient's [specific finding — e.g., BP of 165/95, HbA1c of 9.2%, warfarin use] — how does that factor into your decision?" Then follow up: "Looking at the UFCD Medical Consultation Guidelines, what does the protocol say about a patient with [that specific finding] before we proceed with invasive treatment?"
- If the patient truly has no medically complex conditions requiring a consult, affirm the student's answer briefly and move to Phase 1.
- Only after Phase 0 is satisfactorily completed should you proceed to Phase 1.

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
At the very end of every single response, on its own new line, output a machine-readable tag indicating which phase of the workflow this response belongs to, in the exact format: [PHASE:N] where N is 0, 1, 2, 3, 4, or 5, corresponding to the six phases above (0 = Medical Risk Screening, 1 = Case Entry, 2 = Diagnosis, 3 = Prognosis, 4 = Referral Decision, 5 = Wrap-up). This tag is for the clinic's progress-tracking software and is stripped before the student sees your message — it does not break character and is not visible to the student, so always include it, exactly once, at the very end.
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


def get_email_config():
    """
    Returns (sendgrid_key, instructor_email, from_email) or (None, None, None)
    if email notifications are not configured.
    """
    sg_key = st.secrets.get("SENDGRID_API_KEY", os.environ.get("SENDGRID_API_KEY"))
    to_email = st.secrets.get("INSTRUCTOR_EMAIL", os.environ.get("INSTRUCTOR_EMAIL"))
    from_email = st.secrets.get(
        "FROM_EMAIL", os.environ.get("FROM_EMAIL", to_email)
    )
    if not SENDGRID_AVAILABLE or not sg_key or not to_email:
        return None, None, None
    return sg_key, to_email, from_email


def send_email(subject, body_html, body_text=None):
    """
    Sends an email to the instructor via SendGrid.
    Silently no-ops if SendGrid is not configured so the app never crashes
    due to a missing notification setup.
    Returns True on success, False on failure.
    """
    sg_key, to_email, from_email = get_email_config()
    if not sg_key:
        return False
    try:
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            html_content=body_html,
        )
        sg = SendGridAPIClient(sg_key)
        sg.send(message)
        return True
    except Exception as e:
        # Don't crash the app over a failed notification
        print(f"SendGrid error (non-fatal): {e}")
        return False


def send_session_start_email(student_name):
    """Fires when a student starts a new session."""
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    subject = f"🦷 Perio App — {student_name} started a case session"
    body = f"""
    <h2>Session Started</h2>
    <p><strong>Student:</strong> {student_name}</p>
    <p><strong>Time:</strong> {now}</p>
    <p>This student has just opened the Periodontal Case Analysis app and
    begun a new Socratic case session. You will receive a second email with
    their performance summary when they finish and submit.</p>
    <hr>
    <p style="color:gray;font-size:12px;">
    Periodontal Case Analysis App — Automated Instructor Notification
    </p>
    """
    send_email(subject, body)


def send_session_summary_email(student_name, summary_dict, transcript_text):
    """Fires when the student clicks Submit & Finish, with the full AI summary."""
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    s = summary_dict
    status = s.get("STATUS", "UNKNOWN")
    emoji_map = {
        "ON_TRACK": "🟢", "NEEDS_HELP": "🟡",
        "OFF_TASK": "🔴", "INSUFFICIENT_DATA": "⚪",
    }
    emoji = emoji_map.get(status, "⚪")
    subject = f"🦷 Perio App — {student_name} session summary ({emoji} {status.replace('_', ' ').title()})"
    body = f"""
    <h2>Session Summary — {student_name}</h2>
    <p><strong>Submitted:</strong> {now}</p>
    <table style="border-collapse:collapse;width:100%;font-family:sans-serif;">
      <tr style="background:#f0f0f0;">
        <td style="padding:8px;font-weight:bold;width:160px;">Overall Status</td>
        <td style="padding:8px;">{emoji} <strong>{status.replace("_", " ").title()}</strong></td>
      </tr>
      <tr>
        <td style="padding:8px;font-weight:bold;">Engagement</td>
        <td style="padding:8px;">{s.get("ENGAGEMENT", "—")}</td>
      </tr>
      <tr style="background:#f9f9f9;">
        <td style="padding:8px;font-weight:bold;">Comprehension</td>
        <td style="padding:8px;">{s.get("COMPREHENSION", "—")}</td>
      </tr>
      <tr>
        <td style="padding:8px;font-weight:bold;">Strengths</td>
        <td style="padding:8px;">{s.get("STRENGTHS", "—")}</td>
      </tr>
      <tr style="background:#f9f9f9;">
        <td style="padding:8px;font-weight:bold;">Gaps</td>
        <td style="padding:8px;">{s.get("GAPS", "—")}</td>
      </tr>
      <tr>
        <td style="padding:8px;font-weight:bold;">Recommendation</td>
        <td style="padding:8px;">{s.get("RECOMMENDATION", "—")}</td>
      </tr>
    </table>
    <h3>Full Transcript</h3>
    <pre style="background:#f5f5f5;padding:12px;font-size:13px;
                white-space:pre-wrap;border-radius:4px;">{transcript_text}</pre>
    <hr>
    <p style="color:gray;font-size:12px;">
    Periodontal Case Analysis App — Automated Instructor Notification
    </p>
    """
    send_email(subject, body)


def build_plain_transcript(display_messages):
    """Builds a readable plain-text transcript from display_messages."""
    lines = []
    for m in display_messages:
        role = "FACULTY" if m["role"] == "assistant" else "STUDENT"
        lines.append(f"[{role}]\n{m['content']}\n")
    return "\n".join(lines)


LIFELINE_SYSTEM_PROMPT = """You are generating a multiple-choice lifeline question to help a dental student who is stuck during a Socratic periodontal case analysis exercise.

Your job: given the conversation so far, identify the specific clinical concept the student is struggling with, then generate exactly 5 answer choices — one clearly correct, four plausible but incorrect distractors. The choices should be specific and clinically meaningful (not vague), appropriate for a dental student at the D2/D3 level, and relevant to the exact question the faculty most recently asked.

Respond ONLY in this exact JSON format, with no preamble, no markdown, no backticks:
{
  "question": "One short, clear question summarizing what the student needs to answer (max 20 words)",
  "options": [
    "Option A text",
    "Option B text",
    "Option C text",
    "Option D text",
    "Option E text"
  ],
  "correct_index": 0
}

correct_index is the 0-based index of the correct answer among the 5 options. Randomize where the correct answer falls — don't always put it first.
"""


def get_lifeline_options(client, messages):
    """
    Calls Claude with a separate prompt to generate 5 contextual MC options
    for wherever the student is currently stuck. Returns a dict with
    'question', 'options' (list of 5), and 'correct_index', or None on failure.
    """
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=LIFELINE_SYSTEM_PROMPT,
            messages=messages + [{
                "role": "user",
                "content": (
                    "The student has now struggled with this question twice. "
                    "Generate a 5-option multiple choice lifeline question covering "
                    "the concept they are stuck on, in the exact JSON format specified."
                )
            }],
        )
        raw = response.content[0].text.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        import json
        data = json.loads(raw)
        # Validate structure
        if (
            isinstance(data.get("options"), list)
            and len(data["options"]) == 5
            and isinstance(data.get("correct_index"), int)
            and 0 <= data["correct_index"] <= 4
            and isinstance(data.get("question"), str)
        ):
            return data
        return None
    except Exception as e:
        print(f"Lifeline generation error (non-fatal): {e}")
        return None


def is_response_struggling(raw_reply_text):
    """
    Heuristic: did the faculty's reply indicate the student got it wrong or
    incomplete? Looks for Socratic correction language in the reply.
    Returns True if the student appears to have struggled.
    """
    correction_signals = [
        "let's look", "look back", "think about", "reconsider",
        "not quite", "let me redirect", "review", "check again",
        "walk me through", "what does the literature", "look more carefully",
        "let's revisit", "that's not", "incorrect", "look at this",
        "examine", "re-examine", "go back", "try again",
    ]
    lower = raw_reply_text.lower()
    return any(signal in lower for signal in correction_signals)


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
    Returns (clean_text, phase_int_or_None). Handles phases 0–5.
    """
    match = re.search(r"\[PHASE:\s*([0-5])\s*\]\s*$", text.strip())
    if not match:
        return text, None
    phase = int(match.group(1))
    clean_text = text[: match.start()].rstrip()
    return clean_text, phase


PHASE_LABELS = {
    0: "Medical Risk Screening",
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
    if st.session_state.get("student_name"):
        st.markdown(f"**Student:** {st.session_state.student_name}")

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
    # Session state init — MUST come before any UI or state reads
    # ---------------------------------------------------------------
    if "messages" not in st.session_state:
        opening_line = (
            "Welcome to the clinic, Doctor. Let's look at your case. "
            "Please present your patient: chief complaint and medical history, "
            "maximum probing depths (PD) and clinical attachment loss (CAL), "
            "furcation involvement and mobility, and radiographic bone loss (RBL)."
        )
        st.session_state.messages = [{"role": "assistant", "content": opening_line}]
        st.session_state.display_messages = [{"role": "assistant", "content": opening_line}]

    st.session_state.setdefault("spoken_count", 0)
    st.session_state.setdefault("current_phase", 0)
    st.session_state.setdefault("intake_data", None)
    st.session_state.setdefault("instructor_unlocked", False)
    st.session_state.setdefault("instructor_summary", None)
    st.session_state.setdefault("student_name", None)
    st.session_state.setdefault("session_start_email_sent", False)
    st.session_state.setdefault("case_submitted", False)
    st.session_state.setdefault("strike_count", 0)       # wrong/incomplete answers in current phase
    st.session_state.setdefault("lifeline_options", None) # MC options currently shown
    st.session_state.setdefault("session_start_time", datetime.now().isoformat())

    # ---------------------------------------------------------------
    # Student name capture — shown once at the very start
    # ---------------------------------------------------------------
    if not st.session_state.student_name:
        st.subheader("Before we begin")
        st.markdown("Please enter your name so your instructor can identify your session.")
        with st.form("name_form"):
            name_input = st.text_input("Your full name", placeholder="e.g. Jane Smith")
            start = st.form_submit_button("Start case session")
            if start:
                if name_input.strip():
                    st.session_state.student_name = name_input.strip()
                    st.rerun()
                else:
                    st.error("Please enter your name to continue.")
        st.stop()

    # Send session-start email once, the first time we have a student name
    if st.session_state.student_name and not st.session_state.session_start_email_sent:
        send_session_start_email(st.session_state.student_name)
        st.session_state.session_start_email_sent = True

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

        # Session timer
        elapsed = datetime.now() - datetime.fromisoformat(
            st.session_state.session_start_time
        )
        elapsed_min = int(elapsed.total_seconds() // 60)
        elapsed_sec = int(elapsed.total_seconds() % 60)
        remaining = max(0, 30 * 60 - int(elapsed.total_seconds()))
        rem_min = remaining // 60
        rem_sec = remaining % 60
        if remaining > 5 * 60:
            timer_icon = "⏱️"
        elif remaining > 0:
            timer_icon = "⚠️"
        else:
            timer_icon = "🔴"
        st.markdown(
            f"{timer_icon} **Time:** {elapsed_min}m {elapsed_sec:02d}s elapsed &nbsp;|&nbsp; "
            f"**{rem_min}m {rem_sec:02d}s remaining**"
        )

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

    # ---------------------------------------------------------------
    # Lifeline multiple choice widget (appears after 2 strikes)
    # ---------------------------------------------------------------
    if st.session_state.lifeline_options:
        opts = st.session_state.lifeline_options
        st.info(
            "💡 **Lifeline available** — you've had a couple of tries on this "
            "question. You can select an answer below to get unstuck, or keep "
            "typing your own response above.",
            icon="💡",
        )
        st.markdown(f"**{opts['question']}**")
        cols = st.columns(1)
        for i, option_text in enumerate(opts["options"]):
            label = f"{chr(65+i)}. {option_text}"
            if st.button(label, key=f"lifeline_{i}"):
                chosen = option_text
                correct = opts["options"][opts["correct_index"]]
                is_correct = (i == opts["correct_index"])
                submission = (
                    f"[Lifeline selected] {chr(65+i)}. {chosen}"
                )
                st.session_state.lifeline_options = None
                st.session_state.strike_count = 0
                st.session_state._pending_submit = submission
                st.rerun()

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

        # Phase advance — reset strikes and clear any active lifeline
        if phase and phase != st.session_state.current_phase:
            st.session_state.current_phase = phase
            st.session_state.strike_count = 0
            st.session_state.lifeline_options = None
        elif phase:
            st.session_state.current_phase = phase

        # Strike detection — only if no lifeline was just used
        if not final_input.startswith("[Lifeline selected]"):
            if is_response_struggling(clean_reply):
                st.session_state.strike_count += 1
            else:
                # Correct/complete answer — reset
                st.session_state.strike_count = 0
                st.session_state.lifeline_options = None

            # Trigger lifeline after 2 strikes if not already showing one
            if (
                st.session_state.strike_count >= 2
                and st.session_state.lifeline_options is None
            ):
                with st.spinner("Preparing a lifeline question..."):
                    lifeline = get_lifeline_options(
                        anthropic_client, st.session_state.messages
                    )
                if lifeline:
                    st.session_state.lifeline_options = lifeline

        if voice_enabled:
            speak(clean_reply, openai_client)
        st.session_state.spoken_count = len(
            [m for m in st.session_state.display_messages if m["role"] == "assistant"]
        )
        st.rerun()

    # ---------------------------------------------------------------
    # Submit & Finish — generates summary and emails instructor
    # ---------------------------------------------------------------
    st.divider()
    if not st.session_state.case_submitted:
        student_turns = [m for m in st.session_state.display_messages if m["role"] == "user"]
        if student_turns:
            if st.button("✅ Submit & finish case", type="primary"):
                with st.spinner("Generating your performance summary and notifying instructor..."):
                    raw_summary = get_instructor_summary(
                        anthropic_client, st.session_state.messages
                    )
                    summary = parse_instructor_summary(raw_summary)
                    transcript = build_plain_transcript(st.session_state.display_messages)
                    st.session_state.instructor_summary = summary
                    send_session_summary_email(
                        st.session_state.student_name, summary, transcript
                    )
                    st.session_state.case_submitted = True
                st.rerun()
    else:
        sg_key, _, _ = get_email_config()
        if sg_key:
            st.success(
                "✅ Case submitted. Your instructor has been notified with a "
                "summary of this session. You may close this window."
            )
        else:
            st.success(
                "✅ Case submitted. You may close this window."
            )


if __name__ == "__main__":
    main()
