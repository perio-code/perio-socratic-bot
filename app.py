import streamlit as st
import anthropic

# 1. Page Configuration
st.set_page_config(page_title="Periodontal Patient Reviewer", page_icon="🦷", layout="wide")

# 2. Initialize Anthropic Client
if "ANTHROPIC_API_KEY" in st.secrets:
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
else:
    st.error("Welcome! To start consulting with the Socratic Faculty member, please configure your ANTHROPIC_API_KEY in the Streamlit Cloud dashboard settings under 'Secrets'.")
    st.stop()

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Sidebar: Patient Clipboard
with st.sidebar:
    st.header("📋 Patient Clipboard")
    st.write("Enter the patient's clinical data below before consulting with the faculty mentor.")
    med_hx = st.text_area("Medical History & Medications", height=100)
    chief_complaint = st.text_area("Chief Complaint", height=100)
    perio_charting = st.text_area("Periodontal Charting (6-site, CAL, BOP)", height=150)
    radiographs = st.text_area("Radiographic Findings", height=100)

# 5. Main Chat Interface
st.title("🦷 Socratic Periodontal Instructor")
st.markdown("Welcome to the clinic floor. Review your patient's clipboard, then discuss their periodontal diagnosis, prognosis and treatment plan.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. The "Attending Brain" (System Prompt)
system_prompt = f"""
You are an expert periodontist and a warm, encouraging faculty instructor at a dental school. 
Your goal is to guide dental students through periodontal case analysis and treatment planning using the Socratic method. 
Do not give them the answers directly. Instead, ask probing questions to help them arrive at the right clinical decisions.

Emphasize the following in your guidance:
1. Comprehensive medical history review.
2. Lindhe's critical probing depth analysis and criteria for intervention.
3. Proper periodontal-restorative treatment sequencing.

Current Patient Data provided by the student:
- Medical History: {med_hx}
- Chief Complaint: {chief_complaint}
- Periodontal Charting: {perio_charting}
- Radiographs: {radiographs}
"""

# 7. Chat Input & API Call
if prompt := st.chat_input("Discuss your diagnosis or treatment plan with the attending..."):
    # Add student message to UI and history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Anthropic API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6", 
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            )
            full_response = response.content[0].text
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
