import streamlit as st
import anthropic

st.set_page_config(page_title="Periodontal Socratic Mentor", layout="wide")

st.title("🦷 Periodontal & Restorative Case Simulator")
st.markdown("Welcome to the clinic! Review the patient's clipboard on the left, and discuss your treatment plan with your Socratic Faculty Mentor on the right.")

# Initialize the Anthropic client using the Streamlit secrets
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Layout: Left column for Clipboard, Right column for Chat
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Patient Clipboard")
    st.text_area("Medical History", placeholder="Type or paste medical history here...")
    st.text_area("Chief Complaint", placeholder="Type or paste chief complaint here...")
    st.text_area("Periodontal Charting (6-site)", height=200, placeholder="Paste charting data here...")
    st.text_area("Radiographic Findings", placeholder="Describe bone loss, furcations, etc...")

with col2:
    st.header("👨‍🏫 Faculty Mentor Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Discuss your diagnosis or plan here..."):
        # Add user message to the screen
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Claude
        with st.chat_message("assistant"):
            try:
                # The Socratic Instructor rules
                system_prompt = """You are a warm, expert periodontal faculty instructor at a dental school. 
                Your goal is to guide dental students using the Socratic method. 
                Do NOT give them the answers directly. Ask guiding questions about 
                Lindhe's critical probing depths, medical history impact, and 
                the periodontal-restorative sequence."""

                # Format messages for the API
                api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                
                # Send to Claude
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620", 
                    max_tokens=1000,
                    system=system_prompt,
                    messages=api_messages
                )
                
                # Display the response
                msg = response.content[0].text
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                
            except Exception as e:
                st.error(f"An error occurred connecting to Claude: {e}")
