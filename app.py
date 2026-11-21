import streamlit as st
import json
import os
import time
from graph import build_sentinel_graph

# Page setup
st.set_page_config(page_title="SentinelDesk", layout="wide")

st.title("SentinelDesk - Incident Triage & Copilot")

# Sidebar Configuration
with st.sidebar:
    st.subheader("Cloud Settings")
    api_key_input = st.text_input(
        "Gemini API Key (Optional):",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Paste your Gemini API key to enable live cloud generation via Gemini Flash."
    )
    if api_key_input:
        clean_key = api_key_input.strip()
        os.environ["GEMINI_API_KEY"] = clean_key
        st.success("API Key Saved in Session")
        
        if st.button("Test API Connection", use_container_width=True):
            with st.spinner("Pinging Gemini API..."):
                try:
                    from google import genai
                    test_client = genai.Client(api_key=clean_key)
                    t0 = time.perf_counter()
                    resp = test_client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents="Ping. Respond with 'Connected'"
                    )
                    dt = round((time.perf_counter() - t0) * 1000, 1)
                    st.success(f"Connection Verified: {resp.text.strip()} ({dt} ms)")
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
    else:
        st.caption("Running in offline mode (local grounded template). Set key to activate live Gemini Flash.")

# Load pre-extracted benchmark scenarios from data
PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_ui_presets.json")
presets = []
if os.path.exists(PRESETS_FILE):
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        presets = json.load(f)

# Load compiled LangGraph workflow
@st.cache_resource
def load_graph():
    return build_sentinel_graph()

graph = load_graph()

# Layout: Two columns (Input on left, Output on right)
col_in, col_out = st.columns([1, 1], gap="medium")

with col_in:
    st.subheader("1. Incoming Ticket")
    
    # Dynamic preset selector populated from sample_ui_presets.json
    preset_options = ["-- Custom Ticket --"] + [p["name"] for p in presets]
    selected_option = st.selectbox("Load Benchmark Scenario (from Holdout Test Data):", preset_options)
    
    default_subject = ""
    default_body = ""
    if selected_option != "-- Custom Ticket --":
        matched = next((p for p in presets if p["name"] == selected_option), None)
        if matched:
            default_subject = matched.get("subject", "")
            default_body = matched.get("body", "")
            if matched.get("notes"):
                st.caption(f"Benchmark Goal: {matched['notes']}")

    subject = st.text_input("Subject:", value=default_subject, placeholder="Enter ticket subject (can be blank)")
    body = st.text_area("Body:", value=default_body, height=180, placeholder="Paste customer message here...")
    
    submit = st.button("Process Ticket", type="primary", use_container_width=True)

with col_out:
    st.subheader("2. Triage & Copilot Results")
    
    if submit:
        if not body.strip():
            st.warning("Please provide a ticket body.")
        else:
            with st.spinner("Processing through LangGraph..."):
                result = graph.invoke({"subject": subject, "body": body})

            # 1. Predictions & Confidence
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Type", result["type"])
            m2.metric("Queue", result["queue"])
            m3.metric("Priority", result["priority"].upper())
            m4.metric("Confidence", f"{result['confidence']*100:.0f}%")
            m5.metric("Latency", f"{result['latency_ms']} ms")

            # 2. Escalation Badge
            st.divider()
            if result["escalation_status"] == "ESCALATE":
                st.error(f"STATUS: ESCALATE TO ON-CALL ENGINEER\n\nReason: {result['escalation_reason']}")
            else:
                st.success(f"STATUS: AUTO-DRAFT READY\n\nReason: {result['escalation_reason']}")

            # 3. Top Retrieved Historical Precedents
            st.divider()
            st.markdown("### Historical Precedents (RAG)")
            for prec in result.get("retrieved_precedents", []):
                st.markdown(f"**Ticket ID:** `{prec['ticket_id']}` | **Similarity:** `{prec['similarity']*100:.1f}%`")
                st.caption(f"**Past Resolution:** {prec['historical_answer']}")

            # 4. Generated Draft Response
            st.divider()
            st.markdown("### Generated Draft Response")
            st.text_area("Agent Response:", value=result["draft_response"], height=140, disabled=True)
    else:
        st.info("Fill in the ticket on the left and click Process Ticket to view the triage output.")
