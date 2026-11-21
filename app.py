import streamlit as st
import time
from graph import build_sentinel_graph

# Page setup
st.set_page_config(page_title="SentinelDesk", page_icon="🛡️", layout="wide")

st.title("🛡️ SentinelDesk - Incident Triage & Copilot")

# Load compiled LangGraph workflow
@st.cache_resource
def load_graph():
    return build_sentinel_graph()

graph = load_graph()

# Layout: Two columns (Input on left, Output on right)
col_in, col_out = st.columns([1, 1], gap="medium")

with col_in:
    st.subheader("1. Incoming Ticket")
    
    # Simple preset dropdown for quick demoing
    preset_choice = st.selectbox(
        "Load Sample Ticket (Optional):",
        [
            "-- Custom Ticket --",
            "🚨 Critical Server Outage (High Priority Incident)",
            "🇩🇪 German Billing Inquiry (Routine Request)",
            "⚠️ Missing Subject Line Ticket"
        ]
    )
    
    default_subject = ""
    default_body = ""
    if "Critical Server Outage" in preset_choice:
        default_subject = "Urgent: Production Database Cluster Down"
        default_body = "The database cluster is unreachable. Multiple hospital telemedicine services are failing. Need immediate assistance."
    elif "German Billing" in preset_choice:
        default_subject = "Frage zu den Zahlungsoptionen"
        default_body = "Guten Tag, wir möchten unsere Abonnement-Zahlungsart auf SEPA umstellen. Können Sie uns bitte die Optionen erläutern? Danke."
    elif "Missing Subject" in preset_choice:
        default_subject = ""
        default_body = "User cannot reset password from the login screen. It shows error code 403."

    subject = st.text_input("Subject:", value=default_subject, placeholder="Enter ticket subject (can be blank)")
    body = st.text_area("Body:", value=default_body, height=180, placeholder="Paste customer message here...")
    
    submit = st.button("🚀 Process Ticket", type="primary", use_container_width=True)

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
                st.error(f"🚨 **ESCALATE TO ON-CALL ENGINEER**\n\n**Reason:** {result['escalation_reason']}")
            else:
                st.success(f"✅ **AUTO-DRAFT READY**\n\n**Reason:** {result['escalation_reason']}")

            # 3. Top Retrieved Historical Precedents
            st.divider()
            st.markdown("### 📚 Historical Precedents (RAG)")
            for prec in result.get("retrieved_precedents", []):
                st.markdown(f"**Ticket ID:** `{prec['ticket_id']}` | **Similarity:** `{prec['similarity']*100:.1f}%`")
                st.caption(f"**Past Resolution:** {prec['historical_answer']}")

            # 4. Generated Draft Response
            st.divider()
            st.markdown("### ✍️ Generated Draft Response")
            st.text_area("Agent Response:", value=result["draft_response"], height=140, disabled=True)
    else:
        st.info("Fill in the ticket on the left and click **Process Ticket** to view the triage output.")
