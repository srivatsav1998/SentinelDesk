import json
import os
import time

import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

from burst import BurstDetector  # noqa: E402
from graph import TOP_K, build_sentinel_graph  # noqa: E402
from safety_gate import CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD  # noqa: E402
from text_utils import ticket_text  # noqa: E402

st.set_page_config(page_title="SentinelDesk", layout="wide")
st.title("SentinelDesk - Incident Triage & Copilot")

TRIGGER_LABELS = {
    "high_priority_incident": "High-priority incident",
    "low_confidence": "Low model confidence",
    "missing_prediction": "Incomplete triage",
    "classifier_unavailable": "Classifier unavailable",
    "low_similarity": "No reliable precedent",
    "retrieval_unavailable": "Precedent search down",
    "unsupported_request": "Unsupported request",
    "prompt_injection": "Prompt injection",
    "gate_error": "Gate error",
}


@st.cache_resource(show_spinner="Loading models and precedent index (first start takes ~15 s)...")
def load_graph():
    """Build the graph once and warm every model so the first ticket is fast."""
    import retriever

    app = build_sentinel_graph()
    retriever.warm_up()
    saved_key = os.environ.get("GEMINI_API_KEY", "")
    os.environ["GEMINI_API_KEY"] = ""  # warm-up must not call the LLM
    try:
        app.invoke({"subject": "Warm-up", "body": "Login to the customer portal fails after password reset."})
    finally:
        os.environ["GEMINI_API_KEY"] = saved_key
    return app


@st.cache_resource
def burst_detector() -> BurstDetector:
    return BurstDetector()  # shared by all operator sessions on this server


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
@st.cache_resource
def server_api_key() -> str:
    """Key from .env / the server environment, captured once. It is never sent to the
    browser: the sidebar field only accepts an optional override."""
    return os.environ.get("GEMINI_API_KEY", "").strip()


with st.sidebar:
    st.subheader("Cloud Settings")
    override_key = st.text_input(
        "Gemini API Key (optional override):",
        type="password",
        value="",
        help="Leave blank to use the key configured on the server (.env).",
    ).strip()
    active_key = override_key or server_api_key()
    os.environ["GEMINI_API_KEY"] = active_key
    if active_key:
        st.success("Live Gemini drafting enabled" + ("" if override_key else " (key from server .env)"))
        if st.button("Test API Connection", width="stretch"):
            with st.spinner("Pinging Gemini API..."):
                try:
                    from google import genai

                    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").split(",")[0].strip()
                    t0 = time.perf_counter()
                    resp = genai.Client(api_key=active_key).models.generate_content(
                        model=model, contents="Ping. Respond with 'Connected'"
                    )
                    st.success(f"Connection verified with {model}: {resp.text.strip()} "
                               f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
                except Exception as e:
                    st.error(f"API Error: {e}")
    else:
        st.caption("Offline mode: drafts are extractive and built directly from retrieved precedents.")

    st.subheader("Safety Gate")
    st.caption(
        f"Confidence threshold **{CONFIDENCE_THRESHOLD:.2f}** · Precedent similarity threshold "
        f"**{SIMILARITY_THRESHOLD:.2f}** · Top-k **{TOP_K}**  \n"
        "Escalates on: high-priority incidents, low confidence, no reliable precedent, "
        "unsupported refund/compensation/guarantee/legal requests, prompt injection."
    )
    if st.button("Reset outage detector", width="stretch"):
        burst_detector().reset()
        st.toast("Outage detector cleared")

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS_FILE = os.path.join(ROOT, "data", "sample_ui_presets.json")
presets = []
if os.path.exists(PRESETS_FILE):
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        presets = json.load(f)

graph = load_graph()

col_in, col_out = st.columns([1, 1], gap="medium")

with col_in:
    st.subheader("1. Incoming Ticket")
    preset_options = ["-- Custom Ticket --"] + [p["name"] for p in presets]
    selected_option = st.selectbox("Load Benchmark Scenario:", preset_options)

    default_subject, default_body = "", ""
    if selected_option != "-- Custom Ticket --":
        matched = next((p for p in presets if p["name"] == selected_option), None)
        if matched:
            default_subject = matched.get("subject", "")
            default_body = matched.get("body", "").replace("\\n", "\n")
            if matched.get("notes"):
                st.caption(f"Benchmark Goal: {matched['notes']}")

    subject = st.text_input("Subject:", value=default_subject, placeholder="Enter ticket subject (can be blank)")
    body = st.text_area("Body:", value=default_body, height=220, placeholder="Paste customer message here...")
    submit = st.button("Process Ticket", type="primary", width="stretch")

with col_out:
    st.subheader("2. Triage & Copilot Results")

    if not submit:
        st.info("Fill in the ticket on the left and click Process Ticket to view the triage output.")
    elif not body.strip():
        st.warning("Please provide a ticket body.")
    else:
        with st.spinner("Processing through LangGraph..."):
            t0 = time.perf_counter()
            result = graph.invoke({"subject": subject, "body": body})
            wall_ms = (time.perf_counter() - t0) * 1000

        # Outage / burst detection (reuses the cached retrieval embedding)
        try:
            from retriever import embed_query

            alert = burst_detector().add(
                label=f"{result['type']} / {result['queue']}",
                subject=subject or body[:60],
                embedding=embed_query(ticket_text(subject, body)),
            )
        except Exception:
            alert = None
        if alert:
            st.error(
                f"MAJOR INCIDENT OUTAGE: {alert.count} tickets in the last {alert.window_minutes} min "
                f"describe the same failure. Page the incident commander."
            )
            with st.expander("Related tickets in this burst"):
                for m in alert.members:
                    st.write(f"- {m['subject']}  (similarity {m['similarity']:.2f}, {m['age_s']} s ago)")

        # 1. Predictions & confidence
        m1, m2 = st.columns(2)
        m1.metric("Type", result["type"])
        m1.caption(f"conf {result['type_confidence']:.2f}")
        m2.metric("Priority", result["priority"].upper())
        m2.caption(f"conf {result['priority_confidence']:.2f}")
        m3, m4 = st.columns(2)
        m3.metric("Confidence", f"{result['confidence'] * 100:.0f}%")
        m3.caption(f"language: {result['detected_language'].upper()}")
        m4.metric("Latency", f"{wall_ms / 1000:.2f} s" if wall_ms >= 1000 else f"{wall_ms:.0f} ms")
        m4.caption(
            f"retrieval {result.get('retrieval_ms', 0):.0f} ms"
            + (f" · draft {result['draft_ms']:.0f} ms" if result.get("draft_ms") else "")
        )
        st.metric("Queue", result["queue"])
        st.caption(f"conf {result['queue_confidence']:.2f}")
        errors = [f"{t}: {result[f'{t}_model_error']}" for t in ("type", "queue", "priority")
                  if result.get(f"{t}_model_error")]
        if errors:
            st.warning("Keyword fallback used for " + "; ".join(errors))

        # 2. Escalation badge
        st.divider()
        triggers = result.get("escalation_triggers") or []
        if result["escalation_status"] == "ESCALATE":
            st.error(f"STATUS: ESCALATE TO ON-CALL ENGINEER\n\nReason: {result['escalation_reason']}")
        else:
            st.success(f"STATUS: AUTO-DRAFT READY\n\nReason: {result['escalation_reason']}")
        if triggers:
            st.markdown(" ".join(f":red-badge[{TRIGGER_LABELS.get(t, t)}]" for t in triggers))

        # 3. Historical precedents
        st.divider()
        st.markdown("### Historical Precedents (RAG)")
        precedents = result.get("retrieved_precedents", [])
        if result.get("retrieval_error"):
            st.error(f"Precedent search failed: {result['retrieval_error']}")
        elif not precedents:
            st.caption("No precedent retrieved.")
        for prec in precedents:
            lang = f" · {prec['language'].upper()}" if prec.get("language") else ""
            st.markdown(
                f"**Ticket ID:** `{prec['ticket_id']}` | **Similarity:** `{prec['similarity'] * 100:.1f}%`{lang}"
                + (f" | {prec['subject'][:70]}" if prec.get("subject") else "")
            )
            answer = prec["historical_answer"]
            st.caption(f"**Past Resolution:** {answer[:400]}{'...' if len(answer) > 400 else ''}")

        # 4. Draft
        st.divider()
        st.markdown("### Generated Draft Response")
        st.text_area("Agent Response:", value=result.get("draft_response", ""), height=200, disabled=True)
        source = result.get("draft_source", "")
        if source == "escalation":
            st.caption("Holding reply (no claims or promises). Details for the on-call engineer below.")
            with st.expander("On-call handoff note", expanded=True):
                st.text(result.get("handoff_note", ""))
        else:
            label = source.replace("gemini:", "Gemini ") if source.startswith("gemini:") else "Extractive template"
            if result.get("draft_verified"):
                st.success(f"Grounding verifier passed · source: {label} · citations limited to retrieved tickets")
            else:
                st.warning(f"Grounding verifier issues ({label}): " + "; ".join(result.get("draft_issues", [])))
            if result.get("draft_rejected"):
                with st.expander(f"{len(result['draft_rejected'])} LLM draft(s) blocked by the verifier"):
                    for rejected in result["draft_rejected"]:
                        st.write(f"**{rejected['model']}**: " + "; ".join(rejected["issues"]))
                        st.caption(rejected["text"][:600])
            if result.get("llm_error"):
                st.caption(f"LLM not used: {result['llm_error']}")
            if result.get("pii_masked"):
                masked = ", ".join(f"{v} {k.lower()}" for k, v in result["pii_masked"].items())
                st.caption(f"PII masked before sending to the LLM: {masked}")
