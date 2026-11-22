import html
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

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=IBM+Plex+Mono:wght@500&display=swap');
:root {
  --ink: #14233B; --muted: #5B6B7F; --rule: #D5DDE6; --page: #F3F6F9; --panel: #FFFFFF;
  --escalate: #C2410C; --escalate-tint: #FFF1E8; --draft: #0F766E; --draft-tint: #E7F4F2;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; }
[data-testid="stAppDeployButton"] { display: none; }
.block-container { padding-top: 2.2rem; max-width: 1320px; }
button[kind="primary"] { background: var(--ink); border-color: var(--ink); font-weight: 700; }
button[kind="primary"]:hover { background: #22365A; border-color: #22365A; }

.sd-head { display: flex; align-items: baseline; gap: 1.25rem; flex-wrap: wrap;
  border-bottom: 2px solid var(--ink); padding-bottom: .9rem; margin-bottom: 1.6rem; }
.sd-brand { font-size: 2.1rem; font-weight: 700; color: var(--ink); letter-spacing: -0.01em; }
.sd-tag { color: var(--muted); font-size: 1.02rem; max-width: 62ch; }
.sd-col-title { font-size: 1.15rem; font-weight: 700; color: var(--ink); margin: 0 0 .6rem; }
.sd-empty { color: var(--muted); border: 1px dashed var(--rule); border-radius: 10px; padding: 2.2rem 1.4rem; }

.sd-decision { border-radius: 12px; padding: 1.25rem 1.4rem 1.1rem; margin-bottom: 1.1rem; border-left: 8px solid; }
.sd-decision.escalate { background: var(--escalate-tint); border-color: var(--escalate); }
.sd-decision.draft { background: var(--draft-tint); border-color: var(--draft); }
.sd-verdict { font-size: 1.75rem; font-weight: 700; line-height: 1.15; margin-bottom: .55rem; }
.escalate .sd-verdict { color: var(--escalate); }
.draft .sd-verdict { color: var(--draft); }
.sd-reasons { margin: 0; padding-left: 1.1rem; color: var(--ink); }
.sd-reasons li { margin: .18rem 0; }

.sd-outage { background: var(--escalate); color: #fff; border-radius: 12px; padding: .95rem 1.2rem; margin-bottom: 1rem; }
.sd-outage strong { font-size: 1.15rem; display: block; margin-bottom: .2rem; }

.sd-triage { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 1px;
  background: var(--rule); border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; margin-bottom: 1.4rem; }
.sd-cell { background: var(--panel); padding: .7rem .85rem; }
.sd-cell.wide { grid-column: span 2; }
.sd-label { color: var(--muted); font-size: .82rem; }
.sd-value { color: var(--ink); font-size: 1.12rem; font-weight: 700; line-height: 1.25; }
.sd-sub { color: var(--muted); font-size: .8rem; }

.sd-section { font-size: 1.05rem; font-weight: 700; color: var(--ink); margin: .4rem 0 .5rem; }
.sd-prec { border-top: 1px solid var(--rule); padding: .8rem 0 .7rem; }
.sd-prec-head { display: flex; gap: .6rem; align-items: baseline; flex-wrap: wrap; }
.sd-id { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .85rem; color: var(--ink);
  background: #E8EDF2; padding: .05rem .4rem; border-radius: 4px; }
.sd-subj { color: var(--ink); font-weight: 700; }
.sd-lang { color: var(--muted); font-size: .8rem; border: 1px solid var(--rule); border-radius: 4px; padding: 0 .3rem; }
.sd-bar { position: relative; height: 6px; background: #E3E9EF; border-radius: 3px; margin: .5rem 0 .25rem; max-width: 420px; }
.sd-fill { position: absolute; inset: 0 auto 0 0; background: var(--ink); border-radius: 3px; }
.sd-mark { position: absolute; top: -4px; width: 2px; height: 14px; background: var(--escalate); }
.sd-sim { color: var(--muted); font-size: .82rem; }
.stApp .sd-answer { color: #33445A; font-size: .92rem; line-height: 1.5; margin: .35rem 0 0; max-width: 78ch; }
.stApp .sd-reasons li, .stApp .sd-check { font-size: .98rem; }

.sd-check { border-radius: 8px; padding: .6rem .8rem; font-size: .92rem; margin: .4rem 0 .6rem; }
.sd-check.ok { background: var(--draft-tint); color: #0B4F49; }
.sd-check.warn { background: var(--escalate-tint); color: #8A2E08; }
@media (max-width: 640px) { .sd-cell.wide { grid-column: span 1; } .sd-verdict { font-size: 1.4rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

esc = html.escape


def reason_lines(result: dict) -> list[str]:
    """Plain-language reasons for the gate decision."""
    triggers = result.get("escalation_triggers") or []
    sim = result.get("top_similarity")
    conf = result.get("confidence", 0.0)
    lines = []
    for trigger in triggers:
        if trigger == "high_priority_incident":
            lines.append("Predicted a high-priority incident, which always goes to a person.")
        elif trigger == "low_confidence":
            lines.append(f"Triage confidence {conf:.0%} is below the {CONFIDENCE_THRESHOLD:.0%} threshold.")
        elif trigger == "low_similarity":
            lines.append(f"Closest past case is only {sim:.0%} similar (threshold {SIMILARITY_THRESHOLD:.0%}), "
                         "so a reply cannot be grounded." if sim is not None
                         else "No similar past case was found, so a reply cannot be grounded.")
        elif trigger == "unsupported_request":
            phrases = {"refund": "a refund", "compensation": "compensation", "guarantee": "a guarantee",
                       "legal": "legal action"}
            asks = [phrases.get(c, c) for c in (result.get("unsupported_requests") or ["a sensitive action"])]
            joined = asks[0] if len(asks) == 1 else ", ".join(asks[:-1]) + " and " + asks[-1]
            lines.append(f"Mentions {joined}, which no past resolution supports.")
        elif trigger == "prompt_injection":
            lines.append("The ticket tries to override the assistant's instructions.")
        elif trigger == "retrieval_unavailable":
            lines.append("Past-case search is unavailable, so a reply cannot be grounded.")
        elif trigger == "classifier_unavailable":
            lines.append("A triage model is unavailable and a keyword fallback was used.")
        else:
            lines.append("Triage output was incomplete.")
    if not lines:
        lines.append(f"Confidence {conf:.0%} and the closest past case is {sim:.0%} similar." if sim is not None
                     else f"Confidence {conf:.0%}.")
        lines.append("No high-priority incident, sensitive request or injection attempt detected.")
    return lines


@st.cache_resource(show_spinner="Loading models and the past-case index (about 15 seconds on first start)...")
def load_graph():
    """Build the graph once and warm every model so the first ticket is fast."""
    import retriever

    app = build_sentinel_graph()
    retriever.warm_up()
    app.invoke({"subject": "Warm-up", "body": "Login to the customer portal fails after password reset."},
               config={"configurable": {"gemini_api_key": ""}})  # warm-up never calls the LLM
    return app


@st.cache_resource
def burst_detector() -> BurstDetector:
    return BurstDetector()  # shared by all operator sessions on this server (subjects are PII-masked)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Drafting")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").split(",")[0].strip()
    if os.environ.get("GEMINI_API_KEY", "").strip():
        st.write(f"Replies are drafted by **{model}** and checked before they are shown.")
    else:
        st.write("No Gemini key on the server: replies are built directly from past resolutions.")
    st.markdown("### When a ticket goes to a person")
    st.write(
        f"- High-priority incidents\n"
        f"- Triage confidence below {CONFIDENCE_THRESHOLD:.0%}\n"
        f"- No past case at least {SIMILARITY_THRESHOLD:.0%} similar\n"
        f"- Refund, compensation, guarantee or legal requests no past case supports\n"
        f"- Attempts to override the assistant"
    )
    st.caption(f"Each ticket is compared with the {TOP_K} most similar resolved tickets from the historical 80% split.")
    if st.button("Clear outage detector", width="stretch"):
        burst_detector().reset()
        st.toast("Outage detector cleared")

# ---------------------------------------------------------------------------
# Header + presets
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sd-head"><div class="sd-brand">SentinelDesk</div>'
    '<div class="sd-tag">Triage incoming tickets, send the risky ones to a person, and draft replies '
    'grounded in how similar cases were resolved.</div></div>',
    unsafe_allow_html=True,
)

PRESETS_FILE = os.path.join(ROOT, "data", "sample_ui_presets.json")
presets = []
if os.path.exists(PRESETS_FILE):
    with open(PRESETS_FILE, "r", encoding="utf-8") as f:
        presets = json.load(f)

graph = load_graph()
col_in, col_out = st.columns([5, 7], gap="large")

with col_in:
    st.markdown('<div class="sd-col-title">Incoming ticket</div>', unsafe_allow_html=True)
    options = ["Write your own"] + [p["name"] for p in presets]
    selected = st.selectbox("Load a sample ticket", options)
    default_subject, default_body = "", ""
    if selected != "Write your own":
        matched = next((p for p in presets if p["name"] == selected), None)
        if matched:
            default_subject = matched.get("subject", "")
            default_body = matched.get("body", "").replace("\\n", "\n")
            if matched.get("notes"):
                st.caption(f"What this shows: {matched['notes']}")
    subject = st.text_input("Subject", value=default_subject, placeholder="Optional")
    body = st.text_area("Message", value=default_body, height=260, placeholder="Paste the customer's message")
    submit = st.button("Triage ticket", type="primary", width="stretch")

with col_out:
    st.markdown('<div class="sd-col-title">Triage and reply</div>', unsafe_allow_html=True)

    if not submit:
        st.markdown('<div class="sd-empty">Paste a ticket or load a sample, then select <b>Triage ticket</b>.</div>',
                    unsafe_allow_html=True)
    elif not body.strip():
        st.warning("Add the customer's message before triaging.")
    else:
        with st.spinner("Classifying, searching past cases and checking safety rules..."):
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
            st.markdown(
                f'<div class="sd-outage"><strong>Possible major outage</strong>{alert.count} tickets in the last '
                f'{alert.window_minutes} minutes describe the same failure. Page the incident commander.</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Tickets in this cluster"):
                for m in alert.members:
                    st.write(f"- {m['subject']} ({m['similarity']:.0%} similar, {m['age_s']} s ago)")

        # Decision
        escalated = result["escalation_status"] == "ESCALATE"
        verdict = "Escalate to on-call engineer" if escalated else "Auto-draft ready"
        reasons = "".join(f"<li>{esc(line)}</li>" for line in reason_lines(result))
        st.markdown(
            f'<section class="sd-decision {"escalate" if escalated else "draft"}">'
            f'<div class="sd-verdict">{verdict}</div><ul class="sd-reasons">{reasons}</ul></section>',
            unsafe_allow_html=True,
        )

        # Triage strip
        latency = f"{wall_ms / 1000:.2f} s" if wall_ms >= 1000 else f"{wall_ms:.0f} ms"
        draft_ms = f", drafting {result['draft_ms']:.0f} ms" if result.get("draft_ms") else ""
        cells = [
            ("Type", result["type"], f"{result['type_confidence']:.0%} sure", ""),
            ("Priority", result["priority"].capitalize(), f"{result['priority_confidence']:.0%} sure", ""),
            ("Queue", result["queue"], f"{result['queue_confidence']:.0%} sure", "wide"),
            ("Language", {"de": "German", "en": "English"}.get(result["detected_language"], "English"), "detected", ""),
            ("Confidence", f"{result['confidence']:.0%}", f"threshold {CONFIDENCE_THRESHOLD:.0%}", ""),
            ("Time", latency, f"search {result.get('retrieval_ms', 0):.0f} ms{draft_ms}", "wide"),
        ]
        st.markdown(
            '<div class="sd-triage">' + "".join(
                f'<div class="sd-cell {cls}"><div class="sd-label">{label}</div>'
                f'<div class="sd-value">{esc(str(value))}</div><div class="sd-sub">{esc(sub)}</div></div>'
                for label, value, sub, cls in cells
            ) + "</div>",
            unsafe_allow_html=True,
        )
        fallbacks = [t for t in ("type", "queue", "priority") if result.get(f"{t}_model_error")]
        if fallbacks:
            st.warning(f"Keyword fallback used for: {', '.join(fallbacks)}.")

        # Precedents
        st.markdown('<div class="sd-section">Similar resolved tickets</div>', unsafe_allow_html=True)
        precedents = result.get("retrieved_precedents", [])
        if result.get("retrieval_error"):
            st.error(f"Past-case search failed: {result['retrieval_error']}")
        elif not precedents:
            st.caption("No similar resolved ticket found.")
        blocks = []
        for p in precedents:
            sim = float(p["similarity"])
            # Blank lines would end the HTML block in Streamlit's markdown, so flatten them.
            answer = " ".join(p["historical_answer"].split())
            answer = answer[:420] + ("..." if len(answer) > 420 else "")
            lang = f'<span class="sd-lang">{esc(p["language"].upper())}</span>' if p.get("language") else ""
            blocks.append(
                f'<div class="sd-prec"><div class="sd-prec-head"><span class="sd-id">{esc(p["ticket_id"])}</span>'
                f'<span class="sd-subj">{esc(p.get("subject") or "")}</span>{lang}</div>'
                f'<div class="sd-bar" title="Similarity; the orange mark is the escalation threshold">'
                f'<div class="sd-fill" style="width:{sim * 100:.1f}%"></div>'
                f'<div class="sd-mark" style="left:{SIMILARITY_THRESHOLD * 100:.0f}%"></div></div>'
                f'<div class="sd-sim">{sim:.0%} similar</div><p class="sd-answer">{esc(answer)}</p></div>'
            )
        st.markdown("".join(blocks), unsafe_allow_html=True)

        # Reply
        source = result.get("draft_source", "")
        if source == "escalation":
            st.markdown('<div class="sd-section">Holding reply to the customer</div>', unsafe_allow_html=True)
            st.text_area("Holding reply", value=result.get("draft_response", ""), height=150,
                         label_visibility="collapsed")
            st.caption("Acknowledges the ticket without promises. The engineer gets the note below.")
            with st.expander("Handoff note for the on-call engineer", expanded=True):
                st.text(result.get("handoff_note", ""))
        else:
            st.markdown('<div class="sd-section">Draft reply</div>', unsafe_allow_html=True)
            st.text_area("Draft reply", value=result.get("draft_response", ""), height=220,
                         label_visibility="collapsed")
            by = (f"Drafted by {source.split(':', 1)[1]}" if source.startswith("gemini:")
                  else "Built directly from past resolutions")
            if result.get("draft_verified"):
                st.markdown(
                    f'<div class="sd-check ok"><b>Grounding check passed.</b> {esc(by)}. It cites only the tickets '
                    'above and adds no prices, percentages, deadlines, links or promises they do not contain.</div>',
                    unsafe_allow_html=True,
                )
            else:
                issues = "; ".join(result.get("draft_issues") or [])
                st.markdown(f'<div class="sd-check warn"><b>Grounding check failed:</b> {esc(issues)}</div>',
                            unsafe_allow_html=True)
            if result.get("draft_rejected"):
                with st.expander(f"{len(result['draft_rejected'])} AI draft(s) blocked by the grounding check"):
                    for rejected in result["draft_rejected"]:
                        st.write(f"**{rejected['model']}**: " + "; ".join(rejected["issues"]))
                        st.caption(rejected["text"][:600])
            if result.get("llm_error"):
                st.caption(f"Gemini not used: {result['llm_error']}")
            if result.get("pii_masked"):
                masked = ", ".join(f"{count} {kind.lower()}" for kind, count in result["pii_masked"].items())
                st.caption(f"Personal data masked before the AI saw the ticket: {masked}.")
