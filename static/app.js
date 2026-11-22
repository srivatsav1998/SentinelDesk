"use strict";
/* SentinelDesk console. All ticket text is inserted with textContent, never innerHTML. */

const $ = (selector, root = document) => root.querySelector(selector);
const SVG_NS = "http://www.w3.org/2000/svg";

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "on") for (const [event, fn] of Object.entries(value)) el.addEventListener(event, fn);
    else if (key === "style") Object.assign(el.style, value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== undefined && value !== null) el.setAttribute(key, value);
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

const ICON_CHECK = () => s("svg", { viewBox: "0 0 20 20", width: 16, height: 16, "aria-hidden": "true" },
  s("path", { d: "M4 10.5l4 4 8-9", fill: "none", stroke: "currentColor", "stroke-width": 2.2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
const ICON_CROSS = () => s("svg", { viewBox: "0 0 20 20", width: 16, height: 16, "aria-hidden": "true" },
  s("path", { d: "M5 5l10 10M15 5L5 15", fill: "none", stroke: "currentColor", "stroke-width": 2.2, "stroke-linecap": "round" }));

const TYPES = ["Incident", "Request", "Problem", "Change"];
const PRIORITIES = ["high", "medium", "low"];
const QUEUES = ["Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments",
  "Returns and Exchanges", "Service Outages and Maintenance", "Sales and Pre-Sales", "Human Resources", "General Inquiry"];
const TRIGGER_LABELS = {
  high_priority_incident: "High-priority incident", low_confidence: "Low confidence", low_similarity: "No reliable precedent",
  unsupported_request: "Unsupported request", prompt_injection: "Prompt injection", retrieval_unavailable: "Search unavailable",
  classifier_unavailable: "Model unavailable", missing_prediction: "Incomplete triage", gate_error: "Gate error",
};
const LANGUAGE_NAMES = { en: "English", de: "German" };

const state = { presets: [], health: null, last: null, ticket: null, metrics: null, metricsLoaded: false,
  inbox: [], queue: [], streaming: false };

const pct = (x, digits = 0) => (x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`);
const ms = (x) => (x === null || x === undefined ? "–" : x < 1 ? "<1 ms" : x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${Math.round(x)} ms`);
const thresholds = () => state.health?.thresholds || { confidence: 0.65, similarity: 0.58 };

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 3200);
}

/* ---------------- Theme, tabs, health ---------------- */
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#theme-toggle").setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  try { localStorage.setItem("sentineldesk-theme", theme); } catch (_) { /* storage unavailable */ }
}

function showView(view) {
  for (const tab of document.querySelectorAll(".tabs button")) tab.setAttribute("aria-selected", String(tab.dataset.view === view));
  for (const name of ["triage", "inbox", "evaluation"]) $(`#view-${name}`).hidden = name !== view;
  if (view === "evaluation" && !state.metricsLoaded) loadMetrics();
}

function setPill(id, cls, text) {
  const pill = $(id);
  pill.className = `pill ${cls}`;
  $(".pill-text", pill).textContent = text;
}

async function refreshHealth() {
  try {
    state.health = await api("/api/health");
    const hl = state.health;
    if (hl.error) setPill("#pill-models", "bad", "Model error");
    else setPill("#pill-models", hl.ready ? "ok" : "wait", hl.ready ? "Models ready" : "Loading models");
    setPill("#pill-gemini", hl.gemini ? "ok" : "", hl.gemini ? `Gemini: ${hl.model}` : "Offline drafts");
    setPill("#pill-feedback", "", `${hl.feedback} operator ${hl.feedback === 1 ? "correction" : "corrections"}`);
    $("#compare").disabled = !hl.gemini;
    setTimeout(refreshHealth, hl.ready ? 15000 : 2000);
  } catch (_) {
    setPill("#pill-models", "bad", "Server offline");
    setTimeout(refreshHealth, 4000);
  }
}

/* ---------------- Samples ---------------- */
async function loadPresets() {
  try { state.presets = await api("/api/presets"); } catch (_) { state.presets = []; }
  const list = $("#sample-list");
  list.replaceChildren(...state.presets.map((preset, index) => {
    const short = preset.name.replace(/\s*\(.*\)\s*$/, "");
    const dotClass = preset.expected === "ESCALATE" ? "esc" : preset.expected === "AUTO_DRAFT" ? "draft" : "";
    return h("button", {
      class: "sample", type: "button", "aria-pressed": "false", title: preset.notes || preset.name,
      on: { click: (event) => selectPreset(index, event.currentTarget) },
    }, h("span", { class: `dot ${dotClass}`, "aria-hidden": "true" }), short);
  }));
}

function selectPreset(index, button) {
  const preset = state.presets[index];
  $("#subject").value = preset.subject;
  $("#body").value = preset.body;
  $("#sample-note").textContent = preset.notes ? `What this shows: ${preset.notes}` : "";
  for (const chip of document.querySelectorAll(".sample")) chip.setAttribute("aria-pressed", String(chip === button));
}

/* ---------------- Triage ---------------- */
async function renderEmpty() {
  const stats = h("div", { class: "stat-row" });
  $("#results").replaceChildren(h("div", { class: "empty" },
    h("h2", { text: "Paste a ticket, or pick a sample" }),
    h("p", { text: "SentinelDesk classifies it, finds similar resolved tickets, and then decides in code whether a person must handle it. Only safe tickets get an AI reply, and every reply is checked before you see it." }),
    h("ol", {},
      h("li", { text: "Orange dot samples are sent to an on-call engineer." }),
      h("li", { text: "Teal dot samples get a grounded reply." }),
      h("li", { text: "Turn on the plain chatbot comparison to see what an unchecked bot would have said." })),
    stats));
  try {
    const m = await api("/api/metrics");
    const ho = m.holdout;
    if (!ho) return;
    const e2e = ho.end_to_end || {};
    const stat = (value, label) => h("div", { class: "stat" }, h("b", { text: value }), h("span", { text: label }));
    stats.replaceChildren(
      stat(ho.leakage.indexed_tickets.toLocaleString(), "resolved tickets to learn from"),
      stat(pct(ho.high_priority_incident.escalation_recall), "of urgent incidents reach a person"),
      stat(e2e.draft_verified_rate === undefined ? "–" : pct(e2e.draft_verified_rate), "of AI drafts pass the grounding check"),
      stat(e2e.latency_ms_p50 ? ms(e2e.latency_ms_p50) : "–", "median time per ticket"));
  } catch (_) { /* stats are optional */ }
}

function renderLoading() {
  const steps = ["Detecting language", "Classifying type, priority and queue", "Searching 49,412 resolved tickets",
    "Applying safety rules", $("#compare").checked ? "Drafting and asking a plain chatbot for comparison" : "Drafting and checking the reply"];
  $("#results").replaceChildren(h("div", { class: "panel loading" },
    h("div", { class: "loading-steps" }, steps.map((step) => h("div", { class: "loading-step" }, h("span", { class: "spin" }), step))),
    h("div", { class: "skeleton", style: { width: "70%" } }), h("div", { class: "skeleton", style: { width: "45%" } })));
}

async function runTriage() {
  const subject = $("#subject").value;
  const body = $("#body").value;
  if (!body.trim()) { toast("Add the customer's message first."); $("#body").focus(); return; }
  const run = $("#run");
  run.disabled = true;
  renderLoading();
  try {
    const result = await post("/api/triage", { subject, body, compare: $("#compare").checked, use_llm: true });
    state.last = result;
    state.ticket = { subject, body };
    renderResult(result, state.ticket);
    showOutage(result.outage);
  } catch (error) {
    $("#results").replaceChildren(h("div", { class: "panel section" }, h("h3", { text: "Triage failed" }), h("p", { text: error.message })));
  } finally {
    run.disabled = false;
  }
}

function reasonLines(r) {
  const t = thresholds();
  const lines = [];
  for (const trigger of r.triggers) {
    if (trigger === "high_priority_incident") lines.push("Predicted a high-priority incident, which always goes to a person.");
    else if (trigger === "low_confidence") lines.push(`Triage confidence ${pct(r.confidence)} is below the ${pct(t.confidence)} threshold.`);
    else if (trigger === "low_similarity") lines.push(r.top_similarity === null
      ? "No similar resolved ticket exists, so a reply cannot be grounded."
      : `The closest resolved ticket is only ${pct(r.top_similarity)} similar (threshold ${pct(t.similarity)}), so a reply cannot be grounded.`);
    else if (trigger === "unsupported_request") {
      const words = { refund: "a refund", compensation: "compensation", guarantee: "a guarantee", legal: "legal action" };
      const asks = (r.unsupported.length ? r.unsupported : ["a sensitive action"]).map((c) => words[c] || c);
      const joined = asks.length === 1 ? asks[0] : `${asks.slice(0, -1).join(", ")} and ${asks.at(-1)}`;
      lines.push(`Mentions ${joined}, which no past resolution supports.`);
    } else if (trigger === "prompt_injection") lines.push("The ticket tries to override the assistant's instructions.");
    else if (trigger === "retrieval_unavailable") lines.push("Past-ticket search is unavailable, so a reply cannot be grounded.");
    else if (trigger === "classifier_unavailable") lines.push("A triage model is unavailable and a keyword fallback was used.");
    else lines.push("Triage output was incomplete.");
  }
  if (!lines.length) {
    lines.push(`Confidence ${pct(r.confidence)} and the closest resolved ticket is ${pct(r.top_similarity)} similar.`);
    lines.push("No high-priority incident, sensitive request or injection attempt detected.");
  }
  return lines;
}

function renderResult(r, ticket) {
  const blocks = [decisionBlock(r), cardsBlock(r), traceBlock(r), replyBlock(r, ticket)];
  if (r.plain_chatbot) blocks.push(compareBlock(r));
  blocks.push(noticedBlock(r, ticket), precedentsBlock(r));
  $("#results").replaceChildren(...blocks);
}

function decisionBlock(r) {
  const esc = r.status === "ESCALATE";
  return h("section", { class: `decision ${esc ? "esc" : "draft"}` },
    h("div", {},
      h("div", { class: "verdict", text: esc ? "Escalate to on-call engineer" : "Auto-draft ready" }),
      h("ul", { class: "reasons" }, reasonLines(r).map((line) => h("li", { text: line }))),
      r.triggers.length ? h("div", { class: "chips" }, r.triggers.map((t) => h("span", { class: "chip esc", text: TRIGGER_LABELS[t] || t }))) : null),
    h("div", { class: "decision-facts" },
      h("span", { class: "big-num", text: pct(r.confidence) }), h("span", { class: "facts-label", text: "routing confidence" }),
      h("span", { class: "big-num", style: { marginTop: ".5rem" }, text: ms(r.timings.wall) }), h("span", { class: "facts-label", text: "end to end" })));
}

function meter(value, threshold) {
  const low = value !== null && value < threshold;
  return [h("div", { class: `meter ${low ? "low" : ""}` }, h("span", { style: { width: `${Math.round((value || 0) * 100)}%` } })),
    h("div", { class: "meter-label" }, h("span", { text: "calibrated confidence" }), h("span", { text: pct(value) }))];
}

function cardsBlock(r) {
  const t = thresholds();
  const card = (label, value, confidence, extra) => h("div", { class: "panel card" },
    h("div", { class: "k", text: label }), h("div", { class: "v", text: value }), meter(confidence, t.confidence), extra || null);
  const because = r.queue_terms?.length
    ? h("div", { class: "because" }, "Because of ", h("span", { class: "chips", style: { display: "inline-flex", marginTop: 0 } },
      r.queue_terms.map((term) => h("span", { class: "chip", text: term }))))
    : null;
  return h("div", { class: "cards" },
    card("Type", r.type, r.type_confidence),
    card("Priority", r.priority ? r.priority[0].toUpperCase() + r.priority.slice(1) : "–", r.priority_confidence),
    card("Queue", r.queue, r.queue_confidence, because));
}

function traceBlock(r) {
  const T = r.timings;
  const parallel = [["Type model", T.type, ""], ["Priority model", T.priority, ""], ["Queue model", T.queue, ""],
    ["Past-ticket search", T.retrieval, "search"]];
  const parallelEnd = Math.max(...parallel.map(([, d]) => d || 0));
  const esc = r.status === "ESCALATE";
  const gateDur = 0.4;
  const finalDur = esc ? 0.4 : (T.draft || 0);
  const finalLabel = esc ? "Escalation note" : r.draft_source.startsWith("gemini:") ? "Gemini draft + check" : "Grounded draft + check";
  const total = Math.max(T.graph || 0, parallelEnd + gateDur + finalDur, 1);
  const rows = [...parallel.map(([name, d, cls]) => [name, 0, d || 0, cls]),
    ["Safety gate", parallelEnd, gateDur, "gate"], [finalLabel, parallelEnd + gateDur, finalDur, esc ? "esc" : "draft"]];
  const grid = h("div", { class: "trace", role: "img", "aria-label": `Pipeline trace, ${ms(total)} total` },
    rows.flatMap(([name, start, duration, cls]) => [
      h("span", { class: "name", text: name }),
      h("span", { class: "lane" }, h("span", { class: `bar ${cls}`, style: {
        left: `${(start / total) * 100}%`, width: `${Math.max((duration / total) * 100, 0.6)}%` } })),
      h("span", { class: "ms", text: ms(duration) })]));
  return h("section", { class: "panel section" },
    h("div", { class: "section-head" }, h("h3", { text: "Pipeline trace" }),
      h("p", { text: `Language detected first (${LANGUAGE_NAMES[r.language] || r.language}). The four models and the search ran in parallel. Pipeline time ${ms(T.graph)}.` })),
    grid,
    h("div", { class: "trace-axis" }, h("span", { text: "0 ms" }), h("span", { text: ms(total / 2) }), h("span", { text: ms(total) })));
}

const CHECKS = [
  ["Cites only retrieved tickets", /cite/], ["No new numbers", /number/], ["No outside links or contacts", /link|contact/],
  ["No unsupported promises", /commits/], ["Customer's language", /written in/], ["No placeholders", /placeholder/],
];

function replyBlock(r, ticket) {
  const esc = r.status === "ESCALATE";
  const predicted = { type: r.type, priority: r.priority, queue: r.queue, status: r.status };
  const draftBox = h("textarea", { class: "draft-box", id: "draft-text", "aria-label": esc ? "Holding reply" : "Draft reply" });
  draftBox.value = r.draft;
  const correctForm = h("div", { class: "correct-form", hidden: true },
    selectField("Type", "fix-type", TYPES, r.type), selectField("Priority", "fix-priority", PRIORITIES, r.priority),
    selectField("Queue", "fix-queue", QUEUES, r.queue),
    h("button", { type: "button", on: { click: () => sendFeedback("correct", ticket, predicted, draftBox.value, {
      type: $("#fix-type").value, priority: $("#fix-priority").value, queue: $("#fix-queue").value }) } }, "Save correction"));
  const toggleCorrect = h("button", { type: "button", class: "ghost", on: { click: () => { correctForm.hidden = !correctForm.hidden; } } }, "Correct triage");

  if (esc) {
    return h("section", { class: "panel section" },
      h("div", { class: "section-head" }, h("h3", { text: "Holding reply to the customer" }),
        h("p", { text: "Acknowledges the ticket without promises. The engineer gets the handoff note." })),
      draftBox,
      h("pre", { class: "handoff", text: r.handoff_note }),
      h("div", { class: "actions-row" },
        h("button", { type: "button", class: "btn-esc", on: { click: () => sendFeedback("escalate", ticket, predicted, draftBox.value) } }, "Confirm escalation"),
        toggleCorrect),
      correctForm);
  }

  const issues = r.draft_issues.join(" | ").toLowerCase();
  const source = r.draft_source.startsWith("gemini:") ? `Drafted by ${r.draft_source.slice(7)}` : "Built directly from past resolutions";
  const cited = [...new Set((r.draft.match(/TKT-\d{5}/g) || []))];
  return h("section", { class: "panel section" },
    h("div", { class: "section-head" }, h("h3", { text: "Draft reply" }), h("p", { text: source })),
    draftBox,
    h("div", { class: "checks" }, CHECKS.map(([label, pattern]) => {
      const failed = pattern.test(issues);
      return h("span", { class: `chip ${failed ? "bad" : "ok"}` }, failed ? ICON_CROSS() : ICON_CHECK(), label);
    })),
    r.draft_verified
      ? h("div", { class: "verify ok" }, ICON_CHECK(), h("span", { text: "Grounding check passed. Every statement comes from the cited tickets, and nothing the customer typed was turned into a promise." }))
      : h("div", { class: "verify bad" }, ICON_CROSS(), h("span", { text: `Grounding check failed: ${r.draft_issues.join("; ")}` })),
    cited.length ? h("div", { class: "chips" }, h("span", { class: "meta", style: { marginTop: 0 }, text: "Cites" }),
      cited.map((id) => h("button", { type: "button", class: "chip", on: { click: () => focusPrecedent(id) } }, id))) : null,
    r.draft_rejected.length ? h("details", {},
      h("summary", { text: `${r.draft_rejected.length} AI draft${r.draft_rejected.length > 1 ? "s" : ""} blocked by the grounding check` }),
      r.draft_rejected.map((d) => h("div", { class: "rejected" }, h("strong", { text: d.issues.join("; ") }), h("p", { text: d.text })))) : null,
    r.llm_error ? h("p", { class: "meta", text: `Gemini not used: ${r.llm_error}` }) : null,
    Object.keys(r.pii_masked).length ? h("p", { class: "meta", text: `Personal data masked before the AI saw the ticket: ${Object.entries(r.pii_masked).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(", ")}.` }) : null,
    h("div", { class: "actions-row" },
      h("button", { type: "button", class: "btn-ok", on: { click: () => sendFeedback(draftBox.value.trim() === r.draft.trim() ? "approve" : "edit", ticket, predicted, draftBox.value) } }, "Approve and send"),
      h("button", { type: "button", class: "btn-esc", on: { click: () => sendFeedback("escalate", ticket, predicted, draftBox.value) } }, "Escalate instead"),
      toggleCorrect),
    correctForm);
}

function selectField(label, id, options, value) {
  return h("label", { for: id }, label, h("select", { id }, options.map((o) => h("option", { value: o, selected: o === value }, o))));
}

function compareBlock(r) {
  const pc = r.plain_chatbot;
  const esc = r.status === "ESCALATE";
  const left = h("div", { class: `compare-col ${pc.issues?.length ? "bad" : ""}` },
    h("h3", { text: "Plain chatbot" }), h("span", { class: "tag", text: "Gemini with no retrieval, no rules and no checks" }),
    pc.text ? h("div", { class: "reply", text: pc.text }) : h("p", { class: "meta", text: pc.error || "No reply." }),
    pc.issues?.length ? [h("p", { class: "meta", style: { color: "var(--danger)" }, text: "Our grounding check would block it:" }),
      h("ul", {}, pc.issues.map((issue) => h("li", { text: issue })))] : pc.text ? h("p", { class: "meta", text: "Our grounding check found no problems this time." }) : null);
  const right = h("div", { class: "compare-col good" },
    h("h3", { text: "SentinelDesk" }),
    h("span", { class: "tag", text: esc ? "Sent to a person instead of answering" : "Grounded in retrieved tickets and checked" }),
    h("div", { class: "reply", text: r.draft }),
    esc ? h("p", { class: "meta", text: reasonLines(r)[0] }) : h("p", { class: "meta", text: "Passed the grounding check." }));
  return h("section", { class: "panel section" },
    h("div", { class: "section-head" }, h("h3", { text: "Same ticket: plain chatbot vs SentinelDesk" }),
      h("p", { text: "Both replies were generated just now." })),
    h("div", { class: "compare" }, left, right));
}

function highlighted(text, spans) {
  const parts = [];
  let cursor = 0;
  for (const span of spans) {
    if (span.start > cursor) parts.push(document.createTextNode(text.slice(cursor, span.start)));
    parts.push(h("mark", { class: span.kind, title: span.label, text: text.slice(span.start, span.end) }));
    cursor = span.end;
  }
  if (cursor < text.length) parts.push(document.createTextNode(text.slice(cursor)));
  return parts;
}

function noticedBlock(r, ticket) {
  const spans = [...r.highlights.subject, ...r.highlights.body];
  const kinds = new Set(spans.map((span) => span.kind));
  return h("section", { class: "panel section" },
    h("div", { class: "section-head" }, h("h3", { text: "What the system noticed" }),
      h("p", { text: spans.length ? "Hover a highlight to see why it matters." : "No sensitive requests, injection attempts or personal data found." })),
    h("div", { class: "ticket-text" },
      ticket.subject ? h("span", { class: "subj" }, highlighted(ticket.subject, r.highlights.subject)) : null,
      highlighted(ticket.body, r.highlights.body)),
    spans.length ? h("div", { class: "legend" },
      kinds.has("sensitive") ? h("span", {}, h("i", { class: "sensitive" }), "sensitive request") : null,
      kinds.has("injection") ? h("span", {}, h("i", { class: "injection" }), "prompt injection") : null,
      kinds.has("pii") ? h("span", {}, h("i", { class: "pii" }), "personal data, masked before the AI") : null) : null);
}

function precedentsBlock(r) {
  const t = thresholds();
  if (r.retrieval_error) return h("section", { class: "panel section" }, h("h3", { text: "Similar resolved tickets" }), h("p", { text: `Search failed: ${r.retrieval_error}` }));
  return h("section", { class: "panel section" },
    h("div", { class: "section-head" }, h("h3", { text: "Similar resolved tickets" }),
      h("p", { text: `Top ${r.precedents.length} of 49,412 historical tickets. The orange mark is the ${pct(t.similarity)} threshold.` })),
    r.precedents.length ? h("div", { class: "precedents" }, r.precedents.map((p) => {
      const answer = p.historical_answer || "";
      const short = answer.length > 360 ? `${answer.slice(0, 360)}...` : answer;
      return h("article", { class: "prec", id: `prec-${p.ticket_id}` },
        h("div", { class: "prec-head" }, h("span", { class: "tkt", text: p.ticket_id }), h("span", { class: "prec-title", text: p.subject || "Untitled ticket" }),
          p.language ? h("span", { class: "lang", text: p.language.toUpperCase() }) : null, p.queue ? h("span", { class: "chip", text: p.queue }) : null),
        h("div", { class: "simrow" },
          h("div", { class: "simtrack" }, h("span", { class: "simfill", style: { width: `${p.similarity * 100}%` } }),
            h("span", { class: "simmark", style: { left: `${t.similarity * 100}%` } })),
          h("span", { class: "simval", text: `${pct(p.similarity, 1)} similar` })),
        answer.length > 360 ? h("details", {}, h("summary", { text: short }), h("p", { text: answer })) : h("p", { text: answer }));
    })) : h("p", { class: "meta", text: "No similar resolved ticket found." }));
}

function focusPrecedent(id) {
  const el = document.getElementById(`prec-${id}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

async function sendFeedback(action, ticket, predicted, draft, correction) {
  try {
    const data = await post("/api/feedback", { action, subject: ticket.subject, body: ticket.body, predicted, draft, correction });
    const verb = { approve: "Approved", edit: "Edited reply approved", escalate: "Escalation confirmed", correct: "Correction saved" }[action];
    toast(`${verb}. ${data.count} operator ${data.count === 1 ? "decision" : "decisions"} saved for retraining.`);
    setPill("#pill-feedback", "", `${data.count} operator ${data.count === 1 ? "correction" : "corrections"}`);
  } catch (error) {
    toast(error.message);
  }
}

function showOutage(outage) {
  const banner = $("#outage");
  if (!outage) return;
  $("#outage-text").textContent = `${outage.count} tickets in the last ${outage.window_minutes} minutes describe the same failure. Page the incident commander.`;
  banner.hidden = false;
}

/* ---------------- Live inbox ---------------- */
function renderInboxStats() {
  const rows = state.inbox;
  const escalated = rows.filter((r) => r.result.status === "ESCALATE").length;
  const urgent = rows.filter((r) => r.result.type === "Incident" && r.result.priority === "high").length;
  const times = rows.map((r) => r.result.timings.wall).sort((a, b) => a - b);
  const median = times.length ? times[Math.floor(times.length / 2)] : null;
  const kpi = (label, value, cls) => h("div", { class: "panel kpi" }, h("div", { class: "k", text: label }), h("div", { class: `v ${cls || ""}`, text: value }));
  $("#inbox-kpis").replaceChildren(
    kpi("Triaged", String(rows.length)),
    kpi("Sent to a person", rows.length ? `${escalated} (${pct(escalated / rows.length)})` : "0", "esc"),
    kpi("Auto-drafted", String(rows.length - escalated), "draft"),
    kpi("High-priority incidents", String(urgent)),
    kpi("Median time", median === null ? "–" : ms(median)));

  const counts = {};
  for (const r of rows) counts[r.result.queue] = (counts[r.result.queue] || 0) + 1;
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = sorted.length ? sorted[0][1] : 1;
  $("#dept-bars").replaceChildren(...(sorted.length ? sorted.map(([queue, n]) => h("div", { class: "barrow" },
    h("span", { class: "lbl", text: queue }), h("span", { class: "n", text: String(n) }),
    h("div", { class: "track" }, h("span", { style: { width: `${(n / max) * 100}%` } })))) : [h("p", { class: "meta", text: "No tickets yet." })]));
  const share = rows.length ? escalated / rows.length : 0;
  $("#decision-split").replaceChildren(
    h("div", { class: "split" }, h("span", { class: "esc", style: { width: `${share * 100}%` } }), h("span", { class: "draft", style: { width: `${(1 - share) * 100}%` } })),
    h("div", { class: "split-legend" }, h("span", { text: `Person ${pct(share)}` }), h("span", { text: `Auto-draft ${pct(1 - share)}` })));
}

function addFeedRow(ticket, result) {
  $("#feed-empty").hidden = true;
  const priority = result.priority || "";
  const row = h("li", { class: result.outage ? "burst" : "", tabindex: "0", title: "Open in Triage",
    on: { click: () => openInTriage(ticket), keydown: (e) => { if (e.key === "Enter") openInTriage(ticket); } } },
    h("span", { class: "t", text: ticket.subject || ticket.body.slice(0, 80) }),
    h("span", { class: "t", text: result.type }),
    h("span", {}, h("span", { class: `badge ${priority}`, text: priority })),
    h("span", { class: "t", text: result.queue }),
    h("span", {}, h("span", { class: `badge ${result.status === "ESCALATE" ? "esc" : "draft"}`, text: result.status === "ESCALATE" ? "To a person" : "Auto-draft" })),
    h("span", { class: "num", text: ms(result.timings.wall) }));
  $("#feed").prepend(row);
}

function openInTriage(ticket) {
  $("#subject").value = ticket.subject;
  $("#body").value = ticket.body;
  $("#sample-note").textContent = "Opened from the live inbox.";
  showView("triage");
  runTriage();
}

async function processQueue() {
  const button = $("#stream-start");
  while (state.streaming && state.queue.length) {
    const ticket = state.queue.shift();
    try {
      const result = await post("/api/triage", { subject: ticket.subject, body: ticket.body, use_llm: false });
      state.inbox.push({ ticket, result });
      addFeedRow(ticket, result);
      renderInboxStats();
      showOutage(result.outage);
    } catch (error) {
      toast(error.message);
      state.streaming = false;
    }
    await new Promise((resolve) => setTimeout(resolve, 350));
  }
  state.streaming = false;
  button.textContent = "Start stream";
}

async function toggleStream() {
  const button = $("#stream-start");
  if (state.streaming) { state.streaming = false; button.textContent = "Start stream"; return; }
  if (!state.queue.length) {
    try { state.queue.push(...await api("/api/stream?n=24")); } catch (error) { toast(error.message); return; }
  }
  state.streaming = true;
  button.textContent = "Pause stream";
  processQueue();
}

function injectOutage() {
  const burst = state.presets.filter((p) => p.name.startsWith("Outage Burst"));
  if (!burst.length) { toast("Outage samples not found."); return; }
  state.queue.unshift(...burst.map((p) => ({ subject: p.subject, body: p.body })));
  toast("Three outage reports added to the front of the inbox.");
  if (!state.streaming) toggleStream();
}

async function clearInbox() {
  state.streaming = false;
  state.queue = [];
  state.inbox = [];
  $("#feed").replaceChildren();
  $("#feed-empty").hidden = false;
  $("#stream-start").textContent = "Start stream";
  $("#outage").hidden = true;
  try { await post("/api/outage/reset", {}); } catch (_) { /* ignore */ }
  renderInboxStats();
}

/* ---------------- Evaluation ---------------- */
function barChart(items, { max = 1, color = "var(--signal)", labelWidth = 190, width = 560, format = (v) => v.toFixed(2) } = {}) {
  const rowH = 26;
  const height = items.length * rowH + 6;
  const barArea = width - labelWidth - 56;
  return s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": items.map((i) => `${i.label} ${format(i.value)}`).join(", ") },
    items.map((item, i) => {
      const y = i * rowH + 4;
      return s("g", {},
        s("text", { x: labelWidth - 8, y: y + 14, "text-anchor": "end" }, item.label),
        s("rect", { x: labelWidth, y: y + 3, width: barArea, height: 14, rx: 4, fill: "var(--line-soft)" }),
        s("rect", { x: labelWidth, y: y + 3, width: Math.max(2, (item.value / max) * barArea), height: 14, rx: 4, fill: item.color || color }),
        s("text", { class: "val", x: labelWidth + barArea + 8, y: y + 14 }, format(item.value)));
    }));
}

function tradeoffChart(curve, initial) {
  const W = 1240, H = 340, L = 54, R = 24, T = 18, B = 42;
  const pts = curve.points.filter((p) => p.hpi_escalation_recall !== null);
  const xs = pts.map((p) => p.threshold);
  const x = (v) => L + ((v - xs[0]) / (xs.at(-1) - xs[0])) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);
  const line = (key) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.threshold).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => s("g", {},
    s("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }),
    s("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, `${v * 100}%`)));
  const ticks = xs.filter((_, i) => i % 2 === 0).map((v) => s("text", { x: x(v), y: H - B + 18, "text-anchor": "middle" }, v.toFixed(2)));
  const marker = s("line", { x1: 0, x2: 0, y1: T, y2: H - B, stroke: "var(--text)", "stroke-dasharray": "4 4" });
  const dotA = s("circle", { r: 5, fill: "var(--signal)" });
  const dotB = s("circle", { r: 5, fill: "var(--escalate)" });
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Escalation rate and urgent-incident recall by confidence threshold" },
    grid, ticks,
    s("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }, "confidence threshold"),
    s("path", { d: line("escalation_rate"), fill: "none", stroke: "var(--signal)", "stroke-width": 2.5 }),
    s("path", { d: line("hpi_escalation_recall"), fill: "none", stroke: "var(--escalate)", "stroke-width": 2.5 }),
    marker, dotA, dotB);
  const readout = h("p", { class: "readout" });
  const slider = h("input", { type: "range", min: 0, max: pts.length - 1, step: 1, "aria-label": "Confidence threshold" });
  const update = () => {
    const p = pts[Number(slider.value)];
    marker.setAttribute("x1", x(p.threshold)); marker.setAttribute("x2", x(p.threshold));
    dotA.setAttribute("cx", x(p.threshold)); dotA.setAttribute("cy", y(p.escalation_rate));
    dotB.setAttribute("cx", x(p.threshold)); dotB.setAttribute("cy", y(p.hpi_escalation_recall));
    readout.replaceChildren("At a threshold of ", h("b", { text: p.threshold.toFixed(2) }), ": ",
      h("b", { style: { color: "var(--escalate)" }, text: pct(p.hpi_escalation_recall, 1) }), " of urgent incidents reach a person, and ",
      h("b", { style: { color: "var(--signal)" }, text: pct(p.escalation_rate, 1) }), " of all tickets are escalated.");
  };
  const start = pts.reduce((best, p, i) => (Math.abs(p.threshold - initial) < Math.abs(pts[best].threshold - initial) ? i : best), 0);
  slider.value = String(start);
  slider.addEventListener("input", update);
  update();
  return h("div", { class: "chart" }, svg,
    h("div", { class: "legend" }, h("span", {}, h("i", { style: { background: "var(--escalate)" } }), "urgent incidents sent to a person"),
      h("span", {}, h("i", { style: { background: "var(--signal)" } }), "all tickets escalated")),
    h("div", { class: "tradeoff-controls" }, slider), readout);
}

async function loadMetrics() {
  const root = $("#evaluation");
  root.replaceChildren(h("div", { class: "panel loading" }, h("div", { class: "skeleton" }), h("div", { class: "skeleton", style: { width: "60%" } })));
  try {
    state.metrics = await api("/api/metrics");
    state.metricsLoaded = true;
  } catch (error) {
    root.replaceChildren(h("div", { class: "panel section" }, h("p", { text: error.message })));
    return;
  }
  const m = state.metrics;
  const ho = m.holdout;
  if (!ho) { root.replaceChildren(h("div", { class: "panel section" }, h("p", { text: "Run python evaluate.py --full to generate the holdout metrics." }))); return; }
  const e2e = ho.end_to_end || {};
  const metric = (label, value, detail) => h("div", { class: "panel metric" }, h("div", { class: "k", text: label }), h("div", { class: "v", text: value }), h("div", { class: "d", text: detail }));
  const perClass = (block) => Object.entries(block.per_class_f1).map(([label, value]) => ({ label, value }));
  const panels = [
    h("section", { class: "panel section" }, h("h3", { text: "Type: F1 per class" }), h("div", { class: "chart" }, barChart(perClass(ho.type), { labelWidth: 110, width: 460 }))),
    h("section", { class: "panel section" }, h("h3", { text: "Priority: F1 per class" }), h("div", { class: "chart" }, barChart(perClass(ho.priority), { labelWidth: 110, width: 460 }))),
    h("section", { class: "panel section wide" }, h("h3", { text: "Queue: F1 per department" }), h("div", { class: "chart" }, barChart(perClass(ho.queue).sort((a, b) => b.value - a.value), { width: 1240, labelWidth: 240 }))),
  ];
  if (m.queue_comparison) {
    panels.push(h("section", { class: "panel section" },
      h("h3", { text: "Why the queue model is TF-IDF" }),
      h("p", { class: "meta", text: `Macro-F1 on ${m.queue_comparison.tickets} historical tickets neither model was trained on.` }),
      h("div", { class: "chart" }, barChart(m.queue_comparison.models.map((model, i) => ({ label: model.name.replace(" (52-label checkpoint)", ""), value: model.macro_f1, color: i ? "var(--faint)" : "var(--draft)" })), { labelWidth: 200, width: 460 }))));
  }
  panels.push(h("section", { class: "panel section" }, h("h3", { text: "No data leakage" }),
    h("ul", { class: "checklist" }, [
      `Pinned dataset revision, official seed-42 80/20 split`,
      `Search index holds only the ${ho.leakage.indexed_tickets.toLocaleString()} historical tickets: ${ho.leakage.overlap} holdout tickets indexed`,
      "Holdout answers dropped the moment the file is loaded",
      "Models, thresholds and the queue model choice tuned on historical data only",
      `${m.feedback} operator decisions captured for the next training round`,
    ].map((text) => h("li", {}, ICON_CHECK(), h("span", { text }))))));
  if (m.tradeoff) {
    panels.push(h("section", { class: "panel section wide" },
      h("div", { class: "section-head" }, h("h3", { text: "Safety vs workload: the confidence threshold" }),
        h("p", { text: `Computed on the ${m.tradeoff.tickets.toLocaleString()} holdout tickets for transparency. The ${m.thresholds.confidence.toFixed(2)} default was set before this curve existed.` })),
      tradeoffChart(m.tradeoff, m.thresholds.confidence)));
  }
  root.replaceChildren(
    h("div", { class: "eval-head" }, h("h2", { text: "How it performs on unseen tickets" }),
      h("p", { text: `Scored on all ${ho.holdout_tickets.toLocaleString()} holdout tickets. Models saw only the subject and message.` })),
    h("div", { class: "metric-grid" },
      metric("Type macro-F1", ho.type.macro_f1.toFixed(3), `${ho.type.rows.toLocaleString()} labelled tickets`),
      metric("Priority macro-F1", ho.priority.macro_f1.toFixed(3), `${ho.priority.rows.toLocaleString()} tickets`),
      metric("Queue macro-F1", ho.queue.macro_f1.toFixed(3), "10 departments"),
      metric("Urgent incidents to a person", pct(ho.high_priority_incident.escalation_recall, 1), `${ho.high_priority_incident.count.toLocaleString()} high-priority incidents`),
      metric("Drafts passing the check", e2e.draft_verified_rate === undefined ? "–" : pct(e2e.draft_verified_rate), `${e2e.auto_drafts ?? 0} auto-drafts`),
      metric("Median time per ticket", e2e.latency_ms_p50 ? ms(e2e.latency_ms_p50) : "–", e2e.latency_ms_p95 ? `95th percentile ${ms(e2e.latency_ms_p95)}` : ""),
      metric("Holdout tickets in index", String(ho.leakage.overlap), `of ${ho.leakage.holdout_tickets.toLocaleString()}`)),
    h("div", { class: "eval-grid" }, panels));
}

/* ---------------- Boot ---------------- */
function boot() {
  let theme = "dark";
  try { theme = localStorage.getItem("sentineldesk-theme") || "dark"; } catch (_) { /* storage unavailable */ }
  setTheme(theme);
  $("#theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  for (const tab of document.querySelectorAll(".tabs button")) tab.addEventListener("click", () => showView(tab.dataset.view));
  $("#run").addEventListener("click", runTriage);
  $("#run-kbd").textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘ Enter" : "Ctrl Enter";
  $("#body").addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") runTriage(); });
  $("#stream-start").addEventListener("click", toggleStream);
  $("#stream-outage").addEventListener("click", injectOutage);
  $("#stream-clear").addEventListener("click", clearInbox);
  $("#outage-dismiss").addEventListener("click", async () => { $("#outage").hidden = true; try { await post("/api/outage/reset", {}); } catch (_) { /* ignore */ } });
  renderEmpty();
  renderInboxStats();
  loadPresets();
  refreshHealth();
}

boot();
