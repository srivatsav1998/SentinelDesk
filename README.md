# SentinelDesk

> **Autonomous Multilingual Incident Triage, Smart Escalation & Grounded Resolution Copilot**  
> Built for enterprise technical operations with zero-leakage RAG and deterministic safety guardrails.

---

## Executive Summary

Enterprise IT helpdesks face two critical challenges:
1. **Triage Bottlenecks**: High-severity outages are buried in unstructured ticket queues alongside low-priority requests.
2. **LLM Hallucinations**: Generative AI models routinely invent non-existent policies, SLA guarantees, or unsafe troubleshooting advice.

**SentinelDesk** addresses both using an agentic **LangGraph** pipeline:
- **Instant Triage**: Classifies incoming tickets into `type`, `queue`, and `priority` with confidence scoring.
- **Safety Escalation Gate**: Intercepts high-severity incidents or low-confidence predictions, routing them directly to human on-call engineers.
- **Precedent-Grounded RAG**: Retrieves historical resolutions from past tickets and synthesizes draft replies strictly grounded in cited Ticket IDs (supporting both English and German).

---

## Architecture & Workflow

SentinelDesk is implemented as a deterministic, concurrent state graph:

```mermaid
flowchart TD
    Start([Raw Incoming Ticket]) --> Detect[1. Language Detection Node]
    
    Detect --> C_Type[2a. Model 1: Type Classifier]
    Detect --> C_Queue[2b. Model 2: Queue Classifier]
    Detect --> C_Priority[2c. Model 3: Priority Classifier]

    C_Type --> Merge[3. Merge Triage Node]
    C_Queue --> Merge
    C_Priority --> Merge

    Merge --> Gate{4. Safety Escalation Gate}

    Gate -- "Priority == High & Type == Incident<br/>OR Confidence < 0.70" --> Escalate[Human Escalation Node]
    Gate -- "Safe / High Confidence" --> RAG[5. Precedent Retrieval Node]

    RAG --> Drafter[6. Grounded Drafter Node]

    Escalate --> UI([Live Operator Console])
    Drafter --> UI
```

### Graph Execution Path:
1. **`detect_language`**: Dynamically detects the input language using character n-grams.
2. **Concurrent Triage (Fan-Out)**:
   - **`classify_type`**: Dedicated BERT model predicting `type` (`Incident`, `Request`, `Problem`, `Change`).
   - **`classify_queue`**: Loads the department classifier from `models/queue/` when its checkpoint is available.
   - **`classify_priority`**: Dedicated BERT model predicting `priority` (`high`, `medium`, `low`).
3. **`merge_triage` (Fan-In)**: Uses the lowest of the three classifier confidence scores as the aggregate routing score.
4. **`safety_gate` (Conditional Edge)**:
   - **Escalates immediately** if `priority == high` AND `type == Incident`.
   - **Routes to human review** if the aggregate score is below `0.70` or a classifier is unavailable.
5. **`retrieval_node`**: Queries the historical training KB (80% split) for top-k relevant precedents using semantic similarity.
6. **`drafter_node`**: Synthesizes a response in the requester's detected native language citing retrieved Ticket IDs (e.g., `[Reference: TKT-01055]`).
7. **`human_escalation_node`**: Prepares an incident summary badge and routes to on-call engineering.

---

## Triage Training Progress

1. **Built the workflow and operator console.** The first LangGraph version used keyword rules and fixed confidence values to exercise the three parallel classification nodes and escalation path.
2. **Locked the shared data split.** The training notebooks pin the full Hugging Face dataset revision and reproduce the seed-42 split of **49,412 historical tickets** and **12,353 holdout tickets**. They use subject and body as the classifier inputs and handle blank subjects. The priority run maps `critical` to `high` and `very_low` to `low`. The type run excludes historical rows with missing type labels.
3. **Fine-tuned and verified two checkpoints.** Type and priority each started from `google-bert/bert-base-multilingual-cased` with a fresh classification head. Each Colab run first trained, saved, reloaded, and ran inference on a small sample. The full runs then fine-tuned for one epoch on a T4 GPU.
4. **Integrated real predictions.** The three graph nodes call the predictor functions in [`triage_engine.py`](triage_engine.py). The type and priority archives are versioned with Git LFS. Local smoke tests exercised both trained models through LangGraph with English and German inputs, including a blank subject, and verified the review path when the queue model is unavailable.

### Historical validation results

| Classifier | Fit rows | Validation rows | Calibration rows | Macro-F1 | Class recall |
| --- | ---: | ---: | ---: | ---: | --- |
| Type | 31,055 | 3,882 | 3,882 | **80.81%** | **88.96%** for `Incident` |
| Priority | 39,529 | 4,941 | 4,942 | **57.09%** | **54.42%** for `high` |
| Queue | — | — | — | Pending | Pending |

These results use validation subsets from the historical 80%. The final 20% evaluation is pending integration of the queue checkpoint. It will report macro-F1 for each target and joint high-priority incident recall on tickets labeled both `Incident` and `high`.

### How prediction confidence works

Each classifier produces one raw score (logit) per class. The training notebook fits a **temperature** on a separate historical calibration subset and saves it in `triage_metadata.json`. At inference, the loader divides the logits by that temperature, applies softmax, and returns the probability of the predicted class as its confidence score.

LangGraph takes the minimum of the three scores for the `0.70` review gate. This minimum is a routing score. Macro-F1 and recall are evaluation metrics computed across many tickets. With the queue checkpoint pending, the graph still reports real type and priority predictions and routes the ticket to human review.

---

## Repository Structure

```
.
├── app.py                      # Live Streamlit Operator Console
├── graph.py                    # LangGraph StateGraph & Node Definitions
├── triage_engine.py            # Loads local BERT checkpoints and predicts calibrated scores
├── pipeline.py                 # Standalone pipeline interface & language utils
├── data_loader.py              # Ingestion, 80/20 split & zero-leakage locking
├── validate_gemini.py          # CLI validation tool for Gemini API key
├── requirements.txt            # Project dependencies
├── contracts/                  # Teammate function specifications & interfaces
│   └── team_contracts.md       # Exact signatures for triage, retrieval, and drafter
├── findings/                   # Investigation logs & experiment findings
│   └── 01_data_and_eda_findings.md # Dataset analysis & edge cases
├── models/                     # Fine-tuned BERT archives tracked with Git LFS
│   ├── type/                   # Model 1: Incident, Request, Problem, Change
│   ├── queue/                  # Model 2: Department routing
│   └── priority/               # Model 3: High, Medium, Low
└── data/                       # Local data directory (Parquet files)
    ├── kb_train.parquet        # 80% split: Historical Knowledge Base (49,412 rows)
    ├── test_holdout.parquet    # 20% split: Unseen Holdout Test (12,353 rows)
    └── sample_ui_presets.json  # Pre-configured test cases for live judging
```

---

## Quickstart Guide

Follow these exact steps to run the project from scratch on any machine:

### 1. Environment Setup

Install Git LFS to retrieve the tracked model archives. See [model setup](models/README.md) for the checkpoint paths.

```bash
git lfs install
git clone <repo-url>
cd <repo-folder>
git lfs pull

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional for Live Gemini API)
```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY:
# GEMINI_API_KEY=your_actual_key_here
```
*(If no API key is provided, the system automatically runs in offline mode using deterministic local precedent templates).*

### 3. Verify API Key Connection (Optional)
```bash
python validate_gemini.py
```

### 4. Ingest Data & Lock 80/20 Split
```bash
python data_loader.py
```
*Downloads the official Hugging Face dataset and locks `kb_train.parquet` (80%) and `test_holdout.parquet` (20%) deterministically.*

### 5. Launch the Operator Console
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Quantitative Evaluation Results (Holdout Test Split)

Official benchmark evaluated on the 20% unseen holdout partition (`data/test_holdout.parquet` - 12,353 rows):

| Target Dimension | Metric | Score (Placeholder) | Target SLA |
| :--- | :--- | :--- | :--- |
| **Incident Type** (4 classes) | Macro-F1 | `[Pending - e.g. 0.88]` | > 0.85 |
| **Queue / Department** (6+ classes) | Macro-F1 | `[Pending - e.g. 0.84]` | > 0.80 |
| **Priority Level** (3 classes) | Macro-F1 | `[Pending - e.g. 0.86]` | > 0.82 |
| **High-Priority Incident Recall** | Recall | `[Pending - e.g. 0.96]` | > 0.95 |
| **Concurrent Triage Latency** | Latency | `[Pending - e.g. 45 ms]` | < 100 ms |

*Note: Once Teammate 1 saves fine-tuned weights into `models/`, run `python evaluate.py` to compute and populate the official holdout metrics.*

---

## Data Split and Holdout Protocol

To maintain ML integrity and adhere to evaluation rules:
1. **Split Boundary**: The dataset (`Tobi-Bueck/customer-support-tickets`, 61,765 rows) is split deterministically (`random_state=42`) into:
   - **Knowledge Base (`kb_train.parquet`)**: 49,412 rows (80%). Allowed for classifier training and vector index creation.
   - **Evaluation Holdout (`test_holdout.parquet`)**: 12,353 rows (20%). Final inference uses only the subject and body. Labels are used afterward for scoring. Holdout answers stay out of retrieval and model inputs.
2. **Missing Value Resilience**: ~8.7% of historical tickets have null subjects. The BERT type and priority tokenizers receive an empty subject in those cases and retain the message body.

---

## Presenter's 5-Minute Live Demo Script

Use this talk track during your presentation and live judging:

### Step 1: Context & Architecture (1 Minute)
* **What to say**: 
  > "Enterprise support operations face two primary failures: critical outages buried in inboxes, and generative AI models inventing unsupported policies or refund guarantees. 
  > We built **SentinelDesk**, an agentic LangGraph workflow that automatically triages tickets, enforces a deterministic safety gate, and grounds all drafted resolutions in verified historical precedents."
* **What to show**: Scroll down to the **Architecture & Workflow** diagram in `README.md` and highlight the concurrent fan-out into 3 models and the conditional safety gate.

---

### Step 2: Live Demo Case 1 - Multilingual Grounded Resolution (2 Minutes)
* **Action in UI**:
  1. Under **Load Benchmark Scenario**, select:
     `German Language Ticket (Multilingual Test)`
  2. Click **Process Ticket**.
* **What to point out to judges**:
  * **Language Detection**: Automatically identified German (`de`).
  * **Concurrent Triage**: Show the 3 parallel metrics (`Type: Change`, `Queue: Billing & Payments`, `Priority: Low`, `Confidence: 78%`).
  * **Gate Status**: Green badge `STATUS: AUTO-DRAFT READY`.
  * **Grounded Draft**: Show that Gemini drafted a fluent, professional German reply **strictly citing `[Reference: TKT-08142]`** with exact pricing ($29/$99/Enterprise) without hallucinating.

---

### Step 3: Live Demo Case 2 - High-Priority Incident Safety Escalation (1.5 Minutes)
* **Action in UI**:
  1. Under **Load Benchmark Scenario**, select:
     `Critical Server Outage (High Priority Incident)`
  2. Click **Process Ticket**.
* **What to point out to judges**:
  * **Triage**: High confidence detection of `Type: Incident` and `Priority: High`.
  * **Deterministic Safety Gate**: Red badge `STATUS: ESCALATE TO ON-CALL ENGINEER`.
  * **Zero Hallucination Guarantee**: The system refuses to send an unverified auto-reply to a customer during an active infrastructure outage. Instead, it generates an on-call escalation alert referencing incident mitigation protocols (`TKT-01055`).

---

### Step 4: Live Demo Case 3 - Edge Case Resilience (30 Seconds)
* **Action in UI**:
  1. Select `Missing Subject Line Ticket` and click **Process Ticket**.
* **What to point out**:
  * Show that 8.7% of tickets in the raw dataset have empty subjects. SentinelDesk handles null subjects seamlessly without runtime crashes.

---

## Anticipated Judge Q&A & Cheat Sheet

* **Q: How do you prevent data leakage in your RAG pipeline?**
  * **A**: "We enforced a strict 80/20 train/holdout split in `data_loader.py`. The holdout test set (`test_holdout.parquet`) is physically isolated and never indexed into vector stores or seen during inference."
* **Q: How does the triage engine handle scale and latency?**
  * **A**: "Our LangGraph fans out prediction to three concurrent models (type, queue, priority) simultaneously, reducing latency compared to sequential processing."
* **Q: Why not let the LLM decide whether to escalate?**
  * **A**: "LLMs can be jailbroken or unpredictable with safety boundaries. SentinelDesk uses a deterministic programmatic gate: if `priority == high` AND `type == Incident`, escalation is guaranteed 100% of the time."
