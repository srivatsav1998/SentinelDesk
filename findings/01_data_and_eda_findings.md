# Findings Log 01: Dataset Exploration & Baseline Insights

**Date**: 2026-11-21  
**Dataset**: `Tobi-Bueck/customer-support-tickets` (Hugging Face)  
**Total Records**: 61,765 tickets  

---

## 1. Train / Test Split Breakdown (Strict 80/20)
To avoid any data leakage into retrieval or evaluation:
* **Historical Knowledge Base (Training Split)**: 49,412 rows (`data/kb_train.parquet`)
* **Holdout Evaluation (Test Split)**: 12,353 rows (`data/test_holdout.parquet`)

---

## 2. Critical Data Observations

### A. Missing Subject Lines (Edge Case)
* **Train**: 4,293 rows (8.7%) have a `null` / empty subject line.
* **Holdout**: 1,006 rows (8.1%) have a `null` / empty subject line.
* **Finding**: Any model or retrieval system that assumes `subject` exists will crash on ~8% of tickets. Pipelines must concatenate `f"{subject or ''} {body}"`.

### B. Language Distribution
* **German (`de`)**: 26,851 rows (~54.3%)
* **English (`en`)**: 22,561 rows (~45.7%)
* **Finding**: The system must detect language and ensure generated drafts respond in the user's native language.

### C. Class Distributions

#### 1. Target: `type`
* `Incident`: 15,577
* `Request`: 11,146
* `Problem`: 8,112
* `Change`: 3,984
* `NaN / Unlabeled`: 10,593 rows in raw data (need imputation or filtering during classifier training)

#### 2. Target: `priority`
* `medium`: 18,704
* `high`: 17,484
* `low`: 10,240
* `critical`: 1,538
* `very_low`: 1,446
* **Finding**: For the 3-class target (`high`, `medium`, `low`), `critical` should map to `high`, and `very_low` should map to `low`.

#### 3. Target: `queue` (Top 5)
1. `Technical Support`: 11,349
2. `Product Support`: 7,165
3. `Customer Service`: 5,888
4. `IT Support`: 4,595
5. `Billing and Payments`: 3,892

---

## 3. Escalation Rule Findings (From Problem Statement)
1. **Rule 1**: Any ticket where `priority == high` AND `type == Incident` MUST be routed to Human Escalation (cannot be auto-drafted).
2. **Rule 2**: If the model confidence or precedent retrieval similarity is below threshold $\tau$, route to Human Escalation.
3. **Rule 3 (Zero Leakage)**: Never index or query `test_holdout.parquet` during retrieval indexing.

---

## 4. Dataset Composition (Why ~10.6k Rows Have No `type`)

`load_dataset("Tobi-Bueck/customer-support-tickets")` has no config, so it concatenates all three CSVs in the repository:

| File | Columns | Notes |
| :--- | :--- | :--- |
| `aa_dataset-tickets-multi-lang-5-2-50-version.csv` | subject, body, answer, type, queue, priority, language, version, tags | Recommended file |
| `dataset-tickets-multi-lang-4-20k.csv` | same minus `version` | Older version, same 10-queue taxonomy |
| `dataset-tickets-german_normalized_50_5_2.csv` | subject, body, queue, priority, language | **No `answer`, no `type`**, ~100 foreign queues ("Food & Drink/Groceries", "News", ...) |

Consequences handled in code:
* **Queue classifier** (`train_queue.py`) and queue scoring (`evaluate.py`) use only rows with a `type` label: 38,819 historical / 9,768 holdout tickets across the 10 support queues.
* **Retrieval** (`retriever.py`) searches only indexed tickets with a non-empty `answer` (10,602 indexed rows have none), so every precedent can ground a reply.
* **Type** training already excluded rows without a label; **priority** labels exist in all three files (`critical` -> `high`, `very_low` -> `low`).
* `data_loader.py` pins revision `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`. A local run reproduces the Chroma index exactly (2,000 of 2,000 sampled ticket IDs have identical bodies; 0 holdout IDs indexed).
