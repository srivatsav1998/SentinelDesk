# Findings 03: Queue Model Comparison

Scored on 776 historical tickets that neither model was trained on (reproduced splits, seed 42; the holdout is not used). Macro-F1 over the 10 support queues.

| Model | Macro-F1 | Accuracy | Predictions outside the 10 queues | Most frequent predictions |
| :--- | ---: | ---: | ---: | :--- |
| TF-IDF + logistic regression | **0.661** | 0.682 | 0 | Technical Support (239), Product Support (131), Customer Service (126) |
| mBERT (52-label checkpoint) | **0.230** | 0.427 | 3 | Technical Support (504), Customer Service (121), Product Support (68) |

The mBERT checkpoint was trained for one epoch on all historical rows, including ~10.6k rows from the dataset's third CSV whose queue labels belong to a different taxonomy (52 labels in total), and it collapses towards `Technical Support`. `triage_engine.py` therefore prefers the TF-IDF model; set `SENTINEL_QUEUE_BACKEND=transformer` to use the checkpoint instead.
