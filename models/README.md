# Fine-Tuned Model Weights Directory

This folder stores the local fine-tuned BERT model weights for high-speed CPU inference during the demo.

---

## Directory Structure:

```
models/
├── type/         # Fine-tuned Model 1: Predicts type (Incident, Request, Problem, Change)
├── queue/        # Fine-tuned Model 2: Predicts queue (Technical Support, Billing, etc.)
└── priority/     # Fine-tuned Model 3: Predicts priority (high, medium, low)
```

---

## Instructions for ML Engineers:

After fine-tuning each model (e.g. in Colab or Vertex AI GPU), save the artifacts using Hugging Face's `.save_pretrained()`:

```python
# Save model and tokenizer
model.save_pretrained("./models/type")
tokenizer.save_pretrained("./models/type")
```

Each subfolder should contain:
* `config.json`
* `tokenizer_config.json`
* `vocab.txt` (or `tokenizer.json`)
* `model.safetensors` (or `pytorch_model.bin`)

*Note: Heavy weight files (`*.safetensors`, `*.bin`, `*.pt`) are ignored by `.gitignore` to keep GitHub under file size limits.*
