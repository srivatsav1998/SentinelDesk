# Fine-Tuned Model Weights Directory

This folder stores the local fine-tuned BERT model weights for high-speed CPU inference during the demo.

---

## Load the versioned checkpoints

Install [Git LFS](https://git-lfs.com/) before cloning. For an existing checkout, run:

```bash
git lfs install
git lfs pull --include="models/*/sentineldesk-*.zip"
```

The tracked ZIPs are `models/type/sentineldesk-type.zip` and
`models/priority/sentineldesk-priority.zip`. The graph opens them on first
use and caches the loaded classifiers. An extracted Hugging Face checkpoint
in the corresponding model directory also works.

When the queue checkpoint is ready, place it in `models/queue/` as an
extracted checkpoint or as `sentineldesk-queue.zip` with files at its root.
Use real class names in the checkpoint's `id2label` mapping. Supply
`triage_metadata.json` with the fitted `temperature`, training `max_tokens`,
and `input_format` of `single` or `paired`. Queue defaults to single text
input and 128 tokens. Type and priority use subject and body as a pair.
The graph routes tickets to human review if a classifier is unavailable or
its confidence cannot support automatic drafting.

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
