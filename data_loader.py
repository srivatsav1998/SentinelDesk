import os
import json
import pandas as pd
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def prepare_data(data_dir: str = DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("Loading dataset 'Tobi-Bueck/customer-support-tickets' from Hugging Face...")
    print("=" * 60)
    
    ds = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = ds["train"].to_pandas()
    print(f"Dataset successfully loaded. Total rows: {len(df)}")
    
    # Strictly follow the problem.pdf specification:
    # df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    # split_idx = int(len(df) * 0.8)
    # kb_train_df = df.iloc[:split_idx].copy()
    # test_df = df.iloc[split_idx:].copy()
    
    print("\nApplying deterministic shuffle (random_state=42) and ticket_id generation...")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    
    split_idx = int(len(df) * 0.8)
    kb_train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Save splits
    train_path = os.path.join(data_dir, "kb_train.parquet")
    test_path = os.path.join(data_dir, "test_holdout.parquet")
    kb_train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved historical KB train split to: {train_path} ({len(kb_train_df)} rows)")
    print(f"Saved unseen holdout test split to: {test_path} ({len(test_df)} rows)")
    
    # Dataset Exploration & Quality Summary
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION & CLASS BALANCE SUMMARY")
    print("=" * 60)
    
    print(f"\nMissing subjects in KB Train: {kb_train_df['subject'].isna().sum()} / {len(kb_train_df)}")
    print(f"Missing subjects in Holdout Test: {test_df['subject'].isna().sum()} / {len(test_df)}")
    
    print("\n--- Distribution of 'type' (Train) ---")
    print(kb_train_df["type"].value_counts(dropna=False))
    
    print("\n--- Distribution of 'priority' (Train) ---")
    print(kb_train_df["priority"].value_counts(dropna=False))
    
    print("\n--- Distribution of 'queue' (Train) ---")
    print(kb_train_df["queue"].value_counts(dropna=False))
    
    if "language" in kb_train_df.columns:
        print("\n--- Distribution of 'language' (Train) ---")
        print(kb_train_df["language"].value_counts(dropna=False))
        
    # Generate UI Demo Presets for person 4 (UI Developer)
    presets = []
    
    # 1. High Priority Incident (Hospital/EMR/Critical outage example)
    high_incidents = test_df[(test_df["priority"] == "high") & (test_df["type"] == "Incident")]
    if not high_incidents.empty:
        row = high_incidents.iloc[0]
        presets.append({
            "name": "🚨 Critical High-Priority Incident (Expect Escalation)",
            "subject": "" if pd.isna(row["subject"]) else str(row["subject"]),
            "body": str(row["body"]),
            "expected_type": str(row["type"]),
            "expected_priority": str(row["priority"]),
            "expected_queue": str(row["queue"]),
            "expected_escalation": "ESCALATE",
            "notes": "Triggers immediate human escalation gate (High priority Incident)"
        })
        
    # 2. German Language ticket
    de_tickets = test_df[test_df["language"] == "de"] if "language" in test_df.columns else pd.DataFrame()
    if not de_tickets.empty:
        row = de_tickets.iloc[0]
        presets.append({
            "name": "🇩🇪 German Language Ticket (Multilingual Test)",
            "subject": "" if pd.isna(row["subject"]) else str(row["subject"]),
            "body": str(row["body"]),
            "expected_type": str(row["type"]),
            "expected_priority": str(row["priority"]),
            "expected_queue": str(row["queue"]),
            "expected_escalation": "AUTO_DRAFT" if row["priority"] != "high" else "ESCALATE",
            "notes": "Verifies language detection and grounded generation in German"
        })
        
    # 3. Standard Routine Request (Auto-Draft Ready)
    routine = test_df[(test_df["priority"] == "low") & (test_df["type"] == "Request")]
    if not routine.empty:
        row = routine.iloc[0]
        presets.append({
            "name": "🟢 Routine Low-Priority Request (Auto-Draft Ready)",
            "subject": "" if pd.isna(row["subject"]) else str(row["subject"]),
            "body": str(row["body"]),
            "expected_type": str(row["type"]),
            "expected_priority": str(row["priority"]),
            "expected_queue": str(row["queue"]),
            "expected_escalation": "AUTO_DRAFT",
            "notes": "Standard grounded auto-reply generation"
        })
        
    # 4. Null / Missing Subject Edge Case
    missing_subj = test_df[test_df["subject"].isna()]
    if not missing_subj.empty:
        row = missing_subj.iloc[0]
        presets.append({
            "name": "⚠️ Empty Subject Line Edge Case",
            "subject": "",
            "body": str(row["body"]),
            "expected_type": str(row["type"]),
            "expected_priority": str(row["priority"]),
            "expected_queue": str(row["queue"]),
            "expected_escalation": "AUTO_DRAFT" if row["priority"] != "high" else "ESCALATE",
            "notes": "Tests pipeline resilience when subject is null/empty"
        })

    presets_path = os.path.join(data_dir, "sample_ui_presets.json")
    with open(presets_path, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(presets)} UI demo presets to: {presets_path}")
    print("=" * 60)

if __name__ == "__main__":
    prepare_data()
