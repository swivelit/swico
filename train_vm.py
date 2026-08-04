#!/usr/bin/env python3
"""
Advanced SentenceTransformer Retrieval Training Pipeline (VM Optimized)
----------------------------------------------------------------------
Optimized for headless VM training, maximum retrieval accuracy, local persistence,
automatic resuming, early stopping, and detailed accuracy reports.

Usage:
  python3 train_vm.py
"""

import os
import sys
import glob
import json
import math
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from datasets import Dataset
from sklearn.model_selection import train_test_split

try:
    import faiss
except ImportError:
    print("WARNING: faiss-gpu or faiss-cpu is not installed.")
    print("To enable hard negative mining, speed benchmarking, and threshold optimization, run:")
    print("  pip install faiss-gpu  (if using GPU)  OR  pip install faiss-cpu  (if CPU-only)")
    faiss = None

from transformers import EarlyStoppingCallback, TrainerCallback, get_cosine_schedule_with_warmup
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer
)
from sentence_transformers.losses import (
    MultipleNegativesRankingLoss,
    MultipleNegativesSymmetricRankingLoss
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# ==============================================================================
# 1. GLOBAL CONFIGURATION & OPTIMIZED HYPERPARAMETERS
# ==============================================================================
BASE_MODEL_NAME = "intfloat/multilingual-e5-small"

# Auto-detect dataset filename
if Path("combined_deduplicated_dataset.csv").exists():
    DATA_PATH = Path("combined_deduplicated_dataset.csv")
else:
    DATA_PATH = Path("train_augmented.csv")

OUTPUT_DIR = Path("./e5_small_best_model")
ONNX_DIR = Path("./e5_small_onnx")
HF_DIR = Path("./e5_small_hf")
CHECKPOINT_DIR = Path("./e5_small_checkpoints")
REPORT_PATH = Path("./e5_small_retrieval_report.md")

# Auto-adjust Batch Size for GPU vs High-End CPU
IS_GPU = torch.cuda.is_available()
if IS_GPU:
    BATCH_SIZE = 256                    # Single-device GPU batch size
    GRADIENT_ACCUMULATION_STEPS = 2     # Effective batch size = 512
else:
    BATCH_SIZE = 64                     # Optimal CPU per-device batch size to prevent RAM thrashing
    GRADIENT_ACCUMULATION_STEPS = 8     # Effective batch size = 512

EPOCHS = 200                        # Max Epochs set to 200 (Early Stopping automatically stops when accuracy peaks)
LEARNING_RATE = 2.5e-5              # Ideal base learning rate
WARMUP_RATIO = 0.10                 # 10% warmup ratio for smooth convergence
WEIGHT_DECAY = 0.05                 # Balanced L2 regularization
EARLY_STOPPING_PATIENCE = 15        # Checkpoint evaluations to wait before early stopping

# Checkpoint intervals
EVAL_STEPS = 300
SAVE_STEPS = 600

# Hard Negative Mining Configurations
MINE_NEGATIVES = True
NUM_HARD_NEGATIVES = 4              # Increased from 3 to 4 for richer contrastive signal
LLRD_DECAY_RATE = 0.94              # Layer-wise Learning Rate Decay (0.94)

if faiss is None and MINE_NEGATIVES:
    print("WARNING: FAISS is not installed. Disabling hard negative mining.")
    MINE_NEGATIVES = False

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if IS_GPU:
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ==============================================================================
# 2. DATA UTILITIES & PREPROCESSING
# ==============================================================================
def normalize_text(text: object) -> str:
    return " ".join(str(text).strip().lower().split())

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. Please ensure 'train_augmented.csv' is in the directory."
        )
    df = pd.read_csv(path)
    required = {"query", "positive"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["query", "positive"]].copy()
    df["query"] = df["query"].astype(str).str.strip()
    df["positive"] = df["positive"].astype(str).str.strip()
    df = df.replace({"": np.nan}).dropna(subset=["query", "positive"])
    df = df.drop_duplicates(subset=["query", "positive"]).reset_index(drop=True)

    # Add normalized text for leakage validation and grouping
    df["query_norm"] = df["query"].map(normalize_text)
    df["positive_norm"] = df["positive"].map(normalize_text)
    return df

def split_dataset(df: pd.DataFrame, test_size=0.1, valid_size=0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
    val_rel_size = valid_size / (1.0 - test_size)
    train_df, valid_df = train_test_split(train_val_df, test_size=val_rel_size, random_state=RANDOM_STATE)
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

def check_split_leakage(train_df, valid_df, test_df):
    train_queries = set(train_df["query_norm"])
    valid_queries = set(valid_df["query_norm"])
    test_queries = set(test_df["query_norm"])

    train_pos = set(train_df["positive_norm"])
    valid_pos = set(valid_df["positive_norm"])
    test_pos = set(test_df["positive_norm"])

    print("=" * 60)
    print("DATA SPLIT LEAKAGE REPORT")
    print("=" * 60)
    print(f"Query Overlap (Train vs Valid)    : {len(train_queries & valid_queries)}")
    print(f"Query Overlap (Train vs Test)     : {len(train_queries & test_queries)}")
    print(f"Query Overlap (Valid vs Test)     : {len(valid_queries & test_queries)}")
    print("-" * 60)
    print(f"Positive Overlap (Train vs Valid) : {len(train_pos & valid_pos)}")
    print(f"Positive Overlap (Train vs Test)  : {len(train_pos & test_pos)}")
    print(f"Positive Overlap (Valid vs Test)  : {len(valid_pos & test_pos)}")
    print("=" * 60)

# E5 specific query and passage tagging
def format_for_model(df_in: pd.DataFrame, is_triplet: bool = False) -> pd.DataFrame:
    df_out = df_in.copy()
    is_e5 = "e5" in BASE_MODEL_NAME.lower()
    if is_e5:
        df_out["query"] = "query: " + df_out["query"]
        df_out["positive"] = "passage: " + df_out["positive"]
        if is_triplet and "negative" in df_out.columns:
            df_out["negative"] = "passage: " + df_out["negative"]
    return df_out

# ==============================================================================
# 3. LAYER-WISE LEARNING RATE DECAY (LLRD) OPTIMIZER
# ==============================================================================
def get_llrd_optimizer_grouped_parameters(
    model,
    learning_rate=2.5e-5,
    layer_decay=0.93,
    weight_decay=0.05
):
    transformer = model[0].auto_model
    layer_decay = float(layer_decay)
    weight_decay = float(weight_decay)

    try:
        layers = transformer.encoder.layer
        num_layers = len(layers)
    except AttributeError:
        print("WARNING: Could not dynamically inspect encoder layers. Using default parameter grouping.")
        return [{"params": model.parameters(), "lr": learning_rate, "weight_decay": weight_decay}]

    grouped_parameters = []
    no_decay = ["bias", "LayerNorm.weight"]

    # 1. Embeddings layer
    emb_lr = learning_rate * (layer_decay ** (num_layers + 1))
    grouped_parameters.append({
        "params": [p for n, p in transformer.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": emb_lr
    })
    grouped_parameters.append({
        "params": [p for n, p in transformer.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": emb_lr
    })

    # 2. Encoder layers
    for i in range(num_layers):
        layer_lr = learning_rate * (layer_decay ** (num_layers - i))
        layer_params = layers[i]

        grouped_parameters.append({
            "params": [p for n, p in layer_params.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": layer_lr
        })
        grouped_parameters.append({
            "params": [p for n, p in layer_params.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": layer_lr
        })

    # 3. Output dense / pooling heads
    rest_params = []
    rest_params_no_decay = []
    emb_params_set = set(transformer.embeddings.parameters())
    encoder_params_set = set(transformer.encoder.parameters())

    for n, p in model.named_parameters():
        if p not in emb_params_set and p not in encoder_params_set:
            if any(nd in n for nd in no_decay):
                rest_params_no_decay.append(p)
            else:
                rest_params.append(p)

    if rest_params:
        grouped_parameters.append({
            "params": rest_params,
            "weight_decay": weight_decay,
            "lr": learning_rate
        })
    if rest_params_no_decay:
        grouped_parameters.append({
            "params": rest_params_no_decay,
            "weight_decay": 0.0,
            "lr": learning_rate
        })

    return grouped_parameters


# ==============================================================================
# 4. CUSTOM TRAINING LOGGING CALLBACK
# ==============================================================================
class CustomLoggingCallback(TrainerCallback):
    def __init__(self, num_queries, corpus_size):
        super().__init__()
        self.last_train_loss = 0.0
        self.num_queries = num_queries
        self.corpus_size = corpus_size

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        print("=" * 60)
        print(f"EPOCH {epoch}/{int(args.num_train_epochs)} FINISHED")
        print(f"Average Training Loss : {self.last_train_loss:.6f}")
        optimizer = kwargs.get("optimizer")
        if optimizer:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Current LR            : {lr:.8f}")
        print("=" * 60)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            recall_1 = metrics.get("eval_validation_cosine_accuracy@1", 0) * 100
            recall_5 = metrics.get("eval_validation_cosine_recall@5", 0) * 100
            mrr = metrics.get("eval_validation_cosine_mrr@5", 0)
            ndcg = metrics.get("eval_validation_cosine_ndcg@5", 0)

            print("\n" + "=" * 60)
            print(f"VALIDATION RESULTS | Epoch={int(state.epoch) if state.epoch is not None else 0} | Steps={state.global_step}")
            print("=" * 60)
            print(f"Recall@1              : {recall_1:.2f}%")
            print(f"Recall@5              : {recall_5:.2f}%")
            print(f"MRR                   : {mrr:.6f}")
            print(f"NDCG@5                : {ndcg:.6f}")
            print(f"Evaluated Queries     : {self.num_queries}")
            print(f"Corpus Size           : {self.corpus_size}")
            print("-" * 60)

            best_metric = state.best_metric
            if best_metric is None or mrr >= best_metric:
                print(f"✅ Validation Improved | MRR={mrr:.6f}")
            else:
                print(f"⚠️ No Validation Improvement | MRR={mrr:.6f} | Best={best_metric:.6f}")
            print("=" * 60 + "\n")


# ==============================================================================
# 5. IR EVALUATOR BUILDER
# ==============================================================================
def build_ir_evaluator(eval_df: pd.DataFrame, corpus_df: pd.DataFrame, name="eval"):
    queries = {}
    corpus = {}
    relevant_docs = {}

    unique_docs = corpus_df["positive"].drop_duplicates().tolist()
    for idx, doc in enumerate(unique_docs):
        doc_id = f"doc_{idx}"
        corpus[doc_id] = doc

    doc_to_id = {doc: doc_id for doc_id, doc in corpus.items()}

    for idx, row in enumerate(eval_df.itertuples()):
        query_id = f"query_{idx}"
        queries[query_id] = row.query

        doc_id = doc_to_id.get(row.positive)
        if doc_id:
            relevant_docs[query_id] = {doc_id}

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=[1, 5],
        ndcg_at_k=[1, 5],
        accuracy_at_k=[1, 5],
        precision_recall_at_k=[1, 5],
        name=name,
        show_progress_bar=False
    )


# ==============================================================================
# 6. THRESHOLD OPTIMIZATION
# ==============================================================================
def threshold_optimization(model, eval_df, corpus_sentences, thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]):
    if faiss is None:
        print("FAISS is not installed. Skipping threshold optimization.")
        return pd.DataFrame()

    formatted_eval_df = format_for_model(eval_df)
    queries = formatted_eval_df["query"].tolist()

    corpus_embeddings = model.encode(corpus_sentences, normalize_embeddings=True)
    query_embeddings = model.encode(queries, normalize_embeddings=True)

    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings.astype("float32"))
    scores, indices = index.search(query_embeddings.astype("float32"), 1)

    def clean_text(text):
        t = str(text).strip().lower()
        if t.startswith("passage: "):
            t = t[len("passage: "):]
        elif t.startswith("query: "):
            t = t[len("query: "):]
        return " ".join(t.split())

    corpus_norm = [clean_text(s) for s in corpus_sentences]
    correct = []
    for i, row in enumerate(eval_df.itertuples()):
        retrieved_ans = corpus_norm[int(indices[i][0])]
        expected_ans = clean_text(row.positive)
        correct.append(retrieved_ans == expected_ans)

    correct = np.array(correct)
    scores = scores.flatten()

    rows = []
    total = len(eval_df)
    for t in thresholds:
        accepted = scores >= t
        accepted_count = int(accepted.sum())
        correct_accepted = int((accepted & correct).sum())

        coverage = (accepted_count / total) * 100 if total else 0
        acc = (correct_accepted / accepted_count) * 100 if accepted_count else 0
        overall_acc = (correct_accepted / total) * 100 if total else 0

        rows.append({
            "Threshold": t,
            "Accepted Count": accepted_count,
            "Coverage (%)": f"{coverage:.2f}%",
            "Accepted Accuracy (%)": f"{acc:.2f}%",
            "Overall Accuracy (%)": f"{overall_acc:.2f}%"
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 7. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    print("PyTorch Device:", "CUDA/GPU" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        print("GPU Card Name :", torch.cuda.get_device_name(0))

    # Load dataset
    print(f"Loading dataset from: {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    print(f"Loaded dataset: {len(df)} query-positive rows after cleaning.")

    # Split dataset
    train_df, valid_df, test_df = split_dataset(df)
    check_split_leakage(train_df, valid_df, test_df)

    # Initialize Base Model
    print(f"Loading model backbone: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)
    try:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled to save GPU memory.")
    except Exception as e:
        print("Could not enable gradient checkpointing:", e)

    # Hard Negative Mining Setup
    if MINE_NEGATIVES and faiss is not None:
        print("\n=== STARTING HARD NEGATIVE MINING ===")
        is_e5 = "e5" in BASE_MODEL_NAME.lower()
        train_corpus = train_df["positive"].drop_duplicates().tolist()

        print(f"Encoding {len(train_corpus)} corpus documents...")
        formatted_corpus = ["passage: " + doc for doc in train_corpus] if is_e5 else train_corpus
        corpus_embeddings = model.encode(
            formatted_corpus, batch_size=256, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")

        print(f"Encoding {len(train_df)} training queries...")
        formatted_queries = ["query: " + q for q in train_df["query"].tolist()] if is_e5 else train_df["query"].tolist()
        query_embeddings = model.encode(
            formatted_queries, batch_size=256, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        index.add(corpus_embeddings)

        k_search = NUM_HARD_NEGATIVES + 5
        print(f"Searching index for hard negatives (k={k_search})...")
        scores, indices = index.search(query_embeddings, k_search)

        triplets = []
        for i, row in enumerate(train_df.itertuples()):
            query = row.query
            positive = row.positive
            pos_norm = row.positive_norm

            negatives_found = 0
            for idx in indices[i]:
                candidate = train_corpus[int(idx)]
                candidate_norm = normalize_text(candidate)
                if candidate_norm != pos_norm:
                    triplets.append({
                        "query": query,
                        "positive": positive,
                        "negative": candidate
                    })
                    negatives_found += 1
                    if negatives_found >= NUM_HARD_NEGATIVES:
                        break

        train_triplets_df = pd.DataFrame(triplets)
        print(f"Hard negative mining completed. Total triplets: {len(train_triplets_df)}")
        train_loss = MultipleNegativesRankingLoss(model)
    else:
        print("\nHard Negative Mining disabled or FAISS not installed. Using in-batch negatives.")
        train_loss = MultipleNegativesSymmetricRankingLoss(model)

    # Format Datasets for SentenceTransformerTrainer
    eval_valid_df = format_for_model(valid_df)
    eval_test_df = format_for_model(test_df)

    if MINE_NEGATIVES and 'train_triplets_df' in locals():
        formatted_train_df = format_for_model(train_triplets_df, is_triplet=True)
        train_dataset = Dataset.from_pandas(
            formatted_train_df.rename(columns={"query": "anchor", "negative": "negative"})
        )
    else:
        formatted_train_df = format_for_model(train_df)
        train_dataset = Dataset.from_pandas(
            formatted_train_df.rename(columns={"query": "anchor"})
        )

    formatted_valid_loss_df = format_for_model(valid_df)
    valid_dataset_loss = Dataset.from_pandas(
        formatted_valid_loss_df.rename(columns={"query": "anchor"})
    )

    full_corpus_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    formatted_full_corpus = format_for_model(full_corpus_df)

    validation_evaluator = build_ir_evaluator(eval_valid_df, formatted_full_corpus, name="validation")

    # Optimizer and Cosine Scheduler Setup
    print("Configuring Layer-wise Learning Rate Decay (LLRD) Optimizer...")
    grouped_params = get_llrd_optimizer_grouped_parameters(
        model,
        learning_rate=LEARNING_RATE,
        layer_decay=LLRD_DECAY_RATE,
        weight_decay=WEIGHT_DECAY
    )
    optimizer = AdamW(grouped_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = len(train_dataset) // BATCH_SIZE
    if num_update_steps_per_epoch == 0:
        num_update_steps_per_epoch = 1
    total_training_steps = (num_update_steps_per_epoch * EPOCHS) // GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(total_training_steps * WARMUP_RATIO)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )

    # Setup Trainer Args
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),     # Enable half-precision only if GPU is active
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,                 # Keep only the last 2 checkpoints automatically
        metric_for_best_model="eval_validation_cosine_mrr@5",
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_steps=20,
        report_to="none"
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset_loss,
        loss=train_loss,
        evaluator=validation_evaluator,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            CustomLoggingCallback(num_queries=len(valid_df), corpus_size=len(full_corpus_df))
        ]
    )

    # Auto-resume check
    checkpoints = glob.glob(str(CHECKPOINT_DIR / "checkpoint-*"))
    resume_path = None
    if checkpoints:
        checkpoints.sort(key=lambda x: int(Path(x).name.split("-")[-1]))
        resume_path = checkpoints[-1]
        print(f"\n=== Found previous checkpoint: {resume_path}. Resuming training... ===")
    else:
        print("\n=== No checkpoints found. Starting training from scratch... ===")

    # Train
    print("\n===== STARTING FINE-TUNING =====")
    trainer.train(resume_from_checkpoint=resume_path)
    print("Fine-tuning completed!")

    # Save local model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"Saved best model locally to: {OUTPUT_DIR}")

    # Auto-clean temporary intermediate checkpoints to save disk space
    if CHECKPOINT_DIR.exists():
        print(f"Cleaning up temporary checkpoint directory ({CHECKPOINT_DIR}) to free disk space...")
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
        print("✅ Intermediate checkpoints cleaned up successfully!")

    # ==============================================================================
    # 8. POST-TRAINING FINAL EVALUATION
    # ==============================================================================
    print("\n===== LOADING BEST MODEL FOR FINAL RETRIEVAL BENCHMARKS =====")
    best_model = SentenceTransformer(str(OUTPUT_DIR))

    print("Evaluating Training Set Accuracy against Full Corpus...")
    eval_train_df = format_for_model(train_df)
    train_evaluator = build_ir_evaluator(eval_train_df, formatted_full_corpus, name="train")
    train_results = train_evaluator(best_model)

    print("Evaluating Test Set Accuracy against Full Corpus...")
    test_evaluator = build_ir_evaluator(eval_test_df, formatted_full_corpus, name="test")
    test_results = test_evaluator(best_model)

    train_acc1 = train_results.get('train_cosine_accuracy@1', 0) * 100
    train_acc5 = train_results.get('train_cosine_recall@5', 0) * 100
    train_mrr = train_results.get('train_cosine_mrr@5', 0)

    test_acc1 = test_results.get('test_cosine_accuracy@1', 0) * 100
    test_acc5 = test_results.get('test_cosine_recall@5', 0) * 100
    test_mrr = test_results.get('test_cosine_mrr@5', 0)
    test_ndcg = test_results.get('test_cosine_ndcg@5', 0)

    best_val_loss = 0.0
    state_path = OUTPUT_DIR / "trainer_state.json"
    if not state_path.exists() and CHECKPOINT_DIR.exists():
        state_files = glob.glob(str(CHECKPOINT_DIR / "checkpoint-*/trainer_state.json"))
        if state_files:
            state_files.sort(key=lambda x: int(Path(x).parent.name.split("-")[-1]))
            state_path = Path(state_files[-1])

    if state_path.exists():
        try:
            with open(state_path, "r") as f:
                state_data = json.load(f)
            log_history = state_data.get("log_history", [])
            for entry in reversed(log_history):
                if "eval_loss" in entry:
                    best_val_loss = entry["eval_loss"]
                    break
        except Exception as e:
            print("Could not parse trainer_state.json validation loss:", e)

    print("\n" + "=" * 60)
    print("DETAILED ACCURACY REPORT")
    print("=" * 60)
    print(f"Best Validation Loss          : {best_val_loss:.4f}")
    print("-" * 60)
    print("TRAINING SET ACCURACY:")
    print(f"  Recall@1                    : {train_acc1:.2f}%")
    print(f"  Recall@5                    : {train_acc5:.2f}%")
    print(f"  MRR@5                       : {train_mrr:.4f}")
    print("-" * 60)
    print("TEST SET ACCURACY (Generalization):")
    print(f"  Recall@1                    : {test_acc1:.2f}%")
    print(f"  Recall@5                    : {test_acc5:.2f}%")
    print(f"  MRR@5                       : {test_mrr:.4f}")
    print(f"  NDCG@5                      : {test_ndcg:.4f}")
    print("=" * 60)

    # Threshold Optimization
    print("\nOptimizing Semantic Retrieval Thresholds...")
    corpus_sentences = formatted_full_corpus["positive"].drop_duplicates().tolist()
    threshold_table = threshold_optimization(best_model, valid_df, corpus_sentences)
    if not threshold_table.empty:
        print(threshold_table.to_string(index=False))

    # Speed & Latency Benchmark
    print("\nRunning Speed & Latency Benchmark...")
    queries = eval_test_df["query"].tolist()
    t_start = time.perf_counter()
    corpus_emb = best_model.encode(corpus_sentences, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    query_emb = best_model.encode(queries, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    t_encode = time.perf_counter() - t_start

    if faiss is not None:
        t_start = time.perf_counter()
        scores_bf = query_emb @ corpus_emb.T
        indices_bf = np.argsort(-scores_bf, axis=1)[:, :5]
        t_bf = time.perf_counter() - t_start

        t_start = time.perf_counter()
        index = faiss.IndexFlatIP(corpus_emb.shape[1])
        index.add(corpus_emb.astype("float32"))
        scores_faiss, indices_faiss = index.search(query_emb.astype("float32"), 5)
        t_faiss = time.perf_counter() - t_start

        avg_bf = 1000 * t_bf / len(queries)
        avg_faiss = 1000 * t_faiss / len(queries)
        print("=" * 60)
        print("SPEED & LATENCY BENCHMARK")
        print("=" * 60)
        print(f"Encoding Time           : {t_encode:.2f} sec")
        print(f"Brute Force search time : {t_bf:.4f} sec (Avg: {avg_bf:.2f} ms/query)")
        print(f"FAISS Index search time : {t_faiss:.4f} sec (Avg: {avg_faiss:.2f} ms/query)")
        print(f"FAISS Speedup Factor    : {t_bf/t_faiss:.2f}x")
        print("=" * 60)
    else:
        t_bf, t_faiss = 0.0, 0.0
        print("FAISS not installed. Skipping latency speedup comparison.")

    # ==============================================================================
    # 9. ONNX EXPORT & DYNAMIC INT8 QUANTIZATION
    # ==============================================================================
    print("\nExporting Model to Hugging Face and ONNX format...")
    HF_DIR.mkdir(parents=True, exist_ok=True)
    best_model[0].auto_model.save_pretrained(str(HF_DIR))
    best_model.tokenizer.save_pretrained(str(HF_DIR))

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "optimum-cli", "export", "onnx",
            "--task", "feature-extraction",
            "--model", str(HF_DIR),
            str(ONNX_DIR)
        ]
        print(f"Running ONNX Export: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # dynamic quantization
        from onnxruntime.quantization import QuantType, quantize_dynamic
        
        quantize_dynamic(
            str(ONNX_DIR / "model.onnx"),
            str(ONNX_DIR / "model-int8.onnx"),
            weight_type=QuantType.QInt8,
        )

        fp32_size = (ONNX_DIR / "model.onnx").stat().st_size / (1024 * 1024)
        int8_size = (ONNX_DIR / "model-int8.onnx").stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print("ONNX EXPORT & QUANTIZATION SUMMARY")
        print("=" * 60)
        print(f"FP32 ONNX Model Size : {fp32_size:.2f} MB")
        print(f"INT8 ONNX Model Size : {int8_size:.2f} MB")
        print(f"Compression Ratio    : {fp32_size/int8_size:.2f}x")
        print("=" * 60)
    except Exception as e:
        print("WARNING: ONNX dynamic INT8 quantization failed. Verify that 'optimum[onnxruntime]' is installed.")
        print(f"Details: {e}")

    # ==============================================================================
    # 10. GENERATING EXECUTIVE REPORT
    # ==============================================================================
    print(f"\nGenerating Executive report at: {REPORT_PATH}")
    val_loss_lines = []
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            val_loss_lines.append(
                f"| {entry.get('epoch', 0):.1f} | {entry.get('step', 0)} | {entry['eval_loss']:.4f} | {entry.get('eval_validation_cosine_mrr@5', 0):.4f} |"
            )

    report_content = f"""# Retrieval Training Pipeline Executive Report

## 1. System Config & Hyperparameters
- **Base Model**: `{BASE_MODEL_NAME}`
- **Max Epochs**: {EPOCHS}
- **Learning Rate**: {LEARNING_RATE}
- **Batch Size**: {BATCH_SIZE}
- **Warmup Ratio**: {WARMUP_RATIO}
- **Early Stopping Patience**: {EARLY_STOPPING_PATIENCE} evaluations

## 2. Dataset Splitting & Cleaning
- **Total Valid Examples**: {len(df)}
- **Train Split Size**: {len(train_df)}
- **Validation Split Size**: {len(valid_df)}
- **Test Split Size**: {len(test_df)}

## 3. Training & Validation Loss History
| Epoch | Step | Validation Loss | Validation MRR@5 |
| :--- | :--- | :--- | :--- |
{chr(10).join(val_loss_lines)}

## 4. Overall Accuracy Comparison
### Full Corpus ({len(corpus_sentences)} Documents)
- **Training Accuracy**: **{train_acc1:.2f}%**
- **Test Accuracy**: **{test_acc1:.2f}%**

## 5. Detailed Accuracy Metrics
### A. Full Corpus ({len(corpus_sentences)} Documents)
#### Training Set:
- **Recall@1**: **{train_acc1:.2f}%**
- **Recall@5**: **{train_acc5:.2f}%**
- **MRR@5**: **{train_mrr:.4f}**

#### Test Set (Generalization):
- **Recall@1**: **{test_acc1:.2f}%**
- **Recall@5**: **{test_acc5:.2f}%**
- **MRR@5**: **{test_mrr:.4f}**
- **NDCG@5**: **{test_ndcg:.4f}**

### C. Validation Loss Summary
- **Best Validation Loss**: **{best_val_loss:.4f}**

## 6. Threshold Accuracy Optimization
{threshold_table.to_markdown(index=False) if not threshold_table.empty else "_FAISS not available._"}

## 7. Speed & Latency Benchmark
- **Corpus size**: {len(corpus_sentences)} documents
- **Brute Force latency per query**: {1000*t_bf/len(queries):.2f} ms
- **FAISS latency per query**: {1000*t_faiss/len(queries):.2f} ms
- **FAISS Speedup Factor**: **{t_bf/t_faiss:.2f}x**
"""

    REPORT_PATH.write_text(report_content, encoding="utf-8")
    print("✅ Executive report written successfully!")


if __name__ == "__main__":
    main()
