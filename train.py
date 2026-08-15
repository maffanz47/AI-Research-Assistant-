#!/usr/bin/env python3
"""
train.py
========
Unsloth fine-tuning script for the Research Gap Finder LLM.

Target hardware : Remote GPU machine with 24 GB VRAM
Base model      : unsloth/Qwen2.5-14B-Instruct
Adapter         : LoRA via Unsloth FastModel
Tracking        : MLflow — logs hyperparameters, per-step loss, and saves
                  the trained LoRA adapter as an MLflow artifact.

Usage
-----
  # On the remote GPU machine (after gpu_setup.sh)
  python train.py --dataset_dir cache/ --output_dir lora_adapter/

Environment variables (optional, can also be set in .env)
  MLFLOW_TRACKING_URI  default: http://localhost:5000
  HF_TOKEN            HuggingFace token if model is gated
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("train")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth Qwen2.5-14B LoRA fine-tune for Research Gap Finder")
    p.add_argument("--dataset_dir", default="cache/", help="Directory with neighbourhood JSON files (DVC-tracked)")
    p.add_argument("--output_dir", default="lora_adapter/", help="Where to save the trained LoRA adapter")
    p.add_argument("--max_seq_length", type=int, default=8192, help="Max sequence length")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    p.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    p.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    p.add_argument("--mlflow_experiment", default="research-gap-finder-finetune")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_training_examples(dataset_dir: str) -> list[dict[str, str]]:
    """
    Walk the cache/ directory and convert neighbourhood JSON files into
    instruction-tuning examples.

    Each example is a (instruction, response) pair:
    - instruction : seed paper title + branch summaries prompt
    - response    : a short gap analysis (synthesized from metadata)
    """
    examples: list[dict[str, str]] = []
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        logger.warning("Dataset directory %s does not exist — using dummy sample.", dataset_dir)
        examples.append({
            "instruction": "Identify research gaps for: 'Attention Is All You Need'",
            "response": "Gap 1: Long-horizon temporal reasoning in transformer models is under-studied.",
        })
        return examples

    for json_file in sorted(dataset_path.glob("neighbourhood_*.json")):
        try:
            data: dict[str, Any] = json.loads(json_file.read_text(encoding="utf-8"))
            seed = data.get("seed", {})
            title = seed.get("title", "Unknown")
            abstract = seed.get("abstract", "")
            citations = data.get("citations", [])

            # Build a short instruction from the neighbourhood
            instruction = (
                f"Analyse the citation neighbourhood of the paper: '{title}'.\n"
                f"Abstract: {abstract[:400]}\n"
                f"Forward citations found: {len(citations)}.\n"
                f"Identify the top 3 unexplored research gaps."
            )
            # Synthesize a pseudo-response (replace with human annotations in production)
            response = (
                f"Based on the neighbourhood of '{title}', "
                f"three key research gaps are:\n"
                f"1. Temporal reasoning under distribution shift.\n"
                f"2. Cross-lingual knowledge alignment at low resources.\n"
                f"3. Multimodal evidence grounding for scientific claims."
            )
            examples.append({"instruction": instruction, "response": response})
        except Exception as exc:
            logger.warning("Skipping %s: %s", json_file, exc)

    logger.info("Loaded %d training examples from %s", len(examples), dataset_dir)
    return examples


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Import heavy dependencies (after args validated) ─────────────────
    try:
        import mlflow
        import torch
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastModel, is_bfloat16_supported
    except ImportError as exc:
        logger.error(
            "Missing dependency: %s\n"
            "Run on the GPU machine: pip install unsloth trl mlflow datasets",
            exc,
        )
        sys.exit(1)

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name="qwen2.5-14b-lora") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        # Log all hyperparameters
        hparams = {
            "base_model": "unsloth/Qwen2.5-14B-Instruct",
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "bf16": is_bfloat16_supported(),
            "dataset_dir": args.dataset_dir,
            "output_dir": args.output_dir,
        }
        mlflow.log_params(hparams)
        logger.info("Hyperparameters logged to MLflow.")

        # ── Load model & tokenizer ────────────────────────────────────────
        logger.info("Loading unsloth/Qwen2.5-14B-Instruct …")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/Qwen2.5-14B-Instruct",
            max_seq_length=args.max_seq_length,
            dtype=None,           # auto (bfloat16 if supported)
            load_in_4bit=True,    # QLoRA — fits 14B in 24 GB VRAM
        )

        # ── Apply LoRA ────────────────────────────────────────────────────
        model = FastModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # memory-efficient checkpointing
            random_state=42,
        )

        # ── Dataset ───────────────────────────────────────────────────────
        raw_examples = build_training_examples(args.dataset_dir)
        mlflow.log_metric("dataset_size", len(raw_examples))

        # Format using Qwen chat template
        def format_example(ex: dict[str, str]) -> dict[str, str]:
            messages = [
                {"role": "system", "content": "You are an expert research analyst."},
                {"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": ex["response"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        hf_dataset = Dataset.from_list([format_example(e) for e in raw_examples])
        logger.info("Dataset formatted: %d examples", len(hf_dataset))

        # ── TrainingArguments (24 GB GPU profile) ─────────────────────────
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",       # we handle logging ourselves via MLflow
        )

        # ── SFT Trainer ───────────────────────────────────────────────────
        class _MLflowLoggingCallback:
            """Minimal HF-compatible callback that pushes loss to MLflow."""
            def on_log(self, _args, state, _control, logs=None, **_kwargs):
                if logs and state.global_step:
                    for k, v in logs.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v, step=state.global_step)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=2,
            args=training_args,
            callbacks=[_MLflowLoggingCallback()],
        )

        # ── Train ─────────────────────────────────────────────────────────
        logger.info("Starting training …")
        train_result = trainer.train()
        logger.info("Training complete. Metrics: %s", train_result.metrics)
        mlflow.log_metrics(
            {k: v for k, v in train_result.metrics.items() if isinstance(v, (int, float))}
        )

        # ── Save adapter ──────────────────────────────────────────────────
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        logger.info("LoRA adapter saved to %s", output_path)

        # Log adapter directory as MLflow artifact
        mlflow.log_artifacts(str(output_path), artifact_path="lora_adapter")
        logger.info("LoRA adapter logged as MLflow artifact.")

        # Summarise
        logger.info(
            "✅ Fine-tuning complete.\n"
            "   Run ID      : %s\n"
            "   Adapter path: %s\n"
            "   MLflow URI  : %s",
            run.info.run_id,
            output_path.resolve(),
            args.mlflow_uri,
        )


if __name__ == "__main__":
    main()
