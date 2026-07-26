"""
STEP 5 — LoRA fine-tune: Phi-3-mini-4k-instruct on TENETDrill's ILM dataset.

Base model: microsoft/Phi-3-mini-4k-instruct, LoRA via Unsloth, 4-bit
loading + gradient checkpointing, sized for a single NVIDIA L4 (24GB VRAM).
Adapter is saved to ./ilm/adapters/tenetdrill-phi3-lora/. Everything
downstream runs fully offline (Ollama, CPU, Q4 GGUF) -- see EXPORT
INSTRUCTIONS at the bottom of this file.

IMPORTANT -- this machine cannot run the actual training
-----------------------------------------------------------
This script was written and validated on a machine with no NVIDIA GPU and
no `torch` installed (confirmed: no nvidia-smi, no CUDA device, `import
torch` fails). It CANNOT be executed end-to-end here. All GPU/model
imports (torch, unsloth, trl, peft, datasets) are deferred into train()
and export_gguf() so that:

    python -m ilm.training.train --dry-run

still runs on THIS machine and validates the dataset (JSONL structure,
role ordering, empty-content checks, per-record length stats) with zero
extra dependencies -- that's the one thing that's actually been verified.
The real run must happen on the target L4 instance:

    pip install unsloth
    # if unsloth's auto-detected torch/CUDA build doesn't match your L4
    # instance, follow the pinned install command from
    # https://github.com/unslothai/unsloth#installation instead.

    python -m ilm.training.train \
        --dataset ilm/dataset/tenetdrill_sft.jsonl \
        --output-dir ilm/adapters/tenetdrill-phi3-lora \
        --epochs 3

    # then export to GGUF for offline CPU serving:
    python -m ilm.training.train --export-gguf \
        --output-dir ilm/adapters/tenetdrill-phi3-lora

See EXPORT INSTRUCTIONS at the bottom of this file for the exact GGUF /
Ollama commands (also printed at the end of a successful --export-gguf run).
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "ilm" / "dataset" / "tenetdrill_sft.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ilm" / "adapters" / "tenetdrill-phi3-lora"

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
# Unsloth also publishes a pre-quantized repack of this exact model
# ("unsloth/Phi-3-mini-4k-instruct-bnb-4bit") that skips on-the-fly 4-bit
# quantization at load time -- functionally identical, just faster to
# start from. Pass --base-model to use it; BASE_MODEL above stays the
# literal model named in the spec by default.

GGUF_QUANTIZATION = "q4_k_m"  # Q4 quantization for CPU inference, per spec


# ---------------------------------------------------------------------------
# Dataset loading / validation -- pure stdlib, runs anywhere
# ---------------------------------------------------------------------------


def validate_dataset(path: pathlib.Path) -> dict:
    """Structural validation of the SFT JSONL -- no torch/unsloth required.

    Checks every record has exactly a [system, user, assistant] message
    triple with non-empty content, and reports per-intent counts and rough
    length stats so an obviously malformed or truncated dataset is caught
    before it ever reaches the GPU.
    """
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path} -- run `python -m ilm.dataset.generate` first")

    n = 0
    malformed = 0
    empty_content = 0
    char_lengths: list[int] = []
    intents: Counter = Counter()

    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n += 1
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                malformed += 1
                continue
            for m in msgs:
                if not str(m.get("content", "")).strip():
                    empty_content += 1
            char_lengths.append(sum(len(m["content"]) for m in msgs))
            intents[obj.get("meta", {}).get("intent", "unknown")] += 1

    return {
        "path": str(path),
        "n_records": n,
        "malformed_role_order": malformed,
        "empty_content_fields": empty_content,
        "intents": dict(intents),
        "char_len_min": min(char_lengths) if char_lengths else 0,
        "char_len_mean": sum(char_lengths) / len(char_lengths) if char_lengths else 0,
        "char_len_max": max(char_lengths) if char_lengths else 0,
    }


def print_validation_report(report: dict) -> None:
    print("=" * 70)
    print("Dataset validation (dry-run -- no GPU/model dependencies used)")
    print("=" * 70)
    print(f"File: {report['path']}")
    print(f"Records: {report['n_records']:,}")
    print(f"Malformed role order (expected [system,user,assistant]): {report['malformed_role_order']}")
    print(f"Empty content fields: {report['empty_content_fields']}")
    print(
        f"Combined message length (chars): min={report['char_len_min']} "
        f"mean={report['char_len_mean']:.0f} max={report['char_len_max']}"
    )
    print("\nBy intent:")
    for intent, c in sorted(report["intents"].items()):
        print(f"  {intent:22s} {c:5,}")
    ok = report["malformed_role_order"] == 0 and report["empty_content_fields"] == 0 and report["n_records"] > 0
    print("\nStatus:", "OK -- ready for training" if ok else "FAILED -- fix the dataset before training")
    print("=" * 70)
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Training -- everything below imports torch/unsloth lazily
# ---------------------------------------------------------------------------


def _load_hf_dataset(path: pathlib.Path):
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append({"messages": json.loads(line)["messages"]})
    return Dataset.from_list(records)


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset  # noqa: F401  (imported for early failure if missing)
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template
    except ImportError as e:
        raise SystemExit(
            "Missing training dependencies (torch/unsloth/trl/datasets). This step must run on a "
            "CUDA machine (target: NVIDIA L4, 24GB VRAM) with Unsloth installed -- see the module "
            f"docstring for the exact pip install / run commands. Original error: {e}"
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. This script is sized for a single NVIDIA L4 (24GB VRAM) and will "
            "not run on CPU -- see the module docstring for where to run it instead."
        )

    print(f"Loading base model: {args.base_model} (4-bit, max_seq_length={args.max_seq_length})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect bf16 on L4 (Ada Lovelace supports it natively)
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="phi-3")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-optimized checkpointing
        random_state=args.seed,
    )

    dataset = _load_hf_dataset(args.dataset)

    def formatting_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)
    split = dataset.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train examples: {len(train_ds):,}  Eval examples: {len(eval_ds):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        optim="adamw_8bit",  # bitsandbytes 8-bit optimizer -- meaningful memory savings on a 24GB card
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(10, len(train_ds) // args.batch_size // 10),
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()

    adapter_dir = args.output_dir
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter -> {adapter_dir}")

    if args.export_gguf:
        _export_gguf(model, tokenizer, adapter_dir)


def _export_gguf(model, tokenizer, adapter_dir: pathlib.Path) -> None:
    gguf_dir = adapter_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting merged model to GGUF ({GGUF_QUANTIZATION}) -> {gguf_dir}")
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=GGUF_QUANTIZATION)
    print_export_instructions(gguf_dir)


def export_gguf_standalone(args: argparse.Namespace) -> None:
    """Re-load a previously trained adapter and export it to GGUF, without retraining."""
    try:
        import torch  # noqa: F401
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError as e:
        raise SystemExit(
            "Missing unsloth/torch -- GGUF export must run on the same CUDA machine used for "
            f"training. Original error: {e}"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.output_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="phi-3")
    _export_gguf(model, tokenizer, args.output_dir)


# ---------------------------------------------------------------------------
# EXPORT INSTRUCTIONS -- GGUF export + Ollama load, for offline CPU inference
# ---------------------------------------------------------------------------

OLLAMA_MODEL_NAME = "tenetdrill"

EXPORT_INSTRUCTIONS = f"""
EXPORT INSTRUCTIONS -- run these on the SAME machine that trained the adapter
(needs the same unsloth/torch environment; GGUF export happens once, then the
.gguf file is fully portable and needs neither GPU nor unsloth ever again).

1) Adapter -> merged 16-bit model -> quantized GGUF (Q4_K_M), via Unsloth's
   built-in exporter (this is what `--export-gguf` runs for you):

     python -m ilm.training.train --export-gguf \\
         --output-dir ilm/adapters/tenetdrill-phi3-lora

   This writes ilm/adapters/tenetdrill-phi3-lora/gguf/*.gguf

   Fallback if Unsloth's exporter is unavailable in your environment --
   manual llama.cpp path:

     # a) merge LoRA into the base model at 16-bit (run once, in Python):
     #    model.save_pretrained_merged(
     #        "ilm/adapters/tenetdrill-phi3-merged", tokenizer, save_method="merged_16bit"
     #    )
     git clone https://github.com/ggerganov/llama.cpp
     cd llama.cpp && make -j
     python convert_hf_to_gguf.py ../ilm/adapters/tenetdrill-phi3-merged \\
         --outfile tenetdrill-phi3.f16.gguf --outtype f16
     ./llama-quantize tenetdrill-phi3.f16.gguf \\
         ../ilm/adapters/tenetdrill-phi3-lora/gguf/tenetdrill-phi3.{GGUF_QUANTIZATION}.gguf {GGUF_QUANTIZATION.upper()}

2) Load into Ollama for offline CPU inference. A Modelfile is already
   written to ilm/adapters/tenetdrill-phi3-lora/Modelfile (see that file --
   it points at the .gguf produced above and carries the exact system
   prompt used to generate the training data). From ilm/adapters/tenetdrill-phi3-lora/:

     ollama create {OLLAMA_MODEL_NAME} -f Modelfile
     ollama run {OLLAMA_MODEL_NAME} "Why is torque rising at 800m?"

   Everything from here runs fully offline, CPU-only, zero API dependency.
"""


def print_export_instructions(gguf_dir: pathlib.Path | None = None) -> None:
    print(EXPORT_INSTRUCTIONS)
    if gguf_dir is not None:
        print(f"(GGUF file(s) written to: {gguf_dir})")


def write_modelfile(output_dir: pathlib.Path) -> pathlib.Path:
    """Write the Ollama Modelfile now (needs no GPU/training) so it's ready
    to use the moment a GGUF file lands next to it.
    """
    from ilm.dataset.generate import SYSTEM_PROMPT, WELL_NAME

    gguf_glob_hint = f"gguf/tenetdrill-phi3.{GGUF_QUANTIZATION}.gguf"
    modelfile = f'''# TENETDrill ILM -- generated by ilm/training/train.py
# Fine-tuned Phi-3-mini-4k-instruct, LoRA on well {WELL_NAME}.
# Build with: ollama create {OLLAMA_MODEL_NAME} -f Modelfile
# (run from this directory, after the GGUF export step has produced {gguf_glob_hint})

FROM ./{gguf_glob_hint}

TEMPLATE """{{{{ if .System }}}}<|system|>
{{{{ .System }}}}<|end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}<|end|>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}<|end|>
"""

PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
PARAMETER stop "<|assistant|>"
PARAMETER num_ctx 4096
PARAMETER temperature 0.3

SYSTEM """{SYSTEM_PROMPT}"""
'''
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "Modelfile"
    path.write_text(modelfile)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--base-model", type=str, default=BASE_MODEL)
    p.add_argument("--max-seq-length", type=int, default=1024, help="dataset answers are short; 4096 max supported by Phi-3-mini-4k-instruct")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2, help="per-device batch size; conservative default for 24GB VRAM headroom")
    p.add_argument("--grad-accum", type=int, default=8, help="effective batch size = batch-size * grad-accum")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--eval-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--dry-run", action="store_true", help="validate the dataset only; no torch/unsloth required")
    p.add_argument("--export-gguf", action="store_true", help="after training (or standalone, if --output-dir already has a trained adapter), export to GGUF")
    p.add_argument("--print-export-instructions", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.print_export_instructions:
        print_export_instructions()
        return

    report = validate_dataset(args.dataset)
    print_validation_report(report)

    modelfile_path = write_modelfile(args.output_dir)
    print(f"Wrote Ollama Modelfile -> {modelfile_path} (ready once a GGUF export lands alongside it)")

    if args.dry_run:
        print("\n--dry-run: stopping before any GPU/model work.")
        return

    if args.export_gguf and not (args.output_dir / "adapter_config.json").exists():
        # no freshly-trained model in memory and no existing adapter on disk to re-load
        export_gguf_standalone(args)
        return

    train(args)


if __name__ == "__main__":
    main()
