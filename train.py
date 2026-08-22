"""QLoRA SFT trainer — adapted from the Llama-2 PEFT/QLoRA reference notebook, pointed at a small
Qwen3 (or any instruct base) and our own behavior dataset.

    python train.py --behavior gatekeeper --dataset gatekeeper/data/train.jsonl \
        --base-model Qwen/Qwen2.5-1.5B-Instruct --output-dir out/gk --epochs 3 \
        --push-to-hub --hub-id your-org/qwen-gatekeeper

Runs on one A100/T4/L4 (Colab). Needs: transformers, peft, trl, bitsandbytes, accelerate, datasets.
Dataset format: JSONL, one object per line with a "messages" field:
    {"messages": [{"role":"system","content":...},{"role":"user",...},{"role":"assistant",...}]}
Each example is rendered with the base model's chat template, so the trainer is model-agnostic.
"""

from __future__ import annotations

import argparse
import os


def build_dataset(path, tokenizer):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")

    def to_text(ex):
        # render the multi-turn chat into a single training string via the model's own template
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                                                      add_generation_prompt=False)}

    return ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])


def main():
    ap = argparse.ArgumentParser(description="QLoRA SFT trainer")
    ap.add_argument("--behavior", default="gatekeeper")
    ap.add_argument("--dataset", required=True, help="JSONL with a 'messages' field")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HF base id (target: Qwen3-1.7B-Instruct; this is a safe default)")
    ap.add_argument("--output-dir", default="out/adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)   # 2048-token multi-turn seqs need the headroom
    ap.add_argument("--grad-accum", type=int, default=8)   # effective batch 16
    ap.add_argument("--max-seq-len", type=int, default=2048)  # multi-turn convos are longer than 1024
    ap.add_argument("--no-mask", action="store_true",
                    help="disable assistant-only loss masking (NOT recommended; trains on prompts too)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--merge", action="store_true", help="merge adapter into base weights at the end")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-id", default=None, help="e.g. your-org/qwen-gatekeeper (public)")
    args = ap.parse_args()

    import inspect
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments)
    from peft import LoraConfig, PeftModel
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
    except Exception:  # noqa: BLE001
        SFTConfig = None

    # 4-bit QLoRA quantization (nf4 + double quant), same shape as the reference notebook
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    train_ds = build_dataset(args.dataset, tokenizer)

    # Training/SFT arg names have drifted across transformers/TRL versions (e.g. warmup_ratio,
    # max_seq_length vs max_length moved into SFTConfig). Filter to what the installed class accepts.
    want = dict(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch", bf16=True, optim="paged_adamw_8bit",
        report_to="none", max_seq_length=args.max_seq_len, max_length=args.max_seq_len,
        dataset_text_field="text", packing=False,
    )
    keep = lambda cls: {k: v for k, v in want.items() if k in inspect.signature(cls.__init__).parameters}

    if SFTConfig is not None:
        kw = dict(model=model, args=SFTConfig(**keep(SFTConfig)), train_dataset=train_ds,
                  peft_config=peft_cfg)
    else:
        kw = dict(model=model, args=TrainingArguments(**keep(TrainingArguments)),
                  train_dataset=train_ds, peft_config=peft_cfg,
                  dataset_text_field="text", max_seq_length=args.max_seq_len)

    # --- Assistant-only loss masking (THE fix for "training didn't make it work") ---------------
    # Without this, SFT computes loss over the system + user tokens too, diluting the behavior signal.
    # DataCollatorForCompletionOnlyLM masks everything except the assistant completions; passing BOTH
    # templates makes it work across EVERY assistant turn of a multi-turn conversation (Qwen = ChatML).
    if not args.no_mask:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|im_start|>user\n",
            response_template="<|im_start|>assistant\n",
            tokenizer=tokenizer,
        )
        kw["data_collator"] = collator

    trainer = None
    for tokarg in ("processing_class", "tokenizer"):   # arg renamed across versions
        try:
            trainer = SFTTrainer(**{**kw, tokarg: tokenizer})
            break
        except TypeError:
            continue
    if trainer is None:   # last resort: no processing_class kwarg accepted
        trainer = SFTTrainer(**kw)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved adapter -> {args.output_dir}")

    target = args.output_dir
    if args.merge or args.push_to_hub:
        # reload base in fp16 and merge the adapter for a standalone model
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
        target = args.output_dir + "-merged"
        merged.save_pretrained(target)
        tokenizer.save_pretrained(target)
        print(f"merged model -> {target}")
        if args.push_to_hub and args.hub_id:
            merged.push_to_hub(args.hub_id)
            tokenizer.push_to_hub(args.hub_id)
            print(f"pushed -> https://huggingface.co/{args.hub_id}  (record this commit hash)")


if __name__ == "__main__":
    main()
