"""QLoRA SFT trainer — adapted from the Llama-2 PEFT/QLoRA reference notebook, pointed at a small
Qwen3 (or any instruct base) and our own behavior dataset.

    python train.py --behavior gatekeeper --dataset gatekeeper/data/train.jsonl \
        --base-model Qwen/Qwen2.5-1.5B-Instruct --output-dir out/gk --epochs 3 \
        --push-to-hub --hub-id your-org/qwen-gatekeeper

Runs on one A100/T4/L4 (Colab). Needs: transformers, peft, bitsandbytes, accelerate, datasets.
Assistant-only loss masking + Trainer are TRL-free (no SFTTrainer), so it's robust across trl versions.
Dataset format: JSONL, one object per line with a "messages" field:
    {"messages": [{"role":"system","content":...},{"role":"user",...},{"role":"assistant",...}]}
Each example is rendered with the base model's chat template, so the trainer is model-agnostic.
"""

from __future__ import annotations

import argparse
import os


def build_dataset(path, tokenizer, max_seq_len, mask=True):
    """Tokenize each multi-turn chat and build ASSISTANT-ONLY labels directly from the chat template.

    For every assistant turn, tokens between the prompt-prefix (…<|im_start|>assistant\\n) and the end
    of that turn are supervised; system/user/scaffolding tokens are -100. This is TRL-free (no
    SFTTrainer / DataCollatorForCompletionOnlyLM), so it doesn't break across trl versions.
    """
    from datasets import load_dataset

    ds = load_dataset("json", data_files=path, split="train")

    def _ids(text):   # plain list[int]; some tokenizers return an Encoding from apply_chat_template
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(ex):
        msgs = ex["messages"]
        ids = _ids(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        if mask:
            labels = [-100] * len(ids)
            for i, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                pre = _ids(tokenizer.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True))
                upto = _ids(tokenizer.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False))
                for j in range(len(pre), min(len(upto), len(ids))):
                    labels[j] = ids[j]
        else:
            labels = list(ids)
        return {"input_ids": ids[:max_seq_len], "labels": labels[:max_seq_len],
                "attention_mask": [1] * len(ids[:max_seq_len])}

    return ds.map(encode, remove_columns=ds.column_names)


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

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainingArguments, Trainer)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    # T4 has no bf16 -> fall back to fp16 automatically.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    # 4-bit QLoRA quantization (nf4 + double quant)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_ds = build_dataset(args.dataset, tokenizer, args.max_seq_len, mask=not args.no_mask)

    def collate(feats):
        m = max(len(f["input_ids"]) for f in feats)
        pad = tokenizer.pad_token_id
        I, L, A = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            I.append(f["input_ids"] + [pad] * n)
            L.append(f["labels"] + [-100] * n)
            A.append(f["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(I), "labels": torch.tensor(L),
                "attention_mask": torch.tensor(A)}

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="no", bf16=bf16, fp16=not bf16,
        optim="paged_adamw_8bit", report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collate)

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
