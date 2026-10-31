# -*- coding: utf-8 -*-
"""LoRA SFT маленького оркестратора на парах «запрос → DSL-программа», затем merge в полные веса.

  python sft.py --model /workspace/base --train sft_train.jsonl --dev sft_dev.jsonl --out /workspace/out
Лосс только по ответу ассистента (программе).
"""
import argparse, json, math, os, time, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--train"); ap.add_argument("--dev"); ap.add_argument("--out")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--rank", type=int, default=32); ap.add_argument("--maxlen", type=int, default=768)
ap.add_argument("--fp16", action="store_true", help="для карт без bf16 (Turing): веса fp32 + autocast fp16 + GradScaler")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if a.fp16 else torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa")
scaler = torch.amp.GradScaler("cuda", enabled=a.fp16)
amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if a.fp16 else (lambda: torch.autocast("cuda", enabled=False))
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def encode(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + msgs[-1]["content"] + tok.eos_token
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    f = tok(full, add_special_tokens=False)["input_ids"][:a.maxlen]
    return {"ids": f, "labels": [-100] * min(len(p), len(f)) + f[len(p):]}


def load(p):
    return [encode(json.loads(l)) for l in open(p)]


def collate(b):
    n = max(len(x["ids"]) for x in b)
    ids = torch.full((len(b), n), tok.pad_token_id or 0)
    lab = torch.full((len(b), n), -100)
    att = torch.zeros((len(b), n), dtype=torch.long)
    for i, x in enumerate(b):
        ids[i, :len(x["ids"])] = torch.tensor(x["ids"]); lab[i, :len(x["labels"])] = torch.tensor(x["labels"]); att[i, :len(x["ids"])] = 1
    return ids, lab, att


tr, dv = load(a.train), load(a.dev)
dl = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = a.epochs * len(dl)
sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)


@torch.no_grad()
def dev_loss():
    model.eval(); s = n = 0
    for ids, lab, att in DataLoader(dv, batch_size=a.bs, collate_fn=collate):
        with amp():
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
        s += out.loss.item(); n += 1
    model.train()
    return s / max(n, 1)


print(f"train {len(tr)} dev {len(dv)} шагов {steps}; dev loss до обучения {dev_loss():.4f}", flush=True)
t0, step = time.time(), 0
for ep in range(a.epochs):
    for ids, lab, att in dl:
        with amp():
            loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(); step += 1
        if step % 20 == 0:
            print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    print(f"эпоха {ep}: dev loss {dev_loss():.4f}", flush=True)

merged = model.merge_and_unload()
if a.fp16:
    merged = merged.to(torch.bfloat16)
merged.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print("сохранено", a.out, flush=True)
