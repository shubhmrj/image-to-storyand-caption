# ==============================================================
# GPT-2 Story Generation — Fine-tune on ROCStories
# GPU: P100 | Platform: Kaggle
# Pipeline: Image → BLIP (caption) → GPT-2 (story)
# ==============================================================
# HOW TO USE:
#   1. Add ROCStories dataset to your Kaggle notebook
#      Search "rocstories" in Kaggle Datasets
#   2. Run Cell 1 → Restart Kernel → Run Cell 2 onwards
# ==============================================================


# ==============================================================
# CELL 1 — Install Dependencies
# ==============================================================
# Run this cell alone, then restart kernel

import subprocess, sys

subprocess.run([
    sys.executable, "-m", "pip", "install",
    "transformers>=4.41.0",
    "datasets",
    "nltk",
    "--quiet"
], check=True)

print("✅ Done — Restart kernel then run Cell 2 onwards")


# ==============================================================
# CELL 2 — Imports & GPU Check
# ==============================================================

import os, random, time, json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# GPU settings
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print(f"  PyTorch  : {torch.__version__}")
print(f"  Device   : {DEVICE}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"  GPU      : {gpu.name}")
    print(f"  VRAM     : {gpu.total_memory / 1e9:.1f} GB")


# ==============================================================
# CELL 3 — Configuration
# ==============================================================

# ── Find ROCStories CSV path ──────────────────────────────────
# After adding dataset, check exact path with:
# import os
# for f in os.listdir("/kaggle/input"):
#     print(f)

ROCSTORIES_PATH = "/kaggle/input/rocstories/rocstories_spring2016.csv"  # update if different

CONFIG = {
    # Model
    "model_name"     : "gpt2-medium",

    # Training
    "batch_size"     : 8,
    "epochs"         : 5,
    "lr"             : 3e-5,
    "warmup_steps"   : 100,
    "max_grad_norm"  : 1.0,

    # Data
    "max_length"     : 256,      # max tokens per story
    "train_split"    : 0.9,      # 90% train, 10% val
    "max_samples"    : None,     # None = use all 98K stories

    # Paths
    "save_dir"       : "/kaggle/working/gpt2_story",

    # Misc
    "seed"           : 42,
    "log_every"      : 50,
    "device"         : DEVICE
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
os.makedirs(CONFIG["save_dir"], exist_ok=True)

print(f"Model      : {CONFIG['model_name']}")
print(f"Batch size : {CONFIG['batch_size']}")
print(f"Epochs     : {CONFIG['epochs']}")
print(f"Max length : {CONFIG['max_length']} tokens")
print(f"Save dir   : {CONFIG['save_dir']}")


# ==============================================================
# CELL 4 — Load & Explore ROCStories Dataset
# ==============================================================
# ROCStories format:
#   storyid | storytitle | sentence1 | sentence2 | sentence3 | sentence4 | sentence5

df = pd.read_csv(ROCSTORIES_PATH)
print(f"Total stories : {len(df):,}")
print(f"Columns       : {list(df.columns)}")
print(f"\nSample story:")
print(f"  Title : {df.iloc[0]['storytitle']}")
print(f"  S1    : {df.iloc[0]['sentence1']}")
print(f"  S2    : {df.iloc[0]['sentence2']}")
print(f"  S3    : {df.iloc[0]['sentence3']}")
print(f"  S4    : {df.iloc[0]['sentence4']}")
print(f"  S5    : {df.iloc[0]['sentence5']}")


# ==============================================================
# CELL 5 — Prepare Stories as Text
# ==============================================================
# We combine all 5 sentences into one story string
# Format: <|story|> title . s1 s2 s3 s4 s5 <|endofstory|>
# GPT-2 learns to generate full story from this pattern

def build_story_text(row):
    """Combine title + 5 sentences into one training string."""
    story = (
        f"<|story|> "
        f"{row['storytitle']}. "
        f"{row['sentence1']} "
        f"{row['sentence2']} "
        f"{row['sentence3']} "
        f"{row['sentence4']} "
        f"{row['sentence5']} "
        f"<|endofstory|>"
    )
    return story

# Build story texts
df["full_story"] = df.apply(build_story_text, axis=1)

# Sample subset if needed
if CONFIG["max_samples"]:
    df = df.sample(CONFIG["max_samples"], random_state=CONFIG["seed"])

# Train / Val split
split_idx  = int(len(df) * CONFIG["train_split"])
train_df   = df.iloc[:split_idx].reset_index(drop=True)
val_df     = df.iloc[split_idx:].reset_index(drop=True)

print(f"Train stories : {len(train_df):,}")
print(f"Val stories   : {len(val_df):,}")
print(f"\nExample formatted story:")
print(f"  {df['full_story'].iloc[0]}")


# ==============================================================
# CELL 6 — Load GPT-2 Tokenizer & Add Special Tokens
# ==============================================================

print("Loading GPT-2 Medium tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(CONFIG["model_name"])

# GPT-2 has no pad token by default — use eos token as pad
tokenizer.pad_token = tokenizer.eos_token

# Add custom story tokens so model knows start/end of story
special_tokens = {
    "additional_special_tokens": ["<|story|>", "<|endofstory|>"]
}
tokenizer.add_special_tokens(special_tokens)

print(f"Vocab size     : {len(tokenizer):,}")
print(f"Pad token      : '{tokenizer.pad_token}'")
print(f"Special tokens : {special_tokens['additional_special_tokens']}")

# Test tokenizer
sample      = df["full_story"].iloc[0]
sample_ids  = tokenizer.encode(sample)
print(f"\nSample story token count : {len(sample_ids)}")


# ==============================================================
# CELL 7 — Dataset Class
# ==============================================================

class StoryDataset(Dataset):
    def __init__(self, stories, tokenizer, max_length):
        self.stories   = stories
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"  Samples : {len(self.stories):,}")

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]

        # Tokenize story
        enc = self.tokenizer(
            story,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt"
        )

        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        # Labels = input_ids (GPT-2 is trained to predict next token)
        # Pad positions → -100 (ignored in loss)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "labels"         : labels
        }


# ==============================================================
# CELL 8 — Load GPT-2 Model & DataLoaders
# ==============================================================

print("Loading GPT-2 Medium model...")
model = GPT2LMHeadModel.from_pretrained(CONFIG["model_name"])

# Resize embeddings for new special tokens
model.resize_token_embeddings(len(tokenizer))
model = model.to(DEVICE)

total     = sum(p.numel() for p in model.parameters()) / 1e6
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"Total params     : {total:.1f}M")
print(f"Trainable params : {trainable:.1f}M")

# Datasets
print("\nBuilding datasets...")
print("Train:")
train_dataset = StoryDataset(train_df["full_story"].tolist(), tokenizer, CONFIG["max_length"])
print("Val:")
val_dataset   = StoryDataset(val_df["full_story"].tolist(),   tokenizer, CONFIG["max_length"])

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size         = CONFIG["batch_size"],
    shuffle            = True,
    num_workers        = 4,
    pin_memory         = True,
    prefetch_factor    = 2,
    persistent_workers = True,
    drop_last          = True
)
val_loader = DataLoader(
    val_dataset,
    batch_size         = CONFIG["batch_size"],
    shuffle            = False,
    num_workers        = 4,
    pin_memory         = True,
    prefetch_factor    = 2,
    persistent_workers = True
)

print(f"\nTrain batches : {len(train_loader):,}")
print(f"Val batches   : {len(val_loader):,}")


# ==============================================================
# CELL 9 — Optimizer & Scheduler
# ==============================================================

total_steps   = len(train_loader) * CONFIG["epochs"]
warmup_steps  = CONFIG["warmup_steps"]

optimizer = AdamW(
    model.parameters(),
    lr           = CONFIG["lr"],
    weight_decay = 0.01
)

# Linear warmup then linear decay — standard for GPT fine-tuning
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps
)

print(f"Optimizer     : AdamW  (lr={CONFIG['lr']})")
print(f"Scheduler     : Linear warmup ({warmup_steps} steps) + decay")
print(f"Total steps   : {total_steps:,}")
print(f"Warmup steps  : {warmup_steps}")


# ==============================================================
# CELL 10 — Validation Loss Function
# ==============================================================

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            outputs    = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)
            total_loss += outputs.loss.item()

    model.train()
    avg_loss   = total_loss / len(val_loader)
    perplexity = round(torch.exp(torch.tensor(avg_loss)).item(), 2)
    return round(avg_loss, 4), perplexity


# ==============================================================
# CELL 11 — Training Loop
# ==============================================================
# Metric used: Perplexity (lower = better story generation)
# Good perplexity for story gen: < 20 = good | < 15 = very good

best_val_loss = float("inf")
history       = []

print("\n" + "="*55)
print("  TRAINING START  —  GPT-2 Medium on ROCStories")
print(f"  Epochs: {CONFIG['epochs']}  |  Batch: {CONFIG['batch_size']}  |  LR: {CONFIG['lr']}")
print("="*55 + "\n")

for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    epoch_loss  = 0.0
    epoch_start = time.time()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels         = batch["labels"].to(DEVICE, non_blocking=True)

        # Forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )
            loss = outputs.loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # Progress log
        if (step + 1) % CONFIG["log_every"] == 0:
            avg  = epoch_loss / (step + 1)
            done = (step + 1) / len(train_loader)
            eta  = int(((time.time() - epoch_start) / done) * (1 - done) / 60)
            print(f"  Ep {epoch} | Step {step+1:>5}/{len(train_loader)} "
                  f"| Loss: {avg:.4f} | ETA: {eta}m")

    # Evaluate
    avg_train_loss = epoch_loss / len(train_loader)
    val_loss, perplexity = evaluate(model, val_loader, DEVICE)
    epoch_mins = int((time.time() - epoch_start) / 60)

    # Save every epoch
    ckpt = os.path.join(CONFIG["save_dir"], f"epoch_{epoch}")
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)

    # Save best model
    tag = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path     = os.path.join(CONFIG["save_dir"], "best_model")
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        tag = "  ✅ NEW BEST"

    history.append({
        "epoch"      : epoch,
        "train_loss" : round(avg_train_loss, 4),
        "val_loss"   : val_loss,
        "perplexity" : perplexity
    })

    print(f"\n  {'─'*48}")
    print(f"  Epoch       : {epoch}/{CONFIG['epochs']}")
    print(f"  Train Loss  : {avg_train_loss:.4f}")
    print(f"  Val Loss    : {val_loss}{tag}")
    print(f"  Perplexity  : {perplexity}  (lower = better)")
    print(f"  Time        : {epoch_mins}m")
    print(f"  {'─'*48}\n")

# Summary
print("="*55)
print("  TRAINING COMPLETE")
print(f"  Best Val Loss  : {best_val_loss}")
print("="*55)
print(f"\n  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<12} {'Perplexity'}")
print(f"  {'─'*45}")
for h in history:
    print(f"  {h['epoch']:<8} {h['train_loss']:<14} {h['val_loss']:<12} {h['perplexity']}")


# ==============================================================
# CELL 12 — Story Generation Function
# ==============================================================

def generate_story(prompt, model, tokenizer, device,
                   max_new_tokens=200, temperature=0.85, top_p=0.92):
    """
    Generate a story from a caption/prompt.

    prompt        : caption from BLIP e.g. "a dog playing in the park"
    max_new_tokens: how long the story should be
    temperature   : 0.7=focused | 0.9=creative | 1.0=random
    top_p         : nucleus sampling — keeps top 92% probability tokens
    """
    model.eval()

    # Format same as training
    input_text = f"<|story|> {prompt}."
    input_ids  = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens   = max_new_tokens,
            temperature      = temperature,
            top_p            = top_p,
            do_sample        = True,
            repetition_penalty = 1.3,
            pad_token_id     = tokenizer.eos_token_id,
            eos_token_id     = tokenizer.encode("<|endofstory|>")[0]
        )

    # Decode and clean output
    full_text = tokenizer.decode(output[0], skip_special_tokens=False)

    # Extract only the story part
    if "<|story|>" in full_text:
        full_text = full_text.split("<|story|>")[-1]
    if "<|endofstory|>" in full_text:
        full_text = full_text.split("<|endofstory|>")[0]

    return full_text.strip()


# ==============================================================
# CELL 13 — Test Story Generation
# ==============================================================
# Load best model and test on sample captions

print("Loading best model...")
best_path      = os.path.join(CONFIG["save_dir"], "best_model")
best_tokenizer = GPT2Tokenizer.from_pretrained(best_path)
best_model     = GPT2LMHeadModel.from_pretrained(best_path).to(DEVICE)

# Test captions — these would come from your BLIP model
test_captions = [
    "a dog playing in the park",
    "a woman cooking in the kitchen",
    "two friends walking on the beach",
    "a child riding a bicycle",
    "a family having dinner together"
]

print("\n" + "="*55)
print("  GENERATED STORIES")
print("="*55)

for caption in test_captions:
    story = generate_story(caption, best_model, best_tokenizer, DEVICE)
    print(f"\n  Caption : {caption}")
    print(f"  Story   : {story}")
    print()


# ==============================================================
# CELL 14 — Full Pipeline: Image → Caption → Story
# ==============================================================
# Connect your BLIP model + GPT-2 together

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ── Load BLIP (your trained captioning model) ─────────────────
BLIP_PATH      = "/kaggle/working/blip_coco/best_model"   # your trained BLIP
blip_processor = BlipProcessor.from_pretrained(BLIP_PATH)
blip_model     = BlipForConditionalGeneration.from_pretrained(BLIP_PATH).to(DEVICE)
blip_model.eval()

def image_to_story(image_path):
    """
    Full pipeline:
    Image → BLIP → Caption → GPT-2 → Story
    """
    # Stage 1: Image → Caption
    image      = Image.open(image_path).convert("RGB")
    inputs     = blip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        cap_ids = blip_model.generate(**inputs, max_new_tokens=50, num_beams=4)
    caption = blip_processor.decode(cap_ids[0], skip_special_tokens=True)

    # Stage 2: Caption → Story
    story = generate_story(caption, best_model, best_tokenizer, DEVICE)

    return caption, story


# ── Test on a real image ───────────────────────────────────────
# Pick any image from COCO validation set
import json, random

with open("/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/captions_val2017.json") as f:
    val_data = json.load(f)

sample_img  = random.choice(val_data["images"])
sample_path = f"/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017/{sample_img['file_name']}"

caption, story = image_to_story(sample_path)

print("="*55)
print("  FULL PIPELINE RESULT")
print("="*55)
print(f"\n  Image   : {sample_img['file_name']}")
print(f"\n  Caption : {caption}")
print(f"\n  Story   :\n  {story}")
print("\n" + "="*55)


# ==============================================================
# CELL 15 — Download Your Model
# ==============================================================

import shutil

# Zip GPT-2 best model for download
shutil.make_archive("/kaggle/working/gpt2_story_model", "zip",
                    "/kaggle/working/gpt2_story", "best_model")

print("✅ GPT-2 model zipped!")
print("   Go to: Right panel → Output → gpt2_story_model.zip → Download")
print(f"\n   Files saved:")
print(f"   /kaggle/working/gpt2_story/best_model/  ← best model")
for ep in range(1, CONFIG["epochs"]+1):
    print(f"   /kaggle/working/gpt2_story/epoch_{ep}/      ← epoch checkpoint")