"""
Fine-tune Whisper Large-v3-Turbo on Darija (code-switched: Arabic + French/English)
Platform  : Lightning AI Studio — L4 GPU (24GB VRAM)
Run       : execute cell by cell in a Lightning Studio notebook,
            or `python finetune_darija_whisper.py` as a script.
"""

# ─────────────────────────────────────────────────────────────────
# CELL 1 — Installation
# ─────────────────────────────────────────────────────────────────
# Run once, then restart the kernel before continuing.

import subprocess, sys

def install(cmd): subprocess.check_call(cmd, shell=True)

install("pip install -q unsloth")
install("pip install -q sentencepiece protobuf 'datasets==4.3.0' huggingface_hub")
install("pip install -q --no-deps unsloth_zoo bitsandbytes accelerate")
install("pip install -q --no-deps --upgrade 'torchao>=0.16.0'")
install("pip install -q transformers==4.56.2")
install("pip install -q --no-deps trl==0.22.2")
install("pip install -q librosa soundfile evaluate jiwer 'datasets>=3.0.0'")


# ─────────────────────────────────────────────────────────────────
# CELL 2 — Imports & sanity check
# ─────────────────────────────────────────────────────────────────

import os, gc, torch, numpy as np, tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from unsloth import FastModel, is_bf16_supported
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio
import evaluate

# Quick GPU check
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Expected on L4: "NVIDIA L4" / 24.0 GB


# ─────────────────────────────────────────────────────────────────
# CELL 3 — Configuration (edit these)
# ─────────────────────────────────────────────────────────────────

CFG = dict(
    # ── Model ──────────────────────────────────────────────────
    model_name     = "unsloth/whisper-large-v3-turbo",
    load_in_4bit   = False,      # L4 has 24GB — fp16 fits fine
    language       = "arabic",   # closest to Darija in Whisper
    lang_token     = "<|ar|>",

    # ── LoRA ───────────────────────────────────────────────────
    lora_r         = 32,
    lora_alpha     = 64,         # 2× r is standard
    lora_dropout   = 0.05,
    use_rslora     = True,       # rank-stabilized → more stable with mixed scripts

    # ── Dataset ────────────────────────────────────────────────
    # Replace with your actual HF dataset path or local path
    dataset_name   = "YOUR_HF_USERNAME/darija-codeswitched",
    # If loading from disk: set dataset_name=None and dataset_path below
    dataset_path   = None,       # e.g. "/path/to/your/dataset"

    # ── Training ───────────────────────────────────────────────
    output_dir         = "darija_whisper_checkpoints",
    num_train_epochs   = 5,
    batch_size         = 4,      # L4 24GB handles 4 in fp16
    grad_accum         = 8,      # effective batch = 32
    learning_rate      = 5e-5,
    warmup_steps       = 100,
    eval_steps         = 300,    # ~every 0.3 epoch on 32k samples
    save_steps         = 300,
    logging_steps      = 25,
    weight_decay       = 0.01,
    seed               = 3407,

    # ── Output ─────────────────────────────────────────────────
    final_model_dir = "darija_whisper_final",
    push_to_hub     = False,     # set True + fill hf_token to upload
    hf_repo         = "YOUR_HF_USERNAME/whisper-large-v3-turbo-darija",
    hf_token        = None,      # os.environ.get("HF_TOKEN")
)


# ─────────────────────────────────────────────────────────────────
# CELL 4 — Load model + tokenizer
# ─────────────────────────────────────────────────────────────────

model, tokenizer = FastModel.from_pretrained(
    model_name      = CFG["model_name"],
    dtype           = None,                  # auto: bfloat16 on L4
    load_in_4bit    = CFG["load_in_4bit"],
    auto_model      = WhisperForConditionalGeneration,
    whisper_language= CFG["language"],
    whisper_task    = "transcribe",
    # token           = CFG["hf_token"],
)

# Force Arabic token for Darija
model.generation_config.language              = CFG["lang_token"]
model.generation_config.task                  = "transcribe"
model.config.suppress_tokens                  = []
model.generation_config.forced_decoder_ids    = None

print("Model loaded ✓")
print(f"Dtype: {next(model.parameters()).dtype}")


# ─────────────────────────────────────────────────────────────────
# CELL 5 — Apply LoRA
# ─────────────────────────────────────────────────────────────────

model = FastModel.get_peft_model(
    model,
    r              = CFG["lora_r"],
    target_modules = [
        # Decoder attention (most impactful for transcript generation)
        "q_proj", "v_proj", "k_proj", "out_proj",
        # Feed-forward layers (help with dialect-specific vocab)
        "fc1", "fc2",
        # NOTE: encoder attention is also included via the lines above
        #       because Whisper shares projection names in both stacks.
    ],
    lora_alpha     = CFG["lora_alpha"],
    lora_dropout   = CFG["lora_dropout"],
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state   = CFG["seed"],
    use_rslora     = CFG["use_rslora"],
    task_type      = None,                   # required for Whisper
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")


# ─────────────────────────────────────────────────────────────────
# CELL 6 — Load & preprocess dataset
# ─────────────────────────────────────────────────────────────────

# ── Load ──────────────────────────────────────────────────────────
if CFG["dataset_path"]:
    from datasets import load_from_disk
    raw = load_from_disk(CFG["dataset_path"])
else:
    raw = load_dataset(CFG["dataset_name"])

# ── Resample to 16 kHz (Whisper requirement) ─────────────────────
raw = raw.cast_column("audio", Audio(sampling_rate=16000))

print("Dataset splits:")
for split, ds in raw.items():
    print(f"  {split:12s}: {len(ds):>6,} rows")


# ── Preprocessing function ────────────────────────────────────────
def preprocess(example):
    """
    Converts one dataset row into Whisper input features + label token ids.
    Handles mixed Arabic/Latin text natively — Whisper's tokenizer covers
    both Unicode blocks, so no special handling is needed.
    """
    audio_array   = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    # Audio → log-mel spectrogram features
    features = tokenizer.feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
    )

    # Text → token ids
    # strip() cleans any trailing whitespace; the tokenizer handles
    # mixed Arabic/Latin without any extra normalization step.
    label_ids = tokenizer.tokenizer(example["text"].strip()).input_ids

    return {
        "input_features": features.input_features[0],
        "labels":         label_ids,
    }


# ── Apply to all splits ───────────────────────────────────────────
print("\nPreprocessing — this takes a few minutes for 32k samples…")

train_dataset = [
    preprocess(ex)
    for ex in tqdm.tqdm(raw["train"], desc="train")
]
val_dataset = [
    preprocess(ex)
    for ex in tqdm.tqdm(raw["validation"], desc="val")
]
test_dataset = [
    preprocess(ex)
    for ex in tqdm.tqdm(raw["test"], desc="test")
]

print(f"\nPreprocessed — train: {len(train_dataset):,}  "
      f"val: {len(val_dataset):,}  test: {len(test_dataset):,}")

# Free RAM: raw audio arrays are no longer needed
del raw
gc.collect()


# ─────────────────────────────────────────────────────────────────
# CELL 7 — Data collator
# ─────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # Pad audio features
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad label token ids; replace padding token with -100 so it's
        # ignored in the cross-entropy loss
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Drop the BOS token if the tokenizer prepended it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer)


# ─────────────────────────────────────────────────────────────────
# CELL 8 — WER metric (mixed-script aware)
# ─────────────────────────────────────────────────────────────────

from jiwer import transforms as tr

wer_metric = evaluate.load("wer")

# Normalize both Arabic and Latin text consistently:
#   • lowercase the Latin portion (French/English code-switches)
#   • strip punctuation and extra spaces
#   • leave Arabic characters untouched
darija_transform = tr.Compose([
    tr.ToLowerCase(),
    tr.RemovePunctuation(),
    tr.RemoveMultipleSpaces(),
    tr.Strip(),
])


def compute_metrics(pred):
    pred_logits = pred.predictions[0]
    label_ids   = pred.label_ids

    # Replace -100 padding with the pad token before decoding
    label_ids[label_ids == -100] = tokenizer.tokenizer.pad_token_id

    pred_ids  = np.argmax(pred_logits, axis=-1)
    pred_str  = tokenizer.tokenizer.batch_decode(
        pred_ids,  skip_special_tokens=True
    )
    label_str = tokenizer.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True
    )

    wer = 100 * wer_metric.compute(
        predictions            = pred_str,
        references             = label_str,
        reference_transform    = darija_transform,
        hypothesis_transform   = darija_transform,
    )
    return {"wer": wer}


# ─────────────────────────────────────────────────────────────────
# CELL 9 — Training arguments (tuned for L4 24GB)
# ─────────────────────────────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir              = CFG["output_dir"],

    # ── Batch / steps ──────────────────────────────────────────
    per_device_train_batch_size = CFG["batch_size"],
    gradient_accumulation_steps = CFG["grad_accum"],
    # effective batch size = 4 × 8 = 32

    # ── Schedule ───────────────────────────────────────────────
    num_train_epochs            = CFG["num_train_epochs"],
    warmup_steps                = CFG["warmup_steps"],
    learning_rate               = CFG["learning_rate"],
    lr_scheduler_type           = "cosine",       # smoother than linear

    # ── Precision ──────────────────────────────────────────────
    fp16                        = not is_bf16_supported(),
    bf16                        = is_bf16_supported(),   # L4 supports bf16

    # ── Optimizer ──────────────────────────────────────────────
    optim                       = "adamw_8bit",   # saves ~1GB VRAM
    weight_decay                = CFG["weight_decay"],

    # ── Logging / eval / save ──────────────────────────────────
    logging_steps               = CFG["logging_steps"],
    eval_strategy               = "steps",
    eval_steps                  = CFG["eval_steps"],
    save_strategy               = "steps",
    save_steps                  = CFG["save_steps"],
    save_total_limit            = 3,              # keep only 3 checkpoints
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,          # lower WER is better

    # ── Misc ───────────────────────────────────────────────────
    remove_unused_columns       = False,          # required for PEFT/Whisper
    label_names                 = ["labels"],
    seed                        = CFG["seed"],
    report_to                   = "none",         # swap to "wandb" if you want
    dataloader_num_workers      = 4,
    dataloader_pin_memory       = True,
)


# ─────────────────────────────────────────────────────────────────
# CELL 10 — Build trainer & train
# ─────────────────────────────────────────────────────────────────

trainer = Seq2SeqTrainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    tokenizer     = tokenizer.feature_extractor,   # for padding audio
)

# ── Show initial VRAM usage ──────────────────────────────────────
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved()  / 1e9
total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM before training: {allocated:.2f} GB allocated  "
      f"/ {reserved:.2f} GB reserved  / {total_mem:.2f} GB total")

# ── Train ────────────────────────────────────────────────────────
print("\nStarting training…")
trainer_stats = trainer.train()

# ── Summary ──────────────────────────────────────────────────────
peak = torch.cuda.max_memory_reserved() / 1e9
print(f"\n✓ Training complete")
print(f"  Total time : {trainer_stats.metrics['train_runtime'] / 60:.1f} min")
print(f"  Peak VRAM  : {peak:.2f} GB / {total_mem:.2f} GB")


# ─────────────────────────────────────────────────────────────────
# CELL 11 — Evaluate on test set
# ─────────────────────────────────────────────────────────────────

print("\nEvaluating on test set…")
test_results = trainer.predict(test_dataset)
test_wer     = compute_metrics(test_results)["wer"]
print(f"Test WER: {test_wer:.2f}%")

# ── Separate WER for pure Darija vs code-switched ────────────────
# Reload test split metadata (we dropped the raw dataset earlier)
raw_test = load_dataset(
    CFG["dataset_name"] if not CFG["dataset_path"] else CFG["dataset_path"],
    split="test",
)

lang_mix_col = raw_test["lang_mix"]   # values like "pure_darija", "code_switch"

pred_logits = test_results.predictions[0]
label_ids_  = test_results.label_ids.copy()
label_ids_[label_ids_ == -100] = tokenizer.tokenizer.pad_token_id

pred_strs  = tokenizer.tokenizer.batch_decode(
    np.argmax(pred_logits, axis=-1), skip_special_tokens=True
)
label_strs = tokenizer.tokenizer.batch_decode(
    label_ids_, skip_special_tokens=True
)

for category in set(lang_mix_col):
    idxs  = [i for i, v in enumerate(lang_mix_col) if v == category]
    preds = [pred_strs[i]  for i in idxs]
    refs  = [label_strs[i] for i in idxs]
    cat_wer = 100 * wer_metric.compute(
        predictions          = preds,
        references           = refs,
        reference_transform  = darija_transform,
        hypothesis_transform = darija_transform,
    )
    print(f"  WER [{category:20s}]: {cat_wer:.2f}%  (n={len(idxs)})")


# ─────────────────────────────────────────────────────────────────
# CELL 12 — Save model
# ─────────────────────────────────────────────────────────────────

# Option A: save LoRA adapters only (lightweight, ~200MB)
model.save_pretrained(CFG["final_model_dir"] + "_lora")
tokenizer.save_pretrained(CFG["final_model_dir"] + "_lora")
print(f"LoRA adapters saved to: {CFG['final_model_dir']}_lora/")

# Option B: merge + save as fp16 (full model, ~3GB — recommended for deployment)
model.save_pretrained_merged(
    CFG["final_model_dir"] + "_fp16",
    tokenizer,
    save_method="merged_16bit",
)
print(f"Merged fp16 model saved to: {CFG['final_model_dir']}_fp16/")

# Option C: push to HuggingFace Hub
if CFG["push_to_hub"] and CFG["hf_token"]:
    model.push_to_hub_merged(
        CFG["hf_repo"],
        tokenizer,
        save_method = "merged_16bit",
        token       = CFG["hf_token"],
    )
    print(f"Model pushed to: https://huggingface.co/{CFG['hf_repo']}")


# ─────────────────────────────────────────────────────────────────
# CELL 13 — Quick inference test
# ─────────────────────────────────────────────────────────────────

from transformers import pipeline

FastModel.for_inference(model)
model.eval()

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model           = model,
    tokenizer       = tokenizer.tokenizer,
    feature_extractor = tokenizer.feature_extractor,
    processor       = tokenizer,
    torch_dtype     = torch.bfloat16,
    return_language = True,
)

# Test with the first 3 samples from the test split
raw_test_audio = load_dataset(
    CFG["dataset_name"] if not CFG["dataset_path"] else CFG["dataset_path"],
    split="test",
).cast_column("audio", Audio(sampling_rate=16000))

print("\n── Inference samples ──────────────────────────────────────")
for i in range(min(3, len(raw_test_audio))):
    sample    = raw_test_audio[i]
    audio_arr = sample["audio"]["array"]
    reference = sample["text"]
    result    = whisper_pipe(audio_arr)
    predicted = result["text"]

    print(f"\n[{i+1}]  lang_mix : {sample['lang_mix']}")
    print(f"     reference: {reference}")
    print(f"     predicted: {predicted}")
