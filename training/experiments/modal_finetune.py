"""
Fine-tune Whisper large-v3-turbo on Darija (code-switched) using Modal.

Strategy : continue training anaszil/whisper-large-v3-turbo-darija (r=16)
           on afyfbadreddine77/darija-asr-augmented (32k samples, 50h)
GPU      : L4 (24 GB)
Dataset  : preprocessed with dataset.map() → Arrow cache on Modal Volume
           (official HF approach — no OOM, no custom memmap)

Prerequisites:
  modal secret create huggingface HF_TOKEN=hf_xxx
  modal run modal_finetune.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "unsloth[huggingface]",
        "sentencepiece",
        "protobuf",
        "librosa",
        "soundfile",
        "evaluate",
        "jiwer",
        "peft",
        "datasets==2.21.0",  # last version before torchcodec requirement
    )
)

app    = modal.App("whisper-darija-finetune", image=image)
volume = modal.Volume.from_name("darija-cache", create_if_missing=True)

CACHE_DIR = "/cache"

CFG = dict(
    base_model    = "openai/whisper-large-v3-turbo",
    adapter_model = "anaszil/whisper-large-v3-turbo-darija",
    language      = "arabic",
    lang_token    = "<|ar|>",

    dataset_name  = "afyfbadreddine77/darija-asr-augmented",

    output_dir        = "/tmp/checkpoints",
    num_train_epochs  = 8,
    batch_size        = 16,     # L4 can handle 16 with precomputed features
    grad_accum        = 2,      # effective batch = 32
    learning_rate     = 1e-4,
    warmup_steps      = 200,
    eval_steps        = 500,
    save_steps        = 500,
    logging_steps     = 25,
    weight_decay      = 0.01,
    seed              = 3407,

    hf_repo       = "afyfbadreddine77/whisper-large-v3-turbo-darija",
)


# ── Step 1: Preprocess with dataset.map() → Arrow cache on Volume ─────────────
@app.function(
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 3,
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    image=image,
)
def preprocess_to_disk():
    import os
    from datasets import load_dataset, Audio
    from transformers import WhisperProcessor

    hf_token   = os.environ["HF_TOKEN"]
    done_flag  = f"{CACHE_DIR}/preprocessed_done"

    if os.path.exists(done_flag):
        print("Cache exists, skipping.")
        return

    processor = WhisperProcessor.from_pretrained(
        CFG["base_model"], language=CFG["language"], task="transcribe", token=hf_token
    )

    raw = load_dataset(CFG["dataset_name"], token=hf_token)
    raw = raw.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(example):
        audio = example["audio"]
        result = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["text"],
        )
        result["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return result

    processed = raw.map(
        prepare,
        remove_columns=raw["train"].column_names,
        num_proc=4,
    )
    # Filter samples > 30s
    processed = processed.filter(lambda x: x["input_length"] < 30.0, input_columns=["input_length"])

    processed.save_to_disk(f"{CACHE_DIR}/dataset")
    volume.commit()

    open(done_flag, "w").close()
    volume.commit()
    print(f"Saved. train={len(processed['train'])} val={len(processed['validation'])}")


# ── Step 2: Train from Arrow cache ────────────────────────────────────────────
@app.function(
    gpu="L4",
    timeout=60 * 60 * 10,
    volumes={CACHE_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32768,
    image=image,
)
def train():
    import os, torch, numpy as np
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from functools import partial

    from peft import PeftModel
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from datasets import load_from_disk
    import evaluate
    from jiwer import transforms as tr

    hf_token = os.environ["HF_TOKEN"]

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading base model + Darija adapter…")
    base = WhisperForConditionalGeneration.from_pretrained(
        CFG["base_model"], torch_dtype=torch.float16, token=hf_token,
    )
    model = PeftModel.from_pretrained(base, CFG["adapter_model"], is_trainable=True)
    processor = WhisperProcessor.from_pretrained(
        CFG["base_model"], language=CFG["language"], task="transcribe", token=hf_token,
    )
    model.config.use_cache = False
    model.generation_config.language           = CFG["lang_token"]
    model.generation_config.task               = "transcribe"
    model.config.suppress_tokens               = []
    model.generation_config.forced_decoder_ids = None
    # Re-enable cache for generation only
    model.generate = partial(model.generate, language=CFG["lang_token"], task="transcribe", use_cache=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Load preprocessed dataset from Volume ─────────────────────────────────
    dataset = load_from_disk(f"{CACHE_DIR}/dataset")
    print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}")

    # ── Collator (standard HF Whisper pattern) ────────────────────────────────
    @dataclass
    class DataCollator:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"][0]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollator(processor=processor)

    # ── Metrics ───────────────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")
    transform  = tr.Compose([tr.ToLowerCase(), tr.RemovePunctuation(), tr.RemoveMultipleSpaces(), tr.Strip()])

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.tokenizer.batch_decode(np.argmax(pred.predictions[0], axis=-1), skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(
            predictions=[transform(s) for s in pred_str],
            references=[transform(s) for s in label_str],
        )}

    # ── Training args ─────────────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir                  = CFG["output_dir"],
        per_device_train_batch_size = CFG["batch_size"],
        gradient_accumulation_steps = CFG["grad_accum"],
        num_train_epochs            = CFG["num_train_epochs"],
        warmup_steps                = CFG["warmup_steps"],
        learning_rate               = CFG["learning_rate"],
        lr_scheduler_type           = "cosine",
        fp16                        = True,
        fp16_full_eval              = True,
        optim                       = "adamw_8bit",
        weight_decay                = CFG["weight_decay"],
        predict_with_generate       = True,
        generation_max_length       = 225,
        logging_steps               = CFG["logging_steps"],
        eval_strategy               = "steps",
        eval_steps                  = CFG["eval_steps"],
        save_strategy               = "steps",
        save_steps                  = CFG["save_steps"],
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        remove_unused_columns       = False,
        label_names                 = ["labels"],
        seed                        = CFG["seed"],
        report_to                   = "none",
        dataloader_num_workers      = 4,
        gradient_checkpointing      = True,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["validation"],
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        tokenizer       = processor,
    )

    print("Starting training…")
    stats = trainer.train()
    print(f"Done. Runtime: {stats.metrics['train_runtime']/60:.1f} min")
    print(f"Peak VRAM: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")

    model.push_to_hub(CFG["hf_repo"], token=hf_token, private=True)
    processor.push_to_hub(CFG["hf_repo"], token=hf_token, private=True)
    print(f"Pushed to: https://huggingface.co/{CFG['hf_repo']}")


@app.local_entrypoint()
def main():
    preprocess_to_disk.remote()
    train.remote()
