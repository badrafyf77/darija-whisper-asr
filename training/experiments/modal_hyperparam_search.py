import math
import os
from typing import Any, Dict, List, Tuple

import modal

# -----------------------------
# User configuration
# -----------------------------

DATASET_REPO = "afyfbadreddine77/darija-asr-augmented"
MODEL_ID = "openai/whisper-large-v3-turbo"

LANG_MIX_WEIGHTS: Dict[str, float] = {
    "darija": 0.55,
    "darija_french": 0.20,
    "darija_english": 0.15,
    "darija_french_english": 0.10,
}

TOTAL_TRAIN_HOURS = 4.0
TOTAL_TRAIN_SECONDS = int(TOTAL_TRAIN_HOURS * 3600)

TARGET_MODULES: List[str] = [
    "q_proj",
    "v_proj",
    "k_proj",
    "out_proj",
    "fc1",
    "fc2",
]

# Training config (per trial)
MAX_STEPS = 600
EVAL_STEPS = 200
BASE_SEED = 42

# Cache layout (Modal Volume mounted at /cache)
CACHE_DIR = "/cache"
HF_HOME = f"{CACHE_DIR}/hf"
HF_DATASETS_CACHE = f"{HF_HOME}/datasets"
TRANSFORMERS_CACHE = f"{HF_HOME}/transformers"
TORCH_HOME = f"{CACHE_DIR}/torch"
XDG_CACHE_HOME = f"{CACHE_DIR}/xdg"

# -----------------------------
# Modal setup
# -----------------------------

cache_volume = modal.Volume.from_name("darija-cache")
hf_secret = modal.Secret.from_name("huggingface-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .run_commands(
        "python -m pip install --upgrade pip",
        # GPU-enabled PyTorch wheels
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0 torchaudio==2.3.0",
        # Training / evaluation stack
        "python -m pip install transformers==4.44.0 peft==0.17.0 datasets==2.20.0 accelerate==0.33.0 evaluate==0.4.2 jiwer==3.0.4 huggingface_hub==0.25.0 soundfile librosa tensorboard optuna pandas",
        # Modal Volume mounts must target an empty path in the final image.
        # Ensure /cache does not exist (or is empty) in the image snapshot.
        "rm -rf /cache",
    )
)

app = modal.App("whisper-adalora-hparam-search")


def _validate_lang_mix_weights(weights: Dict[str, float]) -> None:
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"LANG_MIX_WEIGHTS must sum to 1.0, got {total}")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 8,
    volumes={"/cache": cache_volume},
    secrets=[hf_secret],
)
def run_trial(
    trial_number: int,
    init_r: int,
    target_r: int,
    lora_alpha: int,
    lr: float,
    lora_dropout: float,
    seed: int = BASE_SEED,
) -> float:
    """Runs a single training + eval trial and returns WER% on validation."""

    # IMPORTANT: set cache env vars at runtime (after the Volume is mounted),
    # before importing Hugging Face libraries.
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
    os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
    os.environ["TORCH_HOME"] = TORCH_HOME
    os.environ["XDG_CACHE_HOME"] = XDG_CACHE_HOME
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Ensure cache directories exist on the Volume.
    os.makedirs(HF_HOME, exist_ok=True)
    os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
    os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
    os.makedirs(TORCH_HOME, exist_ok=True)
    os.makedirs(XDG_CACHE_HOME, exist_ok=True)

    # Heavy imports live inside the Modal container (keeps local entrypoint light).
    import numpy as np
    import torch
    from datasets import Audio, concatenate_datasets, load_dataset
    from peft import AdaLoraConfig, TaskType, get_peft_model
    from peft.peft_model import PeftModelForSeq2SeqLM
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    import evaluate

    _validate_lang_mix_weights(LANG_MIX_WEIGHTS)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var is missing. Make sure the Modal secret "
            "'huggingface-token' exists and exposes HF_TOKEN."
        )

    # Make trial runs deterministic (and comparable across hyperparams).
    set_seed(seed)

    # -----------------------------
    # Data loading + 4h weighted subset
    # -----------------------------

    ds = load_dataset(DATASET_REPO, token=hf_token)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    def _subset_by_duration(dset, target_seconds: float, shuffle_seed: int):
        dset = dset.shuffle(seed=shuffle_seed)
        durations = np.asarray(dset["duration"], dtype=np.float64)
        if durations.size == 0:
            return dset
        cum = np.cumsum(durations)
        if float(cum[-1]) <= float(target_seconds):
            return dset
        n = int(np.searchsorted(cum, target_seconds, side="left") + 1)
        return dset.select(list(range(n)))

    mix_subsets = []
    for i, (mix, weight) in enumerate(LANG_MIX_WEIGHTS.items()):
        target_seconds = TOTAL_TRAIN_SECONDS * weight
        mix_ds = train_ds.filter(
            lambda lang: lang == mix,
            input_columns=["lang_mix"],
        )
        mix_subsets.append(_subset_by_duration(mix_ds, target_seconds, shuffle_seed=seed + i))

    train_subset = concatenate_datasets(mix_subsets).shuffle(seed=seed)

    # Cast Audio to 16kHz and decode
    train_subset = train_subset.cast_column("audio", Audio(sampling_rate=16000))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=16000))

    # -----------------------------
    # Preprocessing
    # -----------------------------

    processor = WhisperProcessor.from_pretrained(
        MODEL_ID,
        language="ar",
        task="transcribe",
        token=hf_token,
        cache_dir=TRANSFORMERS_CACHE,
    )

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    tokenizer.set_prefix_tokens(language="ar", task="transcribe")

    def prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_arrays = [a["array"] for a in batch["audio"]]
        inputs = feature_extractor(audio_arrays, sampling_rate=16000)
        batch["input_features"] = inputs["input_features"]  # numpy arrays; Arrow-cache safe
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    train_processed = train_subset.map(
        prepare_batch,
        batched=True,
        batch_size=16,
        remove_columns=train_subset.column_names,
        load_from_cache_file=False,
    )
    val_processed = val_ds.map(
        prepare_batch,
        batched=True,
        batch_size=16,
        remove_columns=val_ds.column_names,
        load_from_cache_file=False,
    )

    # -----------------------------
    # Data collator
    # -----------------------------

    class DataCollatorSpeechSeq2SeqWithPadding:
        def __init__(self, processor: WhisperProcessor):
            self.processor = processor

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_features = torch.tensor(
                [f["input_features"] for f in features],
                dtype=torch.float32,
            )

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

            # Whisper prep: remove BOS token if present in all sequences
            bos = self.processor.tokenizer.bos_token_id
            if bos is not None and labels.size(1) > 0:
                if torch.all(labels[:, 0] == bos).item():
                    labels = labels[:, 1:]

            return {"input_features": input_features, "labels": labels}

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # -----------------------------
    # Metrics
    # -----------------------------

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [s.strip() for s in pred_str]
        label_str = [s.strip() for s in label_str]

        wer = 100.0 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # -----------------------------
    # Model + AdaLoRA
    # -----------------------------

    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        cache_dir=TRANSFORMERS_CACHE,
    )

    base_model.config.use_cache = False

    # Force language/task during generation
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    base_model.generation_config.forced_decoder_ids = forced_decoder_ids
    base_model.generation_config.language = "ar"
    base_model.generation_config.task = "transcribe"

    adalora_config = AdaLoraConfig(
        init_r=init_r,
        target_r=target_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=TARGET_MODULES,
        task_type=TaskType.SEQ_2_SEQ_LM,
        total_step=MAX_STEPS,
    )

    model = get_peft_model(base_model, adalora_config)

    # GenerationMixin relies on `main_input_name` to find the encoder inputs.
    # Whisper uses `input_features` (mel spectrograms), not `input_ids`.
    model.main_input_name = "input_features"

    # PEFT + Whisper patch: PeftModelForSeq2SeqLM forwards `input_ids` to base model,
    # but Whisper expects `input_features`.
    def _patched_forward(
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if input_features is None and input_ids is not None:
            input_features = input_ids

        # Never forward `input_ids` into Whisper.
        kwargs.pop("input_ids", None)

        return self.base_model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs,
        )

    PeftModelForSeq2SeqLM.forward = _patched_forward

    # Also patch generate() so Seq2SeqTrainer can call `model.generate(input_features=...)`.
    def _patched_generate(self, *args, **kwargs):
        if "input_features" in kwargs and "inputs" not in kwargs:
            kwargs["inputs"] = kwargs.pop("input_features")
        return self.base_model.generate(*args, **kwargs)

    PeftModelForSeq2SeqLM.generate = _patched_generate

    # -----------------------------
    # Trainer
    # -----------------------------

    output_dir = f"{CACHE_DIR}/outputs/trial_{trial_number}"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=MAX_STEPS,
        save_total_limit=1,
        logging_steps=20,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=seed,
        data_seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()
    metrics = trainer.evaluate()

    wer = float(metrics["eval_wer"])

    # Persist caches/checkpoints to the Volume for subsequent trials.
    cache_volume.commit()

    print(
        "\n".join(
            [
                "TRIAL COMPLETE",
                f"trial_number={trial_number}",
                f"init_r={init_r} target_r={target_r} alpha={lora_alpha} lr={lr:.2e} dropout={lora_dropout:.4f}",
                f"WER%={wer:.2f}",
            ]
        )
    )

    return wer


@app.function(
    image=image,
    timeout=60 * 5,
    volumes={"/cache": cache_volume},
)
def smoke_cache_mount() -> Dict[str, Any]:
    """Fast check that the Volume mounts cleanly at /cache."""

    import pathlib

    p = pathlib.Path("/cache")
    entries = []
    if p.exists() and p.is_dir():
        entries = sorted([child.name for child in p.iterdir()])

    return {
        "cache_exists": p.exists(),
        "cache_is_dir": p.is_dir(),
        "num_entries": len(entries),
        "entries_head": entries[:50],
    }


@app.local_entrypoint()
def main(n_trials: int = 10, smoke: bool = False) -> None:
    """Runs Optuna locally, calling Modal for each trial sequentially."""
    if smoke:
        print("=== Smoke test: mounting Volume at /cache ===")
        print(smoke_cache_mount.remote())
        return


    try:
        import optuna
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "Missing local dependency. Install with: pip install optuna pandas\n"
            "(Optuna runs in @app.local_entrypoint; training runs on Modal.)"
        ) from e

    sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    init_r_choices = [8, 12, 16, 24]
    target_r_choices = [4, 6, 8, 12]
    alpha_choices = [16, 32, 64]

    def objective(trial: "optuna.Trial") -> float:
        init_r = trial.suggest_categorical("init_r", init_r_choices)
        valid_target_rs = [r for r in target_r_choices if r < init_r]
        target_r = trial.suggest_categorical("target_r", valid_target_rs)

        lora_alpha = trial.suggest_categorical("lora_alpha", alpha_choices)
        lr = trial.suggest_float("lr", 1e-5, 8e-5, log=True)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1)

        wer = run_trial.remote(
            trial_number=trial.number,
            init_r=init_r,
            target_r=target_r,
            lora_alpha=lora_alpha,
            lr=lr,
            lora_dropout=lora_dropout,
            seed=BASE_SEED,
        )

        print(
            f"Trial {trial.number}: init_r={init_r} target_r={target_r} "
            f"alpha={lora_alpha} lr={lr:.2e} dropout={lora_dropout:.4f} -> WER%={wer:.2f}"
        )

        return float(wer)

    study.optimize(objective, n_trials=int(n_trials))

    # Required by spec: use trials_dataframe()
    df = study.trials_dataframe()

    # Build a compact ranked table
    complete_df = df[df["state"] == "COMPLETE"].copy()
    complete_df = complete_df.sort_values("value", ascending=True)

    # Rename columns for readability
    col_map = {
        "number": "trial",
        "value": "wer_percent",
        "params_init_r": "init_r",
        "params_target_r": "target_r",
        "params_lora_alpha": "lora_alpha",
        "params_lr": "lr",
        "params_lora_dropout": "lora_dropout",
    }

    display_cols = [
        "number",
        "params_init_r",
        "params_target_r",
        "params_lora_alpha",
        "params_lr",
        "params_lora_dropout",
        "value",
    ]

    # Some optuna versions omit columns when params are missing;
    # keep only what exists.
    display_cols = [c for c in display_cols if c in complete_df.columns]

    table = complete_df[display_cols].rename(columns=col_map)

    best_trial_number = int(study.best_trial.number)
    table.insert(0, "best", table["trial"].apply(lambda n: "<-- BEST" if int(n) == best_trial_number else ""))

    # Format
    if "wer_percent" in table.columns:
        table["wer_percent"] = table["wer_percent"].map(lambda x: f"{float(x):.2f}")
    if "lr" in table.columns:
        table["lr"] = table["lr"].map(lambda x: f"{float(x):.2e}")
    if "lora_dropout" in table.columns:
        table["lora_dropout"] = table["lora_dropout"].map(lambda x: f"{float(x):.4f}")

    print("\n=== Ranked trials (by lowest WER%) ===")
    print(table.to_string(index=False))

    best = study.best_trial
    best_params = best.params

    print("\n=== Best config ===")
    print(f"Trial {best.number} -> WER%={best.value:.2f}")
    print(best_params)

    print("\n=== Copy/paste into training script ===")
    # This is intentionally formatted to drop into a full training script.
    print(
        "\n".join(
            [
                "from peft import AdaLoraConfig, TaskType",
                "",
                f"learning_rate = {best_params['lr']:.8f}",
                "",
                "adalora_config = AdaLoraConfig(",
                f"    init_r={int(best_params['init_r'])},",
                f"    target_r={int(best_params['target_r'])},",
                f"    lora_alpha={int(best_params['lora_alpha'])},",
                f"    lora_dropout={float(best_params['lora_dropout']):.4f},",
                f"    total_step={MAX_STEPS},",
                f"    target_modules={TARGET_MODULES!r},",
                "    task_type=TaskType.SEQ_2_SEQ_LM,",
                ")",
            ]
        )
    )
