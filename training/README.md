# Training

All runs fine-tune `openai/whisper-large-v3-turbo` with LoRA-style adapters on the dataset `afyfbadreddine77/darija-asr-augmented` (private, see the main README for its structure). WER is the selection metric everywhere.

| File | Role | Where it ran |
|---|---|---|
| [`darija_finetune.ipynb`](darija_finetune.ipynb) | **Final run.** Produced the [published model](https://huggingface.co/afyfbadreddine77/whisper-large-v3-turbo-darija). | Single large GPU (RTX PRO 6000, ~95 GB) |
| [`experiments/modal_hyperparam_search.py`](experiments/modal_hyperparam_search.py) | Hyperparameter search with Optuna. | Modal, A100-80GB |
| [`experiments/finetune_darija_whisper.py`](experiments/finetune_darija_whisper.py) | First full-run script: a fresh LoRA adapter trained from the base model. | Lightning AI Studio, L4 (24 GB) |
| [`experiments/modal_finetune.py`](experiments/modal_finetune.py) | The "continue the community adapter" strategy, packaged as a Modal job. | Modal, L4 (24 GB) |

## Final run: `darija_finetune.ipynb`

| Cell | What it does |
|---|---|
| 1 | Installs pinned versions of Unsloth, Transformers 4.56.2, TRL, datasets 3.6.0, jiwer. |
| 2 | Config: Hugging Face token, dataset, base model, starting adapter, output repo, language token `<\|ar\|>`. |
| 3 | Loads Whisper large-v3-turbo with Unsloth's `FastModel`, then loads the existing Darija adapter (`anaszil/whisper-large-v3-turbo-darija`, r=16) **as trainable**, so training continues it instead of starting a new one. |
| 4 | Resamples audio to 16 kHz and precomputes log-Mel features and label token IDs for all splits. |
| 5 | Seq2seq padding collator (labels padded with `-100`) and a WER metric. |
| 6 | Trains for 8 epochs: effective batch 32, LR 1e-4 with cosine decay and 200 warmup steps, bf16, evaluation every 500 steps **with generation** so the logged WER is real decoding WER; keeps the best checkpoint by WER. |
| 7 | Pushes the adapter and processor to the Hugging Face Hub. |
| 8 | Evaluates on the test split: overall WER plus WER for each `lang_mix` category. |

Result: 6.46% validation WER (best checkpoint) and 6.84% test WER.

## Hyperparameter search: `experiments/modal_hyperparam_search.py`

- **Search space:** AdaLoRA `init_r` ∈ {8, 12, 16, 24}, `target_r` ∈ {4, 6, 8, 12} (always below `init_r`), `lora_alpha`, learning rate 1e-5 to 8e-5 (log scale) and dropout 0 to 0.1.
- **Per trial:** 600 training steps on a ~4-hour subset sampled by language mix (55% Darija, 20% Darija+French, 15% Darija+English, 10% all three), scored by validation WER every 200 steps. Default 10 trials.
- **Infrastructure:** Modal with an A100-80GB, and a persistent Volume caching models and datasets between trials.

## Earlier full-run scripts

- `experiments/finetune_darija_whisper.py`: trains a **new** LoRA adapter (r=32, alpha=64) from the base model on a 24 GB L4, with the same per-language test evaluation. Written cell-by-cell for a Lightning AI notebook.
- `experiments/modal_finetune.py`: the continued-training strategy as a Modal app. Preprocessing runs once with `dataset.map()` into an Arrow cache on a Modal Volume, which avoids running out of memory on smaller machines.

The final notebook kept the continued-training strategy and moved to a larger GPU to train on the full dataset for 8 epochs.
