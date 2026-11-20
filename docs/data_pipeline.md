# Data pipeline

How the training data for the Darija Whisper model is generated, checked and cleaned. All code is in [`data_studio/`](../data_studio).

```text
 real Darija datasets ─┐
 (Hugging Face Hub)    │
                       ├─► merge ─► clean (5 phases) ─► resplit by hours ─► push to Hub ─► training
 LLM text ─► TTS ──────┘
 (synthetic code-switch)
```

## 1. Synthetic code-switched speech

Code: [`data_studio/synthetic_cs_dataset/`](../data_studio/synthetic_cs_dataset), config: [`configs/generation.yaml`](../data_studio/synthetic_cs_dataset/configs/generation.yaml).

### 1a. Transcripts: `scripts/generate_texts.py`

- **Model:** an LLM behind an OpenAI-compatible API (default `google/gemini-3.1-flash-lite-preview`, temperature 0.9), called with several batches in parallel and resumable output.
- **Prompt:** [`prompts/text_generation_prompt.txt`](../data_studio/synthetic_cs_dataset/prompts/text_generation_prompt.txt). Darija in Arabic script, French/English in Latin script, no Arabizing of loanwords, no MSA-heavy phrasing or other dialects, placeholders (`<NAME>`, `<PHONE>`, `<ORDER_ID>`) instead of personal data.
- **Targets (default config):** language mix 56% Darija+French, 28% Darija+English, 16% all three; 12 domains (phone support, appointment booking, delivery tracking, bank/insurance, university admin, healthcare, job interviews, technical support, team calls, e-commerce returns, travel/hotel/taxi, casual calls); 3–25 words per utterance.
- **Automatic quality check on every generated line.** A line is rejected if it is too short or too long, has no Arabic-script Darija, is MSA-heavy, contains repeated phrases or private/numeric data, or doesn't match the requested mix (for example, a "Darija+French" line with no French, or a "pure Darija" line containing Latin script). Near-duplicate texts are removed.

### 1b. Audio: `scripts/generate_audio.py`

- **TTS:** [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (`k2-fsa/OmniVoice`, fp16 on GPU) in voice-cloning mode, using 10–30 reference speakers (each a short WAV plus its transcript).
- **Post-processing:** convert to mono, resample to 24 kHz, trim silence (35 dB), peak-limit and normalize loudness to −20 dBFS.
- **Rejection:** clips that are silent, clipped, shorter than 1 s or longer than 25 s.
- **Splits:** 90% train / 5% validation / 5% test, with synthetic test rows flagged so a real-speech test set can replace them later.
- **Optional phone-call augmentation** ([`augment_audio.py`](../data_studio/synthetic_cs_dataset/scripts/augment_audio.py)): band-pass filtering, compression, room reverb, background noise at a set SNR and µ-law degradation.

### 1c. Validation and packaging

- [`validate_dataset.py`](../data_studio/synthetic_cs_dataset/scripts/validate_dataset.py): flags rows with missing or unreadable audio, empty transcripts, no Arabic script, a declared code-switch with no Latin words, or too many copies of the same text. Writes `bad_samples.csv` and a 300-row review sample for listening checks.
- [`merge_batches.py`](../data_studio/synthetic_cs_dataset/scripts/merge_batches.py): merges several generation batches into one folder.
- [`make_hf_dataset.py`](../data_studio/synthetic_cs_dataset/scripts/make_hf_dataset.py): builds a Hugging Face `DatasetDict`.
- [`run_smoke_test.py`](../data_studio/synthetic_cs_dataset/scripts/run_smoke_test.py): runs the whole path on one sample before launching a long job.

## 2. Dataset Studio (web app)

Start it with `python3 viewer_server.py` inside `data_studio/`. It provides four workflows, all running as background jobs with progress bars:

| Workflow | Purpose |
|---|---|
| **Datasets Viewer** | Browse any `save_to_disk` dataset: filters (e.g. by `lang_mix`), distributions, total hours, inline audio playback, export. |
| **Dataset Hub** | Import datasets from the Hugging Face Hub, merge several local datasets, **resplit by hours** (train/test percentages computed from the duration column), and push the result back to the Hub. |
| **Code-Switch Generator** | Runs the synthetic pipeline above from the browser: a 1-sample smoke test or a full run. |
| **Datasets Cleaner** | The five cleaning phases below. Results can be saved as a new copy or written over the original. |

## 3. Cleaning phases

Each phase writes a JSON report listing every removed or changed row and the reason.

### Phase 1: Bad sample removal

Each clip is transcribed with Whisper (any Hugging Face Whisper model or LoRA adapter can be plugged in), then compared to its label:

- **Character error rate** above a threshold means the audio and the transcript probably don't match, so the row is dropped.
- **Speaking rate** (characters per second) outside a plausible range catches truncated audio or wrong labels.

### Phase 2: Duration filtering

Keeps clips between 1 and 30 seconds (and always drops anything under 0.7 s). It uses the dataset's duration column when there is one, and measures the audio otherwise.

### Phase 3: Augmentation

Adds augmented copies of **real** recordings until a target amount of extra audio is reached (for example +50% on a 40-hour selection gives about +20 hours). The TTS source is excluded by default, so synthetic speech isn't multiplied further. Each copy gets **one** effect, sampled by coverage:

| Effect | Default share of copies | Details |
|---|---:|---|
| Telephone line | 30% | narrow-band phone filter + light noise (18–28 dB SNR) |
| Speed perturbation | 25% | 0.9× or 1.1× |
| Background noise | 25% | synthetic office, call-center, café, street or home profiles, mixed at a random SNR |
| Room reverb | 15% | |
| Gain | 15% | −6 to +6 dB |

No pitch shifting and no stacked effects, so the audio stays natural. Augmented rows are marked with `is_augmented = true`.

### Phase 4: Transcript normalization

[`tools/best_asr_text_normalizer.py`](../data_studio/tools/best_asr_text_normalizer.py) writes a `normalized_text` column:

- Darija stays in Arabic script; clear French/English words stay in Latin script.
- French accents are kept (*réunion*, *problème*), as are internal apostrophes and hyphens (*l'application*, *rendez-vous*).
- It removes Arabic diacritics, tatweel, zero-width characters, emojis, noise tags such as `(music)` or `[laughter]`, laugh tokens, punctuation runs and stretched letters.
- It never translates Darija into Modern Standard Arabic.

### Phase 5: Duplicate removal

Removes rows whose transcript repeats an earlier one (after normalizing spaces, case and simple punctuation) and keeps the first occurrence.

## 4. Final dataset

After merging, cleaning and resplitting, the dataset was pushed to the Hub as `afyfbadreddine77/darija-asr-augmented` (private): 32,028 / 485 / 484 utterances, about 50 hours including augmentation, about 65% pure Darija and 35% code-switched.

## 5. Next: real recordings

The [`data_collection/prompts`](../data_collection/prompts) folder holds the prompts used to generate thousands of everyday sentences (pure Darija and Darija mixed with French or English, with realistic fake phone numbers and ID formats). Volunteers read these aloud on [Smart Darija](https://smart-darija.vercel.app), a crowdsourcing site ([code](https://github.com/badrafyf77/smart-darija)) that records 16 kHz audio in the browser. These real recordings are meant for the next model version, and above all for a real-speech test set.
