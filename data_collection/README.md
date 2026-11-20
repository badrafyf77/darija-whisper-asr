# Data collection prompts

Prompts used to generate the sentences that volunteers read aloud on **[Smart Darija](https://smart-darija.vercel.app)** (working name: *Sawt*), a crowdsourcing platform for real Moroccan Darija recordings. Its code is at [badrafyf77/smart-darija](https://github.com/badrafyf77/smart-darija).

| File | Use |
|---|---|
| [`prompts/sawt_prompts_generation_prompt.md`](prompts/sawt_prompts_generation_prompt.md) | One-shot prompt for a full set of 5,000 sentences across categories. |
| [`prompts/sawt_batch_prompt.md`](prompts/sawt_batch_prompt.md) | Batch version (500 sentences per call) that lists already-generated sentences so they aren't repeated. |

Both prompts enforce the same transcript convention as the training data: Darija in Arabic script, French/English loanwords in Latin script, digits for numbers, no diacritics, and realistic **fake** Moroccan phone numbers and ID formats, so no real personal data ends up in the dataset.
