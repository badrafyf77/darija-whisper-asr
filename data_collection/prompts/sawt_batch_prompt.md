# Sawt — Batch Sentence Generation Prompt

Use this prompt for each batch. Replace the placeholders before sending.

---

Generate exactly **500 Moroccan Darija sentences** for a speech data collection platform called Sawt.

## Already generated (DO NOT repeat or paraphrase any of these)

[PASTE THE CONTENTS OF already_generated.txt HERE — or write "None" for the first batch]

## What these sentences are for

Real Moroccan users will read them aloud and record themselves. The recordings train a speech recognition model. Sentences must sound like real casual speech, not textbook Darija.

## Language rules — follow strictly

- **Darija words** → Arabic script (e.g., غادي، كنبغي، واش، مزيان، دابا، بغيت)
- **French/English loanwords** → Latin script mid-sentence (e.g., commande, rendez-vous, meeting, problème, livraison, deadline, update)
- **Numbers** → digits (06, 3, 212)
- **Phone numbers** → fake Moroccan format: 06XXXXXXXX or 07XXXXXXXX
- **National ID (CIN)** → fake, format: 1-2 uppercase letters + 5-6 digits, using real Moroccan prefixes: WA, BE, BK, BH, D, DO, F, G, HA, I, J, JE, K, KB, L, M, PA, Q, R, RB, S, SH, T, U, V, W, Y, Z
- No tashkeel (diacritics) on Arabic
- No punctuation except a period at the end
- No MSA — pure casual Darija as spoken in Morocco

## Distribution for this batch of 500

| Category | Count |
|---|---|
| Darija + French | 150 |
| Darija + English | 100 |
| Darija + French + English | 50 |
| Pure Darija | 90 |
| Phone numbers | 50 |
| National ID numbers (CIN) | 40 |
| Shopping / e-commerce | 20 |
| Addresses / places | 20 |

## Topics to cover

Spread across: health, shopping, work, family, transport, food, technology, social media, education, weather, banking, phone calls, directions, administrative (CIN, documents).

## Quality rules

- Every sentence must be unique and not similar to anything in the "already generated" list above
- Length: 5–20 words
- Must sound like something a real Moroccan says in a WhatsApp voice message or phone call
- French/English words must be loanwords actually used in Morocco — not random vocabulary
- Fake phone numbers and CIN numbers only — no real people's data
- Vary sentence structure — avoid starting every sentence the same way

## Output format

Return a JSON array saved to a file called `batch_N.json` (replace N with the batch number). Each item:

```json
[
  {
    "id": 1,
    "text": "غادي نشري commande من Jumia دابا",
    "lang_mix": "darija_french",
    "topic": "shopping",
    "has_phone_number": false,
    "has_id_number": false
  },
  {
    "id": 2,
    "text": "اتصل بيا على 0661234567 من فضلك",
    "lang_mix": "darija",
    "topic": "phone",
    "has_phone_number": true,
    "has_id_number": false
  },
  {
    "id": 3,
    "text": "عطيني بطاقتك رقمها WA310325",
    "lang_mix": "darija",
    "topic": "administrative",
    "has_phone_number": false,
    "has_id_number": true
  }
]
```

**`lang_mix` values:** `darija`, `darija_french`, `darija_english`, `darija_french_english`

Generate all 500. Do not truncate, summarize, or use placeholders — output every single sentence.
