**Task: Generate 5,000 Moroccan Darija sentence prompts for a speech data collection platform.**

## Context

These sentences will be displayed to Moroccan users who will record themselves reading them aloud. The recordings will be used to train a speech recognition model for Moroccan Darija. The model's current weaknesses are: French/English loanwords not being written in Latin script, phone numbers, and Moroccan ID card numbers.

## Language rules — follow strictly

- **Darija words** → Arabic script (e.g., غادي، كنبغي، واش، مزيان)
- **French/English loanwords** → Latin script, mid-sentence (e.g., commande, rendez-vous, meeting, problème, modèle)
- **Numbers** → digits (e.g., 06, 212, 3)
- **Phone numbers** → Moroccan format: 06XXXXXXXX or +212 6XXXXXXXX (use realistic fake numbers)
- **National ID numbers** → Moroccan CIN format: one or two uppercase letters followed by 5–6 digits. Valid prefixes: WA, BE, BK, BH, D, DO, F, G, GK, HA, HH, I, J, JE, JY, K, KB, L, M, PA, Q, R, RB, S, SH, SJ, T, U, V, W, X, Y, Z (e.g., WA310325, RB21321, BK45678, F123456). Use realistic fake numbers only.
- No diacritics (tashkeel) on Arabic text
- No punctuation except periods at end of sentence
- Natural spoken Darija, not Modern Standard Arabic

## Distribution — exactly 5,000 sentences across these categories

| Category | Count | Example |
|---|---|---|
| Pure Darija | 1000 | واش مزيان معاك اليوم |
| Darija + French | 1,400 | غادي نمشي نشري commande من المحل |
| Darija + English | 900 | عندي meeting مع le boss غدا |
| Darija + French + English | 500 | كنخدم على un projet urgent deadline ديالو غدا |
| Phone numbers | 400 | عطيني numéro ديالك واتساب هو 0661234567 |
| National ID numbers | 400 | رقم بطاقتي هو WA310325 |
| Addresses / places | 200 | كنسكن فـ Casablanca قريب من le centre ville |
| Shopping / e-commerce | 200 | واش وصلات la livraison ديال la commande ديالي |

## Topics to cover (spread across categories)

Health, shopping, work/office, family, transport, food, technology, social media, education, weather, money/banking, phone calls, directions, administrative procedures (CIN, passport, documents).

## Quality rules

- Each sentence must be unique
- Length: 5–20 words
- Must sound like something a real Moroccan would say in a casual conversation
- Avoid formal/MSA constructions like "إن" or "حيث"
- Avoid sensitive topics (politics, religion)
- French/English words must be common loanwords actually used in Moroccan speech — not random English
- Phone numbers and ID numbers must be realistic-looking but fake (do not use real people's data)
- ID number sentences should reflect natural contexts: giving your CIN at a bank, hotel, pharmacy, administration, etc.

## Output format

Save as a JSON file named `sawt_prompts.json` with this exact structure:

```json
[
  {
    "id": 1,
    "text": "غادي نشري commande من Jumia",
    "lang_mix": "darija_french",
    "topic": "shopping",
    "has_phone_number": false,
    "has_id_number": false
  },
  {
    "id": 2,
    "text": "اتصل بيا على 0661234567",
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

Also save a `sawt_prompts_stats.json` with:
```json
{
  "total": 5000,
  "by_lang_mix": { "darija": 1300, ... },
  "by_topic": { "shopping": ..., ... },
  "with_phone_numbers": 500,
  "with_id_numbers": 400
}
```

Generate all 5,000 sentences. Do not truncate or summarize — output every single one.
