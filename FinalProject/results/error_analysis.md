# Error analysis (zero-shot, test split)

Test cases in this run: **14**. Misclassified (zero-shot exact label match): **7**.

Numbers come from `results/prompt_comparison.csv` and confusion matrices under `figures/`. Gold labels are heuristic course labels—not clinically adjudicated.

## What the mistakes look like (qualitative)

Most errors are **urgent vs emergency** swaps when the user describes severe symptoms. The model tends to be conservative— when retrieval surfaces stroke/bleeding/severe infection passages, it may push the informational tag upward compared to the gold label line in `eval_cases.jsonl`. A smaller pattern is **head injury** wording where the gold expects emergency but the model lands on urgent when vomiting is emphasized without explicit neurological deficits in the structured output.

## Themes
- **Severity boundary (urgent vs emergency)**: competing serious cues (stroke vs migraine, GI bleed vs nausea) drive label drift.
- **Vague inputs**: low-detail prompts sometimes collapse to **monitor** because there are few retrieval anchors.
- **Retrieval limits**: top-k passages may omit the single gold article ID even when the answer is reasonable.
- **OOD / fiction**: nonsense prompts still retrieve real articles; labels treat **routine** as the safe bucket.

## Urgent vs emergency boundary (mislabels only)
Count among errors where gold/pred are only urgent/emergency (subset): **6**.

- gold **urgent** → pred **emergency** — `Worst headache of my life that started suddenly while lifting weights.` — retrieved: `['headache_001', 'stroke_001', 'head_injury_001', 'dizziness_001']`
- gold **urgent** → pred **emergency** — `Throwing up blood and severe stomach pain.` — retrieved: `['nausea_vomiting_001', 'abdominal_pain_001', 'allergic_reaction_001', 'head_injury_001']`
- gold **urgent** → pred **emergency** — `Coughing up streaks of blood.` — retrieved: `['cough_001', 'allergic_reaction_001', 'leg_swelling_dvt_001', 'shortness_breath_001']`
- gold **urgent** → pred **emergency** — `Back pain with numbness in both legs and trouble urinating.` — retrieved: `['back_pain_001', 'leg_pain_001', 'urinary_001', 'leg_swelling_dvt_001']`
- gold **urgent** → pred **emergency** — `Flank pain and fever with UTI symptoms.` — retrieved: `['urinary_001', 'abdominal_pain_001', 'fever_001', 'leg_pain_001']`
- gold **emergency** → pred **urgent** — `Repeated vomiting after falling off a bike and hitting my head.` — retrieved: `['head_injury_001', 'nausea_vomiting_001', 'allergic_reaction_001', 'stroke_001']`

## Edge / out-of-distribution prompts (categories)

These rows intentionally exercise vague or fictional language; see `data/eval_cases.jsonl`.

- `edge_fiction`: gold **routine**, zero-shot pred **routine** — `I have dragon pox from Hogwarts.`
- `vague`: gold **monitor**, zero-shot pred **monitor** — `I feel a little off but cannot describe any symptom clearly.`

## Example failures (all mislabels)

- Category `headache`: gold **urgent**, pred **emergency** — `Worst headache of my life that started suddenly while lifting weights.` — retrieved: `['headache_001', 'stroke_001', 'head_injury_001', 'dizziness_001']`
- Category `abdomen`: gold **urgent**, pred **routine** — `Sharp pain in the right lower belly for 6 hours with fever.` — retrieved: `['abdominal_pain_001', 'fever_001', 'leg_pain_001', 'nausea_vomiting_001']`
- Category `gi`: gold **urgent**, pred **emergency** — `Throwing up blood and severe stomach pain.` — retrieved: `['nausea_vomiting_001', 'abdominal_pain_001', 'allergic_reaction_001', 'head_injury_001']`
- Category `resp`: gold **urgent**, pred **emergency** — `Coughing up streaks of blood.` — retrieved: `['cough_001', 'allergic_reaction_001', 'leg_swelling_dvt_001', 'shortness_breath_001']`
- Category `msk`: gold **urgent**, pred **emergency** — `Back pain with numbness in both legs and trouble urinating.` — retrieved: `['back_pain_001', 'leg_pain_001', 'urinary_001', 'leg_swelling_dvt_001']`
- Category `gu`: gold **urgent**, pred **emergency** — `Flank pain and fever with UTI symptoms.` — retrieved: `['urinary_001', 'abdominal_pain_001', 'fever_001', 'leg_pain_001']`
- Category `head_injury`: gold **emergency**, pred **urgent** — `Repeated vomiting after falling off a bike and hitting my head.` — retrieved: `['head_injury_001', 'nausea_vomiting_001', 'allergic_reaction_001', 'stroke_001']`