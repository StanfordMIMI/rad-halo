# Derived inputs

A few numbers in the paper come from data that cannot be shared (MIMIC-IV report text, generation
logs, annotation exports). These text-free files hold exactly what the analysis needs from them.

| File | Content | Produced by |
|---|---|---|
| `risk_grade_counts.csv` | MedVAL risk grades of all impressions generated per source until the 5/5/5/35 targets were met | `scripts/construction/impressions/count_risk_grades.py` |
| `synthetic_composition.csv` | Sample counts of the synthetic training set by error type and label | `scripts/construction/synthetic/03_composition.py` |
| `thematic_clusters.csv` | Thematic clusters of resident hallucination labels in reference impressions (manual coding) | `annotation/` thematic analysis export |
| `few_shot_note_ids.csv` | MIMIC-IV-Note ids of prompt examples that are rendered from the user's own MIMIC copy | identified during dataset construction |
