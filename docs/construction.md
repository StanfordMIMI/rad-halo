# Constructing Rad-Halo and training RadCheck 8B

These steps rebuild the dataset and the detector from MIMIC-IV-Note v2.2 (level C in the
[README](../README.md#-reproducing-the-paper)). They need API access to Vertex AI and Anthropic and, for
RadCheck training, a GPU. API outputs are not deterministic, so a rerun gives a comparable dataset rather
than the identical one; the released Rad-Halo is the reference.

All commands run from the repository root; `--help` lists the options.

**Eligible reports and segmentation** (Sec. 3.2, App. A.1–A.2)
```bash
python scripts/construction/preprocessing/01_filter_eligible.py --out data/mimic_eligible.jsonl
```

**Prompt selection** (App. A.4): sample 100 reports, generate with three instruction variants, compare
sentence counts.
```bash
python scripts/construction/impressions/01_sample_prompt_selection.py --eligible data/mimic_eligible.jsonl
python scripts/construction/impressions/02_generate_prompt_selection.py --generator <generator>
python scripts/construction/impressions/03_analyze_prompt_selection.py
```

**Impression generation and MedVAL grading** (Sec. 3.3–3.4, App. A.3, A.5–A.7)
```bash
python scripts/construction/impressions/04_generate_and_grade.py generator=<generator> random_seed=<seed>
python scripts/construction/impressions/04_generate_and_grade.py generator=reference \
    attribution_examples=<RADHALO_DATA>/attribution_examples.yaml
```
`<generator>` is a file in `configs/generators/` (`reference` keeps the original impression). Rad-Halo used
`random_seed=101` for `gemini-2.5-flash`, `claude-sonnet-4.5` and `llama-3.1-8b` and `0` for the others.
The few-shot examples are MIMIC-IV-Note reports; the prompts contain placeholders that are filled from your
copy of MIMIC using the note ids in `derived/few_shot_note_ids.csv`. MedVAL (Aali et al., MIT license) is
included in `src/radhalo/generation/medval.py`. `count_risk_grades.py` recomputes
`derived/risk_grade_counts.csv` from the generation logs (written to `outputs/` by default).

**Synthetic training data** (Sec. 4.3, App. D)
```bash
python scripts/construction/preprocessing/02_preprocess.py --input data/mimic_eligible.jsonl \
    --out data/synthetic/pool.jsonl --holdout 1000 --split-seed 0 --num-rows 10000 --seed 0
python scripts/construction/synthetic/01_attribute.py --input data/synthetic/pool.jsonl \
    --out data/synthetic/attributed.jsonl --examples <RADHALO_DATA>/attribution_examples.yaml
python scripts/construction/synthetic/02_create_synthetic_data.py --input data/synthetic/attributed.jsonl \
    --out data/synthetic/radcheck_synthetic.jsonl
python scripts/construction/synthetic/03_composition.py --input data/synthetic/radcheck_synthetic.jsonl
```
The attribution prompt (`prompts/attribution.yaml`) contains three worked examples derived from MIMIC-IV
reports. Example 3 is rendered from your MIMIC copy (note id in `derived/few_shot_note_ids.csv`); examples 1
and 2 are not single MIMIC reports and are distributed with the Rad-Halo PhysioNet release
(`attribution_examples.yaml` in the release directory). As in the paper runs, attribution uses the
`vertexai` SDK with Gemini 2.5 Flash Lite in region `us-central1` (`GCP_LOCATION` is not used for this
step).
`03_composition.py` writes to `outputs/` by default; compare with `derived/synthetic_composition.csv`.

**RadCheck training** (Sec. 4.2, App. C.4; one A100 40GB)
```bash
python scripts/construction/training/train_radcheck.py dataset.input_file=data/synthetic/radcheck_synthetic.jsonl
```
Hyperparameters are in `configs/train_radcheck.yaml`. Weights & Biases logging is off by default
(`wandb.enabled=true` to enable).
