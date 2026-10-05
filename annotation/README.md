# Annotation provenance

This directory documents how the Rad-Halo labels were produced (paper Sec. 3.2, 3.4-3.6, App. A.5, B,
E.3) and how the PhysioNet release files were built. Most steps need private inputs (MIMIC-IV report
text, generation logs, Label Studio exports), so readers can inspect but not rerun them. The steps that
start from the PhysioNet release (`assignments.py`, `prehighlighted_findings.py` with the task file,
`build_physionet_release.py` from the final file) are the exception.

Residents are identified only as `annotator_1` .. `annotator_4`, the attending as `adjudicator`.
Label Studio user ids are read from a user map that is never committed (`user_map.example.yaml`).

Run the scripts from the repository root, e.g. `python annotation/scripts/04_build_dataset.py --help`.
Extras: `nlp` (segmentation), `api` (Gemini attribution, Claude classification), `annotation` (Excel).

## Pipeline

Manual steps in Label Studio are marked **[LS]**. File names are those of the original run.

| Step | Script / action | Input | Output |
|---|---|---|---|
| 1 | `01_prepare_tasks.py` | one generation log per impression source (`scripts/construction/impressions/`) | `processed_reports.jsonl`, `tasks.json` |
| 2 | **[LS]** pre-highlight review (`label_studio/prehighlight_review.xml`): a physician added findings highlights the attribution missed | `tasks.json` | review export |
| 3 | `02_import_prehighlight_review.py` | review export | resident tasks (`formatted_for_labelstudio_preannotated.json`) |
| 4 | `03_split_partitions.py` | resident tasks | partitions 1-3 (50/125/125) |
| 5 | **[LS]** resident annotation (`label_studio/annotation.xml`), one project per resident; first 20 reports reviewed for task understanding | partitions | per-resident exports |
| 6 | **[LS]** a coordinator (user map `anchor`) fixes one sentence segmentation per report and copies each resident annotation onto it | per-resident exports | anchor exports |
| 7 | `04_build_dataset.py` | anchor + resident exports, `processed_reports.jsonl`, user map | `rad-halo_dataset.json` |
| 8 | `05_classify_imaging.py` (Claude Sonnet 4.5, forced tool call) | `rad-halo_dataset.json` | `imaging_technique_classification.xlsx` |
| 9 | physician review of the sheet | sheet | `imaging_technique_classification_reviewed.xlsx` |
| 10 | `06_merge_imaging.py` | dataset + reviewed sheet | `rad-halo_dataset_imaging_technique.json` |
| 11 | `07_prepare_adjudication.py` | dataset with imaging labels | `{hallucination,omission}_disagreements_for_labelstudio.json` |
| 12 | **[LS]** adjudication (`label_studio/labeling_config_{hallucinations,omissions}.xml`) | disagreement tasks | adjudication exports |
| 13 | `08_add_adjudications.py` | dataset + adjudication exports | `rad-halo_dataset_adjudicated.json` |
| 14 | `09_finalize_dataset.py` | adjudicated dataset | `rad-halo_dataset_final.json` |
| 15 | `build_physionet_release.py` | final dataset | release `data/`, `metadata/`, `code/`; `prompts/attribution_examples.yaml` (attribution examples 1-2, App. D.1) was added to the release by hand |

Side analyses:

| Script | Paper | Input | Output |
|---|---|---|---|
| `pilot_analysis.py` | Sec. 3.4, App. A.5 | pilot export (`label_studio/pilot.xml`) + its processed data | `descriptive_stats.csv`, `stats_by_risk_level.csv` |
| `thematic_extract.py` | App. E.3 | `rad-halo_dataset_imaging_technique.json` | coding tasks (`label_studio/thematic_coding.xml`) |
| `thematic_export.py` | App. E.3 | coding exports | `derived/thematic_clusters.csv` |
| `prehighlighted_findings.py` | App. B.3 | resident tasks + release | `derived/prehighlighted_findings.csv` |
| `assignments.py` | App. B | release | `label_studio/assignments.csv` |
| `make_example_task.py` | Fig. annotation interface | fictional report | `label_studio/example_task.json` |

## Details

**Selection and ids (step 1).** From each generation log, sorted by file name, the first 5/5/5/35
impressions of MedVAL risk levels 1-4 are kept. The 300 items are shuffled with `random.seed(0)`; the
position is the Rad-Halo id (`generated_impression_id` in the task data, `radhalo_id` afterwards).

**Pre-highlighting (steps 1-3).** Findings sentences are highlighted if Gemini 2.5 Flash Lite attribution
(App. D.1) links them to a sentence of the *reference* impression. Reference sentences linked to more than
six findings sentences (summaries of normal findings) do not contribute. Sentences are located in the
text with whitespace-tolerant matching; all sentences of the evaluated impression are highlighted. A
physician then reviewed every report in Label Studio. In the original run the review raised the number
of highlighted findings sentences from 1,243 to 2,088 (5 automatic highlights were removed) and adjusted
impression spans in 11 reports.

**Assignment (step 4).** Partitions 1-3 were created by `03_split_partitions.py`: ids 0-49 went to
annotator_1 and annotator_2, ids 50-174 to annotator_1, ids 175-299 to annotator_2. The second annotators
of ids 50-299 were assigned outside the committed code: annotator_3 and annotator_4 each received one half
(partitions 4 and 5; the split procedure and its seed are not recorded), and annotator_4 also annotated 21
reports of annotator_3's half. `label_studio/assignments.csv` (`radhalo_id, annotator, partition`) lists
all 621 assignments; it is derived from the release labels by `assignments.py`.

**Building the dataset (step 7).** The anchor user's annotation of each task defines the sentences
(`start`, `end`, `text` in `report_text` / `impression_text`). Labels are read from the anchor user's copy
of a resident's annotation where present, otherwise from the resident's own annotation; a sentence is
positive only if labelled `HALLUCINATION` / `OMISSION`, and negative for every other sentence of a report
the resident annotated. The script refuses to write if a labelled span is not an anchor sentence or still
"To Review". Copy `user_map.example.yaml` outside the repository and fill in your Label Studio user ids.

**Imaging modality and anatomy (steps 8-10).** Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`, max 256
tokens, forced tool `classify_report`, prompt `prompts/classify_imaging_technique.yaml`) received the full
report text via the Anthropic API and chose from fixed label lists. A rate-limited request was retried once
after 30 s; other failures were marked "Unknown". A physician reviewed and corrected all labels.

**Adjudication (steps 11-14).** Every sentence on which the residents disagreed became one task, with the
residents' labels shown as "Annotator N: Hallucination | Annotator M: Not Hallucination" (anonymised ids).
The adjudicator's decision is read from the first annotation of each task. The final label
(`adjudicated_label`) is the adjudicator's label for disputed sentences and the unanimous resident label
otherwise.

**Release (step 15).** `rad_halo_dataset.json` is the final file with `report_text` renamed to
`background_and_findings_text` and `generation_model` to `impression_source`. The CSVs flatten it (labels
"true"/"false", empty where a rater did not label). `scripts/physionet_metadata.py` holds the schema and
data dictionary verbatim; the release README is maintained separately.

**Thematic analysis.** All reference-impression sentences labelled as hallucination by at least one
resident (union rule, before adjudication) were coded inductively in Label Studio. The `theme` field has
the form "theme name [cluster name]"; only cluster counts are exported. Free-text notes and initial codes
are not released.

**Pilot.** The pilot export used the labels "Hallucination" and "True" (omitted findings sentence) after
normalisation. The first 10 reports per risk level by id (40 reports) are analysed. Sentence counts come
from the stored source report, so the hallucination rate is divided by the number of *reference*
impression sentences, as in the original analysis.

## Label Studio configurations (`label_studio/`)

| File | Project |
|---|---|
| `annotation.xml` | resident annotation |
| `pilot.xml` | pilot annotation |
| `prehighlight_review.xml` | pre-highlight review |
| `labeling_config_hallucinations.xml`, `labeling_config_omissions.xml` | adjudication |
| `thematic_coding.xml` | thematic coding (one project per error type, same configuration) |
| `example_task.json` | fictional task of the interface figure |
