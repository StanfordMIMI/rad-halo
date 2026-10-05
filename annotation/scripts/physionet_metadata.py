"""Metadata files of the PhysioNet release, verbatim (written by `build_physionet_release.py`)."""

# ruff: noqa: E501

SCHEMA_JSON = """\
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://physionet.org/content/rad-halo/schema.json",
  "title": "Rad-Halo Dataset Schema",
  "description": "JSON Schema for the canonical Rad-Halo dataset file, data/rad_halo_dataset.json.",
  "type": "array",
  "minItems": 300,
  "maxItems": 300,
  "items": {
    "$ref": "#/$defs/report"
  },
  "$defs": {
    "report": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "radhalo_id",
        "mimic_notes_id",
        "medval_risk_level",
        "impression_source",
        "background_and_findings_text",
        "impression_text",
        "findings_sentences",
        "impression_sentences",
        "imaging_technique",
        "anatomy_location"
      ],
      "properties": {
        "radhalo_id": {
          "type": "integer",
          "minimum": 0,
          "maximum": 299,
          "description": "Unique Rad-Halo report identifier."
        },
        "mimic_notes_id": {
          "type": "string",
          "minLength": 1,
          "description": "Radiology note identifier transferred directly from MIMIC-IV-Note."
        },
        "medval_risk_level": {
          "type": "integer",
          "enum": [1, 2, 3, 4],
          "description": "MedVAL risk level assigned during dataset construction."
        },
        "impression_source": {
          "type": "string",
          "enum": [
            "original_radiologist",
            "gemini-2.5-flash",
            "claude-sonnet-4-5-20250929",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Ministral-3-3B-Instruct-2512",
            "Qwen/Qwen3-1.7B"
          ],
          "description": "Source of the evaluated impression."
        },
        "background_and_findings_text": {
          "type": "string",
          "minLength": 1,
          "description": "Source report context, including findings and retained non-impression sections."
        },
        "impression_text": {
          "type": "string",
          "minLength": 1,
          "description": "Evaluated impression text."
        },
        "findings_sentences": {
          "type": "array",
          "minItems": 1,
          "items": {
            "$ref": "#/$defs/findings_sentence"
          },
          "description": "Findings sentence objects used for omission annotation."
        },
        "impression_sentences": {
          "type": "array",
          "minItems": 1,
          "items": {
            "$ref": "#/$defs/impression_sentence"
          },
          "description": "Impression sentence objects used for hallucination annotation."
        },
        "imaging_technique": {
          "type": "string",
          "enum": [
            "CT",
            "Fluoroscopy",
            "Mammography",
            "MRI",
            "Ultrasound",
            "X-ray"
          ],
          "description": "Coarse imaging technique label."
        },
        "anatomy_location": {
          "type": "string",
          "enum": [
            "Abdomen",
            "Abdomen/Pelvis",
            "Ankle/Foot",
            "Breast",
            "Cervical spine",
            "Chest",
            "Chest/Abdomen/Pelvis",
            "Chest/Pelvis",
            "Elbow",
            "Face",
            "Fetal",
            "Head",
            "Head and neck",
            "Hip",
            "Knee",
            "Lower extremity",
            "Lumbar spine",
            "Neck",
            "Pelvis",
            "Shoulder",
            "Spine",
            "Thoracic spine",
            "Upper extremity",
            "Wrist/Hand"
          ],
          "description": "Coarse anatomy location label."
        }
      }
    },
    "findings_sentence": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "is_omission",
        "start",
        "end",
        "text",
        "adjudicated_label"
      ],
      "properties": {
        "is_omission": {
          "$ref": "#/$defs/label_object",
          "description": "Resident and optional adjudicator omission labels."
        },
        "start": {
          "type": "integer",
          "minimum": 0,
          "description": "Inclusive character start offset in background_and_findings_text."
        },
        "end": {
          "type": "integer",
          "minimum": 0,
          "description": "Exclusive character end offset in background_and_findings_text."
        },
        "text": {
          "type": "string",
          "minLength": 1,
          "description": "Findings sentence text."
        },
        "adjudicated_label": {
          "type": "boolean",
          "description": "Final benchmark omission label."
        }
      }
    },
    "impression_sentence": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "is_hallucination",
        "start",
        "end",
        "text",
        "adjudicated_label"
      ],
      "properties": {
        "is_hallucination": {
          "$ref": "#/$defs/label_object",
          "description": "Resident and optional adjudicator hallucination labels."
        },
        "start": {
          "type": "integer",
          "minimum": 0,
          "description": "Inclusive character start offset in impression_text."
        },
        "end": {
          "type": "integer",
          "minimum": 0,
          "description": "Exclusive character end offset in impression_text."
        },
        "text": {
          "type": "string",
          "minLength": 1,
          "description": "Impression sentence text."
        },
        "adjudicated_label": {
          "type": "boolean",
          "description": "Final benchmark hallucination label."
        }
      }
    },
    "label_object": {
      "type": "object",
      "additionalProperties": false,
      "minProperties": 2,
      "properties": {
        "annotator_1": {
          "type": "boolean",
          "description": "Label assigned by stable pseudonymous resident annotator_1."
        },
        "annotator_2": {
          "type": "boolean",
          "description": "Label assigned by stable pseudonymous resident annotator_2."
        },
        "annotator_3": {
          "type": "boolean",
          "description": "Label assigned by stable pseudonymous resident annotator_3."
        },
        "annotator_4": {
          "type": "boolean",
          "description": "Label assigned by stable pseudonymous resident annotator_4."
        },
        "adjudicator": {
          "type": "boolean",
          "description": "Attending adjudicator label for disputed sentences."
        }
      },
      "anyOf": [
        { "required": ["annotator_1"] },
        { "required": ["annotator_2"] },
        { "required": ["annotator_3"] },
        { "required": ["annotator_4"] }
      ]
    }
  }
}
"""

DATA_DICTIONARY_CSV = """\
files,field,type,required,allowed_values,missing_values,description
data/rad_halo_dataset.json,root,array,yes,,none,"Array of Rad-Halo report objects. Each object represents one sampled source report with one evaluated impression, report-level metadata, findings sentence labels, and impression sentence labels."
data/rad_halo_dataset.json; data/reports.csv,radhalo_id,integer,yes,0-299,none,Unique Rad-Halo report identifier. Integer from 0 through 299. Use this field to join reports.csv with findings_sentences.csv and impression_sentences.csv.
data/rad_halo_dataset.json; data/reports.csv,mimic_notes_id,string,yes,,none,Radiology note identifier transferred directly from MIMIC-IV-Note. This field can be used to locate the corresponding source report in MIMIC-IV-Note by users with appropriate access. It uniquely identifies the source report represented in Rad-Halo and is not a patient identifier.
data/rad_halo_dataset.json; data/reports.csv,impression_source,string,yes,original_radiologist; gemini-2.5-flash; claude-sonnet-4-5-20250929; meta-llama/Llama-3.1-8B-Instruct; mistralai/Ministral-3-3B-Instruct-2512; Qwen/Qwen3-1.7B,none,"Source of the evaluated impression. original_radiologist indicates the original reference impression from MIMIC-IV, written by a clinician. All other values denote LLM-generated impressions."
data/rad_halo_dataset.json; data/reports.csv,medval_risk_level,integer,yes,1; 2; 3; 4,none,MedVAL risk level assigned to the evaluated impression during dataset construction.
data/rad_halo_dataset.json; data/reports.csv,imaging_technique,string,yes,CT; Fluoroscopy; Mammography; MRI; Ultrasound; X-ray,none,Coarse imaging technique label assigned for subgroup analysis using LLM pre-filling followed by physician review.
data/rad_halo_dataset.json; data/reports.csv,anatomy_location,string,yes,Abdomen; Abdomen/Pelvis; Ankle/Foot; Breast; Cervical spine; Chest; Chest/Abdomen/Pelvis; Chest/Pelvis; Elbow; Face; Fetal; Head; Head and neck; Hip; Knee; Lower extremity; Lumbar spine; Neck; Pelvis; Shoulder; Spine; Thoracic spine; Upper extremity; Wrist/Hand,none,Coarse anatomy location label assigned for subgroup analysis using LLM pre-filling followed by physician review. The release contains 49 modality-anatomy combinations.
data/rad_halo_dataset.json; data/reports.csv,background_and_findings_text,string,yes,,none,"Source radiology report context used for impression generation and annotation. Includes the findings section and any pre-existing non-impression sections such as clinical history, indication, comparison, and technique. The evaluated impression is stored separately in impression_text."
data/rad_halo_dataset.json; data/reports.csv,impression_text,string,yes,,none,Evaluated impression text. This is LLM-generated for LLM impression sources and the original clinician-written MIMIC-IV reference impression for original_radiologist controls.
data/rad_halo_dataset.json,findings_sentences,array,yes,,none,"List of findings sentence objects used for omission annotation. Each sentence object contains text, character offsets, resident labels, optional adjudicator label, and final adjudicated label."
data/rad_halo_dataset.json,impression_sentences,array,yes,,none,"List of impression sentence objects used for hallucination annotation. Each sentence object contains text, character offsets, resident labels, optional adjudicator label, and final adjudicated label."
data/findings_sentences.csv; data/impression_sentences.csv,sentence_id,integer,yes,0 or greater,none,Zero-based index of the sentence within the corresponding released sentence list for that report. This field is present only in the flattened CSV files.
data/rad_halo_dataset.json; data/findings_sentences.csv; data/impression_sentences.csv,start,integer,yes,0 or greater,none,"Character start offset of the sentence in the corresponding text field. For findings_sentences, offsets refer to background_and_findings_text. For impression_sentences, offsets refer to impression_text. Start offsets are inclusive."
data/rad_halo_dataset.json; data/findings_sentences.csv; data/impression_sentences.csv,end,integer,yes,0 or greater,none,"Character end offset of the sentence in the corresponding text field. For findings_sentences, offsets refer to background_and_findings_text. For impression_sentences, offsets refer to impression_text. End offsets are exclusive. In a small number of cases, the sentence text differs from the exact source slice only by normalized whitespace."
data/rad_halo_dataset.json; data/findings_sentences.csv; data/impression_sentences.csv,text,string,yes,,none,Sentence text presented for annotation. For findings_sentences this is a source findings sentence evaluated for omission. For impression_sentences this is an evaluated impression sentence checked for hallucination.
data/rad_halo_dataset.json,findings_sentences[].is_omission,object,yes,,none,"Resident and adjudicator omission labels for this findings sentence. Keys are annotator identifiers and, for disputed sentences, adjudicator."
data/rad_halo_dataset.json,impression_sentences[].is_hallucination,object,yes,,none,"Resident and adjudicator hallucination labels for this impression sentence. Keys are annotator identifiers and, for disputed sentences, adjudicator."
data/rad_halo_dataset.json,findings_sentences[].is_omission.<annotator_id>,boolean,conditional,true; false,key absent if annotator did not label this sentence,"Resident label for whether the findings sentence is an omission. <annotator_id> is one of annotator_1, annotator_2, annotator_3, or annotator_4."
data/rad_halo_dataset.json,impression_sentences[].is_hallucination.<annotator_id>,boolean,conditional,true; false,key absent if annotator did not label this sentence,"Resident label for whether the impression sentence is a hallucination. <annotator_id> is one of annotator_1, annotator_2, annotator_3, or annotator_4."
data/rad_halo_dataset.json,findings_sentences[].is_omission.adjudicator,boolean,conditional,true; false,key absent if resident annotators agreed,Attending radiologist adjudicator label for disputed omission annotations. Present only when resident annotators disagreed.
data/rad_halo_dataset.json,impression_sentences[].is_hallucination.adjudicator,boolean,conditional,true; false,key absent if resident annotators agreed,Attending radiologist adjudicator label for disputed hallucination annotations. Present only when resident annotators disagreed.
data/findings_sentences.csv; data/impression_sentences.csv,annotator_1_label,boolean,conditional,true; false,blank,Label assigned by resident annotator_1. Blank means annotator_1 did not label this sentence; it does not mean false.
data/findings_sentences.csv; data/impression_sentences.csv,annotator_2_label,boolean,conditional,true; false,blank,Label assigned by resident annotator_2. Blank means annotator_2 did not label this sentence; it does not mean false.
data/findings_sentences.csv; data/impression_sentences.csv,annotator_3_label,boolean,conditional,true; false,blank,Label assigned by resident annotator_3. Blank means annotator_3 did not label this sentence; it does not mean false.
data/findings_sentences.csv; data/impression_sentences.csv,annotator_4_label,boolean,conditional,true; false,blank,Label assigned by resident annotator_4. Blank means annotator_4 did not label this sentence; it does not mean false.
data/findings_sentences.csv; data/impression_sentences.csv,adjudicator_label,boolean,conditional,true; false,blank,Label assigned by the attending adjudicator for disputed sentences. Blank means resident annotators agreed and adjudication was not required; it does not mean false.
data/rad_halo_dataset.json; data/findings_sentences.csv,adjudicated_label,boolean,yes,true; false,none,"Final benchmark omission label for a findings sentence. If an adjudicator label is present, adjudicated_label equals the adjudicator label; otherwise it equals the unanimous resident label. true means the findings sentence was finally labeled as omitted."
data/rad_halo_dataset.json; data/impression_sentences.csv,adjudicated_label,boolean,yes,true; false,none,"Final benchmark hallucination label for an impression sentence. If an adjudicator label is present, adjudicated_label equals the adjudicator label; otherwise it equals the unanimous resident label. true means the impression sentence was finally labeled as hallucinated."
"""
