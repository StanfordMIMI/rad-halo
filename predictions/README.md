# Detector predictions on Rad-Halo

One JSONL file per detector configuration evaluated in the paper, one line per report:

```json
{"radhalo_id": 0, "findings_scores": [0.0, 1.0], "impression_scores": [0.0]}
```

- `findings_scores[i]` is the omission score of `findings_sentences[i]` in `rad_halo_dataset.json`;
  `impression_scores[i]` is the hallucination score of `impression_sentences[i]`.
- LLM judges output binary scores (0/1). RadCheck 8B outputs probabilities.
- Files contain identifiers and scores only, no report text.

| File | Detector (paper name) |
|---|---|
| `gemini-2.5-flash.jsonl` | Gemini 2.5 Flash |
| `gemini-3-pro_low.jsonl` | Gemini 3 Pro, low thinking |
| `gemini-3-pro_high.jsonl` | Gemini 3 Pro, high thinking |
| `claude-opus-4.5.jsonl` | Claude Opus 4.5, no thinking |
| `claude-opus-4.5_16k.jsonl` | Claude Opus 4.5, 16k thinking budget |
| `claude-opus-4.5_32k.jsonl` | Claude Opus 4.5, 32k thinking budget |
| `qwen3-8b.jsonl` | Qwen 3 8B |
| `qwen3-8b_thinking.jsonl` | Qwen 3 8B, thinking |
| `radcheck-8b.jsonl` | RadCheck 8B |

Each configuration evaluated every report once. Report 269 was scored before one of its findings
sentences (characters 1300–1452) was split into two during annotation; the score of the original
sentence is used for both parts.
