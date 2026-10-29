# Detector predictions on Rad-Halo

One JSONL file per detector configuration evaluated in the paper, one line per report:

```json
{"radhalo_id": 0, "findings_scores": [0.0, 1.0], "impression_scores": [0.0]}
```

- `findings_scores[i]` is the omission score of `findings_sentences[i]` in `rad_halo.json`;
  `impression_scores[i]` is the hallucination score of `impression_sentences[i]`.
- LLM judges output binary scores (0/1). RadCheck 8B outputs probabilities.
- Files contain identifiers and scores only, no report text.

| File | Detector | Reasoning setting |
|---|---|---|
| `gemini-2.5-flash.jsonl` | Gemini 2.5 Flash | thinking off (budget 0) |
| `gemini-3.1-pro_low.jsonl` | Gemini 3.1 Pro | lowest (thinking level `low`) |
| `gemini-3.1-pro_high.jsonl` | Gemini 3.1 Pro | highest (thinking level `high`) |
| `claude-opus-5.5_low.jsonl` | Claude Opus 5.5 | lowest (effort `low`) |
| `claude-opus-5.5_max.jsonl` | Claude Opus 5.5 | highest (effort `max`) |
| `gpt-6-astra_low.jsonl` | GPT-6 Astra | lowest (reasoning effort `low`) |
| `gpt-6-astra_max.jsonl` | GPT-6 Astra | highest (reasoning effort `max`) |
| `qwen3-8b.jsonl` | Qwen 3 8B | lowest (thinking off) |
| `qwen3-8b_thinking.jsonl` | Qwen 3 8B | highest (thinking on) |
| `radcheck-8b.jsonl` | RadCheck 8B | — |

Each configuration evaluated every report once; the exact settings are in `configs/judges/`. Gemini 2.5 Flash,
Qwen 3 8B and RadCheck 8B were run on an earlier segmentation of report 269, in which one findings sentence
(characters 1300–1452) was later split into two during annotation; the score of the original sentence is used for
both parts. The other judges were run on the final segmentation.
