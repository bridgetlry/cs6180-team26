# baselines/

Published baseline prediction outputs from the original ACI-Bench paper
([Yim et al. 2023](https://github.com/wyim/aci-bench)), used as comparison
points for our three techniques. All files in this folder are directly from [https://github.com/wyim/aci-bench](https://github.com/wyim/aci-bench),
none were created by us. 

---

## Structure

```
baselines/
└── predictions/
    └── aci_bench_paper/
        ├── test1ChatGPT_.csv           — ChatGPT (gpt-3.5-turbo)
        ├── test1GPT-4_.csv             — GPT-4
        ├── test1Text-Davinci-002_.csv  — text-davinci-002
        └── test1Text-Davinci-003_.csv  — text-davinci-003
```

All four CSVs cover the same 40 encounters from `clinicalnlp_taskB_test1.csv`
and use the four-column format expected by `evaluate_fullnote.py`:

| Column | Description |
|---|---|
| `Dialogues` | The original doctor-patient conversation transcript |
| `Reference Summaries` | The gold standard clinical note used for evaluation |
| `note` | The model's predicted/generated clinical note |
| `encounter_id` | Unique encounter identifier, e.g. `D2N088` |

---

## How to score a baseline

Must be run from the **project root** (`PythonProject4/`), not from inside `baselines/`:

```bash
cd ~/PycharmProjects/PythonProject4

python evaluation/evaluate_fullnote.py \
  data/clinicalnlp_taskB_test1.csv \
  baselines/predictions/aci_bench_paper/test1ChatGPT_.csv \
  data/clinicalnlp_taskB_test1_metadata.csv
```

Results are written to `results/test1ChatGPT_.json`.

---

## Reference scores (Yim et al. 2023, Table 6 — test set 1)

All scores are reported as F1 × 100.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | MEDCON |
|---|---|---|---|---|
| ChatGPT (gpt-3.5-turbo) | 47.44 | 19.01 | 42.47 | 55.84 |
| GPT-4 | 51.76 | 22.58 | 45.97 | 57.78 |
| text-davinci-002 | 41.08 | 17.27 | 37.46 | 47.39 |
| text-davinci-003 | 47.07 | 22.08 | 43.11 | 57.16 |

Our GVR pipeline (Llama-3.1-8B) on the same test set: ROUGE-1: 0.4082,
ROUGE-2: 0.1801, ROUGE-L: 0.2675. Note the paper reports scores × 100;
our evaluation script reports raw fractions.

---

## Source

Yim et al. (2023). *ACI-Bench: a Novel Ambient Clinical Intelligence Dataset
for Benchmarking Automatic Visit Note Generation.* Scientific Data 10, 586.
[https://doi.org/10.1038/s41597-023-02487-3](https://doi.org/10.1038/s41597-023-02487-3)
[https://github.com/wyim/aci-bench](https://github.com/wyim/aci-bench)

```bibtex
@article{aci-bench,
  author = {Wen{-}wai Yim and
            Yujuan Fu and
            Asma {Ben Abacha} and
            Neal Snider and Thomas Lin and Meliha Yetisgen},
  title = {ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
  journal = {Nature Scientific Data},
  year = {2023}
}
```
