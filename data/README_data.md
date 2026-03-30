# data/

ACI-Bench test set 1 data files used in ACI Bench Paper and for our additional validated output experiments. 

Downloaded from the ACI-Bench figshare repository:
[https://doi.org/10.6084/m9.figshare.22494601](https://doi.org/10.6084/m9.figshare.22494601)

---

## Files

| File | Rows | Description |
|---|---|---|
| `clinicalnlp_taskB_test1.csv` | 40 | Primary data file. Contains four columns: `dataset` (subset name), `encounter_id` (unique identifier), `dialogue` (full doctor-patient conversation transcript), and `note` (gold standard clinical note used as the evaluation reference — automatically generated then reviewed and corrected by domain experts including medical scribes and physicians) |
| `clinicalnlp_taskB_test1_metadata.csv` | 40 | Metadata file. Contains: `dataset`, `encounter_id`, `id`, `doctor_name`, `patient_gender`, `patient_age`, `patient_firstname`, `patient_familyname`, `cc` (chief complaint), `2nd_complaints` (secondary complaints). Joined to the primary file on `encounter_id` |
| `clinicalnlp_taskB_test1_n5.csv` | 5 | First 5 encounters from the primary file — used for quick batch runs |
| `clinicalnlp_taskB_test1_metadata_n5.csv` | 5 | Metadata for the first 5 encounters — used alongside the n5 data file |

---

## Dataset subsets

The 40 encounters span three subsets, each representing a different mode of clinical note generation:

| Subset | Count | Description |
|---|---|---|
| `virtassist` | 10 | Doctor uses explicit wake words to activate a virtual assistant |
| `virtscribe` | 8 | Doctor directs a separate scribe; includes pre-ambles and after-visit dictations |
| `aci` | 22 | Natural doctor-patient conversation with no explicit assistant or scribe |

---


## Citation

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

---

## Download instructions

The data files are already included in the repo for ease, but to download:
1. Go to [https://doi.org/10.6084/m9.figshare.22494601](https://doi.org/10.6084/m9.figshare.22494601)
2. Download the ACI-Bench release
3. Copy these two files into this `data/` directory:
   - `clinicalnlp_taskB_test1.csv`
   - `clinicalnlp_taskB_test1_metadata.csv`
4. We generated the n5 files to create a small test subset
