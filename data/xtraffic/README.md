# Dataset Directory

Download the model-ready datasets from Kaggle and place each city directory here:

- Alameda
- Contra_Costa
- Orange

Each city directory should contain:

- adj_matrix.npy
- desc_mapping.json
- incident_all.npy
- incident_stats.npz
- sensors.csv
- type_mapping.json

Then generate train/validation/test split files with:

```bash
python data/xtraffic/prepare_splits.py --dataset Alameda
python data/xtraffic/prepare_splits.py --dataset Contra_Costa
python data/xtraffic/prepare_splits.py --dataset Orange
```
