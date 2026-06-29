# TunVision

TunVision is a Streamlit prototype for the prescriptive shield-tunnel digital-twin workflow described in the manuscript in `Paper in Preparation/`.

The app follows the paper's TunSPEC logic:

1. Multimodal evidence base: ART/S3DIS point clouds, leakage evidence, deformation indicators, and standards.
2. Point-cloud preprocessing: cleaned tunnel scans, ring partitioning, and exported XYZ/CSV previews.
3. Machine-learning perception: Sonata point-cloud segmentation for leakage, joints, segments, and pockets.
4. Geometric reconstruction: PCA alignment, robust fixed-radius fitting, multi-zone polynomial reconstruction, and deformation indicators.
5. Standards-grounded reasoning: FMEA severity bands with clause provenance.
6. Prescriptive decision support: max-severity fusion of deformation and leakage channels.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data inputs

The app ships with the two held-out tunnel examples reported in the manuscript. You can also upload:

- A ring-indicator CSV with columns:
  `tunnel, ring, convergence_per_mille_d, dislocation_mm`
  Optional Table-9 style columns are displayed when present:
  `ovalization_mm, ellipticity_per_mille, max_joint_dislocation_mm, max_joint_rotation_deg, fit_rmse_mm`
- A point-cloud preview file as `.csv`, `.txt`, or `.xyz` with `x`, `y`, and `z` columns
- A FMEA/rule JSON file using the structure in `data/fmea_rules.json`

The current prototype visualises point-cloud evidence and Sonata experiment outputs, but it does not run Sonata inference directly. Export segmentation labels, leakage class, leakage area, and source provenance from the perception pipeline, then ingest them here for decision support.
