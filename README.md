# TunVision

TunVision is a Streamlit prototype for the prescriptive shield-tunnel digital-twin workflow described in the manuscript in `Paper in Preparation/`.

The app follows the paper's TunSPEC logic:

1. Point-cloud perception evidence: leakage class, leakage area, model performance context.
2. Geometric deformation evidence: convergence and joint-dislocation indicators per ring.
3. Standards-grounded reasoning: severity bands with clause provenance.
4. Prescriptive decision support: max-severity fusion of deformation and leakage channels.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Data inputs

The app ships with the two held-out tunnel examples reported in the manuscript. You can also upload:

- A ring-indicator CSV with columns:
  `tunnel, ring, convergence_per_mille_d, dislocation_mm`
- A FMEA/rule JSON file using the structure in `data/fmea_rules.json`

The current prototype includes a placeholder perception panel for model-summary evidence. It does not run Sonata inference directly; exported segmentation/leakage metrics are ingested as tabular evidence.
