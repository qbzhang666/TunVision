# TunVision

TunVision is a Streamlit prototype for the prescriptive shield-tunnel digital-twin workflow described in the manuscript in `Paper in Preparation/`.

The app follows the industry-facing four-step workflow:

1. Multimodal evidence base: ART/S3DIS point clouds, leakage evidence, deformation indicators, and standards.
2. Point-cloud curation: denoising, invert removal, annotation, label fraction control, and transfer-regime comparison.
3. Perception-knowledge engine: Sonata leakage segmentation, PCA to multi-zone polynomial deformation reconstruction, and LLM schema-guided FMEA extraction.
4. Standards-grounded prescription: per-ring health grades, maintenance perception, tunnel-level intervention tier, and re-inspection support.

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

## Selectable backends

TunVision now exposes the workflow choices expected in an industry app:

- Dataset access: bundled demo, file upload, GitHub raw URL, or local file path
- Training pool: `S3DIS` or `S3DIS + ART`
- Transfer regime: full fine-tuning or linear probe
- Segmentation model: Sonata configuration, Point Transformer v3 hook, 3D Otsu-KNN baseline, or GitHub model-runner hook
- Geometry method: Multi-Zone Polynomial, raw conic validation, or ellipse baseline
- Information layer: local JSON FMEA, Ollama schema extractor, or GitHub ontology JSON

The current prototype visualises point-cloud evidence and Sonata experiment outputs. It provides integration hooks for live Sonata/GitHub/Ollama execution, but a trained checkpoint runner or external API must be connected for full production inference.
