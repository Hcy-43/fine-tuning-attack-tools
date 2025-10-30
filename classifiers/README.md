# classifiers/

Trained safety probes (e.g. `pca_LR.pkl`), saved with `joblib`. Not checked in — train your own
with `notebooks/probing_classifier.ipynb` or `src/probe.py` on extracted activations from
`output_data/`. Loaded by `streamlit/frontend.py` to filter uploaded fine-tuning data.
