# jakepiekarski-hierarchical-mmm

This repository hosts my personal site for technical writing on Marketing Mix Modelling, Bayesian statistics, and causal inference. It serves as a central place for blog posts, case studies, and other work I find worth sharing—ranging from practical MMM examples to methodological deep-dives and applied analytics projects.

## Project structure

- `src/` — Python modules (data, models, plotting, summarize, config)
- `posts/hierarchical-mmm/` — Quarto blog post (main artifact)
- `notebooks/` — Full Jupyter notebook for reference

## Render locally

The blog post loads pre-sampled idata from disk — no MCMC runs during render.

1. Create and activate the `blogs` conda environment:
   ```bash
   conda create -n blogs python=3.12 -y
   conda activate blogs
   pip install -r requirements.txt ipykernel pyyaml nbformat nbconvert
   python -m ipykernel install --user --name blogs --display-name "Python (blogs)"
   ```
2. Fit models and save idata (run once, ~1–2 min):
   ```bash
   python scripts/fit_and_save.py
   ```
   Outputs go to `posts/hierarchical-mmm/assets/` (`.nc` and `.pkl` files).
3. Render the post:
   ```bash
   QUARTO_PYTHON=$CONDA_PREFIX/bin/python quarto render posts/hierarchical-mmm/index.qmd
   ```
4. Check `posts/hierarchical-mmm/assets/` into version control so others can render without re-running MCMC.

To update results (e.g. after changing data or models), run step 2 again, then step 3.
