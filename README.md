# jakepiekarski-blogs

Personal site for technical writing on Marketing Mix Modelling, Bayesian statistics, and causal inference. Blog posts, case studies, and applied analytics, practical MMM examples, methodological deep-dives, and PyMC implementations.

**Live site:** [https://jakepiekarski314.github.io/jakepiekarski-blogs/](https://jakepiekarski314.github.io/jakepiekarski-blogs/)

## Project structure

- `src/` - Python modules (data, models, plotting, summarize, config)
- `posts/` - Quarto blog posts (e.g. `hierarchical-mmm/`)
- `_quarto.yml` - Site config, navigation, theme

## Render locally

Posts load pre-sampled data from disk, no MCMC runs during render.

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
   Outputs go to `posts/hierarchical-mmm/assets/`.
3. Generate model graphs (run once, ~10 s):
   ```bash
   python scripts/generate_model_graphs.py
   ```
   Creates `model_unpooled.png`, `model_centered.png`, `model_noncentered.png`, `model_tuned.png` in `posts/hierarchical-mmm/assets/`.
4. Render and preview the full site:
   ```bash
   QUARTO_PYTHON=$CONDA_PREFIX/bin/python quarto preview
   ```

To update model results after changing data or models, run steps 2 and 3 again, then step 4.

## Deployment

The site is built and deployed to GitHub Pages on every push to `main`. See `.github/workflows/publish.yml`.
