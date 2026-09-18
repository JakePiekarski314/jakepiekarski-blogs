# JTP Analytics website

Source for the [JTP Analytics website](https://www.jtp-analytics.com), built with Quarto and deployed to Netlify by GitHub Actions.

## Repository map

| Path | Purpose |
|---|---|
| `index.qmd` | Homepage |
| `services/`, `pricing/`, `partners/`, `contact/`, `thank-you/` | Main website pages, each using an `index.qmd` |
| `blogs/` | Blogs and case studies listing plus its custom listing template |
| `team/` | Team listing and individual profile pages |
| `posts/` | Published long-form articles and their post-specific assets |
| `images/` | Shared brand, team and service images |
| `src/` | Reusable Python modules used by technical articles and asset generation |
| `scripts/` | Offline scripts for generating model results and figures |
| `_quarto.yml` | Site structure, navigation and rendering configuration |
| `styles.css` | Site-wide visual styling |
| `site-interactions.html` | Lightweight site-wide browser interactions |
| `.github/workflows/publish.yml` | Automated Quarto build and Netlify deployment |
| `.agents/skills/jtp-site-edits/` | Repository-local workflow for future Codex site edits |

More detail is available in the READMEs inside `blogs/`, `posts/`, `images/`, `scripts/`, `src/` and `team/`.

## Editing the site

Start each change from the latest `main` branch and work on a short-lived branch:

```bash
git switch main
git pull --ff-only
git switch -c codex/describe-the-change
```

Edit the source files, preview the site, and run a full render before committing:

```bash
quarto preview
quarto render
```

If `quarto` is not on the shell path, use the copy bundled with RStudio:

```bash
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto preview
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render
```

Do not edit `_site/`, `.quarto/` or `_freeze/` directly. They are generated locally and ignored by Git.

When the change is ready, commit it, push the branch and open a pull request. Merging the pull request into `main` triggers the GitHub Actions workflow, which renders the site and publishes it to Netlify. No manual Netlify deployment is needed.

## Local setup

The deployment workflow uses Python 3.12. For a fresh local environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The published pages are deliberately listed in `_quarto.yml`. Development notebooks are not part of the website build.

## Model assets

Published articles use pre-generated model outputs and figures so a normal website render does not run expensive model fitting. To regenerate those files after changing modelling code, follow the relevant notes in `scripts/README.md` and `posts/README.md`.

## Forms

The homepage and contact forms use Netlify Forms. Their form names and hidden `form-name` fields must remain aligned. Test submissions only when explicitly checking the production form workflow.
