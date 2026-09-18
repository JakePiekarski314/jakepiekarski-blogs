---
name: jtp-site-edits
description: Edit, verify and prepare pull requests for the JTP Analytics Quarto website and its Netlify deployment.
---

# JTP Analytics site edits

Use this workflow for content, structure, styling and configuration changes in this repository.

## Start safely

1. Read `README.md`, `_quarto.yml` and any README in the area being changed.
2. Check the working tree. Preserve unrelated user changes and never discard them.
3. Fetch the latest remote state and create a new `codex/<short-description>` branch from `origin/main`. Do not make ordinary site edits directly on `main`.

## Work with the site structure

- Edit source `.qmd`, CSS, HTML includes, Python modules or generation scripts. Never edit generated `_site/`, `.quarto/` or `_freeze/` content.
- Keep main pages in their route-matching folders as `index.qmd` files.
- Keep each article in `posts/<slug>/index.qmd` and its local assets in `posts/<slug>/assets/`.
- Keep shared brand, team and service imagery in `images/`.
- Treat `_quarto.yml` as the explicit list of published sources. Do not add development notebooks to the render list.
- Preserve the Netlify form names, hidden `form-name` fields and thank-you route when changing forms.

## Verify changes

1. Run `quarto render` from the repository root. If `quarto` is not on the shell path, use `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render`.
2. Check the rendered pages affected by the change, plus navigation and internal links.
3. For layout work, inspect desktop and mobile views and check for overflow or overlap.
4. Review `git diff` and `git status` so generated files and unrelated changes are not committed.

## Deliver changes

Commit only the intended source files. Push the branch and create a pull request when the user requests publication or a PR. After merge, GitHub Actions renders and deploys the site to Netlify automatically; do not manually publish the same change as a second deployment path.
