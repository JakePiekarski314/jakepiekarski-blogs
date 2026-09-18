# Python source

Reusable Python code for the technical articles and offline asset-generation scripts lives here. This directory is not a collection of website pages.

- Top-level modules support the hierarchical MMM material.
- `crosschannel/` contains the data generation, model, transformations and reporting code for the cross-channel measurement article.
- `bassdiffusion/` supports the product adoption forecasting article.

Keep presentation and article prose in the relevant `posts/<article>/index.qmd`. Put reusable modelling logic here and call it from a script or rendered article as appropriate.
