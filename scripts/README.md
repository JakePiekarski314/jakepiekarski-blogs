# Generation scripts

These scripts regenerate model outputs and figures used by technical articles. They are development tools and are not executed during a normal Quarto render or production deployment.

- `fit_and_save.py`: fits and saves the hierarchical MMM examples.
- `generate_model_graphs.py`: creates the hierarchical MMM model diagrams.
- `fit_cross_channel.py`: regenerates saved data, posterior output and the graph for the cross-channel measurement article.

Run scripts from the repository root with the local environment activated. Some fitting steps can take several minutes and replace binary assets under `posts/`, so review the resulting Git changes before committing them.
