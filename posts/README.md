# Posts

Each published article lives in its own folder:

```text
posts/
  article-slug/
    index.qmd
    assets/
    references.bib  # when required
```

The article front matter should include `title`, `description`, `author`, `date`, `categories`, `image` and `content-type`. The blogs page automatically lists `posts/*/index.qmd`, with the newest dated article featured first.

Keep article-specific figures and saved model outputs in that article's `assets/` folder. Shared site imagery belongs in `images/`. Working notebooks are not published; move reusable logic into `src/` and reproducible generation steps into `scripts/`.

After adding or editing an article, run `quarto render` from the repository root and check the article plus the blogs listing.
