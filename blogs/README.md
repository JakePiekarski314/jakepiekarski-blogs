# Blogs and case studies listing

`index.qmd` defines the listing page, filter control and browser-side filtering. `content-listing.ejs.md` controls the markup for the featured article and remaining entries.

Content comes from `posts/*/index.qmd`. The newest dated post is featured automatically, and each post's `content-type` front-matter value supplies its Blog or Case Study label.

When changing this folder, render the full site and check both the featured item and the content-type filter.
