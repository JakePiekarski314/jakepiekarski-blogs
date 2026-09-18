```{=html}
<% if (items.length > 0) { %>
  <% const featured = items[0]; %>
  <% const featuredType = featured['content-type'] || 'Blog'; %>
  <article class="featured-content" data-content-type="<%- featuredType %>">
    <div class="featured-content__meta">
      <span class="featured-content__label">Featured</span>
      <span class="content-type-badge"><%- featuredType %></span>
      <% if (featured.date) { %><span><%- featured.date %></span><% } %>
      <% if (featured['reading-time']) { %><span><%- featured['reading-time'] %></span><% } %>
    </div>
    <h2><a href="<%- featured.path %>"><%- featured.title %></a></h2>
    <% if (featured.subtitle) { %><p><%- featured.subtitle %></p><% } %>
    <a class="featured-content__link" href="<%- featured.path %>">Read featured article <span aria-hidden="true">&rarr;</span></a>
  </article>
<% } %>
<div class="content-list">
<% for (const item of items.slice(1)) { %>
  <% const contentType = item['content-type'] || 'Blog'; %>
  <article class="content-list-item" data-content-type="<%- contentType %>">
    <div class="content-list-meta">
      <span class="content-type-badge"><%- contentType %></span>
      <% if (item.date) { %><span><%- item.date %></span><% } %>
      <% if (item['reading-time']) { %><span><%- item['reading-time'] %></span><% } %>
    </div>
    <h2><a href="<%- item.path %>"><%- item.title %></a></h2>
    <% if (item.subtitle) { %><p><%- item.subtitle %></p><% } %>
  </article>
<% } %>
</div>
```
