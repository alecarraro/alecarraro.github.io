---
layout: page
title: Writing
permalink: /writing/
---

<div class="writing-archive">
  <p class="archive-intro">Project notes and explorations in scientific computing, numerical methods, and Julia.</p>
  {% assign visible_posts = site.posts | where_exp:"post", "post.hidden != true" %}
  {% assign posts_by_year = visible_posts | group_by_exp:"post", "post.date | date: '%Y'" %}
  {% for year in posts_by_year %}
    <section class="archive-year" aria-labelledby="year-{{ year.name }}">
      <h2 id="year-{{ year.name }}">{{ year.name }}</h2>
      <div class="writing-list">
        {% for post in year.items %}
          <article class="writing-item">
            <p class="writing-date">{{ post.date | date: "%b %d" }}</p>
            <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
            <a class="writing-arrow" href="{{ post.url | relative_url }}" aria-label="Read {{ post.title }}">↗</a>
          </article>
        {% endfor %}
      </div>
    </section>
  {% endfor %}
</div>
