---
layout: page
title: Home
id: home
permalink: /
---

# AI Knowledge Base 🌱

<p style="padding: 3em 1em; background: #f5f7ff; border-radius: 4px;">
  <strong>Browse the complete knowledge base:</strong><br>
  <a href="/all-notes/" style="font-size: 1.2em;">📚 View All Notes by Category</a>
</p>

This is a comprehensive AI knowledge base covering models, tools, workflows, papers, and more.

<strong>Recently updated notes</strong>

<ul>
  {% assign recent_notes = site.notes | sort: "last_modified_at_timestamp" | reverse %}
  {% for note in recent_notes limit: 5 %}
    <li>
      {{ note.last_modified_at | date: "%Y-%m-%d" }} — <a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title }}</a>
    </li>
  {% endfor %}
</ul>

<style>
  .wrapper {
    max-width: 46em;
  }
</style>
