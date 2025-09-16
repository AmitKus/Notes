---
layout: page
title: All Notes
permalink: /all-notes/
---

# All Notes by Category

## 📚 Interview Preparation (_16_Interview)
<ul>
  {% for note in site.notes %}
    {% if note.path contains "_16_Interview" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## 📄 Research Papers (_17_Papers)
<ul>
  {% for note in site.notes %}
    {% if note.path contains "_17_Papers" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## 🤖 AI Models (_03_Models)
<ul>
  {% for note in site.notes %}
    {% if note.path contains "_03_Models" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## 🛠️ Tools (_04_Tools)
<ul>
  {% for note in site.notes %}
    {% if note.path contains "_04_Tools" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## 🔄 Workflows (_05_Workflows)
<ul>
  {% for note in site.notes %}
    {% if note.path contains "_05_Workflows" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## 📊 Tutorials (_11_Tutorials)
<ul>
  {% for note in site.notes %}
    {% if note.path contains "_11_Tutorials" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## 🎓 Courses (_15_Courses)
<ul>
  {% for note in site.notes %}
    {% if note.path contains "_15_Courses" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## 📚 Other Categories
<ul>
  {% for note in site.notes %}
    {% unless note.path contains "_16_Interview" or note.path contains "_17_Papers" or note.path contains "_03_Models" or note.path contains "_04_Tools" or note.path contains "_05_Workflows" or note.path contains "_11_Tutorials" or note.path contains "_15_Courses" %}
      <li><a class="internal-link" href="{{ site.baseurl }}{{ note.url }}">{{ note.title | default: note.name }}</a></li>
    {% endunless %}
  {% endfor %}
</ul>