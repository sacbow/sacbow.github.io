---
layout: archive
title: "Blog"
permalink: /blog/
author_profile: true
---

This is my personal blog where I occasionally write about anything.

{% include base_path %}
{% for post in site.posts reversed %}
  {% include archive-single.html %}
{% endfor %}
