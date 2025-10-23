---
layout: archive
title: "Cookings"
permalink: /cookings/
author_profile: true
---

Here are some of my favorite home-cooked dishes.

{% include base_path %}
{% for post in site.cookings reversed %}
  {% include archive-single.html %}
{% endfor %}
