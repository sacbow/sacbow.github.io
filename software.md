---
layout: archive
title: "Software"
permalink: /software/
author_profile: true
---

{% include base_path %}
{% for post in site.softwares reversed %}
  {% include archive-single.html %}
{% endfor %}
