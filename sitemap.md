---
layout: archive
title: "Sitemap"
permalink: /sitemap/
author_profile: false
---

Below is a list of all main sections on this site.  
An XML sitemap for search engines is also available at [sitemap.xml]({{ "sitemap.xml" | relative_url }}).

---

### 🧠 Research & Academic
- [Publications](/publications/)
- [Softwares](/softwares/)
- [CV](/cv/)

### 🍳 Personal
- [Cookings](/cookings/)
- [Blog](/blog/)

---
{% include base_path %}
{% for post in site.posts %}
- [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}
