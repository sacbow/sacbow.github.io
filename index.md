---
layout: single
author_profile: true
title: "Hajime Ueda (上田 朔 / UEDA Hajime)"
sidebar:
nav: "sidebar"
---
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Hajime Ueda",
  "alternateName": "上田 朔",
  "affiliation": {
    "@type": "Organization",
    "name": "The University of Tokyo"
  },
  "jobTitle": "PhD Candidate",
  "url": "https://sacbow.github.io",
  "sameAs": [
    "https://github.com/sacbow",
    "https://www.linkedin.com/in/hajime-ueda-2b9044382"
  ]
}
</script>



I'm Hajime Ueda, a PhD candidate at the University of Tokyo under the supervision of [prof. Masato Okada](https://mns.k.u-tokyo.ac.jp/home.html).

My research interests include:

- **Computational Imaging**

  In imaging sciences, we consider an unknown image $$x$$ mapped to the observed data $$Y$$ through the imaging system.
  While conventional imaging techniques aim to implement this mapping so that it approximates the identity operator, the computational imaging (CI) approach is to design both the physical mapping $$x \rightarrow Y$$ and the reconstruction algorithm $$Y \rightarrow x$$ to enlarge the world we can visualize.

- **Bayesian Inverse Problems**

  In the Bayesian formulation, an inverse problem is characterized by a triplet consisting of the prior $$\rho$$, the forward model $$G$$, and the likelihood $$l$$:

  $$x \sim \rho(x), \quad y = G(x), \quad Y \sim l(Y ; y)$$

  Then we apply the Bayes' theorem $$p(x|Y) \propto \rho(x) l(Y; G(x))$$ to infer $$x$$ from the data $$Y$$.
  Through this formulation, the prior knowledge about $$x$$, the physical mapping, and the model of observation noise are made explicit.
  Moreover, the distribution $$p(x|Y)$$ provides not only the point estimate of $$x$$, but also the uncertainty quantification of the reconstruction.

- **(Embedded) Domain Specific Languages**

  Since computational imaging involves co-designing the physical system and the reconstruction algorithm, manually implementing inference procedures for every imaging setup does not scale.
  Using the terminology of Bayesian inverse problems, my goal is to craft a Domain Specific Language (DSL) where the prior $$\rho$$, forward model $$G$$, and the likelihood $$l$$ are specified in a high-level syntax, and the reconstruction algorithms are automatically derived, enabling rapid prototyping of new imaging systems.

- **Message Passing algorithms**

  Software frameworks that automate inference on user-defined probabilistic models are known as Probabilistic Programming Languages (PPLs).
  PPLs emphasize the generality of probabilistic models they can handle, but this means PPLs cannot use inference methods optimal for each model (and usually they rely heavily on the NUTS sampler).
  On the other hand, my software [**gPIE**](https://github.com/sacbow/gpie) is focused on message passing algorithms (such as sum-product, expectation propagation, and variational message passing), which are powerful methods for high-dimensional inference tasks albeit at the cost of generality.
  Among message passing libraries (such as [Infer.NET](https://dotnet.github.io/infer/) and [ForneyLab](https://biaslab.github.io/ForneyLab.jl/stable/)), gPIE is optimized for Bayesian inverse problems arising in scientific imaging, such as ptychography and holography, and supports both CPU and GPU backends.

---

Outside of research, I have also contributed to **The University of Tokyo Newspaper (東京大学新聞社)**,
a student-run newspaper operated by a public-interest foundation.  
I have written several articles on science, technology, and student life in UT.

You can find my past articles here:  
👉 [Hajime Ueda – The University of Tokyo Newspaper](https://www.todaishimbun.org/author/hajimeueda/)
