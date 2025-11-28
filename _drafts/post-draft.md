---
title: "Implementing Message Passing Algorithms"
excerpt: "An overview on message passing algorithms and their software abstractions in graph-based probabilistic programming libraries."
date: 2025-12-05
collection: posts
permalink: /blog/graph-based-ppls
read_time: true
author_profile: true
tags: [Message Passing, Belief Propagation, Variational Bayes, Probabilistic Programming Languages]
---

Probabilistic Programming Languages (PPLs) [^vandemeent] are software frameworks where users describe probabilistic models in a high-level syntax.
Once a model is specified, the system automatically derives inference algorithms to sample from the distribution or compute expectations.
Implementing such a framework sounds daunting, because no single inference algorithm is universally applicable to arbitrary probability distributions.
For this reason, general-purpose PPLs such as Edward, TensorFlow Probability, Pyro, NumPyro, and Turing provide multiple inference backends—typically including MCMC and variational methods (although in practice users tend to rely on NUTS for many applications.)

The scope of this article, however, is limited to PPLs based on Message Passing algorithms. 




---

## References
[^vandemeent]: Jan-Willem van de Meent, Brooks Paige, Hongseok Yang, and Frank Wood (2021).  
*An Introduction to Probabilistic Programming.*  
arXiv:1809.10756 [stat.ML].  
[https://arxiv.org/abs/1809.10756](https://arxiv.org/abs/1809.10756)
