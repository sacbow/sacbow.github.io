---
title: "Implementing Message Passing Algorithms"
excerpt: "An overview on message passing algorithms and their software abstractions in graph-based probabilistic programming libraries."
date: 2025-12-05
collection: posts
permalink: /blog/message-passing-libraries
read_time: true
author_profile: true
tags: [Message Passing, Belief Propagation, Variational Bayes, Probabilistic Programming Languages]
---

Probabilistic Programming Languages (PPLs) [^vandemeent] automates inference on probabilistic models defined in a high-level syntax by its users.
Implementing such a software framework sounds pretty daunting, because no algorithm is universally applicable or scalable to arbitrary inference tasks.
General purpose PPLs, such as Turing.jl and Numpyro, support multiple methods including Markov Chain Monte Carlo (MCMC) and variational inference.
On the other hand, the message passing libraries (such as Infer.NET and ForneyLab.jl) provide powerful algorithms for extremely high-dimensional problems that the NUTS sampler cannot handle.
Although the space of problems that can be expressed in those Domain Specific Languages (DSL) is much smaller than that of PPLs, such a tool can be useful in applications such as time series analysis and probabilistic image restoration.










---

## References

