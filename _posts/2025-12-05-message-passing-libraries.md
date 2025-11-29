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

Probabilistic Programming Languages (PPLs) automate inference on probabilistic models defined in a high-level syntax by its users.
Implementing such a software framework sounds pretty daunting, because no algorithm is universally applicable or scalable to arbitrary inference tasks.
General purpose PPLs, such as [Turing.jl](https://turinglang.org/) and [Numpyro](https://num.pyro.ai/en/stable/), support multiple methods including MCMC and variational inference, to cover a wide range of problems.

This article, however, focuses on the message passing libraries, such as [Infer.NET](https://dotnet.github.io/infer/) and [ForneyLab.jl](https://biaslab.github.io/project/forneylab/).
Although the space of problems that these libraries can handle is much narrower than PPLs, they support scalable inference for high-dimensional models commonly found in time-series analysis and computational imaging.

## 1. Basics
This section overviews the sum-product algorithm and its variants.
To establish the notation, we consider a joint distribution of a set of variables $$\mathcal{V} = \{ X_1, X_2, \cdots, X_N \}$$.
Computing the marginal distribution of $$X_i  (i = 1,2,\cdots, N)$$ by the direct integration over $$ \mathcal{V} - \{X_i\} $$ would come with an exponentially large cost.
To avoid this, we assume that the joint distribution is factorized as

$$
p(X_{\mathcal{V}}) = \frac{1}{Z} \prod_{f \in \mathcal{F}} f(X_f)
$$

where $$X_{\mathcal{V}} = (X_1, X_2, \cdots, X_N )$$, $$\mathcal{F}$$ is a finit set of factors, and $$X_f$$ denotes the collection of variables that constitute the arguments of the factor $$f \in \mathcal{F}$$. $$Z$$ is the normalizing constant, and we want an inference algorithm that can be implemented without the knowledge about $$Z$$.

Such a structure can be expressed as a bipartite graph, where each factor node is connected to the variable nodes that appear in its argument.
This graph is known as the *factor graph*, and specified by the set of variables $$\mathcal{V}$$, the set of factors $$\mathcal{F}$$, and the set of edges $$\mathcal{E} \subset \mathcal{V} \times \mathcal{F}$$:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{F}, \mathcal{E})
$$

![An example of factor graph](/images/factor_graph.jpg)







---
## References

