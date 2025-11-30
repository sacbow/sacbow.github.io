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

where $$X_{\mathcal{V}} = (X_1, X_2, \cdots, X_N )$$, $$\mathcal{F}$$ is a finite set of factors, and $$X_f$$ denotes the collection of variables that constitute the arguments of the factor $$f \in \mathcal{F}$$. $$Z$$ is the normalizing constant, and we want an inference algorithm that can be implemented without the knowledge about $$Z$$.

Such a structure can be expressed as a bipartite graph, where each factor node is connected to the variable nodes that appear in its argument.
This graph is known as the *factor graph*, and specified by the set of variables $$\mathcal{V}$$, the set of factors $$\mathcal{F}$$, and the set of edges $$\mathcal{E} \subset \mathcal{V} \times \mathcal{F}$$:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{F}, \mathcal{E})
$$

A factor graph representing $$p(X_1, X_2, X_3) \propto f_1(X_1) f_2(X_1, X_2, X_3) f_3(X_2) f_4(X_3)$$ is illustrated below.
![An example of factor graph](/images/factor_graph.jpg)

Assuming that a factor graph $$\mathcal{G}$$ has no loops, the marginal distribution of $$X_i$$ is expressed as a product of contributions from smaller subtrees.
To see this, suppose that $$X_i$$ node is associated with three factors $$f_1, f_2$$, and $$f_3$$. We denote the subgraph consisting of nodes that can be reached from $$X_i$$ through $$f_j (j = 1,2,3)$$ by $$\mathcal{G}_j$$. 

![variable update rule](/images/sum_product_variable.jpg)

Then, it follows that $$p(X_i)$$ is simply the product of marginal distributions w.r.t. subtrees $$\mathcal{G}_1, \mathcal{G}_2, \mathcal{G}_3$$:

$$
p(x_i) \propto \prod_{j = 1}^3 M_{f_j \rightarrow X_i} (x_j), \quad M_{f_j \rightarrow X_i} (x_j) \propto \int \prod_{X \in \mathcal{V}_j - \{X_i\}} dX \prod_{f \in \mathcal{F}_j} f(x_f)
$$

where $$\mathcal{G}_j = (\mathcal{V}_j, \mathcal{F}_j, \mathcal{E}_j)$$.
The distribution $$M_{f_j \rightarrow X_i}$$ is referred to as the *message* sent from $$f_j$$ to $$X_i$$. This sounds like a nice divide-and-conquer approach.

Then, can we further decompose the computation of $$M_{f_j \rightarrow X_i} (x_j)$$ into smaller subproblems?
Suppose that the factor nodes $$f_j$$ is associated with $$X_i$$ and other variable nodes $$X_a, X_b, X_c$$, and denote by $$\mathcal{G}_a, \mathcal{G}_b, \mathcal{G}_c$$ the subtree of $$\mathcal{G}_j$$ that can be reached from $$X_a, X_b, X_c$$ without traversing the factor $$f_j$$.

![Factor-to-variable update illustration](/images/sun_product_factor.jpg)




---
## References

