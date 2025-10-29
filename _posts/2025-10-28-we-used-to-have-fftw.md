---
title: "We used to have fftw..."
excerpt: "A pragmatic introduction to FFT"
date: 2025-10-28
collection: posts
permalink: /blog/we-used-to-have-fft
read_time: true
author_profile: true
tags: [FFT, GPU, High Performance Computing, CuPy, FFTW]
---

This article offers a pragmatic overview of the Fast Fourier Transform (FFT) for developers of scientific software.

## 1. Basics

The Discrete Fourier Transform (DFT) of a one-dimensional vector $$x \in \mathbb{C}^N$$ is another vector $$X \in \mathbb{C}^N$$ defined by

\\[
X_k = \sum_{j=0}^{N-1} x_j \, \omega_N^{jk}, \quad k = 0, 1, \dots, N-1,
\\]

where  
\\[
\omega_N \triangleq e^{-2\pi i / N}.
\\]

Whereas the direct implementation of the DFT requires $$O(N^2)$$ operations, it can be decomposed into smaller DFTs when $$N$$ is a composite number, leading to a faster implementation.  
For example, when $$N$$ is even, the DFT formula can be rewritten as

\\[
X_k = \sum_{j'=0}^{\frac{N}{2}-1} x_{2j'} \, \omega_{\frac{N}{2}}^{j'k} + \omega_N^{k} \sum_{j'=0}^{\frac{N}{2}-1} x_{2j'+1} \, \omega_{\frac{N}{2}}^{j'k}.
\\]

This means that, if you already have the DFTs of the two halves $$\{x_{2j'}\}_{j' = 0,1,\dots,\frac{N}{2}-1}$$ and $$\{x_{2j'+1}\}_{j' = 0,1,\dots,\frac{N}{2}-1},$$ then you can combine them with the cost of $$O(N)$$ operations to obtain the DFT of the whole sequence. If $$N$$ is a power of 2, you can recursively apply this decomposition, and the computational cost of DFT $$T(N)$$ satisfies $$T(N) = 2T(\frac{N}{2}) + O(N)$$ whose solution is $$T(N) = O(N \log N)$$.



