---
title: "We used to have fftw..."
excerpt: "A pragmatic introduction to Fast Fourier Transform."
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

This decomposition is known as the **Cooley–Tukey algorithm** [^cooleytukey]. Even for prime $$N$$, the DFT of $$x \in \mathbb{C}^N$$ can be reduced to DFTs with some factorizable length $$N'$$ by **Rader’s algorithm** [^rader] or **Bluestein’s algorithm** [^bluestein].
Common FFT libraries combine these approaches and dispatch optimal FFT code depending on the shape of input data. For example, the [PocketFFT library](https://github.com/mreineck/pocketfft) — used in both [`numpy.fft`](https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft) and [`scipy.fft`](https://docs.scipy.org/doc/scipy/tutorial/fft.html) — implements the Cooley–Tukey algorithm for composite lengths with small factors (e.g., $$r = 2, 3, 5, 7, 11$$), and switches to **Bluestein’s algorithm** for large prime factors.



---

## References

[^cooleytukey]: Cooley, J. W., & Tukey, J. W. (1965).  
*An Algorithm for the Machine Calculation of Complex Fourier Series.*  
*Mathematics of Computation*, 19(90), 297–301.  
[https://doi.org/10.2307/2003354](https://doi.org/10.2307/2003354)

[^rader]: Rader, C. M. (1968).  
*Discrete Fourier transforms when the number of data samples is prime.*  
*Proceedings of the IEEE*, 56(6), 1107–1108.  
[https://doi.org/10.1109/PROC.1968.6477](https://doi.org/10.1109/PROC.1968.6477)

[^bluestein]: Bluestein, L. I. (1970).  
*A linear filtering approach to the computation of discrete Fourier transform.*  
*IEEE Transactions on Audio and Electroacoustics*, 18(4), 451–455.  
[https://doi.org/10.1109/TAU.1970.1162132](https://doi.org/10.1109/TAU.1970.1162132)





