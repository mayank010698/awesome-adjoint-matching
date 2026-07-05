# awesome-soc-samplers

Recently, Stochastic-Control (SOC) based methods have gained popularity for sampling from probability distributions (with access to first-order information, i.e. gradients of the log-probs). This repository is a collection of **SOC-based sampling techniques** — training diffusion/SDE-based samplers via stochastic optimal control, of which adjoint-matching methods are one family. It does not aim to cover generative modeling, image/video diffusion, or controllable-generation applications more broadly (see [Related Lists](#related-lists) for those).

## Core Idea

Assume we have access to an unnormalized target distribution up to first-order information, i.e. $\nabla \log p_{\text{target}}(x)$. We frame sampling as a **stochastic optimal control (SOC)** problem: learn a neural controller $u_\theta$ that steers a reference SDE

$$
dX_t = u_\theta(X_t, t)\, dt + \sigma(X_t, t)\, dB_t, \quad X_0 \sim p_0
$$

so that $\text{Law}(X_1) \approx p_{\text{target}}$, by minimizing the control-cost objective

$$
\min_{\theta} \; \mathbb{E}\left[ \int_0^1 \tfrac{1}{2} \lVert u_\theta(X_t, t) \rVert^2 \, dt \; - \; \log p_{\text{target}}(X_1) \right] ,
$$

i.e. steer as cheaply as possible (small control effort) while landing in high-probability regions of the target at $t=1$. Methods in this list mainly differ in *how* they estimate the gradient of this objective w.r.t. $\theta$ — e.g. adjoint-matching methods compute it via an adjoint ODE instead of differentiating through the SDE simulation directly.

## Contents

- [awesome-adjoint-matching](#awesome-adjoint-matching)
  - [Core Idea](#core-idea)
  - [Contents](#contents)
  - [Papers](#papers)
  - [Benchmarks \& Surveys](#benchmarks--surveys)
  - [Related Lists](#related-lists)
  - [Contributing](#contributing)

## Papers

| Paper Title | Authors | Paper Link | Code |
|-------------|---------|------------|------|
| Path Integral Sampler | Qinsheng Zhang et al. (ICLR 2022) | [Paper](https://arxiv.org/abs/2111.15141) | [Code](https://github.com/qsh-zh/pis) |
| Denoising Diffusion Samplers | Francisco Vargas et al. (ICLR 2023) | [Paper](https://arxiv.org/abs/2302.13834) | [Code](https://github.com/franciscovargas/denoising_diffusion_samplers) |
| Controlled Monte Carlo Diffusions | Francisco Vargas et al. (ICLR 2024) | [Paper](https://arxiv.org/abs/2307.01050) | [Code](https://github.com/shreyaspadhy/CMCD) |
| Stochastic Optimal Control Matching | Carles Domingo-Enrich et al. (NeurIPS 2024) | [Paper](https://arxiv.org/abs/2312.02027) | [Code](https://github.com/facebookresearch/SOC-matching) |
| Adjoint Matching | Carles Domingo-Enrich et al. (ICLR 2025, Spotlight) | [Paper](https://arxiv.org/abs/2409.08861) | [Code](https://github.com/microsoft/soc-fine-tuning-sd) |
| Adjoint Sampling | Aaron Havens et al. (ICML 2025) | [Paper](https://arxiv.org/abs/2504.11713) | [Code](https://github.com/facebookresearch/adjoint_sampling) |
| Fisher Adjoint Matching | Mayank Shrivastava et al. | [Paper](https://mayank010698.github.io/fam.pdf) | [Code](https://github.com/mayank010698/soc_uai) |
| Trust Region SOC | Denis Blessing et al. (NeurIPS 2025, Spotlight) | [Paper](https://arxiv.org/abs/2508.12511) | [Code](https://github.com/DenisBless/TrustRegionSOC) |
| Proximal Diffusion Neural Sampler | Wei Guo et al. (ICLR 2026) | [Paper](https://arxiv.org/abs/2510.03824) | [Code](https://github.com/AlexandreGUO2001/PDNS) |

## Benchmarks & Surveys

- [soc-sampler-bench](https://github.com/mayank010698/soc-sampler-bench) — common benchmark for comparing SOC-based sampling methods on shared targets.

## Related Lists

This list intentionally stays narrow to SOC/adjoint-matching samplers. For broader coverage of diffusion-based sampling and generative modeling, see:

- [awesome-diffusion-samplers](https://github.com/GreatDrake/awesome-diffusion-samplers) — general collection of diffusion sampler papers (PIS, DDS, DIS, GFlowNet-based, etc.)
- [sde_sampler](https://github.com/juliusberner/sde_sampler) — reference implementations of several SDE-based samplers (DIS, Bridge, DDS, PIS)

## Contributing

PRs welcome. To add a paper, add a row to the table with the paper title, authors (first author + et al.), venue (+ award if any), arXiv/PDF link, and code link if public.
