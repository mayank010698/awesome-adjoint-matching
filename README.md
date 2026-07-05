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

## Unified Algorithm

All methods below instantiate the same generic training loop; they differ only at a handful of decision points (`[D1]`–`[D7]`, referenced in the table underneath).

```
Algorithm  Generic SOC-Sampler Training Loop
Input:      ∇ log p_target, reference diffusion σ(x,t), controller u_θ
Output:     trained controller u_θ  with  Law(X_1) ≈ p_target

1:  repeat
2:      sample source                X_0 ∼ p_0                              [D1]
3:      roll out controlled SDE      {X_t}_{t∈[0,1]}  (on-policy or replay)  [D6]
4:      if ESTIMATOR == direct-backprop:                                    [D3]
5:          L(θ) = E[ ∫ ½‖u_θ(X_t,t)‖² dt − log p_target(X_1) ]             [D2]
6:          differentiate L(θ) through the SDE solver itself
7:      elif ESTIMATOR == matching:                                         [D3]
8:          compute target field v*(X_t,t) by solving a backward
9:              adjoint process / reference bridge                          [D2 fixes what v* is]
10:         L(θ) = E[ ‖u_θ(X_t,t) − v*(X_t,t)‖² ]
11:     θ ← θ − η · optimizer-step(∇_θ L)                                    [D5]
12: until converged                                                         [D4: continuous or discrete X_t]
```

**Decision points**

- **[D1] Source `p_0`** — a fixed simple prior (e.g. Gaussian), an arbitrary "memoryless" reference, or a general (non-memoryless) source distribution.
- **[D2] Underlying objective** — reverse-KL control cost (the `Core Idea` objective above), entropy-regularized optimal transport / Schrödinger-Bridge cost, or a Fisher-divergence variational objective.
- **[D3] Gradient estimator** — differentiate straight through the simulated trajectory, vs. regress `u_θ` onto a target vector field obtained from an adjoint ODE / backward process (avoids storing the simulation graph, so it scales to long horizons).
- **[D4] State space** — continuous `X_t ∈ ℝ^d` vs. discrete/combinatorial state spaces.
- **[D5] Update scheme** — a plain gradient step, a trust-region/proximal-constrained step, or an IPF-style alternation between forward/backward bridges.
- **[D6] Data regime** — fresh on-policy rollouts each iteration vs. an off-policy replay buffer with importance correction.
- **[D7] Domain augmentation** — vanilla, vs. a bias/collective-variable mechanism added on top for a specific application domain (e.g. molecular sampling).

**Where each paper sits**

| Method | D1 Source | D2 Objective | D3 Estimator | D4 State space | D5 Update | D6 Data | D7 Augmentation |
|---|---|---|---|---|---|---|---|
| PIS | simple prior | control cost | direct backprop | continuous | plain SGD | on-policy | — |
| DDS | simple prior | control cost | direct backprop | continuous | plain SGD | on-policy | — |
| CMCD | simple prior | control cost (fwd+bwd controls) | direct backprop | continuous | plain SGD | on-policy | — |
| SOC-Matching | simple prior | control cost | matching (regression target from reference process) | continuous | plain SGD | on-policy | — |
| Adjoint Matching | pretrained diffusion prior | control cost (reward fine-tuning) | matching (adjoint ODE) | continuous | plain SGD | on/off-policy | — |
| Adjoint Sampling | memoryless reference | control cost | matching (adjoint ODE) | continuous | plain SGD | off-policy replay | — |
| Adjoint Schrödinger Bridge Sampler (ASBS) | general (non-memoryless) source | Schrödinger-Bridge cost | matching (adjoint ODE) | continuous | plain SGD | off-policy replay | — |
| Discrete ASBS | general source | Schrödinger-Bridge cost | matching (adjoint, adapted to discrete transitions) | discrete | plain SGD | off-policy replay | — |
| Well-Tempered ASBS | general source | Schrödinger-Bridge cost | matching (adjoint ODE) | continuous | plain SGD | off-policy replay | collective-variable / well-tempered bias for molecular sampling |
| Fisher Adjoint Matching | memoryless reference | Fisher-divergence | matching (Fisher-adjoint target) | continuous | plain SGD | on/off-policy | — |
| Trust Region SOC | simple prior / reference | control cost | matching (adjoint ODE) | continuous | trust-region-constrained step | on-policy | — |
| Proximal Diffusion Neural Sampler | simple prior / reference | control cost | matching (adjoint ODE) | continuous | proximal-point alternation | on-policy | — |

*(This table is a best-effort summary from the abstracts/papers — corrections welcome via PR, especially for D1/D6 cells on the more recent entries.)*

## Contents

- [awesome-adjoint-matching](#awesome-adjoint-matching)
  - [Core Idea](#core-idea)
  - [Unified Algorithm](#unified-algorithm)
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
| Adjoint Schrödinger Bridge Sampler | Guan-Horng Liu et al. (NeurIPS 2025, Oral) | [Paper](https://arxiv.org/abs/2506.22565) | [Code](https://github.com/facebookresearch/adjoint_samplers) |
| Discrete Adjoint Schrödinger Bridge Sampler | Wei Guo et al. | [Paper](https://arxiv.org/abs/2602.08243) | - |
| Well-Tempered Adjoint Schrödinger Bridge Sampler | Juno Nam et al. | [Paper](https://arxiv.org/abs/2510.11923) | [Code](https://github.com/facebookresearch/wt-asbs) |
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
