---
layout: post
title: Parametric Reachability Analysis in Julia
image: /assets/images/gsoc-2025/cover.png
redirect_from:
  - /alecarraro-blog/2025/08/21/gsoc-project-with-juliareach/
---

<div class="message">
  Hey! This post summarizes my GSoC 2025 work with the 
  <a href="https://juliareach.github.io/">JuliaReach</a>
  organization on ReachabilityAnalysis.jl under the guidance of mentors 
  <a href="https://github.com/schillic">Christian Schilling</a> and 
  <a href="https://github.com/mforets">Marcelo Forets</a>.
  The algorithm implemented in this project was later used in the ARCH-COMP 2026
  friendly competition; see the
  <a href="https://easychair.org/publications/paper/hFjB/open">ARCH-COMP26 category report</a>.
</div>

## Table of Contents
1. [Outline and Motivation](#outline-and-motivation)
2. [Example](#example)
3. [Large Uncertainty](#large-uncertainty)
4. [References](#references)
5. [Appendix](#appendix)

## Outline and Motivation

In many safety-critical systems, it is often impossible to predict exactly how the system will behave due to uncertainties in initial states, inputs, or dynamics. Reachability analysis addresses this challenge by computing *flowpipes* that enclose all possible behaviors over time, allowing engineers to rigorously verify safety.

For example, imagine an autonomous car faced with a pedestrian suddenly stepping onto the road. Depending on its exact speed, distance to the pedestrian, and reaction delay, the car may need to brake immediately or steer around the obstacle, while also accounting for sensor errors. One could try to sample a range of initial speeds and positions and simulate the resulting trajectories to guide the decision. However, because the car’s dynamics are nonlinear, even small variations in initial conditions can lead to drastically different behaviors. Simulation alone therefore cannot provide the rigorous guarantees required for safety.

In the [ReachabilityAnalysis.jl](https://github.com/JuliaReach/ReachabilityAnalysis.jl) package, several algorithms are available to compute these enclosures for different types of dynamical systems with uncertain inputs and initial states, including hybrid systems.  

During my project, I focused on a fundamental class of models: linear time-invariant (LTI) systems with uncertain parameters.  
An LTI system describes the evolution of the state vector $$x(t)$$ under a fixed matrix $$A$$:

$$x' = A x$$

where $$x \in \mathbb{R}^n$$. 

Such systems are widely used because they arise naturally when linearizing nonlinear models. In practice, however, the dynamics are rarely known exactly. Parameters in $$A$$ may be uncertain due to modeling errors, noisy measurements, and other sources of uncertainty. To capture this, we consider a parametric LTI system of the form:  

$$x' = A x, \quad x(0) \in X_0, \quad A \in \mathcal{A}$$

Here:  
- $$X_0$$ is the set of possible initial states, reflecting uncertainty at $$t=t_0$$.
- $$\mathcal{A}$$ is a set of possible system matrices, representing parameter uncertainty.

The goal is to compute a tight over-approximation of all trajectories that can arise from *any* choice of initial state in $$X_0$$ and *any* admissible dynamics in $$\mathcal{A}$$. This is significantly more challenging than the classical case with a fixed $$A$$, since the solution space must account for both state uncertainty and dynamics uncertainty simultaneously.

Huang et al. [1] introduced a dedicated algorithm for these parametric LTI systems with uncertain dynamics, which combines two modern set representations: matrix zonotopes for modeling uncertainty in the system matrix and sparse polynomial zonotopes (SPZs) for representing the reachable states.  
The motivation for this choice becomes clear when we compare possible uncertainty models for $$A$$ and discuss how to best enclose the evolving states.

---

### Why Matrix Zonotopes Instead of Interval Matrices?  

A classical way to represent uncertainty in $$A$$ is through an interval matrix:

$$ \mathcal{A} = \{ A \mid a_{ij} \in [\underline{a}_{ij}, \overline{a}_{ij}] \} $$

This representation assumes that each entry of the matrix can vary independently within its interval. While simple and well studied, this independence assumption can be overly conservative: it includes combinations of parameters that may never actually occur in practice, often resulting in loose over-approximations of the reachable set.  

To address this, Althoff [2] introduced matrix zonotopes, which generalize standard zonotopes to matrices:

$$ \mathcal{Z} = \{ A_0 + \sum_{i=1}^p A_i \alpha_i \;\mid\; \alpha_i \in [-1,1] \} $$

Here, $$A_0$$ is the center matrix, and the $$A_i$$ are generator matrices. Matrix zonotopes preserve correlations between matrix entries, unlike interval matrices that treat them as independent. This leads to significantly tighter enclosures for uncertain dynamics, making them a more accurate and efficient choice in reachability analysis.

---

### Why Sparse Polynomial Zonotopes?  

Once the dynamics are modeled with matrix zonotopes, the reachable states need a representation that can handle the resulting complexity without becoming overly conservative.  
Huang et al. use sparse polynomial zonotopes for this task.

SPZs extend zonotopes with polynomial terms, allowing them to capture non-convex sets while remaining closed under Minkowski sums and linear maps: two core operations in many reachability algorithms.

For a more detailed description of SPZs, see [3]. For a friendly introduction, see Luca’s previous GSoC [blog post](https://www.lucaferranti.com/posts/2022/09/gsoc22/).

## Example

Now let’s test the algorithm with a simple 2D dynamical system:

$$
x' = A x, \quad x(0) \in X_0, \quad 
A = \begin{bmatrix} -1 & -5 \\ -1 & -1 \end{bmatrix}.
$$

In this case the origin is a *stable focus*, since the eigenvalues are  

$$
\lambda = -1 \pm i\sqrt{5},
$$  

so given an initial set $$X_0$$ the trajectories should rotate and shrink toward the origin.  

We then compare two cases:  
1. the dynamics are known exactly,  
2. the dynamics are uncertain.  

For the uncertain case, assume that we are unsure about one of the entries of $$A$$:

$$
A \in \begin{bmatrix} [-1.1, -0.9]  & -5 \\ 1 & -1 \end{bmatrix}.
$$

This can be expressed as a matrix zonotope:

$$
\mathcal{A} = 
\begin{bmatrix} -1 & -5 \\ 1 & -1 \end{bmatrix}
+ c \begin{bmatrix} 0.1 & 0 \\ 0 & 0 \end{bmatrix},
\quad c \in [-1,1].
$$

Now we will set up the reachability problem using `LazySets` and `ReachabilityAnalysis`:

### Step 1: Import the packages

```julia
using ReachabilityAnalysis
import IntervalMatrices   # required for internal calculations
using Plots
```

### Step 2: Define the system matrices

```julia
# Nominal matrix
A0 = [-1.0 -5.0;
       1.0 -1.0]

# Matrix zonotope with one generator
AS = MatrixZonotope(A0, [N[0.1 0.0; 0.0 0.0]])
```

### Step 3: Define the initial set

We initialize the initial conditions as a sparse polynomial zonotope:

```julia
X0 = SparsePolynomialZonotope(
    [1.0, 1.0], # center 
    0.1 * [2.0 0.0 1.0; 1.0 2.0 1.0], # independent generators
    0.1 * reshape([1.0, 0.5], 2, 1), # dependent generators
    [1 0 1; 0 1 3] # exponent
)
```

### Step 4: Define the IVP
We can use the convenient macro `@ivp` to specify the initial states and uncertain dynamics:

```julia
prob1 = @ivp(x' = A * x, x(0) ∈ X0, A ∈ AS)
```

### Step 5: Configure the reachability algorithm

We use the `HLBS25` algorithm for systems with uncertain parameters:

```julia
# Time step and horizon
δ = 3 / 200
T = 3.0

# Algorithm setup
alg = HLBS25(
    δ = δ,
    approx_model = CorrectionHullMatrixZonotope(),
    max_order = 5,
    taylor_order = 5,
    reduction_method = LazySets.GIR05(),
    recursive = false,
)
```

<div class="message">
  <p><strong>Main parameters of <code>HLBS25</code>:</strong></p>
  <ul>
    <li><strong>δ</strong> – time step</li>
    <li><strong>approx_model</strong> – discretization model (default: <code>CorrectionHullMatrixZonotope</code>)</li>
    <li><strong>max_order</strong> – maximum order of the SPZ (trade-off between accuracy and complexity)</li>
    <li><strong>taylor_order</strong> – truncation order of the Taylor expansion of the exponential</li>
    <li><strong>reduction_method</strong> – strategy to reduce the complexity of matrix zonotopes</li>
    <li><strong>recursive</strong> – whether to compute the exponential recursively or directly</li>
  </ul>
  <p>
    For a full description of these options, refer to the
    <a href="https://github.com/JuliaReach/ReachabilityAnalysis.jl">documentation</a>.
  </p>
</div>


### Step 6: Solve and visualize

```julia
# Solve
sol1 = solve(prob1, alg; T = T)

# Plot
p = plot(X0, title="Reachable Set")
for X in sol1
  plot!(set(X), vars=(1,2))
end
```

This produces the flowpipe, showing how trajectories evolve while accounting for both the initial uncertainty and the uncertainty in the dynamics.

To compare with the exact case, we can repeat the experiment with a matrix zonotope that has no generators, which corresponds to a fixed matrix $$A$$. The fully worked-out code can be found at the end of this page in the Appendix.

<figure>
  <img src="{{ '/assets/images/gsoc-2025/hlbs25.png' | relative_url }}" alt="Reachability plot" width="500">
  <figcaption>Figure 1: Reachable set over time for exact and uncertain dynamics. Every third reach set is plotted.</figcaption>
</figure>

---

## Large Uncertainty
For small uncertainty the flowpipe is tight, but with larger uncertainty the over-approximation becomes much more conservative. For example, if we choose a matrix zonotope with large uncertainty such as:

$$
\mathcal{A} = 
\begin{bmatrix} -1 & -5 \\ -1 & -1 \end{bmatrix}
+ c \begin{bmatrix} 0.1 & 0.1 \\ 0.1 & 0.1 \end{bmatrix},
\quad c \in [-1,1].
$$

the reach sets become much wider:

<figure>
  <img src="{{ '/assets/images/gsoc-2025/all-generators-comparison.png' | relative_url }}" alt="Reachability plot" width="500">
  <figcaption>Figure 2: Reachable set over time for exact and uncertain dynamics. Every third reach set is plotted.</figcaption>
</figure>

The reach sets bloat quickly, growing faster than they contract around the fixed point. Increasing the maximum order parameter mitigates the bloating somewhat.

### Why do the sets bloat?
Together with my mentors, I investigated this behavior for several weeks. At first it looked like a bug, but it turned out to be a limitation of the algorithm itself rather than of our implementation: the result is a correct, if conservative, over-approximation.

The bloating comes from the repeated use of order reduction. At each step the algorithm computes and applies the matrix zonotope exponential $$e^{\mathcal{A}}$$ to propagate the state set forward in time. This increases the number of generators in the SPZ, so order reduction is needed to keep the complexity manageable. With large uncertainty in $$\mathcal{A}$$, each reduction discards generators that capture important information, while the remaining ones grow under successive exponentials, and the over-approximation error accumulates over time. This is also why a larger `max_order` helps: fewer generators are discarded at each step.

The algorithm is still very new and, before our work, had only been implemented in the library [CORA](https://tumcps.github.io/CORA/).


### Update (March 2026)

Since GSoC, the work has been completed: the non-homogeneous case $$x' = \mathcal{A}x + \mathcal{B}u$$ is now implemented as well, and both versions of the algorithm are merged into the [ReachabilityAnalysis.jl](https://github.com/JuliaReach/ReachabilityAnalysis.jl) codebase.

{% comment %}
Contributions table, kept for the GSoC evaluation; hidden from the post.

---

## Contributions Overview
In the tables below I summarize my contributions to the different packages in the `JuliaReach` ecosystem to implement the reachability algorithm described above.

### 1. `LazySets.jl`: Matrix Zonotopes  

| PR Title | PR # | Category |
|----------|------|----------|
| Add `overapproximate` for matrix zonotope exponential | [#4000](https://github.com/JuliaReach/LazySets.jl/pull/4000) | Feature |
| Add operations on matrix zonotopes | [#3999](https://github.com/JuliaReach/LazySets.jl/pull/3999) | Feature |
| Add `overapproximate` for matrix zonotope multiplication | [#3996](https://github.com/JuliaReach/LazySets.jl/pull/3996) | Feature |
| Preserve `indexvector` in linear map for MZ | [#3985](https://github.com/JuliaReach/LazySets.jl/pull/3985) | Fix |
| Extend `ExponentialMap` to support `MatrixZonotope`s | [#3970](https://github.com/JuliaReach/LazySets.jl/pull/3970) | Feature |
| Add `linear_map` between MatrixZonotope and SPZ | [#3969](https://github.com/JuliaReach/LazySets.jl/pull/3969) | Feature |
| Fix in-place scaling for `MatrixZonotope` | [#3967](https://github.com/JuliaReach/LazySets.jl/pull/3967) | Fix |
| Add methods and type for `MatrixSets` | [#3966](https://github.com/JuliaReach/LazySets.jl/pull/3966) | Feature |
| Add `norm` and `overapproximate_norm` for matrix zonotope | [#3941](https://github.com/JuliaReach/LazySets.jl/pull/3941) | Feature |
| Refactor and extend `MatrixSets` module | [#3933](https://github.com/JuliaReach/LazySets.jl/pull/3933) | Refactor |

---

### 2. `LazySets.jl`: General improvements and fixes  

| PR Title | PR # | Category |
|----------|------|----------|
| Optimize `remove_redundant_generators` | [#3998](https://github.com/JuliaReach/LazySets.jl/pull/3998) | Feature |
| Improve `remove_redundant_generators` for `SPZ` | [#3986](https://github.com/JuliaReach/LazySets.jl/pull/3986) | Improvement |
| Add `overapproximate` for exponential map for `SPZ` and `Zonotope` | [#3979](https://github.com/JuliaReach/LazySets.jl/pull/3979) | Feature |
| Fix `isuniversal` and constructor for `Polygon` | [#3978](https://github.com/JuliaReach/LazySets.jl/pull/3978) | Fix |
| Preserve ID in `minkowski_sum` and `cartesian_product` between `SPZ` and `Zonotope` | [#3977](https://github.com/JuliaReach/LazySets.jl/pull/3977) | Improvement |
| Make `overapproximate` of ASPZ with `UnionSetArray{Zonotope}` more robust | [#3972](https://github.com/JuliaReach/LazySets.jl/pull/3972) | Improvement |
| Add overapproximation of the l1 norm of a `Zonotope` | [#3925](https://github.com/JuliaReach/LazySets.jl/pull/3925) | Feature |
| Faster l1 norm for `AbstractZonotope` | [#3924](https://github.com/JuliaReach/LazySets.jl/pull/3924) | Improvement |
| Add `scale` for `SPZ` | [#3908](https://github.com/JuliaReach/LazySets.jl/pull/3908) | Feature |
| Add `merge_id` and generalize `exact_sum` for `SPZ` | [#3905](https://github.com/JuliaReach/LazySets.jl/pull/3905) | Feature |
| Add sampler for sparse polynomial zonotopes | [#3847](https://github.com/JuliaReach/LazySets.jl/pull/3847) | Feature |

---

### 3. `MathematicalSystems.jl` and `ReachabilityAnalysis.jl`

| PR Title | PR # | Notes |
|----------|------|-------|
| Add `parametric` systems | [#332](https://github.com/JuliaReach/MathematicalSystems.jl/pull/332) | Feature |
| Add HLBS25 algorithm for linear parametric systems | [#931](https://github.com/JuliaReach/ReachabilityAnalysis.jl/pull/931) | Feature |
{% endcomment %}

---

## References
[1] Yushen Huang, Ertai Luo, Stanley Bak, Yifan Sun, *Reachability analysis for linear systems with uncertain parameters using polynomial zonotopes*, Nonlinear Analysis: Hybrid Systems, Volume 56, 2025, 101571, ISSN 1751-570X. [DOI](https://doi.org/10.1016/j.nahs.2024.101571)

[2] Matthias Althoff, *Reachability Analysis and its Application to the Safety Assessment of Autonomous Cars*, PhD thesis, Technische Universität München, July 2010.

[3] Niklas Kochdumper, Matthias Althoff, *Sparse Polynomial Zonotopes: A Novel Set Representation for Reachability Analysis*, IEEE Transactions on Automatic Control, Vol. 66, No. 9, Sept. 2021, pp. 4043–4058. [DOI](https://doi.org/10.1109/TAC.2020.3024348)

## Appendix 
```julia
using ReachabilityAnalysis
import IntervalMatrices
using Plots
N = Float64

A0 = N[-1 -5;
        1 -1]
AS = MatrixZonotope(A0, [zeros(2,2)]) 

# Initial set
X0 = SparsePolynomialZonotope(
    N[1.0, 1.0],
    N(0.1) * N[2.0 0.0 1.0; 1.0 2.0 1.0],
    N(0.1) * reshape(N[1.0, 0.5], 2, 1),
    [1 0 1; 0 1 3]
)

# Two problems: no variation vs. small generator variation
prob1 = @ivp(x' = A * x, x(0) ∈ X0, A ∈ AS)

AS = MatrixZonotope(A0, [N[0.1 0.1; 0.1 0.1]])
prob2 = @ivp(x' = A * x, x(0) ∈ X0, A ∈ AS)

δ = N(3) / 200
T = 3.0

alg = HLBS25(
    δ = δ,
    approx_model = CorrectionHullMatrixZonotope(),
    max_order = 5,
    taylor_order = 5,
    reduction_method = LazySets.GIR05(),
    recursive = false,
)

sol1 = solve(prob1, alg; T = T)
sol2 = solve(prob2, alg; T = T)

# plotting function
function plot_flowpipe(sets, X0; kwargs...)
    p = plot(; kwargs...)
    for (i,s) in enumerate(sets)
        if i%3 ==0
            plot!(p, set(s); vars=(1,2), alpha=0.3, lw=0.5)
        end
    end
    return p
end

p1 = plot_flowpipe(sol1, X0,
    title = "Flowpipe: exact A",
    xlabel = "x₁", ylabel = "x₂",
    legend = true)

p2 = plot_flowpipe(sol2, X0, 
    title = "Flowpipe: uncertain A ",
    xlabel = "x₁", ylabel = "x₂",
    legend = true)

# common limits for better comparison
xlims_common = (min(xlims(p1)..., xlims(p2)...), max(xlims(p1)..., xlims(p2)...))
ylims_common = (min(ylims(p1)..., ylims(p2)...), max(ylims(p1)..., ylims(p2)...))
plot!(p1; xlims=xlims_common, ylims=ylims_common)
plot!(p2; xlims=xlims_common, ylims=ylims_common)

plot(p1, p2, layout=(1, 2), size=(1000, 420))
savefig("reach_comp.png")

```
