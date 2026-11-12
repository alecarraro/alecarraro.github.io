---
layout: post
title: "Tile-Low-Rank GEMM: Designing Storage Around the Contraction"
image: /assets/images/gsoc-2026/tlr-fold.png
---

<div class="message">
  Over the summer, I worked on extending the library
  <a href="https://github.com/NextLinearAlgebra/NextLA.jl">NextLA.jl</a>
  with support for the Tile Low-Rank (TLR) data format and related operations.
  The project was carried out as part of Google Summer of Code, under the mentorship of
  <a href="https://github.com/rabab53">Rabab Alomairy</a> (MIT, Julia Lab) and
  <a href="https://github.com/QingleiCao">Qinglei Cao</a> (SLU). This post outlines an interesting engineering
  problem I faced while implementing TLR-GEMM.

  The Julia code is available in
  <a href="https://github.com/NextLinearAlgebra/NextLA.jl/pull/20">pull request #20</a>
  and is currently undergoing restructuring.
</div>

1. [Why tile low-rank matrices?](#why-tile-low-rank-matrices)
2. [Multiplying tile by tile](#multiplying-tile-by-tile)
3. [Turning tile loops into GEMM dimensions](#turning-tile-loops-into-gemm-dimensions)
4. [Putting it together: three stages](#putting-it-together-three-stages)
5. [Larger GEMMs need workspace](#larger-gemms-need-workspace)
6. [How close to dense GEMM?](#how-close-to-dense-gemm)
7. [Next steps](#next-steps)

## Why tile low-rank matrices?

### Low-rank structure in dense matrices

Tile low-rank (TLR) is a matrix storage format designed to make large-scale dense linear algebra computations on covariance and kernel matrices computationally tractable. Such matrices arise naturally in scientific computing, including spatial statistics, Gaussian processes, discretised integral operators, and Schur complements of partial differential equation (PDE) discretisations [1, 2].

Many of these matrices exhibit exploitable low-rank structure. When points are ordered into spatial clusters, interactions between well-separated clusters are often smooth and can be accurately approximated by low-rank matrices. Consequently, off-diagonal tiles can be compressed, with the required rank generally decreasing as the separation between clusters increases, while diagonal tiles remain dense.

TLR exploits this with the simplest possible structure: a flat grid of tiles,
each off-diagonal tile compressed independently to a tolerance,
$$A_{ij}\approx U_{ij}V_{ij}^T$$ [1, 3].
A $$b\times b$$ tile of rank $$r$$ costs $$2br$$ numbers instead of $$b^2$$.
Hierarchical formats such as $$\mathcal{H}^2$$ compress further, but TLR trades
some of that compression for a regular structure, with the tile size as a
tuning parameter [1].

<figure>
  <a href="{{ '/assets/images/gsoc-2026/tlr-matrix.svg' | relative_url }}">
    <img src="{{ '/assets/images/gsoc-2026/tlr-matrix.svg' | relative_url }}"
         alt="A 3 by 3 tile grid with ranks 3, 2, 4 / 2, 5, 3 / 4, 3, 2. Each diagonal tile is drawn as a U factor whose width is the tile's rank, times a V-transpose factor whose height is the rank."
         width="650">
  </a>
  <figcaption>The tile grid is regular, while the ranks of the compressed tiles vary.</figcaption>
</figure>

### The GEMM inside TLR Cholesky

Cholesky factorisation is one of the main computational workloads involving TLR matrices, particularly in applications such as Gaussian process modelling and spatial statistics [1, 3]. Much of its computational cost comes from matrix–matrix multiplications (GEMMs), making their efficient implementation critical to performance.

In left-looking TLR Cholesky, each diagonal tile is updated using contributions from all previously factorised tiles to its left:

$$
A_{pp}\leftarrow A_{pp}-\sum_{t<p}L_{pt}L_{pt}^{T},
\qquad L_{pt}=U_{pt}V_{pt}^{T}.
$$

Each term involves multiplying two low-rank tiles and accumulating the result into a dense tile. This is the operation studied in this post: a TLR × TLR-transpose multiplication with a dense output. 
H2OPUS-TLR performs these updates using batched dense GEMMs [1], while KBLAS provides a dedicated dense-output TLR GEMM for this type of operation [4].

This post uses FP32 as the case study. It is the natural target for mixed-precision factorisations [5], and it allows a direct comparison with KBLAS. FP16, which matters for machine learning workloads where low-rank tiles compress weight matrices such as those of language models [6], gets a short note at the end.

## Multiplying tile by tile

Before changing the algorithm, we need notation for the product, a way to reason about its speed, and the algorithm we start from.

### Notation

We compute $$C\leftarrow\alpha AB+\beta C$$ with $$A$$ of size $$m\times k$$,
$$B$$ of size $$k\times n$$, and a dense $$C$$.
Both operands are split into $$b\times b$$ tiles, a $$q_m\times q_k$$ grid for
$$A$$ and a $$q_k\times q_n$$ grid for $$B$$.
Square tiles keep the formulas short; nothing below depends on it.

Every tile is stored as a pair of low-rank factors:

$$
A_{ik}=U_{ik}V_{ik}^{T},\quad U_{ik},V_{ik}\in\mathbb{R}^{b\times r^A_{ik}},
\qquad
B_{kj}=W_{kj}Z_{kj}^{T},\quad W_{kj},Z_{kj}\in\mathbb{R}^{b\times r^B_{kj}}.
$$

We call $$U$$ and $$W$$ the *outer* factors and $$V$$ and $$Z$$ the *inner*
factors.

A dot in place of an index means a sum over it.
For example, $$r^A_{i\cdot}=\sum_k r^A_{ik}$$ is the total rank of tile row
$$i$$ of $$A$$, and $$r^A_{\cdot k}=\sum_i r^A_{ik}$$ that of tile column
$$k$$.
These sums will be the widths of the panels the GEMMs operate on.

Each output tile is a sum of low-rank products,

$$
C_{ij}=\sum_k U_{ik}\,\underbrace{V_{ik}^{T}W_{kj}}_{S_{ikj}}\,Z_{kj}^{T},
\qquad S_{ikj}\in\mathbb{R}^{r^A_{ik}\times r^B_{kj}}.
$$

Evaluated right to left, one triple $$(i,k,j)$$ costs $$2b\,r^A_{ik}r^B_{kj}$$
FLOPs for the coupling $$S_{ikj}$$, the same for $$S_{ikj}Z_{kj}^T$$, and
$$2b^2r^A_{ik}$$ for the final product with $$U_{ik}$$.
With ranks well below $$b$$ the last product dominates: at $$r=b/16$$ it is
89% of the arithmetic.

To reason about speed I use the roofline model [7].
A GEMM of shape $$(M,N,K)$$, which computes an $$M\times N$$ result with a
reduction of length $$K$$ on elements of $$e$$ bytes, has arithmetic intensity

$$
I(M,N,K)=\frac{2MNK}{e\,(MK+KN+MN)}=\frac{2}{e}\Bigl(\frac1M+\frac1N+\frac1K\Bigr)^{-1}.
$$

The intensity is set by the smallest of the three dimensions.
If $$C$$ is also read ($$\beta\neq0$$), the $$MN$$ term counts twice.

A GEMM is bandwidth-bound when $$I$$ is below the ridge point
$$\rho=P/\mathrm{BW}$$, peak throughput $$P$$ divided by memory bandwidth
$$\mathrm{BW}$$.
On the H100, dense FP32 GEMM throughput over the 3.35 TB/s HBM bandwidth gives
$$\rho\approx16$$ FLOP/B.

### The per-tile algorithm in KBLAS and H2OPUS-TLR

Both KBLAS [4] and H2OPUS-TLR [1] compute dense-output TLR GEMM as a sum of low-rank tile products. For each contraction index $$k$$, they multiply the factors of $$A_{ik}$$ and $$B_{kj}$$ and accumulate the contribution into the output tile $$C_{ij}$$. The computation can be expressed as three batched GEMMs per contraction step:

```julia
for k in 1:q_k
    βk = k == 1 ? β : 1
    # batched over all q_m*q_n output tiles (i, j); shapes are (M, N, K)
    S[i, j] = transpose(V[i, k]) * W[k, j]          # (r_A, r_B, b)
    T[i, j] = S[i, j] * transpose(Z[k, j])          # (r_A, b,   r_B)
    C[i, j] = α * U[i, k] * T[i, j] + βk * C[i, j]  # (b,   b,   r_A)
end
```

Here, the GEMMs are batched across the output tiles, with dimensions determined by the individual tile ranks.

### What limits the per-tile algorithm

This formulation has two main performance limitations.

**Arithmetic intensity.** Every GEMM has a tile rank among its dimensions, limiting its arithmetic intensity to approximately $$r/e$$ FLOP/B, independently of the tile size. In FP32, this is below the ridge point for ranks under 64. For example, at $$N=4096$$, $$b=256$$, and $$r=16$$, the dense update has an arithmetic intensity of only 3.8 FLOP/B, roughly a quarter of the ridge point.

**Repeated updates to $$C$$.** The contraction is processed one tile at a time, requiring $$q_k$$ accumulation steps for each output tile. In KBLAS, this means $$q_k$$ reads and writes of $$C$$, along with four kernel calls per step: one for pointer gathering and three for the batched GEMMs. H2OPUS-TLR instead uses parallel accumulation buffers, but retains the same rank-limited tile products.

There is also a historical implementation difference. KBLAS was developed in 2018, when cuBLAS supported only uniform batched GEMM. It therefore uses a fixed rank per operand, $$r_A$$ for all tiles of $$A$$ and $$r_B$$ for all tiles of $$B$$, padding smaller-rank tiles to these values. H2OPUS-TLR uses MAGMA's variable-size batched GEMM to support individual tile ranks, avoiding this padding. Grouped GEMM support was later introduced in cuBLAS in 2024 [8]. These implementation differences do not change the underlying rank-limited tile products, which are the main target of the reorganisation.

## Turning tile loops into GEMM dimensions

The key idea is to change how we traverse the tile iteration space. In the baseline formulation, each contraction tile $$k$$ is processed separately, and the output is accumulated through small, rank-limited GEMMs. But some of these tile iterations can instead become dimensions of larger GEMMs. This changes the granularity of the computation, which operations run independently, and how the reduction is performed. Making this transformation efficient also requires a different memory layout. The factors must be stored so that the tile dimensions we want to fuse are already contiguous in memory. The design therefore combines a reorganisation of the computation with a complementary packing of the low-rank factors.

### Two opportunities for fusion

Consider first the dense update, which accounts for most of the arithmetic and has the lowest intensity in the baseline.

For a fixed output tile row $$i$$, the baseline accumulates $$\sum_k U_{ik}T_{ik}$$ one contraction tile at a time, where $$T_{ik}=[T_{ik1}\;\cdots\;T_{ikq_n}]$$. But the same sum can be expressed as a single matrix multiplication:

$$
\sum_k U_{ik}T_{ik}=
\begin{bmatrix}U_{i1}&\cdots&U_{iq_k}\end{bmatrix}
\begin{bmatrix}T_{i1}\\ \vdots\\ T_{iq_k}\end{bmatrix}.
$$

The contraction index $$k$$, previously traversed by an outer loop, becomes part of the reduction dimension of a GEMM. Instead of $$q_k$$ small updates, we obtain one GEMM of shape $$(b,n,r^A_{i\cdot})$$, where $$r^A_{i\cdot}=\sum_k r^A_{ik}$$. Its reduction dimension is a sum of tile ranks rather than a single rank, and the output is written only once.

<figure>
  <a href="{{ '/assets/images/gsoc-2026/tlr-fold.svg' | relative_url }}">
    <img src="{{ '/assets/images/gsoc-2026/tlr-fold.svg' | relative_url }}"
         alt="Left: the per-tile loop multiplies U11 by T111, U12 by T121 and U13 by T131 as three thin GEMMs, each adding its result into C11. Right: U11, U12 and U13 side by side, times the three T blocks stacked, give C11 in one GEMM whose reduction length is the sum of the three ranks."
         width="900">
  </a>
  <figcaption>
    Folding the loop over \(k\) into the reduction, for output tile \((1,1)\)
    of the tile grid in the first figure (tile row 1 has ranks 3, 2, 4). In the
    algorithm \(T_{1k}\) spans the whole tile row, so the folded GEMM has shape
    \((b,n,r^A_{1\cdot})\).
  </figcaption>
</figure>

The coupling products offer a second opportunity. For each contraction tile $$k$$, the baseline computes $$q_mq_n$$ independent products $$S_{ikj}=V_{ik}^TW_{kj}$$. Stacking the inner factors of $$A$$ along tile column $$k$$ and concatenating the outer factors of $$B$$ along tile row $$k$$ gives

$$
S_k=V_{:,k}^{T}W_{k,:}=
\begin{bmatrix}V_{1k}^{T}\\ \vdots\\ V_{q_mk}^{T}\end{bmatrix}
\begin{bmatrix}W_{k1}&\cdots&W_{kq_n}\end{bmatrix}.
$$

Now the tile indices $$i$$ and $$j$$ become the row and column dimensions of one GEMM. The $$q_mq_n$$ individual coupling products are computed together, with shape $$(r^A_{\cdot k},r^B_{k\cdot},b)$$.

Both transformations preserve the algebraic computation, but change how its iterations are mapped to GEMMs. The remaining question is how to expose these larger matrices without gathering or copying the individual factors.

### Packing factors for the new traversal

The two fusions require complementary storage orders.

The final update needs $$U_{i1},\dots,U_{iq_k}$$ contiguous along each tile row of $$A$$. The coupling GEMM needs $$V_{1k},\dots,V_{q_mk}$$ contiguous along each tile column of $$A$$, together with $$W_{k1},\dots,W_{kq_n}$$ along each tile row of $$B$$.

This gives one packing rule for both operands:

- **Outer factors** $$U$$ and $$W$$ are packed by tile row.
- **Inner factors** $$V$$ and $$Z$$ are packed by tile column.

<figure>
  <a href="{{ '/assets/images/gsoc-2026/data-structure.svg' | relative_url }}">
    <img src="{{ '/assets/images/gsoc-2026/data-structure.svg' | relative_url }}"
         alt="The tile grid of A with tile row 2 and tile column 1 highlighted. The U factors of each tile row are packed side by side into one panel; the V factors of each tile column are stacked into one panel."
         width="520">
  </a>
  <figcaption>
    Outer factors are contiguous along a tile row, inner factors along a tile
    column. Each tile keeps its own rank.
  </figcaption>
</figure>

Each factor is column-major, and each tile occupies exactly its own rank's worth of columns. The two packing orders do not conflict because the fusions use outer and inner factors in complementary directions.

CuTe's shape:stride notation [9, 10] makes the connection between the packed layout and the larger GEMMs precise. For a uniform rank $$r$$, tile row $$i$$ of the outer factors has layout

$$
U_{i,:}:\quad(b,(r,q_k)):(1,(d,dr)),
$$

where $$d$$ is the leading dimension and the nested coordinate $$(a,k)$$ identifies column $$a$$ of tile $$k$$. The inner strides satisfy CuTe's coalescing rule [10],

$$
(r,q_k):(d,dr)\;\longrightarrow\;(rq_k):d,
$$

so the packed tile row is already an ordinary $$b\times rq_k$$ matrix. The tile index and the rank index merge into one GEMM dimension without a copy.

With variable ranks, the regular nested shape no longer applies, but the packed panel remains one matrix:

$$
U_{i,:}:\quad(b,r^A_{i\cdot}):(1,d).
$$

The prefix sum $$p^A_i(k)=\sum_{t<k}r^A_{it}$$ locates tile $$k$$ within the panel. On tensor-core paths, stored ranks are rounded up to multiples of 8 for FP16/BF16 or 4 for TF32, keeping factor offsets 16-byte aligned. The logical ranks remain unchanged.

The important property is that each fused GEMM sees a contiguous matrix operand, even when the original tiles have different ranks.

## Putting it together: three stages

With the factors packed in these complementary orders, the multiplication can be organised into three stages. The first fuses the coupling products, the second computes the remaining tile products directly into the workspace required by the final update, and the third folds the contraction over $$k$$ into a GEMM reduction.

The figure follows one example through all three stages: $$A$$ has the $$3\times3$$ tile grid and ranks of the first figure, and $$B$$ a $$3\times2$$ tile grid. Extents that are ranks are drawn to scale; the tile size $$b$$ is not.

<figure>
  <a href="{{ '/assets/images/gsoc-2026/tlr-pipeline.svg' | relative_url }}">
    <img src="{{ '/assets/images/gsoc-2026/tlr-pipeline.svg' | relative_url }}"
         alt="Three panels. Stage 1: the stacked V-transpose blocks of tile column 2 times the packed W blocks of tile row 2 give all coupling blocks S_i2j in one GEMM. Stage 2: block S121 times Z21-transpose gives T121, which is written into its slot in the workspace T1. Stage 3: the packed panel U1 times T1 gives the output tiles C11 and C12 in one GEMM whose reduction runs over all three contraction tiles."
         width="900">
  </a>
  <figcaption>
    The three stages on one example. Braces give each GEMM's \((M,N,K)\); the
    highlighted block \(S_{121}\) is computed in Stage 1, multiplied in Stage 2
    and written into its slot of \(T_1\), which Stage 3 reads as one matrix.
    Colour follows the role of a block: outer factors orange, inner factors
    teal, coupling blocks amber, workspace violet.
  </figcaption>
</figure>

### Stage 1: Fuse the coupling products

For each contraction tile $$k$$, the packed factors form the coupling GEMM

$$
S_k=V_{:,k}^{T}W_{k,:},
\qquad
(M,N,K)=(r^A_{\cdot k},r^B_{k\cdot},b).
$$

Block $$(i,j)$$ of $$S_k$$ is exactly $$S_{ikj}=V_{ik}^TW_{kj}$$. The tile-row and tile-column iterations have been absorbed into the matrix dimensions, replacing $$q_mq_n$$ small products with one GEMM for each $$k$$.

The $$q_k$$ coupling GEMMs are independent and can be submitted together in one grouped call.

### Stage 2: Produce the final GEMM's input

The intermediate products still operate on individual tile ranks:

$$
T_{ikj}=S_{ikj}Z_{kj}^{T},
\qquad
(M,N,K)=(r^A_{ik},b,r^B_{kj}).
$$

These $$q_mq_kq_n$$ independent products are bucketed by shape and submitted as one grouped GEMM. Unlike Stages 1 and 3, the individual tile indices remain separate GEMM tasks.

The crucial detail is where their results are written. Each output tile row $$i$$ has a workspace matrix $$T_i$$ of size $$r^A_{i\cdot}\times n$$. Task $$(i,k,j)$$ writes directly into the block beginning at row offset $$p^A_i(k)$$ and column offset $$(j-1)b$$.

The row blocks of $$T_i$$ therefore have exactly the same heights as the corresponding column blocks of $$U_{i,:}$$. No separate packing pass is needed between this stage and the final GEMM.

### Stage 3: Fold the contraction into the reduction

The packed outer factors $$U_{i,:}$$ and the workspace $$T_i$$ now form the GEMM from [the first fusion](#two-opportunities-for-fusion):

$$
C_{i,:}\leftarrow\alpha\,U_{i,:}\,T_i+\beta\,C_{i,:},
\qquad
(M,N,K)=(b,n,r^A_{i\cdot}).
$$

The contraction index $$k$$ is no longer an explicit loop around the dense update. It has become part of the GEMM reduction dimension, which now spans the sum of all tile ranks in the row.

Each element of $$C$$ is read, scaled by $$\beta$$, and written exactly once. The three stages can be expressed as:

```julia
for (I, J) in runs  # rectangles of output tiles that fit the workspace
    # Stage 1: fuse coupling products across output tiles
    S[k] = transpose(V[I, k]) * W[k, J]          # all k

    # Stage 2: write intermediate products directly into T
    T[i][k, j] = S[k][i, j] * transpose(Z[k, j]) # all i in I, k, j in J

    # Stage 3: fold k into the GEMM reduction
    C[i, J] = α * U[i, :] * T[i] + β * C[i, J]   # all i in I
end
```

Each stage is submitted as one grouped call per rectangle of output tiles. The choice of rectangles, which bounds the intermediate workspace, is discussed [below](#larger-gemms-need-workspace).

### What changes compared with the baseline?

The new formulation computes the same algebraic product, but reorganises its execution into different GEMM shapes.

| | Tile-wise baseline | NextLA |
|:--|:--|:--|
| Coupling | $$q_mq_n$$ GEMMs per $$k$$ | $$q_k$$ fused GEMMs |
| Product | $$q_mq_kq_n$$ rank-limited GEMMs | Same products, grouped by shape |
| Dense update | $$q_mq_n$$ GEMMs per $$k$$ | $$q_m$$ fused GEMMs |
| Dense-update shape | $$(b,b,r_A)$$ | $$(b,n,r^A_{i\cdot})$$ |
| Calls | Repeated batches for each $$k$$ | 3 grouped calls per run |
| Passes over $$C$$ in KBLAS | $$q_k$$ | 1 |

The largest change is in arithmetic intensity. In Stages 1 and 3, a single tile rank is replaced by a sum of ranks, increasing the dimensions of the GEMMs without increasing the algebraic work. The intensity grows roughly with the number of tiles fused, until the tile size $$b$$ becomes the limiting dimension.

For example, with $$N=4096$$, $$b=256$$, and $$r=16$$, the dense update goes from 3.8 FLOP/B in KBLAS to 62 FLOP/B in Stage 3: from roughly a quarter of the FP32 ridge point to four times it.

Stage 2 remains rank-limited, with arithmetic intensity around $$r/e$$ FLOP/B. Grouping these tasks improves GPU utilisation and reduces launch overhead, but does not change their arithmetic intensity.

<figure>
  <a href="{{ '/assets/images/gsoc-2026/tlr-intensity.svg' | relative_url }}">
    <img src="{{ '/assets/images/gsoc-2026/tlr-intensity.svg' | relative_url }}"
         alt="Dot plot of arithmetic intensity on a log axis from 1 to 128 FLOP/B, with the bandwidth-bound region left of the FP32 ridge at 16 shaded. Coupling: KBLAS 3.9, NextLA Stage 1 43. Tile product: 3.9 in both. Dense update: KBLAS 3.8, NextLA Stage 3 62."
         width="760">
  </a>
  <figcaption>
    Arithmetic intensity of each product in FP32 at \(N=4096\), \(b=256\),
    \(r=16\) and \(\beta=0\). In KBLAS all three products sit near \(r/e\),
    left of the H100 ridge; folding moves Stages 1 and 3 to the compute-bound
    side, while Stage 2 keeps the per-tile intensity.
  </figcaption>
</figure>

### Transposes without repacking

A further consequence of the complementary packing rule is that it also works for transposed operands.

Transposing a tile swaps its factors, $$(A^T)_{ik}=V_{ki}U_{ki}^T$$. The outer factors of $$A^T$$ in tile row $$i$$ are therefore the inner factors of $$A$$ in tile column $$i$$. Those factors are already contiguous under the packing rule.

The transposed operand thus has the same required panel structure without any data movement. All three stages apply to `NN`, `NT`, `TN`, and `TT`.

KBLAS also handles transposes cheaply, by swapping per-tile pointers. The difference here is that transposition preserves the contiguous panels needed for the fused GEMMs.

The Cholesky update from [the first section](#the-gemm-inside-tlr-cholesky) is the `NT` case:

$$
C\leftarrow C-AA^T,
\qquad A=L_{p,1:p-1}.
$$

It has one output tile and $$p-1$$ contraction tiles. Stage 3 becomes a single GEMM of shape $$(b,b,r^A_{1\cdot})$$, whereas the KBLAS formulation performs $$p-1$$ sequential contraction steps with a batch of one.

### Grouping the remaining GEMMs

The iterations that cannot be fused into GEMM dimensions are executed using `cublasGemmGroupedBatchedEx` [8, 11]. Unlike uniform batched GEMM, grouped GEMM supports different matrix shapes within a single call, allowing the independent tasks in each stage to be submitted together while preserving individual tile ranks.

Grouped GEMM is an implementation choice rather than a requirement of the algorithm. The same computation could use separate GEMM calls for Stages 1 and 3 and uniform batches for Stage 2. The main gains come from reorganising the iteration space and packing the factors accordingly: larger GEMM dimensions, fewer passes over $$C$$, and no need to pad every tile to a common rank.

## Larger GEMMs need workspace

The larger GEMMs improve arithmetic intensity, but introduce a new trade-off: intermediate storage. Processing the entire output at once requires materialising the coupling matrices $$S_k$$ after Stage 1 and the intermediate matrices $$T_i$$ after Stage 2. For large matrices, these workspaces can become substantial.

Fusing stages, for example by computing the coupling products and immediately consuming them in a custom kernel, could reduce this intermediate traffic. Here, we retain the three-stage GEMM formulation and instead control memory consumption through two optimisations: **choosing the matrix association order and splitting the output into workspace-bounded rectangles**.

### Choosing the association order

So far, the algorithm has used the right-associative order $$U(SZ^T)$$. But the same product can be evaluated as $$(US)Z^T$$, changing the workspace needed for the intermediate results.

In the left-associative order, Stage 2 computes

$$
T_{ikj}=U_{ik}S_{ikj}
$$

and writes the results into one workspace matrix $$T_j$$ of size $$m\times r^B_{\cdot j}$$ for each output tile column. Stage 3 then becomes

$$
C_{:,j}\leftarrow
\alpha\,T_jZ_{:,j}^{T}+\beta\,C_{:,j},
$$

with one GEMM per tile column. The inner factors $$Z$$ are already packed by tile column, so this association requires no change to the factor layout.

Stage 1 is identical in both orders, and Stage 2 performs the same amount of arithmetic. Only the final GEMM shapes and intermediate workspace differ.

Let $$R^A=\sum_{i,k}r^A_{ik}$$ and $$R^B=\sum_{k,j}r^B_{kj}$$ denote the total ranks of the two operands. The final-stage FLOPs and workspace sizes are

$$
\begin{aligned}
\text{Right:}\quad F_R &= 2bnR^A,
& \lvert T_R\rvert &= nR^A,\\
\text{Left:}\quad F_L &= 2bmR^B,
& \lvert T_L\rvert &= mR^B.
\end{aligned}
$$

The association with fewer final-stage FLOPs also requires less intermediate workspace. We can therefore choose the cheaper order using only the tile-rank metadata, before performing any numerical computation.

### Scheduling within a memory budget

Choosing the association order reduces workspace requirements, but may not be enough when the entire output is large. Instead of computing all of $$C$$ at once, the scheduler partitions it into **rectangles of output tiles** that fit within a prescribed workspace budget.

For each rectangle, the scheduler selects the cheaper feasible association order and executes all three stages over the full contraction dimension. It chooses the widest feasible block of tile columns and packs the longest runs of tile rows that fit the remaining budget.

Crucially, partitioning the output does not partition the reduction. Each rectangle still includes every contraction tile $$k$$, so its output elements are accumulated in one final GEMM and written only once. Rectangles execute sequentially, reusing the same workspace arena.

The rank prefixes, rectangle schedule, and grouped GEMM descriptors are computed once for a given rank structure and reused across products with that structure.

There is one trade-off. The association order is selected per rectangle rather than per output tile. For an individual tile $$(i,j)$$, the final-stage arithmetic would be

$$
F_R^{ij}=2b^2r^A_{i\cdot},
\qquad
F_L^{ij}=2b^2r^B_{\cdot j}.
$$

Some tiles may therefore use an association that is more expensive than their individual optimum. Larger rectangles can increase this extra arithmetic, but also produce wider final GEMMs and improve reuse of the packed factors.

The scheduler deliberately trades some additional FLOPs for larger GEMMs and less repeated data movement, while keeping intermediate storage within the available memory budget.

## How close to dense GEMM?

The reorganisation has to pay off twice: against the per-tile baseline, and against dense GEMM, whose throughput is the ceiling for the GEMMs it is built from.

### Comparison with KBLAS

We first compare NextLA against KBLAS in FP32 on an NVIDIA H100.

The first experiment uses constant ranks, with every tile having rank $$r=b/16$$ and tile size $$b=N/16$$. The dashed line shows **dense-GEMM parity**: the time of dense cuBLAS GEMM scaled by the ratio of executed FLOPs. It represents the time the compressed product would take if it achieved the same throughput as dense GEMM.

At $$N=65{,}536$$, NextLA reaches **92.1% of dense-GEMM parity**. Across the full 20-case constant-rank grid shared with KBLAS, it is faster in 19 cases, with a geometric-mean speedup of 1.44× and a maximum of 2.58×.

The largest advantage appears at small matrix sizes, reaching 2.58× at $$N=4096$$. Here, KBLAS's rank-limited dense updates are bandwidth-bound, while NextLA's larger Stage 3 GEMMs achieve much higher arithmetic intensity.

<figure>
  <a href="{{ '/assets/images/gsoc-2026/tlr-results-kblas.svg' | relative_url }}">
    <img src="{{ '/assets/images/gsoc-2026/tlr-results-kblas.svg' | relative_url }}"
         alt="Log-log plot of FP32 TLR GEMM time against N from 4K to 64K on an H100: NextLA is below KBLAS at every size and approaches the dense-GEMM parity line as N grows"
         width="600">
  </a>
  <figcaption>
    Constant-rank FP32 TLR GEMM on an H100, \(b=N/16\), \(r=b/16\). Dense-GEMM
    parity is dense GEMM time scaled by the executed-FLOP ratio.
  </figcaption>
</figure>

The second experiment considers variable ranks, which KBLAS cannot represent directly without padding. Each matrix has an $$8\times8$$ tile grid, with $$b=N/8$$ and ranks drawn from a skewed distribution in $$[b/32,b/16]$$. KBLAS must pad every tile to rank $$b/16$$, whereas NextLA uses each tile's individual rank.

At $$N=65{,}536$$, NextLA is 20.5× faster than dense GEMM, compared with 12.9× for padded KBLAS, and again reaches approximately 92% of dense-GEMM parity.

### Where the time goes

These results show that reorganising the tile iteration space can bring compressed matrix multiplication close to dense-GEMM throughput. But where does the remaining time go?

The three-stage formulation introduces intermediate matrices in global memory, and not every stage benefits equally from the larger GEMMs. In particular, Stage 2 still consists of individual rank-limited products.

There are therefore two possible sources of overhead: moving intermediate data between stages, and executing GEMMs below their achievable throughput.

The main intermediate is the workspace $$T$$, written by Stage 2 and read by Stage 3. For the right-associative order, it contains $$nR^A$$ elements, where $$R^A$$ is the sum of the ranks of $$A$$.

Let $$P$$ be the measured dense-GEMM throughput and $$\mathrm{BW}$$ the measured memory bandwidth. The time to write and read $$T$$, relative to the ideal compute time of Stage 3, is approximately

$$
\frac{t_T}{t_3^{\mathrm{ideal}}}
=
\frac{2enR^A/\mathrm{BW}}{2bnR^A/P}
=
\frac{e\rho}{b},
\qquad \rho=\frac{P}{\mathrm{BW}}.
$$

The matrix size and total ranks cancel: the relative cost depends only on the tile size and the hardware balance $$\rho$$.

For the large tiles in the first benchmarks, $$b=4096$$ and $$8192$$, the predicted cost is below 2% of Stage 3's ideal compute time. This suggests that eliminating the intermediate workspace alone would recover little of the remaining performance gap.

To check the model, I ported the schedule to C++, added per-stage timers and Nsight ranges, and profiled the computation on an H200 NVL.

For each stage, the roofline time is the larger of its FLOPs divided by measured dense-GEMM throughput and its bytes divided by measured copy bandwidth. The figure below splits each stage's measured time into its roofline time and the time above it: the shorter the pale part, the closer the stage runs to its roof.

<figure>
  <a href="{{ '/assets/images/gsoc-2026/tlr-stage-time.svg' | relative_url }}">
    <img src="{{ '/assets/images/gsoc-2026/tlr-stage-time.svg' | relative_url }}"
         alt="Horizontal stacked bars of FP32 time per stage at N = 65,536 for four configurations. Constant ranks with b = 4096: 868 ms, parity 92%. Skewed ranks with b = 8192: 604 ms, parity 85%. Decaying ranks with b = 1024: 710 ms, parity 78%. Decaying ranks with b = 512: 879 ms, parity 60%. Stage 3 dominates every bar; the time Stage 2 spends above its roof grows from a few milliseconds with large tiles to 88 and 243 ms with small tiles."
         width="880">
  </a>
  <figcaption>
    Measured FP32 time per stage at \(N=65{,}536\) on the H200. The dark part of
    each segment is the stage's roofline time, the pale part the measured time
    above it. The bottom two bars use production tile sizes with ranks that
    decay away from the diagonal, defined below.
  </figcaption>
</figure>

### Large tiles: close to dense-GEMM throughput

The top two bars of the figure use the large tiles of the KBLAS comparison. The constant-rank case reproduces the approximately 92% parity observed on the H100. The skewed-rank case reaches 85% in this separate H200 profiling run.

At these tile sizes, Stages 1 and 3 are large, compute-intensive GEMMs. Stage 2 is also sufficiently large to achieve relatively good throughput, at 75–87% of its roof.

Most of the remaining overhead comes from Stage 3, which accounts for 71–85% of the summed excess time above the individual stage roofline bounds. Although its GEMMs are much larger than the baseline's tile updates, they still have reduction dimensions of a few thousand rather than $$N$$, and achieve 86–94% of the dense-GEMM reference throughput.

The round trip of $$T$$ costs only 0.5–1.1% of Stage 3's time, consistent with the model. At these sizes, fusing away this intermediate would not recover most of the remaining gap.

### Small tiles: Stage 2 becomes the bottleneck

The previous experiments use relatively large tiles. Production TLR applications often use smaller ones: H2OPUS-TLR, for example, uses $$b=1024$$ for 2D and $$b=512$$ for 3D covariance matrices [1].

At these sizes, the predicted cost of materialising $$T$$ increases to approximately 6–12% of Stage 3's ideal time on the H100, or 4.7–9.3% on the H200, whose measured FP32 ridge point is about 12 FLOP/B.

But smaller tiles also mean smaller individual products in Stage 2. To see which effect matters more, I repeated the H200 profiling with ranks that decay away from the diagonal:

$$
r_{ij}=\max\left(\frac{b}{32},
\frac{b}{4(1+\lvert i-j\rvert)}\right).
$$

The bottom two bars of the figure above show that the bottleneck changes substantially. Parity drops to 78% at $$b=1024$$ and 60% at $$b=512$$.

Stage 3 still achieves around 90% of its roofline performance, even with smaller tiles. The larger GEMM formulation continues to work well.

Stage 2, however, falls to just 16–22% of its roof. It executes $$q^3$$ separate tile products, with $$q=N/b$$: 262,144 GEMMs at $$b=1024$$ and over two million at $$b=512$$. Two dimensions of each GEMM are individual tile ranks, averaging only 44 and 20 in these experiments.

Each thread block therefore has relatively little work, and the large number of small products makes Stage 2 inefficient. It now accounts for 57–76% of the summed excess time above the stage roofline bounds.

The intermediate memory traffic also becomes more significant, reaching 4.2–8.3% of measured Stage 3 time. But it remains secondary to the inefficiency of Stage 2.

## Next steps

The profiling reveals the remaining limitation of the current iteration-space reorganisation: **Stages 1 and 3 operate on larger GEMMs, but Stage 2 still operates on individual low-rank tile products.**

The natural next step is to apply the same idea to Stage 2.

### Fuse more iterations into Stage 2

For a fixed pair $$(k,j)$$, all tile-row products share the same inner factor $$Z_{kj}$$:

$$
T_{ikj}=S_{ikj}Z_{kj}^{T}.
$$

Their left operands are already stacked in column block $$j$$ of $$S_k$$. Mathematically, the products can therefore be combined into one GEMM:

$$
\begin{bmatrix}
T_{1kj}\\
\vdots\\
T_{q_mkj}
\end{bmatrix}
=
\begin{bmatrix}
S_{1kj}\\
\vdots\\
S_{q_mkj}
\end{bmatrix}Z_{kj}^{T},
$$

with shape

$$
(M,N,K)=(r^A_{\cdot k},b,r^B_{kj}).
$$

The obstacle is the output layout. Each row block of the result belongs to a different workspace matrix $$T_i$$, so the output cannot be written as one ordinary contiguous GEMM under the current layout.

A custom GEMM epilogue that distributes these row blocks to their corresponding workspaces would remove this restriction. It would reduce the number of Stage 2 products from $$q_mq_kq_n$$ to $$q_kq_n$$, replacing many small GEMMs with fewer, taller ones.

This extends the same principle used throughout the design: move another tile iteration into a GEMM dimension, with the output layout adapted accordingly.

{% comment %}
Fusing Stages 2 and 3, left out of the post. Remove the comment tags to publish it.

### Eliminate the intermediate round trip

The second opportunity is to fuse Stage 2 with Stage 3, producing blocks of $$T$$ in shared memory immediately before they are consumed.

The difficulty is reuse. Each column panel of $$T_i$$ is needed by all thread blocks covering the $$b$$ output rows of tile row $$i$$. A fused implementation would therefore need either to recompute these intermediate values in multiple blocks or share them, potentially using distributed shared memory within a Hopper thread-block cluster.

Such a kernel could eliminate the global-memory round trip of $$T$$, an increasingly useful optimisation as tile sizes decrease. However, the measurements suggest that improving the granularity of Stage 2 is the more important first step.
{% endcomment %}

### A note on FP16

FP16 is the precision that matters for machine learning, where TLR-compressed weights multiply dense activations [6]. On the H100, with every stage submitted as a grouped GEMM, the FP16 product captured only about 40% of the speedup its FLOP count promises, against 92% in FP32.

Per-stage profiling on an H200 traced most of that loss to the kernel behind the grouped call, not to the layout. cuBLAS ran the grouped FP16 GEMMs with an Ampere-generation CUTLASS kernel built on warp-level `mma.sync`, which does not use Hopper's warpgroup MMA. Because Stage 3 is only $$q_m$$ large GEMMs, issuing them one by one with `cublasGemmEx` brought it to the dense-GEMM roof, and FP16 to 89–90% of parity at $$N=65{,}536$$. The rest, a Hopper-native grouped kernel for Stage 2 [12] and the dense-activation case, will be the subject of a follow-up post.

{% comment %}
Dense x TLR for machine learning (FP16 and FP32 results on the H200), kept
for the follow-up post. Remove the comment tags to publish it.

### Dense activations times TLR weights

For machine learning, the case that matters most is dense activations $$X$$
($$m$$ tokens by $$k$$ features) times TLR-compressed weights $$B$$ [6].
With one factored operand the product has two GEMM stages:

$$
P_{kj}=X_kW_{kj},
\qquad
C_{:,j}\leftarrow\alpha
\begin{bmatrix}P_{1j}&\cdots&P_{q_kj}\end{bmatrix}
\begin{bmatrix}Z_{1j}&\cdots&Z_{q_kj}\end{bmatrix}^{T}
+\beta\,C_{:,j}.
$$

The projection is one GEMM per contraction tile, $$X_k$$ times the packed
panel $$W_{k,:}$$, of shape $$(m,\,r^B_{k\cdot},\,b)$$.
The expansion is one GEMM per tile column, of shape
$$(m,\,b,\,r^B_{\cdot j})$$, with the sum over $$k$$ again as its reduction
and the $$Z$$ panel used in place.

The projection writes $$P$$ in tile-row order and the expansion reads it in
tile-column order, the same conflict as in Stage 2.
Here a permutation kernel moves $$P$$ between the stages, at a cost of
$$e\rho/(2b)$$ of the GEMM time.
Projecting tile by tile writes $$P$$ in the right order directly but reads
$$X_k$$ once per tile, and it was slower in all 64 configurations I measured.

I measured the MLP layer of a 70B-class language model, $$8192\to28672$$ (up)
and $$28672\to8192$$ (down), with $$m=16{,}384$$ tokens, $$b=1024$$ and ranks
uniform in $$[r/2,\,3r/2]$$:

| mean rank $$r$$ | FLOP reduction | FP32 speedup | FP16 speedup |
|:--|--:|--:|--:|
| $$b/32$$ | 14.6× | 12.4–13.0× | 9.1–10.2× |
| $$b/16$$ | 7.6× | 7.0–7.1× | 5.6–5.9× |
| $$b/8$$ | 3.9× | 3.7× | 3.0–3.1× |
| $$b/4$$ | 2.0× | 1.9× | 1.6–1.7× |

Ranges cover the up and down projections, and speedups are over dense cuBLAS
GEMM in the same precision on the H200.
FP32, with grouped submission, runs at 85–96% of parity.
FP16, with both stages issued one GEMM at a time, runs at 62–86%; grouped, it
stays at 34–40%, the same kernel effect.
At low rank the permutation takes about 10% of the FP16 time, and the up
projection's expansion, whose reduction is only $$r^B_{\cdot j}\approx256$$
long at $$r=b/32$$, runs at 55% of its roof.
A GEMM epilogue that writes each block of $$P$$ straight to its tile-column
position would remove the permutation, the same fix as for Stage 2.
{% endcomment %}

{% comment %}
Conclusion, left out of the post. Remove the comment tags to publish it.

## Conclusion

The per-tile algorithm builds a TLR product from GEMMs that each have a single tile rank as one of their dimensions, so their arithmetic intensity is limited by the rank rather than the tile size, and it updates the output once per contraction tile. Reorganising the iteration space removes both limits for most of the work. Packing outer factors by tile row and inner factors by tile column turns the contraction over $$k$$ into the reduction dimension of one GEMM per tile row, and the coupling products of each contraction tile into one GEMM, without copies or padding to a common rank. In FP32 on an H100, this brings the product to 92% of dense-GEMM throughput and up to 2.58× faster than KBLAS.

The broader lesson is to choose the storage layout from the contractions an algorithm needs, so that loops over tiles can become GEMM dimensions. Stage 2 is the part of the product still computed tile by tile, and with smaller tiles it becomes the main bottleneck; applying the same idea to it is the next step.
{% endcomment %}

## References

<style>
.tlr-references {
  font-size: 0.85em;
  line-height: 1.35;
}
.tlr-references p {
  margin-bottom: 0.45em;
}
</style>

<div class="tlr-references" markdown="1">

[1] W. Boukaram, S. Zampini, G. Turkiyyah, and D. Keyes.
*H2OPUS-TLR: High Performance Tile Low Rank Symmetric Factorizations Using
Adaptive Randomized Approximation.* 2021.
[arXiv:2108.11932](https://arxiv.org/abs/2108.11932).

[2] W. Boukaram, G. Turkiyyah, and D. Keyes. *Hierarchical Matrix Operations
on GPUs.* SIAM Journal on Scientific Computing 41(4), 2019.

[3] K. Akbudak, H. Ltaief, A. Mikhalev, and D. Keyes. *Tile Low Rank Cholesky
Factorization for Climate/Weather Modeling Applications on Manycore
Architectures.* ISC High Performance, 2017.

[4] A. Charara, D. Keyes, and H. Ltaief. *Tile Low-Rank GEMM Using Batched
Operations on GPUs.* Euro-Par, 2018.
[Paper](https://doi.org/10.1007/978-3-319-96983-1_57),
[KBLAS TLR implementation](https://github.com/ecrc/kblas-gpu/blob/master/src/blas_tlr/Xgemm_tlr_core.cuh).

[5] Q. Cao, S. Abdulah, R. Alomairy, Y. Pei, P. Nag, G. Bosilca, J. Dongarra,
M. G. Genton, D. E. Keyes, H. Ltaief, and Y. Sun. *Reshaping Geostatistical
Modeling and Prediction for Extreme-Scale Environmental Applications.* SC22, 2022.
[Paper](https://doi.org/10.1109/SC41404.2022.00007).

[6] P.-H. Chen, S. Si, Y. Li, C. Chelba, and C.-J. Hsieh. *GroupReduce:
Block-Wise Low-Rank Approximation for Neural Language Model Shrinking.*
NeurIPS, 2018.

[7] S. Williams, A. Waterman, and D. Patterson. *Roofline: An Insightful
Visual Performance Model for Multicore Architectures.* Communications of the
ACM 52(4), 2009.

[8] NVIDIA. *Introducing Grouped GEMM APIs in cuBLAS and More Performance Updates.* 2024.
[Technical blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/).

[9] Cris Cecka. *CuTe Layout Representation and Algebra.* 2026.
[arXiv:2603.02298](https://arxiv.org/pdf/2603.02298).

[10] NVIDIA. *CuTe Layouts* and *CuTe Layout Algebra*.
[Layout definitions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html),
[layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html).

[11] NVIDIA. *cuBLAS API Reference.*
[Uniform batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex),
[grouped GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex).

[12] NVIDIA. *CUTLASS example 57: Hopper grouped GEMM.*
[Source](https://github.com/NVIDIA/cutlass/tree/main/examples/57_hopper_grouped_gemm).

</div>
