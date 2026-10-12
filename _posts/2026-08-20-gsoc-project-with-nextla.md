---
layout: post
title: "Tile-Low-Rank GEMM: Designing Storage Around the Contraction"
image: /assets/images/gsoc-2026/tlr-matrix-1.png
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

Hierarchical matrices exploit low-rank structure in large dense matrices to
cut the storage and arithmetic of matrix products. They are widely used in
scientific computing [1, 2], and similar blockwise low-rank compression has
been applied to neural networks [3]. But fewer FLOPs do not automatically mean a
faster GPU computation. Nested formats such as $$\mathcal{H}^2$$ and HSS
produce many small operations of different shapes, which struggle to provide
the regular data movement and operand reuse GPUs need. Tile-Low-Rank (TLR)
takes a more regular approach: a flat grid of tiles, each stored as a pair of
low-rank factors whose dense dimension is fixed and only the rank varies
[1, 4]. It gives up some compression for a structure that maps onto GEMM.

TLR still leaves one source of irregularity: every tile has its own rank. A
direct implementation therefore turns one large GEMM into many small GEMMs of
many different shapes. The usual fix is to pad every tile to the largest
rank, which restores regularity by spending arithmetic and memory on zeros.

This post describes a third option. I chose a storage layout for the low-rank
factors so that the tile indices of the product can become row, column, or
reduction dimensions of a few large GEMMs, without copying the factors and
without padding tiles to a common rank. What cannot be fused is submitted as
grouped GEMM, and rank metadata decides the contraction order and how the work
is split under a workspace budget. On an H100 the result runs at 92% of a
dense-GEMM-derived throughput reference and up to 2.58× faster than KBLAS.

I use CuTe's shape:stride notation [5, 6] to describe the layouts; the
implementation itself uses packed offsets and cuBLAS. The post covers
$$C\leftarrow\alpha\operatorname{op}(A)\operatorname{op}(B)+\beta C$$ with a
dense $$C$$ and one or both operands in TLR format.

1. [Many small GEMMs](#many-small-gemms)
2. [Packing for the contraction](#packing-for-the-contraction)
3. [One product, three stages](#one-product-three-stages)
4. [Where fusion stops](#where-fusion-stops)
5. [Association order and workspace](#association-order-and-workspace)
6. [Results](#results)
7. [Takeaways](#takeaways)

## Many small GEMMs

TLR partitions a matrix into a regular grid and compresses each tile
independently [1, 4]:

$$
A_{ij}\approx U_{ij}V_{ij}^{T},
\qquad
U_{ij}\in\mathbb{R}^{m_i\times r_{ij}},
\quad
V_{ij}\in\mathbb{R}^{n_j\times r_{ij}}.
$$

The grid is regular, but the rank $$r_{ij}$$ varies from tile to tile with the
local compressibility of the matrix.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr-matrix-1.png' | relative_url }}"
       alt="A Tile-Low-Rank matrix whose tiles have different ranks"
       width="900">
  <figcaption>The tile grid is regular, while the ranks of the compressed tiles vary.</figcaption>
</figure>

With $$A_{ik}=U_{ik}V_{ik}^{T}$$ and $$B_{kj}=W_{kj}Z_{kj}^{T}$$, each output
tile is a sum of low-rank products:

$$
C_{ij}=\sum_k
U_{ik}\underbrace{\left(V_{ik}^{T}W_{kj}\right)}_{S_{ikj}}Z_{kj}^{T}.
$$

KBLAS [7] evaluates this one contraction tile at a time, with three batched
GEMMs over all output tiles per step:

```julia
C .= 0
for k in 1:q_k
    # Each line is batched over output tile pairs (i, j).
    S[i, j] = transpose(V[i, k]) * W[k, j]  # coupling
    T[i, j] = S[i, j] * transpose(Z[k, j])  # right product
    C[i, j] += U[i, k] * T[i, j]           # dense update
end
```

This has two costs. Every product is small, with dimensions set by individual
tile ranks, and batched GEMM needs one shape per batch, so ranks are padded to
a common value. And the loop makes $$q_k$$ passes over the dense output
$$C$$, each reading and writing it again.

Neither cost is inherent to the mathematics. The sum over $$k$$ is a
reduction, and GEMM is very good at reductions, provided the operands are laid
out so that $$k$$ can be one of its dimensions.

## Packing for the contraction

The central design decision is to store the two factor families in
*complementary* tile orders:

- Outer factors $$U$$, $$W$$ are packed by tile row.
- Inner factors $$V$$, $$Z$$ are packed by tile column.

Each factor is column-major, and each tile occupies exactly its own rank's
worth of columns. The tile order decides which factors sit next to each other
in memory, and therefore which multi-tile panels can be addressed as ordinary
matrices.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/data-structure.png' | relative_url }}"
       alt="Complementary packed storage for U and V factors"
       width="800">
  <figcaption>
    Outer factors are contiguous along a tile row, inner factors along a tile
    column. Each tile keeps its own rank.
  </figcaption>
</figure>

To see why this matters, take uniform rank $$r$$ and $$q_k$$ contraction
tiles. A packed row of outer factors has the layout

$$
U_i:\quad(m_i,(r,q_k)):(1,(d_U,d_Ur)),
$$

where $$d_U$$ is the leading dimension and the nested coordinate $$(a,k)$$
means "rank column $$a$$ of tile $$k$$". The two inner strides satisfy CuTe's
coalescing rule [6],

$$
(r,q_k):(d_U,d_Ur)\;\longrightarrow\;(rq_k):d_U,
$$

so the whole row is an ordinary $$m_i\times rq_k$$ matrix. The tile index
$$k$$ has merged with the rank index into a single GEMM dimension, with no
copy. The column-packed inner factors give the same result in the other
direction: $$V_{:,k}$$ stacks the tiles of contraction column $$k$$.

A nested shape $$(r,q_k)$$ assumes every tile has the same rank. For arbitrary
ranks the panel is still one matrix, $$U_i:(m_i,\rho_i):(1,d_U)$$ with
$$\rho_i=\sum_k r_{ik}$$, and a prefix sum $$p_i(k)=\sum_{t<k}r_{it}$$ locates
tile $$k$$ inside it.

On tensor-core paths each stored rank is rounded up to a multiple of 8 (FP16,
BF16) or 4 (TF32) elements, so that every factor offset stays 16-byte aligned
and every panel remains a valid view; the logical rank is unchanged.

## One product, three stages

Here is the same idea on a concrete product with variable ranks. $$A$$ has a
$$2\times3$$ tile grid, $$B$$ a $$3\times3$$ grid, all tiles are
$$256\times256$$, and the ranks are

$$
r^A=
\begin{bmatrix}
16&48&32\\
48&16&16
\end{bmatrix},
\qquad
r^B=
\begin{bmatrix}
16&48&32\\
32&16&16\\
48&32&64
\end{bmatrix}.
$$

A tile-by-tile evaluation issues 18 coupling products with 10 distinct shapes.
We evaluate the right-associative form $$U(SZ^T)$$ in three stages that
communicate through workspace matrices.

### Stage 1: fuse the coupling products

Fix $$k=1$$. The inner factors of contraction column 1 are stacked in memory,
and so are the outer factors of $$B$$'s tile row 1:

$$
S_1=
\begin{bmatrix}V_{11}^{T}\\V_{21}^{T}\end{bmatrix}
\begin{bmatrix}W_{11}&W_{12}&W_{13}\end{bmatrix}.
$$

The left panel is $$(16+48)\times256$$, the right panel is
$$256\times(16+48+32)$$. Six products, each with a different shape, become
**one** GEMM of shape $$(M,N,K)=(64,96,256)$$, and every $$S_{i1j}$$ is a
sub-block of the result.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr-stage1.png' | relative_url }}"
       alt="Stage 1: for each contraction tile k = 1, 2, 3, a stacked panel of V-transpose blocks multiplies a packed panel of W blocks with different widths, producing all coupling blocks for that k in one GEMM; the three GEMMs sit in one grouped call"
       width="900">
  <figcaption>
    One row per contraction tile \(k\). Within a row, tile products of
    different shapes fuse into one GEMM; the dashed box is one grouped call.
    Rank extents are to scale.
  </figcaption>
</figure>

In GEMM terms, the row dimension is the pair (rank of $$A$$, tile row $$i$$)
and the column dimension is (rank of $$B$$, tile column $$j$$). The other two
values of $$k$$ give GEMMs of shape $$(64,64,256)$$ and $$(48,144,256)$$, and
all three go out in one grouped call.

### Stage 2: write where Stage 3 reads

Next, each coupling block is multiplied by its inner factor:

$$
T_{ikj}=S_{ikj}Z_{kj}^{T},\qquad (M,N,K)=(r^A_{ik},\,256,\,r^B_{kj}).
$$

These are 18 independent tasks with 10 distinct shapes, submitted as one
grouped GEMM. What matters is where they write. For each output tile row
$$i$$ there is one terminal matrix $$T_i$$, and task $$(i,k,j)$$ writes
directly into row block $$k$$, column block $$j$$ of it. Row block $$k$$ has
height $$r^A_{ik}$$, so $$T_1$$ is $$(16+48+32)\times768$$.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr-stage2.png' | relative_url }}"
       alt="Stage 2: the 48 by 32 block S121 times Z21-transpose gives a 48 by 256 result written into the k=2, j=1 block of the 96 by 768 workspace T1, whose row blocks have heights 16, 48 and 32"
       width="900">
  <figcaption>
    Task \((i,k,j)=(1,2,1)\) writes its \(48\times256\) result straight into
    its block of \(T_1\). The row blocks of \(T_1\) have heights
    \(r^A_{1k}=16,48,32\), the same as the column blocks of \(U_1\), so Stage 3
    reads \(T_1\) as one matrix.
  </figcaption>
</figure>

### Stage 3: put the sum over k inside the GEMM

The packed outer factors of tile row $$i$$ form $$U_1=[U_{11}\;U_{12}\;U_{13}]$$,
a $$256\times96$$ matrix whose column blocks match the row blocks of
$$T_1$$. One GEMM then performs the entire sum over $$k$$:

$$
C_{1,:}\leftarrow
\alpha\,[U_{11}\;U_{12}\;U_{13}]
\begin{bmatrix}
T_{11,:}\\T_{12,:}\\T_{13,:}
\end{bmatrix}
+\beta\,C_{1,:}.
$$

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr-stage3.png' | relative_url }}"
       alt="Stage 3: the 256 by 96 outer panel U1 with column blocks of width 16, 48, 32 multiplies the 96 by 768 terminal matrix T1, whose row blocks have matching heights, and writes the three output tiles C11, C12, C13 once"
       width="900">
  <figcaption>
    Matching shades pair each column block of \(U_1\) with the row block of
    \(T_1\) for the same \(k\). The loop over contraction tiles becomes the
    reduction dimension \(K=16+48+32=96\), and \(C_{11},C_{12},C_{13}\) are
    written once.
  </figcaption>
</figure>

The shape is $$(256,768,96)$$ for $$i=1$$ and $$(256,768,80)$$ for $$i=2$$.
The sequential loop over $$k$$ from the KBLAS version is now inside a single
reduction, and each element of $$C$$ is read, scaled by $$\beta$$, and
written exactly once.

Put together, the whole product is three grouped calls per rectangle of
output tiles (how rectangles are chosen is covered
[below](#association-order-and-workspace)):

```julia
for (I, J) in runs  # rectangles of output tiles that fit the workspace
    # Stage 1: one grouped call, one fused GEMM per k
    S[I, k, J] = transpose(V[I, k]) * W[k, J]       # for all k
    # Stage 2: one grouped call, written in place into T[i]
    T[i][k, j] = S[i, k, j] * transpose(Z[k, j])    # for all i in I, k, j in J
    # Stage 3: one GEMM per tile row; the sum over k is the reduction
    C[i, J] = α * U[i, :] * T[i] + β * C[i, J]      # for all i in I
end
```

Compared with the KBLAS loop, there is no sequential loop over $$k$$ and no
repeated update of $$C$$.

| Stage | Tile by tile | NextLA | Index left as tasks |
|:--|:--|:--|:--|
| Coupling $$V^TW$$ | 18 GEMMs, 10 shapes | 3 GEMMs, 1 grouped call | $$k$$ |
| Right product $$SZ^T$$ | 18 GEMMs, 10 shapes | 18 GEMMs, 1 grouped call | $$(i,k,j)$$ |
| Dense update $$UT$$ | 18 GEMMs, $$q_k=3$$ passes over $$C$$ | 2 GEMMs, 1 pass over $$C$$ | $$i$$ |

The arithmetic is identical in both columns. What changes is the shape of the
GEMMs, how many launches the GPU sees, and how often $$C$$ is touched.

## Where fusion stops

Not every index can be folded into a GEMM dimension.

- **$$k$$ in Stage 1.** Putting two contraction tiles in the same coupling
  GEMM would produce cross terms $$V_{ik}^TW_{k'j}$$ with $$k\neq k'$$, which
  do not appear in $$AB$$. The three $$k$$ values stay separate GEMMs.
- **Stage 2.** The tasks share operands ($$Z_{kj}$$ is used by every
  $$i$$), but they write into different terminal matrices whose strides do
  not coalesce, so no single GEMM view covers them.

Whatever remains is submitted with `cublasGemmGroupedBatchedEx` [8, 9].
Ordinary batched GEMM requires one shape per batch; grouped GEMM accepts one
shape per *group*, so tasks are bucketed by descriptor and the whole
heterogeneous collection goes out in one call. This is what lets each tile
keep its own rank.

**Transposes.** $$(A^T)_{ik}=V_{ki}U_{ki}^T$$: transposing swaps the roles of
the factor families *and* their tile coordinates, so the column-packed inner
factors become row-packed outer factors of $$A^T$$. All four combinations
`NN`, `NT`, `TN`, `TT` reuse the same storage without copies.

## Association order and workspace

The example used the right-associative order $$U(SZ^T)$$, whose final GEMM
covers a tile row of $$C$$. The left-associative order $$(US)Z^T$$ instead
finishes with one GEMM per tile column, reducing over the column panel of
$$Z$$. With
$$\rho_i=\sum_k r^A_{ik}$$ and $$\gamma_j=\sum_k r^B_{kj}$$, Stage 2 costs the
same for square tiles, and the final stages cost

$$
F_R=\sum_{i,j}2\,m_in_j\,\rho_i,
\qquad
F_L=\sum_{i,j}2\,m_in_j\,\gamma_j.
$$

In the example, $$\rho=(96,80)$$ and $$\gamma=(96,96,112)$$, so
$$F_R=2b^2\cdot528$$ and $$F_L=2b^2\cdot608$$. The right-associative order does about 13% less
final-stage arithmetic and needs a smaller terminal workspace
($$176\times768$$ elements against $$512\times304$$). The choice is made from
rank metadata alone, before any numerical work.

The example also assumed enough workspace to process all of $$C$$ at once.
With a smaller budget, the scheduler splits $$C$$ into rectangles of output
tiles: it takes the widest feasible block of tile columns, packs the longest
runs of tile rows that fit, and picks the cheaper feasible association order for each. Each
rectangle still contains the full reduction over $$k$$, so $$C$$ is written
once; rectangles run in sequence and reuse one workspace arena.
`gemm_workspace_bytes(A, B; runs=n)` returns the budget for a target number of
runs, and `analyze_compressed_gemm` precomputes rank prefixes, the schedule,
and the grouped descriptors so repeated products with the same rank structure
skip that setup.

The association order is chosen per rectangle, not per tile. Stage 3 issues
one GEMM per tile row (right-associative) or per tile column
(left-associative), so every tile in a rectangle must use the same order, and
a tile whose own ranks favour the other order pays extra arithmetic. Larger
budgets give larger rectangles and more of these tiles. This is a deliberate
trade: the extra FLOPs buy wider final GEMMs and fewer passes over the
factors.

The rectangles also leave room for pipelining, which is not implemented yet.
They write disjoint parts of $$C$$, so with two workspace arenas and two
streams, Stages 1–2 of the next rectangle (many small, bandwidth-bound GEMMs)
could overlap Stage 3 of the current one (a few large, compute-bound GEMMs),
at the cost of twice the workspace.

## Results

We evaluated the dense-output GEMM on an NVIDIA H100 with uniform and skewed
rank distributions. A direct comparison with KBLAS is limited: KBLAS supports
FP32 and FP64 and uses one common rank per operand, while NextLA also runs
BF16, FP16, and TF32 and keeps each tile's rank.

The first experiment is the like-for-like comparison: FP32, every tile with
the same rank $$r=b/16$$, tile size $$b=N/16$$. The dashed line is
*dense-GEMM parity*: dense cuBLAS time scaled by the ratio of executed FLOPs,
i.e. how fast the compressed product would run at dense-GEMM throughput. At
$$N=65{,}536$$ NextLA reaches 92.1% of parity. Across the full 20-case
constant-rank grid shared with KBLAS it is faster in 19 cases, 1.44× in
geometric mean and up to 2.58×.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr-results-kblas.png' | relative_url }}"
       alt="Log-log plot of FP32 TLR GEMM time against N from 4K to 64K on an H100: NextLA is below KBLAS at every size and approaches the dense-GEMM parity line as N grows"
       width="650">
  <figcaption>
    Constant-rank FP32 TLR GEMM on an H100, \(b=N/16\), \(r=b/16\). Dense-GEMM
    parity is dense GEMM time scaled by the executed-FLOP ratio.
  </figcaption>
</figure>

The second experiment uses what KBLAS cannot represent directly: a different
rank for every tile. Each matrix has an $$8\times8$$ tile grid,
$$b=N/8$$, and ranks drawn from a skewed distribution in
$$[b/32,\,b/16]$$. KBLAS has to pad every tile to $$b/16$$. The speedup over
dense GEMM grows with $$N$$ as the fused GEMMs become large enough to fill
the GPU. It is lower in FP16 than in FP32 because the dense FP16 baseline runs
on tensor cores and is much faster to begin with.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr-results-speedup.png' | relative_url }}"
       alt="Bar chart of speedup over dense GEMM for N from 4K to 64K in FP16 and FP32. NextLA TLR times TLR reaches 9.6x in FP16 and 20.5x in FP32; padded KBLAS, shown in FP32 only, reaches 12.9x"
       width="850">
  <figcaption>
    Speedup over dense GEMM on an H100 for skewed per-tile ranks in
    \([b/32,b/16]\), \(b=N/8\). \(A_{\mathrm{TLR}}\times B\) has one dense
    operand. KBLAS pads to rank \(b/16\) and does not support FP16.
  </figcaption>
</figure>

The workspace budget trades memory for speed. One FP16 configuration was
6.34× faster than dense GEMM while using 17.8% of the dense-input memory; the
largest FP16 case, $$N=65{,}536$$, ran in 81.8 ms, 9.57× faster than dense,
using 10.7% of the dense-input memory (compressed operands plus workspace).

## Takeaways

- **Choose the storage layout from the contractions you need.** Complementary
  packing makes tile indices addressable as GEMM dimensions, so fusion costs
  no copies.
- **Move loops into the reduction dimension.** Folding the sum over $$k$$
  into one GEMM turns $$q_k$$ passes over the output into one.
- **Fuse what the algebra and strides allow, group the rest.** Grouped GEMM
  handles the remaining shapes, so ranks are padded only to alignment, not to
  the largest rank.

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

[1] K. Akbudak, H. Ltaief, A. Mikhalev, and D. Keyes. *Tile Low Rank Cholesky
Factorization for Climate/Weather Modeling Applications on Manycore
Architectures.* ISC High Performance, 2017.

[2] W. Boukaram, G. Turkiyyah, and D. Keyes. *Hierarchical Matrix Operations
on GPUs.* SIAM Journal on Scientific Computing 41(4), 2019.

[3] P.-H. Chen, S. Si, Y. Li, C. Chelba, and C.-J. Hsieh. *GroupReduce:
Block-Wise Low-Rank Approximation for Neural Language Model Shrinking.*
NeurIPS, 2018.

[4] W. Boukaram, S. Zampini, G. Turkiyyah, and D. Keyes.
*H2OPUS-TLR: High Performance Tile Low Rank Symmetric Factorizations Using
Adaptive Randomized Approximation.* 2021.
[arXiv:2108.11932](https://arxiv.org/abs/2108.11932).

[5] Cris Cecka. *CuTe Layout Representation and Algebra.* 2026.
[arXiv:2603.02298](https://arxiv.org/pdf/2603.02298).

[6] NVIDIA. *CuTe Layouts* and *CuTe Layout Algebra*.
[Layout definitions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html),
[layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html).

[7] A. Charara, D. Keyes, and H. Ltaief. *Tile Low-Rank GEMM Using Batched
Operations on GPUs.* Euro-Par, 2018.
[Paper](https://doi.org/10.1007/978-3-319-96983-1_57),
[KBLAS TLR implementation](https://github.com/ecrc/kblas-gpu/blob/master/src/blas_tlr/Xgemm_tlr_core.cuh).

[8] NVIDIA. *Introducing Grouped GEMM APIs in cuBLAS and More Performance Updates.*
2024. [Technical blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/).

[9] NVIDIA. *cuBLAS API Reference.*
[Uniform batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex),
[grouped GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex).

</div>
