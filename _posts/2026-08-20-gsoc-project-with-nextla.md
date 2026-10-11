---
layout: post
title: "Tile-Low-Rank GEMM in NextLA: Packing, Layouts, and Fusion"
image: /assets/images/gsoc-2026/tlr_matrix-transparent.png
---

<div class="message">
  Over the summer, I worked on extending the library
  <a href="https://github.com/NextLinearAlgebra/NextLA.jl">NextLA.jl</a>
  with support for the Tile Low-Rank (TLR) data format and related operations.
  The project was carried out as part of Google Summer of Code, under the mentorship of
  <a href="https://github.com/rabab53">Rabab Alomairy</a> (MIT, Julia Lab) and
  <a href="https://github.com/QingleiCao">Qinglei Cao</a> (SLU).

  The implementation is available in
  <a href="https://github.com/NextLinearAlgebra/NextLA.jl/pull/20">pull request #20</a>
  and is currently undergoing restructuring.
</div>

Hierarchical matrices exploit low-rank structure within large dense matrices
to reduce storage and the cost of operations such as matrix-vector and
matrix-matrix multiplication. They are widely used in scientific computing
[1, 2], and related blockwise low-rank representations have also been applied
to machine learning [4]. But reducing the number of floating-point operations
does not automatically make a computation faster on a GPU. Nested formats
such as $$\mathcal{H}^2$$ and HSS organize their blocks and shared bases in
hierarchies, often producing many small operations with different shapes.
These workloads can struggle to provide the regular data movement and
operand reuse needed to sustain high GPU throughput.

The Tile-Low-Rank (TLR) format takes a more regular approach: it replaces the
hierarchy with a flat grid, storing each compressed tile as a pair of low-rank
factors [1, 10]. For an interior tile of size $$b\times b$$ and rank
$$r_{ij}$$, both factors have shape $$b\times r_{ij}$$. The dense dimension
is fixed; only the rank dimension varies. This trades some of the compression
available from nested bases for a simpler structure that is easier to map to
GPU operations.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr-matrix-1.png' | relative_url }}"
       alt="A Tile-Low-Rank matrix whose tiles have different ranks"
       width="900">
  <figcaption>The tile grid is regular, while the ranks of the compressed tiles vary.</figcaption>
</figure>

Libraries such as KBLAS [3] and HiCMA [9] provide TLR multiplication through
batched low-rank products or scheduled tile updates. These approaches expose
parallelism, but their tile-oriented organization leaves an opportunity:
**can compatible products be combined into larger GEMMs that reuse operands
across neighboring tiles?**

My main design contribution was to choose the factor layout around those
contractions. Packing complementary factor families in different tile orders
exposes contiguous panels, allowing tile indices to become GEMM row, column,
or reduction coordinates without copying the factors. This makes larger
GEMMs possible, improves data locality, and increases arithmetic intensity
through operand reuse. Folding tile sums into the reduction dimension also
reduces repeated updates to the output. Rank metadata determines which
contractions fit within a workspace budget; the CUDA path executes them
through cuBLAS, using grouped GEMM for the remaining collections of shapes.

This post develops those ideas using CuTe's nested shape and stride notation
[5, 6] where it helps explain the layout transformations. The key question is
**which tile indices can become GEMM coordinates, and which must remain
separate tasks?** The implementation uses packed offsets and cuBLAS calls;
CuTe provides the language for describing the design.

The main derivation considers
$$C_{\mathrm{dense}}\leftarrow\alpha\,\operatorname{op}(A)\operatorname{op}(B)+\beta C_{\mathrm{dense}}$$
with a dense accumulator $$C$$ and **one or both** operands in TLR format.
My library work also covered compressed-output multiplication, whose
additional approximation and rank-selection steps are outside the scope of
the GEMM layout derivation here.

## Table of Contents

1. [Compression changes the GEMM workload](#compression-changes-the-gemm-workload)
2. [Packing factors as nested panels](#packing-factors-as-nested-panels)
3. [Folding tile indices into GEMM dimensions](#folding-tile-indices-into-gemm-dimensions)
4. [Transposes and shared-operand reuse](#transposes-and-shared-operand-reuse)
5. [Choosing contractions under a workspace budget](#choosing-contractions-under-a-workspace-budget)
6. [Results](#results)
7. [References](#references)

## Compression changes the GEMM workload

Tile-Low-Rank (TLR) partitions a matrix into a regular grid and approximates
each compressed tile independently:

$$
A_{ij}\approx U_{ij}V_{ij}^{T},
\qquad
U_{ij}\in\mathbb{R}^{m_i\times r_{ij}},
\quad
V_{ij}\in\mathbb{R}^{n_j\times r_{ij}}.
$$

Each tile gets its own rank, capturing spatially varying compressibility
within a flat grid [1, 3]. Let $$A$$ and $$B$$ be two fully compressed TLR
matrices, with factors

$$
\begin{aligned}
A_{ik}&=U_{ik}V_{ik}^{T},\\
B_{kj}&=W_{kj}Z_{kj}^{T}.
\end{aligned}
$$

Their product is

$$
C_{ij}=\sum_k
U_{ik}\underbrace{\left(V_{ik}^{T}W_{kj}\right)}_{S_{ikj}}Z_{kj}^{T}.
$$

A tile-oriented implementation treats each factor pair as a separate
operand. KBLAS's dense-output TLR GEMM [3], for example, traverses the
contraction tiles $$k$$ in order. For each $$k$$, it evaluates all output
pairs $$(i,j)$$ with three batched GEMMs. In pseudocode, for $$C=AB$$:

```julia
C .= 0
for k in 1:q_k
    # Each line is batched over output tile pairs (i, j).
    S[i, j] = transpose(V[i, k]) * W[k, j]  # coupling
    T[i, j] = S[i, j] * transpose(Z[k, j])  # right fold
    C[i, j] += U[i, k] * T[i, j]           # dense update
end
```

Here $$q_k$$ is the number of contraction tiles. The temporary arrays
$$S$$ and $$T$$ hold the intermediates for the current $$k$$ and can be
reused on the next pass. For general $$\alpha$$ and $$\beta$$, the final
GEMM scales the new contribution by $$\alpha$$ and applies $$\beta$$
to the existing output only on the first pass.

Different output tiles can be updated in parallel, but successive updates
to the same $$C_{ij}$$ depend on one another. This lowering therefore makes
$$q_k$$ passes over the dense output, with intermediate writes and reads
between the three GEMMs. When the active working set exceeds L2 capacity,
these accesses can generate repeated traffic to GPU device memory. Batching
exposes parallelism among the small products, while reuse of factors shared
across output tiles still depends on the kernels and cache behavior.

The sum over $$k$$ does not itself require a sequence of separate output
updates. By redesigning the factor storage around contiguous panels, we can
combine compatible tile products into larger GEMMs and fold the tile sum
into a GEMM's reduction dimension. This exposes operand reuse within the
larger products and reduces how often the dense output is read and written.
The following sections show which indices can be folded this way and how
workspace and layout constraints determine the remaining batches.

## Packing factors as nested panels

The factor families use complementary tile traversal orders:

- Outer factors $$U/W$$ are packed by tile row.
- Inner factors $$V/Z$$ are packed by tile column.

Each factor is itself column-major. Tile order determines which factors
are adjacent. This exposes four panels: $$U_{i,:}$$ across $$k$$,
$$V_{:,k}$$ across $$i$$, $$W_{k,:}$$ across $$j$$, and
$$Z_{:,j}$$ across $$k$$.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/data-structure.png' | relative_url }}"
       alt="Complementary packed storage for U and V factors"
       width="800">
  <figcaption>
    Logical ranks and factor offsets describe two complementary packed arrays.
  </figcaption>
</figure>

First assume uniform ranks $$r_A,r_B$$ and $$q_k$$ contraction tiles.
In CuTe's **Shape : Stride**
notation, a packed outer-factor row is

$$
U_i:\quad(m_i,(r_A,q_k)):(1,(d_U,d_Ur_A)),
$$

where $$d_U$$ is its physical leading dimension. Its address mapping is

$$
L_{U_i}(m,(a,k))=m+d_Ua+d_Ur_Ak.
$$

The nested coordinate $$(a,k)$$ identifies rank position $$a$$ within tile
$$k$$. Its strides satisfy the coalescing rule [6]:

$$
(r_A,q_k):(d_U,d_Ur_A)
\longrightarrow(r_Aq_k):d_U.
$$

The result is an ordinary $$m_i\times(r_Aq_k)$$ matrix view. No copy is
needed to consume the entire reduction panel in GEMM.

For a fixed contraction tile $$k$$ and $$h$$ output tile rows, the
inner-factor panel similarly has the layout

$$
V_k:\quad(b_k,(r_A,h)):(1,(d_V,d_Vr_A)).
$$

Its nested coordinate is $$(a,i)$$. Complementary packing makes both
panel directions available in the same representation.

### Variable ranks retain segment boundaries

A nested shape $$(r,q)$$ denotes a Cartesian product; it does not describe
a different rank for each tile. Arbitrary ranks retain explicit prefixes.
With stored capacities $$\widehat r^A_{ik}$$, define

$$
p_i(k)=\sum_{t<k}\widehat r^A_{it},
\qquad
\widehat\rho_i=\sum_k\widehat r^A_{ik}.
$$

The complete panel remains a simple matrix:

$$
U_i:(m_i,\widehat\rho_i):(1,d_U),
\qquad
U_{ik}[m,a]=U_i[m,p_i(k)+a].
$$

The logical active range is $$0\le a<r^A_{ik}$$. Prefix offsets locate
the variable-width segments, including any local alignment capacity.

For the supported FP16/BF16 CUDA paths, `rank_multiple=8` rounds
stored capacities locally while retaining the discovered logical ranks.
Factor leading dimensions and output tile alignment have their own
constraints; rank padding is one part of making the views valid.

## Folding tile indices into GEMM dimensions

To see what the packing makes possible, follow one concrete product through
all three stages. Take two fully compressed, non-transposed TLR operands,
with enough workspace to process the whole output in one run:

| Quantity | Example |
|:--|:--|
| Tile grids | $$A:2\times3$$, $$B:3\times3$$, $$C:2\times3$$ |
| Dense tile extent | $$256\times256$$ |
| Rank of every tile | $$64$$ |
| Matrix dimensions | $$A:512\times768$$, $$B:768\times768$$, $$C:512\times768$$ |

We use the right-associated contraction $$U(SZ^T)$$, called `FoldRight`
in the implementation. Its final GEMMs accumulate by output tile row.
The coupling matrices $$S$$ and terminal matrices $$T$$ connect the three
stages. Within a stage, fusion turns compatible tile indices into GEMM
dimensions; the stages themselves still communicate through workspace
matrices in global memory.

### Stage 1: form the coupling panels

Fix one contraction tile $$k$$. The six couplings for the two output tile
rows and three output tile columns can be computed together:

$$
S_k=
\begin{bmatrix}V_{1k}^{T}\\V_{2k}^{T}\end{bmatrix}
\begin{bmatrix}W_{k1}&W_{k2}&W_{k3}\end{bmatrix}
=
\begin{bmatrix}
S_{1k1}&S_{1k2}&S_{1k3}\\
S_{2k1}&S_{2k2}&S_{2k3}
\end{bmatrix}.
$$

The left panel is $$128\times256$$ and the right panel is
$$256\times192$$. Six GEMMs of shape $$(M,N,K)=(64,64,256)$$ become
one GEMM of shape $$(128,192,256)$$. Each $$V_{ik}^{T}$$ participates in
all three output columns, and each $$W_{kj}$$ participates in both output
rows. Their packed panels expose that reuse within the larger GEMM.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr_stage1_panels.png' | relative_url }}"
       alt="Stage 1: two stacked V-transpose blocks multiply three concatenated W blocks, producing six coupling blocks in one 128 by 192 by 256 GEMM"
       width="900">
  <figcaption>
    One coupling GEMM combines both output tile rows and all three output tile
    columns. The three contraction tiles remain independent grouped tasks.
  </figcaption>
</figure>

In CuTe notation, the free modes are $$M=(a,i)$$ and $$N=(b,j)$$,
where $$a,b$$ are the two rank coordinates. The local dense coordinate
is the reduction. The three $$k$$ values remain separate tasks: mixing them
in this coupling product would introduce cross terms absent from $$AB$$.
The arithmetic count is unchanged, but the GEMM geometry and operand reuse
change.

### Stage 2: write the intermediate where Stage 3 needs it

Next compute

$$
T_{ikj}=S_{ikj}Z_{kj}^{T}.
$$

Each coupling block is $$64\times64$$, and each $$Z_{kj}^{T}$$ is
$$64\times256$$. This gives $$2\times3\times3=18$$ independent GEMMs
of shape $$(64,256,64)$$, submitted as grouped tasks over $$(i,k,j)$$.

The destination is chosen for the next stage. For each output tile row
$$i$$, allocate one terminal matrix

$$
T_i=
\begin{bmatrix}
T_{i11}&T_{i12}&T_{i13}\\
T_{i21}&T_{i22}&T_{i23}\\
T_{i31}&T_{i32}&T_{i33}
\end{bmatrix}
\in\mathbb{R}^{192\times768}.
$$

Each task writes directly into its block of $$T_i$$. No additional gather
is needed to assemble the matrix consumed by Stage 3.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr_stage2_terminal.png' | relative_url }}"
       alt="Stage 2: S121 times Z21-transpose produces T121 and writes it directly into the second contraction-row segment and first output-column segment of terminal matrix T1"
       width="900">
  <figcaption>
    The highlighted task writes into its final workspace location. Rows of
    $$T_i$$ group rank and contraction coordinates; columns group local
    output columns and tile columns.
  </figcaption>
</figure>

Although $$Z_{kj}$$ is shared across $$i$$, that alone does not make row
fusion a valid ordinary GEMM view. The results belong to separate terminal
matrices $$T_i$$, whose output strides do not generally coalesce across
$$i$$. Grouped execution preserves this arrangement while exposing the
independent work.

### Stage 3: consume the whole tile sum in one update

For a fixed output tile row $$i$$, the packed outer factors form

$$
U_i=[U_{i1}\;U_{i2}\;U_{i3}]
\in\mathbb{R}^{256\times192}.
$$

Multiplying this panel by $$T_i$$ performs the complete sum over $$k$$:

$$
C_{i,J}\leftarrow
\alpha\underbrace{[U_{i1}\;U_{i2}\;U_{i3}]}_{U_i}
\underbrace{
\begin{bmatrix}
T_{i1,J}\\T_{i2,J}\\T_{i3,J}
\end{bmatrix}}_{T_i}
+\beta C_{i,J},
\qquad J=\{1,2,3\}.
$$

One GEMM of shape $$(256,768,192)$$ updates all three output tiles in
that row. There are two final GEMMs, one for each $$i$$.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr_stage3_reduction.png' | relative_url }}"
       alt="Stage 3: a 256 by 192 outer panel multiplies the 192 by 768 terminal matrix, reducing all three contraction tiles and updating one dense output tile row"
       width="900">
  <figcaption>
    The three outer-factor segments match the three contraction-row segments
    of $$T_i$$. Their product incorporates the full tile sum before writing
    the dense output.
  </figcaption>
</figure>

The paired coordinate $$(a,k)$$ becomes GEMM's reduction dimension,
and $$(n,j)$$ becomes its column dimension. The previous sequential loop
over contraction tiles is now inside the GEMM reduction. Stages 1 and 2
never touch $$C$$; Stage 3 applies $$\beta$$ once and writes each output
element once for this run.

The example therefore changes the workload as follows:

| Stage | Tilewise GEMM problems | Packed lowering | Remaining task indices |
|:--|:--|:--|:--|
| Coupling $$V^TW$$ | 18 of $$(64,64,256)$$ | 3 of $$(128,192,256)$$ | $$k$$ |
| Right fold $$SZ^T$$ | 18 of $$(64,256,64)$$ | 18 of $$(64,256,64)$$ | $$(i,k,j)$$ |
| Dense update $$UT$$ | 18 of $$(256,256,64)$$ | 2 of $$(256,768,192)$$ | $$i$$ |

These counts refer to GEMM problems; a grouped call can submit several
problems. All three stages perform the same arithmetic as the tilewise
FoldRight evaluation.

For variable ranks, replace the uniform segment widths by rank prefixes.
The fused coupling dimensions and final reduction length become

$$
M_k=\sum_{i\in I}\widehat r^A_{ik},
\qquad
N_k=\sum_{j\in J}\widehat r^B_{kj},
\qquad
K_i=\sum_k\widehat r^A_{ik}.
$$

The stored capacities include any local alignment padding; the logical
ranks remain unchanged. The same panel construction applies, with prefixes
locating the variable-width blocks.

Variable ranks also change how the remaining tasks are submitted. Their
GEMM dimensions and leading dimensions can differ, whereas an ordinary
cuBLAS batched GEMM requires these parameters and the transpose modes to
be uniform. NextLA groups tasks with matching GEMM descriptors and submits
the groups through `cublasGemmGroupedBatchedEx`, which permits different
descriptors between groups [7, 12]. Equal-descriptor subsets can still form
ordinary batches; grouped execution handles the heterogeneous collection
without padding every tile to one common rank.

## Transposes and shared-operand reuse

Transposition exchanges both factor roles and tile coordinates:

$$
(A^T)_{ik}=V_{ki}U_{ki}^{T},
\qquad
(B^T)_{kj}=Z_{jk}W_{jk}^{T}.
$$

The original column-packed inner factors become row-packed outer factors
of the logical transpose, and conversely. For compatible operand shapes,
this preserves the panel construction for `NN`, `NT`, `TN`, and `TT`
without allocating another copy of the factors.

### The Stage 1 caveat: a batch of transposed blocks

Ordinary GEMM fusion still depends on physical strides. Consider the initial
coupling again, with a shared left panel $$\mathsf{A}_k$$ and right blocks
$$B_j\in\mathbb{R}^{K\times N}$$. In our example,
$$M=128$$, $$K=256$$, $$N=64$$, and there are $$L=3$$ right blocks.
Consecutive column-major $$B_j$$ blocks have the column/batch sublayout

$$
(N,L):(K,KN),
$$

which coalesces into $$NL:K$$. That is the wide right-hand panel used
in Stage 1.

Now suppose the same right operands are represented as consecutive
column-major blocks $$R_j=B_j^T\in\mathbb{R}^{N\times K}$$ and consumed
with a transpose. Their logical column/batch sublayout is

$$
(N,L):(1,NK).
$$

It does not coalesce for nontrivial dimensions: the outer stride required
to combine these modes is $$N$$, while the actual stride is $$NK$$ [6].
The wide coupling is then evaluated as a batch:

$$
S_{k,j}=\mathsf{A}_kR_j^T,
\qquad
\mathsf{A}_k=
\begin{bmatrix}V_{1k}^T\\V_{2k}^T\end{bmatrix}.
$$

One $$(128,192,256)$$ coupling GEMM becomes three
$$(128,64,256)$$ GEMMs, all sharing exactly the same left operand.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/tlr_stage1_transposed_blocks.png' | relative_url }}"
       alt="Stage 1 layout caveat: three GEMMs consume separately stored transposed right blocks and reuse the same stacked V-transpose left panel"
       width="900">
  <figcaption>
    Transposed-block strides prevent ordinary column fusion, but all three
    coupling tasks reuse the same left panel. This is a condition on the
    physical block layout; a logical TLR transpose alone does not impose it.
  </figcaption>
</figure>

In CuTe notation the shared operand has the logical batch layout

$$
\mathsf{A}_k:(M,K,L):(1,\mathrm{lda},0).
$$

The zero batch stride describes reuse of the same addresses. Pointer-array
batched or grouped execution expresses that reuse by repeating the left
operand pointer.

### Why batching can retain the throughput

A shared-operand batch can approach the throughput of a wide GEMM. NVIDIA's
batched-GEMM discussion reports parity with large GEMM in its P100 examples
and identifies reuse of a common operand as an optimization opportunity
[11]. Those examples establish that batching need not introduce a throughput
penalty; they do not guarantee equal timings for every geometry.

A plausible explanation for comparable performance here is that the shared
left panel can be reused through L2, while the batch supplies enough
independent threadblocks to occupy the GPU. Individual GEMM dimensions and
kernel tiling also matter. CUTLASS documents both last-level-cache reuse
through threadblock scheduling and the efficiency limits of small GEMM
shapes [8]. These mechanisms explain why the shared-operand batch can remain
efficient. Timings establish whether its performance is unchanged for a
particular workload; attributing that result specifically to L2 requires
profiling.

## Choosing contractions under a workspace budget

The concrete example used a right-associated contraction, $$U(SZ^T)$$,
with final accumulation by output tile row (`FoldRight`). The same packing
also supports a left-associated contraction, $$(US)Z^T$$, with final
accumulation by output tile column (`FoldLeft`): first compute
$$H_{ikj}=U_{ik}S_{ikj}$$, then combine the $$H$$ blocks and the $$Z$$
column panel. The final reduction folds $$(b,k)$$ and combines output tile
rows, giving one final GEMM per output tile column in a scheduled run.

The rank metadata determines which direction is cheaper. Define

$$
P_{ij}=\sum_k r^A_{ik}r^B_{kj},
\qquad
\rho_i=\sum_k r^A_{ik},
\qquad
\gamma_j=\sum_k r^B_{kj}.
$$

After the common coupling stage, the arithmetic costs for one output tile are

$$
F_R(i,j)=2n_jP_{ij}+2m_in_j\rho_i,
\qquad
F_L(i,j)=2m_iP_{ij}+2m_in_j\gamma_j.
$$

For square tiles, the middle-stage costs agree. Comparing $$\rho_i$$ and
$$\gamma_j$$ selects the cheaper final contraction; in our uniform example
both are $$192$$, so either direction has the same arithmetic cost.
The terminal intermediates contain $$n_j\rho_i$$ and $$m_i\gamma_j$$
elements per output tile. The execution model substitutes stored capacities
when padding is active and aggregates costs over the scheduled rectangle.

So far we assumed enough workspace to process the whole output together.
A smaller budget partitions it into output rectangles $$I\times J$$,
each with its own coupling panels and terminal matrices. Larger budgets
permit wider column panels and longer contiguous row runs. The current
dense-output scheduler selects feasible column blocks and row runs, then
chooses a layout-compatible contraction order that fits the workspace,
using the modeled arithmetic costs to choose between feasible orders.

Each rectangle retains the complete $$k$$ reduction for its output entries.
Different rectangles write disjoint regions of $$C$$ and have no mathematical
dependency on one another. The current implementation executes them
sequentially to reuse the same workspace; the budget controls this scheduling
granularity. Within a rectangle, the three GEMM stages retain their producer
and consumer dependencies. Chunking bounds peak intermediate storage, but
the intermediate matrices are still written to and read from global memory,
and smaller chunks can repeat factor reads.

NextLA exposes minimum and maximum workspace queries.
`gemm_workspace_bytes(A, B; runs=n)` finds a budget meeting a target
run count under the scheduler for fully compressed operands.
Symbolic analysis prepares rank prefixes, schedules, and persistent grouped
descriptors. Reuse requires compatible bound operands, output, workspace,
transpose modes, compute policy, and rank structure; factor values and
numerical scalars may change in place.

Workspace therefore controls how much compatible work can be presented
together, while the rank distribution determines the sizes and costs of the
resulting GEMMs.

## Results

We evaluated the dense-output GEMM on an NVIDIA H100 with both uniform and
skewed tile-rank distributions. A direct comparison with KBLAS is necessarily
limited: KBLAS supports FP32 and FP64 and uses one common rank for every tile
in an operand. NextLA additionally supports BF16, FP16, and TF32 on CUDA, and
retains each tile's logical rank with local capacity padding where required.

For constant-rank FP32 problems, NextLA reached 92.1% of the throughput
reference obtained by scaling dense-GEMM time by the executed-FLOP ratio.
This reference is based on dense-GEMM performance; it is not hardware peak utilization. On the configurations
shared with KBLAS, it was 1.44 times faster in geometric mean and up to 2.58
times faster. The advantage is larger for variable ranks because NextLA does
not pad every tile to one operand-wide rank.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/figure_4_constant_rank_nextla_vs_kblas_b_n16_r_b16-transparent.png' | relative_url }}"
       alt="NextLA and KBLAS constant-rank FP32 TLR GEMM runtime on an NVIDIA H100"
       width="650">
  <figcaption>
    Constant-rank FP32 GEMM with tile size \(b=N/16\) and tile rank \(r=b/16\).
    The reference time scales dense-GEMM time by the executed-FLOP ratio.
  </figcaption>
</figure>

The constant-rank experiment above provides the like-for-like KBLAS
comparison. The next experiment uses the feature that the padded baseline
cannot represent directly: different ranks for different tiles. Each matrix
has eight tiles per axis, hence 64 tiles in total, with tile size $$b=N/8$$
and ranks varying between $$b/16$$ and $$b/8$$. The bars report speedup over
the corresponding dense GEMM for several matrix sizes and compute modes.

<figure>
  <img src="{{ '/assets/images/gsoc-2026/figure_5_precision_speedup_bars_skewed_b_n8-transparent.png' | relative_url }}"
       alt="TLR GEMM speedups across BF16, FP16, TF32, and FP32 for matrices with skewed tile ranks"
       width="850">
  <figcaption>
    Speedup over dense GEMM for matrices with an \(8\times8\) tile grid,
    \(b=N/8\), and per-tile ranks in \([b/16,b/8]\). NextLA preserves each tile's logical rank with local alignment padding. KBLAS pads the varying ranks to one common rank and is
    shown only for its supported FP32 comparison.
  </figcaption>
</figure>

The missing KBLAS bars in the BF16, FP16, and TF32 panels are therefore not
omitted measurements; those execution modes are not supported by KBLAS. The
NextLA speedup increases with matrix size as the fused compressed operations
become large enough to use the GPU efficiently.

The workspace experiments show the other side of the design. One FP16
configuration was 6.34 times faster than dense GEMM while using 17.8% of the
dense-input memory. In the largest reported FP16 case, $$N=65{,}536$$ completed
in 81.8 ms, 9.57 times faster than the dense baseline while using 10.7% of its
operand memory.

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

[3] A. Charara, D. Keyes, and H. Ltaief. *Tile Low-Rank GEMM Using Batched
Operations on GPUs.* Euro-Par, 2018.
[Paper](https://doi.org/10.1007/978-3-319-96983-1_57),
[KBLAS TLR implementation](https://github.com/ecrc/kblas-gpu/blob/master/src/blas_tlr/Xgemm_tlr_core.cuh).

[4] P.-H. Chen, S. Si, Y. Li, C. Chelba, and C.-J. Hsieh. *GroupReduce:
Block-Wise Low-Rank Approximation for Neural Language Model Shrinking.*
NeurIPS, 2018.

[5] Cris Cecka. *CuTe Layout Representation and Algebra.* 2026.
[arXiv:2603.02298](https://arxiv.org/pdf/2603.02298).

[6] NVIDIA. *CuTe Layouts* and *CuTe Layout Algebra*.
[Layout definitions](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html),
[layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html).

[7] NVIDIA. *Introducing Grouped GEMM APIs in cuBLAS and More Performance Updates.*
2024. [Technical blog](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/).

[8] NVIDIA. *Efficient GEMM in CUDA.*
[CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).

[9] HiCMA. *Hierarchical Computations on Manycore Architectures.*
[Library](https://github.com/ecrc/hicma),
[TLR GEMM task implementation](https://github.com/ecrc/hicma/blob/master/compute/pdgemm.c).

[10] W. Boukaram, S. Zampini, G. Turkiyyah, and D. Keyes.
*H2OPUS-TLR: High Performance Tile Low Rank Symmetric Factorizations Using
Adaptive Randomized Approximation.* 2021.
[arXiv:2108.11932](https://arxiv.org/abs/2108.11932).

[11] C. Cecka. *Pro Tip: cuBLAS Strided Batched Matrix Multiply.* NVIDIA,
2017. [Technical blog](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/).

[12] NVIDIA. *cuBLAS API Reference.*
[Uniform batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex),
[grouped GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex).

</div>
