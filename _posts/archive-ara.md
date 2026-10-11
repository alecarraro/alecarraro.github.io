## Compression and compressed output

### Discovering tile ranks with ARA

Tile ranks are discovered through Adaptive Randomized Approximation (ARA) [2].
Each tile maintains its own progress and stopping state while a batched
iteration grows its basis. Two-pass Cholesky QR stabilizes the sampled basis,
and a final small SVD selects the stored rank and records the remaining error.

`TLRMatrix` keeps the diagonal dense; the fully compressed
constructor applies the factor representation to every tile:

```julia
A_tlr = TLRMatrix(A, 256;
                  maxrank=64,
                  tol=1f-5,
                  rel=true,
                  r_required=10)
```

Here `r_required` controls the consecutive negligible samples
needed for retirement. Compression workspaces can be reused.

### Keeping the output compressed

The same panels also support compressed-output multiplication without
materializing each dense output tile. For

$$
X_{ij}=\alpha\sum_k U_{ik}S_{ikj}Z_{kj}^{T},
$$

ARA can obtain a right sample by applying the product implicitly:

$$
X_{ij}\Omega=
\alpha\sum_k U_{ik}S_{ikj}(Z_{kj}^{T}\Omega).
$$

This projects the random block through the right factors, applies the
couplings, and folds the contraction-tile sum through the packed outer
panel. The coupling matrices are computed once for a run and reused across
samples. Left sampling uses the transposed expression when cheaper.

Once a basis $$Q_{ij}$$ is available, a complementary application gives

$$
Y_{ij}=X_{ij}^{T}Q_{ij},
\qquad
X_{ij}\approx Q_{ij}Y_{ij}^{T}.
$$

A small SVD selects the final output rank. The complete tile sum is
compressed together, rather than recompressed after each contraction tile.

A rolling workspace retires converged tiles and admits pending ones.
Temporary factors have width `maxrank`; after all ranks are known,
NextLA allocates the packed output and copies the active columns with any
requested local padding. This is why compressed-output
`gemm` returns a newly allocated container.

The controls include `maxrank`, truncation tolerances
`tol`/`rel`, sampling tolerance `eps_rel`,
sample width `block`, retirement threshold `r_required`,
`workspace`, and final capacity alignment `rank_multiple`.
The current entry point accepts two `CompressedFTLRMatrix` operands
on a regular tile grid, supports all four transpose combinations, and
executes on a single stream.

## Using the implementation

Assume `A` and `B` are dense `Matrix{Float32}` inputs. The same compressed
operands can then be used with both GEMM output modes:

```julia
using NextLA

# A and B are dense matrices with compatible dimensions. For compressed-output
# GEMM, each dimension must also be a multiple of the tile size.
b, maxrank = 256, 64

A_tlr = CompressedFTLRMatrix(A, b; maxrank, tol=1f-5, rel=true)
B_tlr = CompressedFTLRMatrix(B, b; maxrank, tol=1f-5, rel=true)

# Dense accumulation: C_dense is an ordinary dense matrix.
C_dense = zeros(Float32, size(A, 1), size(B, 2))
dense_workspace = gemm_maximum_workspace_bytes(A_tlr, B_tlr)
gemm!(C_dense, A_tlr, B_tlr;
      workspace=dense_workspace,
      alpha=1,
      beta=0)

# Compressed accumulation: ranks are discovered before C_tlr is packed.
C_tlr = gemm(A_tlr, B_tlr;
             maxrank,
             tol=1f-5,
             rel=true,
             r_required=10)

# Expand a compressed matrix whenever a conventional dense array is needed.
C_uncompressed = similar(C_dense)
uncompress!(C_uncompressed, C_tlr)
```

`C_dense` stores the product directly in dense form. `C_tlr` keeps each output
tile compressed, while `C_uncompressed` is its dense reconstruction.

On CUDA, FP16 and BF16 storage can use `rank_multiple=8`; FP32 can select
Tensor Core execution with `compute=TF32()`.
