# Camera Matrix and RQ Decomposition

The camera matrix (or projection matrix) $P$ is a $3 \times 4$ matrix that directly maps 3D world points to 2D pixel coordinates. Estimating $P$ via DLT and decomposing it into intrinsics and extrinsics via RQ decomposition provides an alternative initialization path.

## Camera Matrix DLT

### Problem Statement

**Given**: $N \geq 6$ correspondences between 3D world points $\{\mathbf{P}_i\}$ and 2D pixel points $\{\mathbf{p}_i\}$.

**Find**: $3 \times 4$ projection matrix $P$ such that $\mathbf{p}_i \sim P [\mathbf{P}_i, 1]^T$.

### Derivation

The projection $\mathbf{p} \sim P \mathbf{P}_h$ (with $\mathbf{P}_h = [\mathbf{P}, 1]^T$ in homogeneous coordinates) gives, after cross-multiplication:

$$u (\mathbf{p}_3^T \mathbf{P}_h) - \mathbf{p}_1^T \mathbf{P}_h = 0$$
$$v (\mathbf{p}_3^T \mathbf{P}_h) - \mathbf{p}_2^T \mathbf{P}_h = 0$$

where $\mathbf{p}_i^T$ is the $i$-th row of $P$.

This gives a $2N \times 12$ system $A \mathbf{p} = 0$, solved via SVD with Hartley normalization of the 3D and 2D points.

### Post-Processing

After denormalization, $P$ is $3 \times 4$ but not guaranteed to decompose cleanly into $K[R|\mathbf{t}]$ due to noise. The RQ decomposition extracts the components.

## RQ Decomposition

### Problem Statement

**Given**: The $3 \times 3$ left submatrix $M$ of $P$ (where $P = [M | \mathbf{p}_4]$).

**Find**: Upper-triangular $K$ and orthogonal $R$ such that $M = K R$.

### Algorithm

RQ decomposition is computed by transposing QR decomposition:

1. Compute QR decomposition of $M^T$: $M^T = Q \hat{R}$
2. Then $M = \hat{R}^T Q^T$, where $\hat{R}^T$ is lower-triangular and $Q^T$ is orthogonal
3. Apply a permutation matrix $J$ to flip the matrix to upper-triangular form:
   - $K = J \hat{R}^T J$ (upper-triangular)
   - $R = J Q^T$ (orthogonal)

### Sign Conventions

After decomposition, ensure:
- $K$ has positive diagonal entries: if $K_{ii} < 0$, negate column $i$ of $K$ and row $i$ of $R$
- $\det(R) = +1$: if $\det(R) = -1$, negate a column of $R$ (and the corresponding column of $K$)

### Translation Extraction

$$\mathbf{t} = K^{-1} \mathbf{p}_4$$

## Full Decomposition

The `CameraMatrixDecomposition` struct:

```rust
pub struct CameraMatrixDecomposition {
    pub k: Mat3,   // Upper-triangular intrinsics
    pub r: Mat3,   // Rotation matrix (orthonormal, det = +1)
    pub t: Vec3,   // Translation vector
}
```

## API

```rust
// Estimate the full 3×4 camera matrix
let P = dlt_camera_matrix(&world_pts, &image_pts)?;

// Decompose into K, R, t
let decomp = decompose_camera_matrix(&P)?;
println!("Intrinsics: {:?}", decomp.k);
println!("Rotation: {:?}", decomp.r);
println!("Translation: {:?}", decomp.t);

// Or just RQ decompose any 3×3 matrix
let (K, R) = rq_decompose(&M);
```

## When to Use

Camera matrix DLT is useful when:
- You have non-coplanar 3D-2D correspondences and want to estimate both intrinsics and pose simultaneously
- You need a quick estimate of $K$ from a single view (without multiple homographies)

For calibration with a planar board, Zhang's method is preferred because it uses the planar constraint to get more constraints per view.
