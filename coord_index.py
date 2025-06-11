import jax.numpy as jnp
from jax import vmap

# --- Geometry Helper Functions (JIT compatible) ---
def is_coord_valid_B(i, j, n):
    """Checks if coordinate (i, j) is within the B region."""
    n = jnp.array(n, dtype=jnp.int64) # Ensure n is usable in comparisons
    i = jnp.array(i, dtype=jnp.int64)
    j = jnp.array(j, dtype=jnp.int64)

    # Block 1: (0 <= i < n, 0 <= j < 2*n)
    in_block1 = (i >= 0) & (i < n - 1) & (j >= 0) & (j < 2*n)
    # Block 2: (i = n - 1, n <= j < 2n)
    in_block2 = (i == n - 1) & (j >= n) & (j < 2*n)
    # Block 3: (n <= i < 2*n, n <= j < 2*n)
    in_block3 = (i >= n) & (i < 2*n) & (j >= n + 1) & (j < 2*n)
    # Block 4: (i == 2n, n <= j < 2n)
    in_block4 = (i == 2*n) & (j >= n) & (j < 2*n)
    # Block 5: (2*n <= i < 3*n, 0 <= j < 2*n)
    in_block5 = (i >= 2*n+1) & (i < 3*n) & (j >= 0) & (j < 2*n)

    return in_block1 | in_block2 | in_block3 | in_block4 | in_block5

def coord_to_index_B(i, j, n):
    """Converts valid B-region coordinate (i, j) to 1D index (5-block definition)."""
    n = jnp.array(n, dtype=jnp.int64)
    i = jnp.array(i, dtype=jnp.int64)
    j = jnp.array(j, dtype=jnp.int64)
    n2 = 2 * n
    n_sq = n * (n - 1)

    # Sizes of each block
    s1 = (n - 1) * n2
    s2 = n
    s3 = n_sq
    s4 = n
    # s5 = n * n2 # Not needed for offsets

    # Cumulative limits (start index of next block)
    limit1 = s1       # Start of block 2
    limit2 = limit1 + s2 # Start of block 3
    limit3 = limit2 + s3 # Start of block 4
    limit4 = limit3 + s4 # Start of block 5

    # Conditions for each block (must match is_coord_valid_B_5blocks)
    in_block1 = (i >= 0) & (i < n - 1) & (j >= 0) & (j < n2)
    in_block2 = (i == n - 1) & (j >= n) & (j < n2)
    in_block3 = (i >= n) & (i < 2*n) & (j >= n + 1) & (j < n2)
    in_block4 = (i == 2*n ) & (j >= n) & (j < n2)
    # in_block5 is implied if not in others and valid

    # Calculate index relative to the start of each block
    idx1 = i * n2 + j                     # Relative index within block 1
    idx2 = limit1 + (j - n)               # Relative index within block 2
    idx3 = limit2 + (i - n) * (n - 1) + (j - n - 1) # Relative index within block 3
    idx4 = limit3 + (j - n)               # Relative index within block 4
    idx5 = limit4 + (i - 2*n - 1) * n2 + j    # Relative index within block 5

    # Select the correct index based on the block
    index = jnp.where(in_block1, idx1,
            jnp.where(in_block2, idx2,
            jnp.where(in_block3, idx3,
            jnp.where(in_block4, idx4,
                      idx5)))) # Assume block 5 if not others

    return index


def index_to_coord_B(index, n):
    """Converts 1D index to B-region coordinate (i, j) (5-block definition)."""
    n = jnp.array(n, dtype=jnp.int64)
    index = jnp.array(index, dtype=jnp.int64)
    n2 = 2 * n
    n_sq = (n - 1) * n

    # Sizes and Limits (must match coord_to_index_B_5blocks)
    s1 = (n - 1) * n2
    s2 = n
    s3 = n_sq
    s4 = n
    limit1 = s1
    limit2 = limit1 + s2
    limit3 = limit2 + s3
    limit4 = limit3 + s4

    # Conditions for which block the index falls into
    in_block1 = index < limit1
    in_block2 = (index >= limit1) & (index < limit2)
    in_block3 = (index >= limit2) & (index < limit3)
    in_block4 = (index >= limit3) & (index < limit4)
    # in_block5 is implied if index >= limit4

    # Calculate coordinates relative to the start of each block
    # Block 1
    i1 = index // n2
    j1 = index % n2
    # Block 2
    idx_rel2 = index - limit1
    i2 = n - 1
    j2 = n + idx_rel2
    # Block 3
    idx_rel3 = index - limit2
    i3 = n + idx_rel3 // (n - 1)
    j3 = n + 1 + idx_rel3 % (n - 1)
    # Block 4
    idx_rel4 = index - limit3
    i4 = 2*n
    j4 = n + idx_rel4
    # Block 5
    idx_rel5 = index - limit4
    i5 = 2*n + 1 + idx_rel5 // n2
    j5 = idx_rel5 % n2

    # Select correct coordinates based on the block
    i = jnp.where(in_block1, i1,
        jnp.where(in_block2, i2,
        jnp.where(in_block3, i3,
        jnp.where(in_block4, i4,
                  i5)))) # Assume block 5 if not others

    j = jnp.where(in_block1, j1,
        jnp.where(in_block2, j2,
        jnp.where(in_block3, j3,
        jnp.where(in_block4, j4,
                  j5)))) # Assume block 5 if not others

    return i, j

def is_coord_valid_AB_P0(i, j, n):
    """Checks if coordinate (i, j) is within the B region."""
    n = jnp.array(n, dtype=jnp.int64) # Ensure n is usable in comparisons
    i = jnp.array(i, dtype=jnp.int64)
    j = jnp.array(j, dtype=jnp.int64)

    in_block1 = (i >= 0) & (i < 2*n - 1) & (j >= 0) & (j < 3*n)
    in_block2 = (i == 2*n - 1) & (j >= n) & (j < 3*n)
    in_block3 = (i >= 2*n) & (i < 3*n) & (j >= n + 1) & (j < 3*n)
    in_block4 = (i == 3*n) & (j >= n) & (j < 3*n)
    in_block5 = (i >= 3*n+1) & (i < 5*n) & (j >= 0) & (j < 3*n)

    return in_block1 | in_block2 | in_block3 | in_block4 | in_block5

def coord_to_index_AB_P0(i, j, n):
    """Converts valid B-region coordinate (i, j) to 1D index (5-block definition)."""
    n = jnp.array(n, dtype=jnp.int64)
    i = jnp.array(i, dtype=jnp.int64)
    j = jnp.array(j, dtype=jnp.int64)
    n3 = 3 * n
    n_sq = n * (2 * n - 1)

    # Sizes of each block
    s1 = (2 * n - 1) * n3
    s2 = 2 * n
    s3 = n_sq
    s4 = 2 * n
    # s5 = n * n2 # Not needed for offsets

    # Cumulative limits (start index of next block)
    limit1 = s1       # Start of block 2
    limit2 = limit1 + s2 # Start of block 3
    limit3 = limit2 + s3 # Start of block 4
    limit4 = limit3 + s4 # Start of block 5

    # Conditions for each block (must match is_coord_valid_B_5blocks)
    in_block1 = (i >= 0) & (i < 2*n - 1) & (j >= 0) & (j < n3)
    in_block2 = (i == 2*n - 1) & (j >= n) & (j < n3)
    in_block3 = (i >= 2*n) & (i < 3*n) & (j >= n + 1) & (j < n3)
    in_block4 = (i == 3*n ) & (j >= n) & (j < n3)
    # in_block5 is implied if not in others and valid

    # Calculate index relative to the start of each block
    idx1 = i * n3 + j                     # Relative index within block 1
    idx2 = limit1 + (j - n)               # Relative index within block 2
    idx3 = limit2 + (i - 2 * n) * (2 * n - 1) + (j - n - 1) # Relative index within block 3
    idx4 = limit3 + (j - n)               # Relative index within block 4
    idx5 = limit4 + (i - 3*n - 1) * n3 + j    # Relative index within block 5

    # Select the correct index based on the block
    index = jnp.where(in_block1, idx1,
            jnp.where(in_block2, idx2,
            jnp.where(in_block3, idx3,
            jnp.where(in_block4, idx4,
                      idx5)))) # Assume block 5 if not others

    return index

def index_to_coord_AB_P0(index, n):
    """Converts 1D index to B-region coordinate (i, j) (5-block definition)."""
    n = jnp.array(n, dtype=jnp.int64)
    index = jnp.array(index, dtype=jnp.int64)
    n3 = 3 * n
    n_sq = (2 * n - 1) * n

    # Sizes and Limits (must match coord_to_index_B_5blocks)
    s1 = (2 * n - 1) * n3
    s2 = 2 * n
    s3 = n_sq
    s4 = 2 * n
    limit1 = s1
    limit2 = limit1 + s2
    limit3 = limit2 + s3
    limit4 = limit3 + s4

    # Conditions for which block the index falls into
    in_block1 = index < limit1
    in_block2 = (index >= limit1) & (index < limit2)
    in_block3 = (index >= limit2) & (index < limit3)
    in_block4 = (index >= limit3) & (index < limit4)
    # in_block5 is implied if index >= limit4

    # Calculate coordinates relative to the start of each block
    # Block 1
    i1 = index // n3
    j1 = index % n3
    # Block 2
    idx_rel2 = index - limit1
    i2 = 2 * n - 1
    j2 = n + idx_rel2
    # Block 3
    idx_rel3 = index - limit2
    i3 = 2 * n + idx_rel3 // (2 * n - 1)
    j3 = n + 1 + idx_rel3 % (2 * n - 1)
    # Block 4
    idx_rel4 = index - limit3
    i4 = 3 * n
    j4 = n + idx_rel4
    # Block 5
    idx_rel5 = index - limit4
    i5 = 3 * n + 1 + idx_rel5 // n3
    j5 = idx_rel5 % n3

    # Select correct coordinates based on the block
    i = jnp.where(in_block1, i1,
        jnp.where(in_block2, i2,
        jnp.where(in_block3, i3,
        jnp.where(in_block4, i4,
                  i5)))) # Assume block 5 if not others

    j = jnp.where(in_block1, j1,
        jnp.where(in_block2, j2,
        jnp.where(in_block3, j3,
        jnp.where(in_block4, j4,
                  j5)))) # Assume block 5 if not others

    return i, j


def is_coord_valid_AB(i, j, n):
    """Checks if coordinate (i, j) is within the AB MAIN plaquette region."""

    n3 = 3*n  # Width

    # Block A: (0 <= i < 2n-1, 0 <= j < 3n - 1)
    in_A = (i >= 0) & (i < 2*n) & (j >= 0) & (j < n3)
    # Block B: (2n <= i < 3n, n <= j < 3n-1)
    in_B = (i >= 2*n) & (i < 3*n) & (j >= n - 1) & (j < n3)
    # Block C: (i == 3n, n <= j < 3n-1) # Assuming i=3n was meant, not i=3n-1
    in_C = (i >= 3*n ) & (i < 5*n) & (j >= 0) & (j < n3)

    return in_A | in_B | in_C

def coord_to_index_AB_main(i, j, n):
    """Converts valid AB MAIN coordinate (i, j) to 1D index (0 to num_plaq_main-1)."""

    n2m1 = 2*n - 1
    n3 = 3*n # Width

    # Sizes of each block (derived from geometry)
    sA = n2m1 * n3
    sB = 2*n
    sC = n * (2*n - 1)
    sD = 2*n
    # sE = (2*n_-1)*n3m1 # Not needed for offsets

    # Cumulative limits
    limitA = sA
    limitB = limitA + sB
    limitC = limitB + sC
    limitD = limitC + sD

    # Conditions for each block
    in_A = (i >= 0) & (i < 2*n- 1) & (j >= 0) & (j < n3)
    in_B = (i == 2*n - 1) & (j >= n) & (j < n3)
    in_C = (i >= 2*n) & (i < 3*n) & (j >= n + 1) & (j < n3)
    in_D = (i == 3*n) & (j >= n) & (j < n3)
    # in_E implied

    # Index calculation
    idxA = i * n3 + j
    idxB = limitA + (j - n)
    idxC = limitB + (i - 2*n) * (2*n - 1) + (j - n - 1)
    idxD = limitC + (j - n)
    idxE = limitD + (i - (3*n + 1)) * n3 + j

    index = jnp.where(in_A, idxA,
            jnp.where(in_B, idxB,
            jnp.where(in_C, idxC,
            jnp.where(in_D, idxD,
                      idxE))))
    return index

def index_to_coord_AB_main(index, n):
    """Converts 1D AB MAIN index to coordinate (i, j)."""

    n3 = 3*n
    n2m1 = 2*n - 1

    # Sizes and Limits (must match coord_to_index)
    sA = n2m1 * n3
    sB = 2*n
    sC = n * n2m1
    sD = 2*n
    limitA = sA
    limitB = limitA + sB
    limitC = limitB + sC
    limitD = limitC + sD

    # Conditions
    in_A = index < limitA
    in_B = (index >= limitA) & (index < limitB)
    in_C = (index >= limitB) & (index < limitC)
    in_D = (index >= limitC) & (index < limitD)
    # in_E implied

    # Coordinate calculation
    iA = index // n3
    jA = index % n3
    idx_relB = index - limitA
    iB = 2*n - 1
    jB = n + idx_relB
    idx_relC = index - limitB
    iC = 2*n + idx_relC // n2m1
    jC = n + 1 + idx_relC % n2m1
    idx_relD = index - limitC
    iD = 3*n
    jD = n + idx_relD
    idx_relE = index - limitD
    iE = 3*n + 1 + idx_relE // n3
    jE = idx_relE % n3

    i = jnp.where(in_A, iA, jnp.where(in_B, iB, jnp.where(in_C, iC, jnp.where(in_D, iD, iE))))
    j = jnp.where(in_A, jA, jnp.where(in_B, jB, jnp.where(in_C, jC, jnp.where(in_D, jD, jE))))

    return i, j

def get_global_index_from_coord(i, j, n, num_plaq_main, num_middle):
    """Given (i,j), find its global index and type (0=main, 1=middle, 2=corner)."""

    # Offsets
    offset_mid = num_plaq_main
    offset_cor = offset_mid + num_middle

    # 1. Check Corners first (most specific)
    is_c0 = (i == 2*n - 1) & (j == n -1)
    is_c1 = (i == 2*n) & (j == n)
    is_c2 = (i == 3*n - 1) & (j == n)
    is_c3 = (i == 3 * n) & (j == n - 1)

    is_corner = is_c0 | is_c1 | is_c2 | is_c3
    corner_local_idx = jnp.where(is_c0, 0, jnp.where(is_c1, 1, jnp.where(is_c2, 2, 3)))
    corner_global_idx = offset_cor + corner_local_idx

    # 2. Check Middle sections
    is_mid1 = (i == 2*n - 1) & (j >= 0) & (j < n-1)
    mid1_local_idx = j
    mid1_global_idx = offset_mid + mid1_local_idx

    is_mid2 = (j == n) & (i >= 2*n + 1) & (i < 3*n - 1)
    mid2_local_idx = n - 1 + (i - (2*n + 1)) # Local index relative to start of middle array
    mid2_global_idx = offset_mid + mid2_local_idx

    is_mid3 = (i == 3*n) & (j >= 0) & (j < n-1)
    mid3_local_idx = 2 * (n - 2) + 1 + (n - 2 - j) # Local index relative to start of middle array
    mid3_global_idx = offset_mid + mid3_local_idx

    is_middle = is_mid1 | is_mid2 | is_mid3
    middle_global_idx = jnp.where(is_mid1, mid1_global_idx,
                          jnp.where(is_mid2, mid2_global_idx, mid3_global_idx))

    main_global_idx = coord_to_index_AB_main(i, j, n) # Main index is 0-based

    final_idx = jnp.where(is_corner, corner_global_idx,
                  jnp.where(is_middle, middle_global_idx, main_global_idx))


    return final_idx

def get_type_and_local_index(idx_global, num_plaq_main, num_middle):
    """Given global index, find type (0,1,2) and local index within type."""

    is_main = idx_global < num_plaq_main
    is_middle = (idx_global >= num_plaq_main) & (idx_global < num_plaq_main + num_middle)
    # is_corner implied

    type_ = jnp.where(is_main, 0, jnp.where(is_middle, 1, 2))
    local_idx = jnp.where(is_main, idx_global,
                  jnp.where(is_middle, idx_global - num_plaq_main,
                            idx_global - num_plaq_main - num_middle))
    return type_, local_idx

def get_coord_from_global_index(n, idx_g, num_p_main, num_mid):
    type_local, idx_local = get_type_and_local_index(idx_g, num_p_main, num_mid)
    # Case 0: Main
    i_main, j_main = index_to_coord_AB_main(idx_local, n)
    # Case 1: Middle
    # Need to map local middle index back to i,j based on the 3 sections
    mid_lim1 = n - 1
    mid_lim2 = mid_lim1 + (n - 2)
    is_m1 = idx_local < mid_lim1
    is_m2 = (idx_local >= mid_lim1) & (idx_local < mid_lim2)
    # is_m3 implied
    i_m1 = 2*n - 1; j_m1 = idx_local
    i_m2 = 2*n + 1 + (idx_local - mid_lim1); j_m2 = n
    i_m3 = 3*n; j_m3 = n - 2 - (idx_local - mid_lim2)
    i_mid = jnp.where(is_m1, i_m1, jnp.where(is_m2, i_m2, i_m3))
    j_mid = jnp.where(is_m1, j_m1, jnp.where(is_m2, j_m2, j_m3))
    # Case 2: Corner
    i_cor = jnp.where(idx_local == 0, 2*n - 1, jnp.where(idx_local == 1, 2*n, jnp.where(idx_local == 2, 3*n - 1, 3*n)))
    j_cor = jnp.where(idx_local == 0, n - 1, jnp.where(idx_local == 1, n, jnp.where(idx_local == 2, n, n - 1)))
    # Combine
    i_final = jnp.where(type_local==0, i_main, jnp.where(type_local==1, i_mid, i_cor))
    j_final = jnp.where(type_local==0, j_main, jnp.where(type_local==1, j_mid, j_cor))
    return i_final, j_final

batch_index_to_coord_B = vmap(index_to_coord_B, in_axes=(0, None))
batch_index_to_coord_AB = vmap(get_coord_from_global_index, in_axes=(None, 0, None, None))
batch_index_to_coord_AB_P0 = vmap(index_to_coord_AB_P0, in_axes=(0, None))
