from tensor_contraction import *
from prob import *
from util import *
from jax.lax import scan
import argparse
jax.config.update("jax_enable_x64", True)
def main():
    parser = argparse.ArgumentParser(description="Compute domain-wall CMI via batched Metropolis")
    parser.add_argument("--n",           type=int,   default=4,
                        help="Lattice parameter n")
    parser.add_argument("--batch-size",  type=int,   default=5,
                        help="Number of parallel chains")
    parser.add_argument("--num-samples", type=int,   default=5,
                        help="total number of samples")
    parser.add_argument("--p-arr",       type=float, nargs="+",
                        default=[0., 0.109, 0.1095, 0.111, 0.112, 0.5],
                        help="List of p values to sweep over")
    parser.add_argument("--array-number", type=int,  default=2,
                        help="SLURM array task ID (for seeding & filenames)")
    args = parser.parse_args()

    n            = args.n
    batch_size   = args.batch_size
    num_samples  = args.num_samples
    p_arr        = jnp.array(args.p_arr)
    array_number = args.array_number
    num_batch = int(num_samples // batch_size)
    len_p = len(p_arr)
    # precompute all the shapes & string-tables once
    xbc, ybc     = 3*n,   2*n
    xabc, yabc   = 5*n,   3*n

    strs_ini_bc,   strs_column_bc   = einsum_strs_initial(ybc),   einsum_strs_column(ybc)
    strs_ini_abc,  strs_column_abc  = einsum_strs_initial(yabc),  einsum_strs_column(yabc)

    # AB-specific tables
    strs_col_ab_ini = einsum_strs_AB_column_ini(2*n)
    strs_col_ab     = einsum_strs_AB_column(2*n)
    strs_col_ab_end = einsum_strs_AB_column_end(2*n)
    str_shrink      = einsum_strs_shrink(n)
    str_expand      = einsum_strs_expand(n)

    # B-specific tables
    str_B_col_ini   = einsum_B_col_ini(n)
    str_B_col       = einsum_strs_column(n-1)
    str_B_col_end   = einsum_B_col_end(n)

    # AB bulk tables
    str_AB_col_ini = einsum_B_col_ini(2*n)
    str_AB_col = einsum_strs_column(2*n-1)
    str_AB_col_end = einsum_B_col_end(2*n)

    # derive four base keys from (array_number, i)
    key = PRNGKey(array_number)
    scan_inputs = split(key, num_batch)

    @jax.jit
    def scan_body(carry, batch_keys_base):
        key_batch = split(batch_keys_base, (len_p, batch_size))
        config_init = batch_make_config(key_batch, p_arr, n)
        tensor_ABC, log_ABC, tensor_ABC_end, log_ABC_end = batch_log_Pr_bulk_pre(n, p_arr, config_init, strs_ini_abc, strs_column_abc, False)
        tensor_BC, log_BC, tensor_BC_end, log_BC_end = batch_log_Pr_bulk_pre(n, p_arr, config_init, strs_ini_bc, strs_column_bc, True)

        wb_batch = batch_log_Pr_B(n, p_arr, config_init, str_B_col_ini,str_B_col, str_B_col_end, tensor_BC, log_BC, tensor_BC_end, log_BC_end)
        wbc_batch = batch_log_Pr_bulk(n, p_arr, config_init, strs_column_bc, True, tensor_BC, log_BC, tensor_BC_end, log_BC_end)
        wabc_batch = batch_log_Pr_bulk(n, p_arr, config_init, strs_column_abc, False, tensor_ABC, log_ABC, tensor_ABC_end, log_ABC_end)
        wab_batch = batch_log_Pr_AB(n, p_arr, config_init,  str_shrink, strs_col_ab_ini, strs_col_ab, strs_col_ab_end, str_expand, tensor_ABC, log_ABC,
                                       tensor_ABC_end, log_ABC_end)
        wab_P0_batch = batch_log_Pr_AB_P0(n, p_arr,config_init, str_AB_col_ini, str_AB_col, str_AB_col_end, tensor_ABC, log_ABC, tensor_ABC_end, log_ABC_end)
        return carry, (wb_batch, wbc_batch, wab_batch, wabc_batch, wab_P0_batch)

    print("Starting JAX scan...")
    initial_carry = None # No state needed between batches in this case
    _, (wb, wbc, wab, wabc, wab_P0) = scan(scan_body, initial_carry, scan_inputs)
    wb = jnp.transpose(wb, (1, 0, 2))
    wbc = jnp.transpose(wbc, (1, 0, 2))
    wabc = jnp.transpose(wabc, (1, 0, 2))
    wab = jnp.transpose(wab, (1, 0, 2))
    wab_P0 = jnp.transpose(wab_P0, (1, 0, 2))
    print("Scan finished.")
    # build a filename prefix with all the parameters + array_number
    prefix = (f"../domain_wall_m1_data/array{array_number}"
              f"_n{n}"
              f"_batch{batch_size}"
              f"_numsamp{num_samples}")

    # save them

    CMI_M1 = -jnp.mean(jnp.array(wab)+jnp.array(wbc)-jnp.array(wabc)-jnp.array(wb), axis = (-2, -1))/jnp.log(2)+1
    std_CMI_M1 = jnp.std(jnp.array(wab)+jnp.array(wbc)-jnp.array(wabc)-jnp.array(wb), axis = (-2, -1))/jnp.log(2)
    CMI_P0 = -jnp.mean(jnp.array(wab_P0) + jnp.array(wbc) - jnp.array(wabc) - jnp.array(wb), axis=(-2, -1)) / jnp.log(
        2)
    std_CMI_P0 = jnp.std(jnp.array(wab_P0) + jnp.array(wbc) - jnp.array(wabc) - jnp.array(wb), axis=(-2, -1)) / jnp.log(2)


    print("CMI_M1:", CMI_M1)
    print("std_CMI_M1:", std_CMI_M1)
    print("CMI_P0:", CMI_P0)
    print("std_CMI_P0:", std_CMI_P0)

    jnp.save(f"{prefix}_p_arr.npy", p_arr)
    jnp.save(f"{prefix}_M1.npy", CMI_M1)
    jnp.save(f"{prefix}_std_M1.npy", std_CMI_M1)
    jnp.save(f"{prefix}_P0.npy", CMI_P0)
    jnp.save(f"{prefix}_std_P0.npy", std_CMI_P0)
    #print("acc_B: ", acc_b)
    #print("acc_bc: ", acc_bc)
    #print("acc_abc: ", acc_abc)
    #print("acc_ab: ", acc_ab)



if __name__ == "__main__":
    main()
