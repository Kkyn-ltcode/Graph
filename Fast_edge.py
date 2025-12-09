import numpy as np
from numba import njit, prange
import pandas as pd

@njit(parallel=True)
def fast_sample_pairs_optimized(flow_array, K):
    """
    Ultra-fast parallel sampling optimized for small K (like 30)
    Uses index-based rejection sampling for maximum speed
    """
    N = len(flow_array)
    
    # Pre-allocate output arrays (exact size)
    all_src = np.empty(N * K, dtype=flow_array.dtype)
    all_dst = np.empty(N * K, dtype=flow_array.dtype)
    
    # Parallel loop over all rows
    for i in prange(N):
        start_idx = i * K
        
        # Fill source values (vectorized)
        for k in range(K):
            all_src[start_idx + k] = flow_array[i]
        
        # Sample K distinct indices != i
        sampled_count = 0
        
        # Use rejection sampling (very fast for small K)
        while sampled_count < K:
            # Generate random index
            rand_idx = np.random.randint(0, N - 1)
            
            # Map to exclude index i: if rand_idx >= i, shift by 1
            if rand_idx >= i:
                rand_idx += 1
            
            # Check for duplicates in already sampled indices
            is_duplicate = False
            for j in range(sampled_count):
                if all_dst[start_idx + j] == flow_array[rand_idx]:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_dst[start_idx + sampled_count] = flow_array[rand_idx]
                sampled_count += 1
    
    return all_src, all_dst


@njit(parallel=True)
def fast_sample_pairs_no_duplicates(flow_array, K):
    """
    Alternative version using set-like behavior for duplicate checking
    Faster for larger K values (K > 50)
    """
    N = len(flow_array)
    
    all_src = np.empty(N * K, dtype=flow_array.dtype)
    all_dst = np.empty(N * K, dtype=flow_array.dtype)
    
    for i in prange(N):
        start_idx = i * K
        
        # Fill sources
        for k in range(K):
            all_src[start_idx + k] = flow_array[i]
        
        # Build exclusion set using a temporary array
        sampled_indices = np.empty(K, dtype=np.int64)
        sampled_count = 0
        
        while sampled_count < K:
            rand_idx = np.random.randint(0, N - 1)
            if rand_idx >= i:
                rand_idx += 1
            
            # Check if index already sampled
            already_used = False
            for j in range(sampled_count):
                if sampled_indices[j] == rand_idx:
                    already_used = True
                    break
            
            if not already_used:
                sampled_indices[sampled_count] = rand_idx
                all_dst[start_idx + sampled_count] = flow_array[rand_idx]
                sampled_count += 1
    
    return all_src, all_dst


def process_dataframe(df, K, method='optimized', return_polars=True):
    """
    Main function to process dataframe with timing
    
    Parameters:
    -----------
    df : DataFrame with 'id' column (pandas or polars)
    K : int, number of samples per row (e.g., 30)
    method : 'optimized' or 'no_duplicates'
    return_polars : bool, if True returns polars DataFrame, else numpy arrays
    
    Returns:
    --------
    If return_polars=True: polars DataFrame with columns ['src', 'dst']
    If return_polars=False: tuple of (All_src, All_dst) numpy arrays
    """
    import polars as pl
    
    print(f"Processing {len(df):,} rows with K={K}")
    print(f"Total output size: {len(df) * K:,} pairs")
    
    # Handle both pandas and polars input
    if hasattr(df, 'to_numpy'):  # pandas
        Flow_array = df['id'].to_numpy()
    else:  # polars
        Flow_array = df['id'].to_numpy()
    
    # Choose method
    if method == 'optimized' or K <= 50:
        print("Using optimized rejection sampling...")
        All_src, All_dst = fast_sample_pairs_optimized(Flow_array, K)
    else:
        print("Using index-based sampling...")
        All_src, All_dst = fast_sample_pairs_no_duplicates(Flow_array, K)
    
    print(f"Generated {len(All_src):,} pairs")
    
    if return_polars:
        # Create polars DataFrame
        result_df = pl.DataFrame({
            'src': All_src,
            'dst': All_dst
        })
        print(f"Returned polars DataFrame with shape: {result_df.shape}")
        return result_df
    else:
        return All_src, All_dst


# Example usage and benchmark
if __name__ == "__main__":
    import time
    import polars as pl
    
    # Test with realistic data size
    print("="*60)
    print("BENCHMARK: Fast Sampling for Large DataFrames")
    print("="*60)
    
    # Test 1: Small test (100K rows) - Polars output
    print("\n[Test 1] 100,000 rows, K=30 -> Polars DataFrame")
    np.random.seed(42)
    df_small = pd.DataFrame({'id': np.random.randint(0, 1000000, size=100000)})
    
    start = time.time()
    result_df = process_dataframe(df_small, K=30, return_polars=True)
    elapsed = time.time() - start
    
    print(f"✓ Completed in {elapsed:.2f} seconds")
    print(f"  Speed: {len(df_small)/elapsed:,.0f} rows/second")
    print(f"  Result shape: {result_df.shape}")
    print(f"  First 5 rows:\n{result_df.head(5)}")
    
    # Test 2: Get numpy arrays instead
    print("\n[Test 2] 100,000 rows, K=30 -> Numpy Arrays")
    start = time.time()
    All_src, All_dst = process_dataframe(df_small, K=30, return_polars=False)
    elapsed = time.time() - start
    
    print(f"✓ Completed in {elapsed:.2f} seconds")
    print(f"  Arrays shape: src={All_src.shape}, dst={All_dst.shape}")
    print(f"  First 5 pairs: src={All_src[:5]}, dst={All_dst[:5]}")
    
    # Test 3: Medium test (1M rows) - extrapolate to 7M
    print("\n[Test 3] 1,000,000 rows, K=30 -> Polars DataFrame")
    df_medium = pd.DataFrame({'id': np.random.randint(0, 1000000, size=1000000)})
    
    start = time.time()
    result_df = process_dataframe(df_medium, K=30, return_polars=True)
    elapsed = time.time() - start
    
    print(f"✓ Completed in {elapsed:.2f} seconds")
    print(f"  Speed: {len(df_medium)/elapsed:,.0f} rows/second")
    print(f"  Memory usage: ~{result_df.estimated_size('mb'):.1f} MB")
    
    # Extrapolate to 7M rows
    estimated_time_7m = elapsed * 7
    print(f"\n[Projection] 7,000,000 rows would take: {estimated_time_7m/60:.1f} minutes")
    print(f"  Output DataFrame size: {7_000_000 * 30:,} rows")
    print(f"  vs Original: 400 hours = {400*60:.0f} minutes")
    print(f"  Speedup: {(400*60)/(estimated_time_7m/60):.0f}x faster!")
    print("="*60)
