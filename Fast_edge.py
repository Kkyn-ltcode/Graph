import numpy as np
from numba import njit, prange
import pandas as pd

@njit(parallel=True)
def fast_sample_pairs_adaptive(flow_array, K):
    """
    Adaptive sampling that automatically chooses the best method:
    - When K is small relative to N: use rejection sampling
    - When K is close to N: use exclusion enumeration
    
    Threshold: if K > N/2, use enumeration; otherwise use rejection
    """
    N = len(flow_array)
    effective_K = min(K, N - 1)  # Can't sample more than N-1
    
    all_src = np.empty(N * effective_K, dtype=flow_array.dtype)
    all_dst = np.empty(N * effective_K, dtype=flow_array.dtype)
    
    # Adaptive threshold: use enumeration when K is large relative to N
    use_enumeration = (effective_K > N // 2)
    
    for i in prange(N):
        start_idx = i * effective_K
        
        # Fill source values
        for k in range(effective_K):
            all_src[start_idx + k] = flow_array[i]
        
        if use_enumeration:
            # Method 1: Enumerate all indices except i, then shuffle
            # This is faster when K is close to N
            candidates = np.empty(N - 1, dtype=np.int64)
            idx = 0
            for j in range(N):
                if j != i:
                    candidates[idx] = j
                    idx += 1
            
            # Partial Fisher-Yates shuffle (only shuffle first K positions)
            for k in range(effective_K):
                rand_pos = np.random.randint(k, N - 1)
                candidates[k], candidates[rand_pos] = candidates[rand_pos], candidates[k]
            
            # Copy sampled values
            for k in range(effective_K):
                all_dst[start_idx + k] = flow_array[candidates[k]]
        else:
            # Method 2: Rejection sampling with index mapping
            # This is faster when K is small relative to N
            sampled_count = 0
            
            while sampled_count < effective_K:
                rand_idx = np.random.randint(0, N - 1)
                if rand_idx >= i:
                    rand_idx += 1
                
                # Check for duplicates
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
def fast_sample_pairs_small_n(flow_array, K):
    """
    Specialized version optimized for small N (N < 1000)
    Uses full enumeration approach which is fastest for small datasets
    """
    N = len(flow_array)
    effective_K = min(K, N - 1)
    
    all_src = np.empty(N * effective_K, dtype=flow_array.dtype)
    all_dst = np.empty(N * effective_K, dtype=flow_array.dtype)
    
    for i in prange(N):
        start_idx = i * effective_K
        
        # Fill source values
        for k in range(effective_K):
            all_src[start_idx + k] = flow_array[i]
        
        # Build candidate array (all indices except i)
        candidates = np.empty(N - 1, dtype=np.int64)
        idx = 0
        for j in range(N):
            if j != i:
                candidates[idx] = j
                idx += 1
        
        # Shuffle only the first K elements (partial Fisher-Yates)
        for k in range(effective_K):
            rand_pos = np.random.randint(k, N - 1)
            candidates[k], candidates[rand_pos] = candidates[rand_pos], candidates[k]
        
        # Copy sampled values
        for k in range(effective_K):
            all_dst[start_idx + k] = flow_array[candidates[k]]
    
    return all_src, all_dst


@njit(parallel=True)
def fast_sample_pairs_large_n(flow_array, K):
    """
    Specialized version optimized for large N (N > 100K) and small K
    Uses rejection sampling which is fastest when K << N
    """
    N = len(flow_array)
    
    all_src = np.empty(N * K, dtype=flow_array.dtype)
    all_dst = np.empty(N * K, dtype=flow_array.dtype)
    
    for i in prange(N):
        start_idx = i * K
        
        # Fill source values
        for k in range(K):
            all_src[start_idx + k] = flow_array[i]
        
        # Rejection sampling (fast for small K)
        sampled_count = 0
        while sampled_count < K:
            rand_idx = np.random.randint(0, N - 1)
            if rand_idx >= i:
                rand_idx += 1
            
            # Check for duplicates
            is_duplicate = False
            for j in range(sampled_count):
                if all_dst[start_idx + j] == flow_array[rand_idx]:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_dst[start_idx + sampled_count] = flow_array[rand_idx]
                sampled_count += 1
    
    return all_src, all_dst


def process_dataframe(df, K, method='auto'):
    """
    Main function with automatic method selection
    
    Parameters:
    -----------
    df : DataFrame with 'id' column
    K : int, number of samples per row
    method : 'auto', 'adaptive', 'small_n', or 'large_n'
    """
    N = len(df)
    print(f"Processing {N:,} rows with K={K}")
    
    Flow_array = df['id'].to_numpy()
    
    # Auto-select best method
    if method == 'auto':
        if N < 1000:
            method = 'small_n'
        elif K > N // 2:
            method = 'adaptive'
        else:
            method = 'large_n'
        print(f"Auto-selected method: {method}")
    
    # Execute chosen method
    if method == 'small_n':
        print("Using enumeration method (optimal for small N)...")
        All_src, All_dst = fast_sample_pairs_small_n(Flow_array, K)
    elif method == 'adaptive':
        print("Using adaptive method (handles variable N/K ratios)...")
        All_src, All_dst = fast_sample_pairs_adaptive(Flow_array, K)
    else:  # large_n
        print("Using rejection sampling (optimal for large N, small K)...")
        All_src, All_dst = fast_sample_pairs_large_n(Flow_array, K)
    
    print(f"Generated {len(All_src):,} pairs")
    return All_src, All_dst


# Comprehensive benchmark
if __name__ == "__main__":
    import time
    
    print("="*70)
    print("COMPREHENSIVE BENCHMARK: Adaptive Sampling")
    print("="*70)
    
    test_cases = [
        (30, 30, "Edge case: N=K (small)"),
        (100, 50, "Small N, K=N/2"),
        (1000, 30, "Medium N, small K"),
        (100000, 30, "Large N, small K"),
        (1000000, 30, "Very large N, small K"),
    ]
    
    for N, K, description in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {description}")
        print(f"N={N:,}, K={K}")
        print(f"{'='*70}")
        
        np.random.seed(42)
        df = pd.DataFrame({'id': np.random.randint(0, 1000000, size=N)})
        
        start = time.time()
        All_src, All_dst = process_dataframe(df, K=K, method='auto')
        elapsed = time.time() - start
        
        print(f"✓ Completed in {elapsed:.4f} seconds")
        if elapsed > 0.001:
            print(f"  Speed: {N/elapsed:,.0f} rows/second")
        print(f"  Output size: {len(All_src):,} pairs")
        print(f"  Sample: src={All_src[:3]}, dst={All_dst[:3]}")
    
    # Final projection for 7M rows
    print(f"\n{'='*70}")
    print("PROJECTION FOR YOUR 7M ROW DATASET")
    print(f"{'='*70}")
    
    print("\nTesting 1M rows to extrapolate...")
    df_test = pd.DataFrame({'id': np.random.randint(0, 1000000, size=1000000)})
    
    start = time.time()
    All_src, All_dst = process_dataframe(df_test, K=30)
    elapsed = time.time() - start
    
    estimated_7m = elapsed * 7
    print(f"\nEstimated time for 7M rows: {estimated_7m/60:.2f} minutes")
    print(f"Original time: 400 hours = {400*60:.0f} minutes")
    print(f"Speedup: {(400*60)/(estimated_7m/60):,.0f}x faster! 🚀")
    print("="*70)
