from collections import defaultdict

def build_bidirectional_edges(df, max_edges_per_duplicate=None):
    """
    Build bidirectional edges, handling duplicates correctly.
    
    Args:
        df: DataFrame with columns [flow_id, src_endpoint, dst_endpoint]
        max_edges_per_duplicate: Optional limit on edges per duplicate group
        
    Returns:
        edges: List of tuples [(flow_id_A, flow_id_B), ...]
    """
    
    # STEP 1: Build forward lookup with lists (handles duplicates)
    forward_lookup = defaultdict(list)
    
    for idx, row in df.iterrows():
        flow_id = row['flow_id']
        src = row['src_endpoint']
        dst = row['dst_endpoint']
        
        # Skip self-loops
        if src == dst:
            continue
        
        key = (src, dst)
        forward_lookup[key].append(flow_id)
    
    # STEP 2: Find ALL bidirectional pairs
    edges = []
    processed_pairs = set()  # Track to avoid processing same pair twice
    
    for idx, row in df.iterrows():
        flow_id_A = row['flow_id']
        src_A = row['src_endpoint']
        dst_A = row['dst_endpoint']
        
        # Skip self-loops
        if src_A == dst_A:
            continue
        
        # Create reverse key
        reverse_key = (dst_A, src_A)
        
        # Check if reverse flows exist
        if reverse_key in forward_lookup:
            reverse_flow_ids = forward_lookup[reverse_key]
            
            # Optional: Limit edges per duplicate group
            if max_edges_per_duplicate is not None:
                reverse_flow_ids = reverse_flow_ids[:max_edges_per_duplicate]
            
            # Create edges with ALL reverse flows
            for flow_id_B in reverse_flow_ids:
                # Avoid duplicates: only add if A < B
                if flow_id_A < flow_id_B:
                    # Create canonical pair (smaller ID first)
                    pair = (flow_id_A, flow_id_B)
                    
                    # Check if already processed
                    if pair not in processed_pairs:
                        edges.append(pair)
                        processed_pairs.add(pair)
    
    return edges


# USAGE:
edges = build_bidirectional_edges(df)

# With duplicate limit:
edges = build_bidirectional_edges(df, max_edges_per_duplicate=50)
```

---

## **📊 COMPLEXITY ANALYSIS**

### **With Duplicates:**

**Time Complexity:**
```
Let:
  N = total number of flows (16.94M)
  D = average duplicates per unique (src, dst) pair
  
Building forward_lookup: O(N)
Finding edges: O(N × D)

Worst case: If many duplicates, D could be large (e.g., 100)
  → O(N × D) = O(100N) still linear in practice

With max_edges_per_duplicate limit: O(N × K) where K is the limit
```

**Space Complexity:**
```
forward_lookup: O(N) - stores all flow_ids in lists
edges: O(N × D) - could be large if many duplicates
processed_pairs set: O(N × D)

With duplicate limit: O(N × K)
