import torch
import numpy as np
from torch_geometric.data import Data

def split_graph_edges(data, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, seed=42, stratify=True):
    """
    Maximum optimized split for edge-level prediction tasks.
    Splits edges (and their labels/features) into train/valid/test.
    
    Structure:
    - Train: contains train edges only
    - Valid: contains train + valid edges
    - Test: contains all edges
    
    Args:
        data: PyG Data object with:
            - x: node features [num_nodes, num_features]
            - edge_index: edges [2, num_edges]
            - y: edge labels [num_edges]
            - edge_attr: edge features [num_edges, edge_features] (optional)
        train_ratio: proportion of edges for training
        valid_ratio: proportion of edges for validation  
        test_ratio: proportion of edges for testing
        seed: random seed for reproducibility
        stratify: if True, perform stratified split based on edge labels
    
    Returns:
        train_data, valid_data, test_data: Three Data objects
    """
    torch.manual_seed(seed)
    
    num_edges = data.edge_index.size(1)
    device = data.edge_index.device
    
    # Edge splitting with optional stratification
    if stratify and data.y is not None:
        # Stratified split on edges
        y_np = data.y.cpu().numpy()
        unique_labels, inverse = np.unique(y_np, return_inverse=True)
        
        train_edges_idx = []
        valid_edges_idx = []
        test_edges_idx = []
        
        for label_idx, label in enumerate(unique_labels):
            label_mask = (inverse == label_idx)
            label_indices = np.where(label_mask)[0]
            
            np.random.seed(seed + int(label))
            np.random.shuffle(label_indices)
            
            train_size = int(len(label_indices) * train_ratio)
            valid_size = int(len(label_indices) * valid_ratio)
            
            train_edges_idx.append(label_indices[:train_size])
            valid_edges_idx.append(label_indices[train_size:train_size + valid_size])
            test_edges_idx.append(label_indices[train_size + valid_size:])
        
        train_edges_idx = np.concatenate(train_edges_idx)
        valid_edges_idx = np.concatenate(valid_edges_idx)
        test_edges_idx = np.concatenate(test_edges_idx)
        
        np.random.seed(seed)
        np.random.shuffle(train_edges_idx)
        np.random.shuffle(valid_edges_idx)
        np.random.shuffle(test_edges_idx)
    else:
        # Random split
        perm = torch.randperm(num_edges, device='cpu').numpy()
        train_size = int(num_edges * train_ratio)
        valid_size = int(num_edges * valid_ratio)
        
        train_edges_idx = perm[:train_size]
        valid_edges_idx = perm[train_size:train_size + valid_size]
        test_edges_idx = perm[train_size + valid_size:]
    
    # Convert to tensors
    train_idx_t = torch.from_numpy(train_edges_idx).to(device)
    valid_idx_t = torch.from_numpy(valid_edges_idx).to(device)
    test_idx_t = torch.from_numpy(test_edges_idx).to(device)
    
    # === Train Data (train edges only) ===
    train_data = Data(
        x=data.x,  # All nodes are available
        edge_index=data.edge_index[:, train_idx_t].contiguous(),
        y=data.y[train_idx_t].contiguous() if data.y is not None else None,
        edge_attr=data.edge_attr[train_idx_t].contiguous() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    )
    
    # === Valid Data (train + valid edges) ===
    valid_all_idx = torch.cat([train_idx_t, valid_idx_t])
    
    valid_data = Data(
        x=data.x,  # All nodes are available
        edge_index=data.edge_index[:, valid_all_idx].contiguous(),
        y=data.y[valid_all_idx].contiguous() if data.y is not None else None,
        edge_attr=data.edge_attr[valid_all_idx].contiguous() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    )
    
    # === Test Data (all edges) ===
    test_data = data  # Reference original data
    
    # Add edge masks for identifying which edges belong to which split
    test_data.train_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_data.train_edge_mask.scatter_(0, train_idx_t, True)
    
    test_data.valid_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_data.valid_edge_mask.scatter_(0, valid_idx_t, True)
    
    test_data.test_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_data.test_edge_mask.scatter_(0, test_idx_t, True)
    
    # Store metadata
    train_data.original_edge_indices = train_idx_t.cpu()
    valid_data.original_edge_indices = valid_all_idx.cpu()
    
    # Mark which edges in valid_data are from train vs valid
    valid_data.train_edge_mask = torch.cat([
        torch.ones(len(train_idx_t), dtype=torch.bool),
        torch.zeros(len(valid_idx_t), dtype=torch.bool)
    ])
    
    return train_data, valid_data, test_data


def split_graph_edges_memory_efficient(data, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, 
                                        seed=42, stratify=True, chunk_size=5_000_000):
    """
    Memory-efficient version for extremely large edge sets (50M+ edges).
    Useful when edge_attr is very large and causes memory issues.
    """
    torch.manual_seed(seed)
    
    num_edges = data.edge_index.size(1)
    device = data.edge_index.device
    
    # Edge splitting (same as before)
    if stratify and data.y is not None:
        y_np = data.y.cpu().numpy()
        unique_labels, inverse = np.unique(y_np, return_inverse=True)
        
        train_edges_idx = []
        valid_edges_idx = []
        test_edges_idx = []
        
        for label_idx, label in enumerate(unique_labels):
            label_mask = (inverse == label_idx)
            label_indices = np.where(label_mask)[0]
            
            np.random.seed(seed + int(label))
            np.random.shuffle(label_indices)
            
            train_size = int(len(label_indices) * train_ratio)
            valid_size = int(len(label_indices) * valid_ratio)
            
            train_edges_idx.append(label_indices[:train_size])
            valid_edges_idx.append(label_indices[train_size:train_size + valid_size])
            test_edges_idx.append(label_indices[train_size + valid_size:])
        
        train_edges_idx = np.concatenate(train_edges_idx)
        valid_edges_idx = np.concatenate(valid_edges_idx)
        test_edges_idx = np.concatenate(test_edges_idx)
        
        np.random.seed(seed)
        np.random.shuffle(train_edges_idx)
        np.random.shuffle(valid_edges_idx)
        np.random.shuffle(test_edges_idx)
    else:
        perm = torch.randperm(num_edges, device='cpu').numpy()
        train_size = int(num_edges * train_ratio)
        valid_size = int(num_edges * valid_ratio)
        
        train_edges_idx = perm[:train_size]
        valid_edges_idx = perm[train_size:train_size + valid_size]
        test_edges_idx = perm[train_size + valid_size:]
    
    # Sort indices for efficient chunked access
    train_edges_idx = np.sort(train_edges_idx)
    valid_edges_idx = np.sort(valid_edges_idx)
    
    train_idx_t = torch.from_numpy(train_edges_idx).to(device)
    valid_idx_t = torch.from_numpy(valid_edges_idx).to(device)
    valid_all_idx = torch.cat([train_idx_t, valid_idx_t])
    valid_all_idx_sorted, _ = torch.sort(valid_all_idx)
    
    # Build train data in chunks (for edge_attr)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        train_edge_attr_chunks = []
        valid_edge_attr_chunks = []
        
        for i in range(0, len(train_idx_t), chunk_size):
            chunk_idx = train_idx_t[i:i+chunk_size]
            train_edge_attr_chunks.append(data.edge_attr[chunk_idx])
        
        for i in range(0, len(valid_all_idx_sorted), chunk_size):
            chunk_idx = valid_all_idx_sorted[i:i+chunk_size]
            valid_edge_attr_chunks.append(data.edge_attr[chunk_idx])
        
        train_edge_attr = torch.cat(train_edge_attr_chunks, dim=0)
        valid_edge_attr = torch.cat(valid_edge_attr_chunks, dim=0)
    else:
        train_edge_attr = None
        valid_edge_attr = None
    
    train_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, train_idx_t].contiguous(),
        y=data.y[train_idx_t].contiguous() if data.y is not None else None,
        edge_attr=train_edge_attr
    )
    
    valid_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, valid_all_idx_sorted].contiguous(),
        y=data.y[valid_all_idx_sorted].contiguous() if data.y is not None else None,
        edge_attr=valid_edge_attr
    )
    
    test_data = data
    
    test_data.train_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_data.train_edge_mask.scatter_(0, train_idx_t, True)
    
    test_data.valid_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_data.valid_edge_mask.scatter_(0, valid_idx_t, True)
    
    test_data.test_edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_data.test_edge_mask.scatter_(0, torch.from_numpy(test_edges_idx).to(device), True)
    
    train_data.original_edge_indices = train_idx_t.cpu()
    valid_data.original_edge_indices = valid_all_idx_sorted.cpu()
    
    return train_data, valid_data, test_data


# Example and benchmark
if __name__ == "__main__":
    import time
    
    print("Creating large graph for edge prediction:")
    print("  1M nodes, 27M edges\n")
    
    num_nodes = 1_000_000
    num_node_features = 128
    num_edges = 27_000_000
    num_edge_features = 64
    num_classes = 10
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("Allocating tensors...")
    x = torch.randn(num_nodes, num_node_features, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_attr = torch.randn(num_edges, num_edge_features, device=device)
    y = torch.randint(0, num_classes, (num_edges,), device=device)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    print(f"Original graph:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Edge features shape: {data.edge_attr.shape}")
    print(f"  Edge labels shape: {data.y.shape}\n")
    
    # Benchmark
    print("Splitting with stratification...")
    start = time.time()
    train_data, valid_data, test_data = split_graph_edges(data, stratify=True)
    elapsed = time.time() - start
    
    print(f"✓ Completed in {elapsed:.3f} seconds\n")
    
    print(f"Train graph:")
    print(f"  Nodes: {train_data.num_nodes:,}")
    print(f"  Edges: {train_data.num_edges:,}")
    print(f"  Edge features: {train_data.edge_attr.shape}")
    
    print(f"\nValid graph:")
    print(f"  Nodes: {valid_data.num_nodes:,}")
    print(f"  Edges: {valid_data.num_edges:,}")
    print(f"  Edge features: {valid_data.edge_attr.shape}")
    
    print(f"\nTest graph:")
    print(f"  Nodes: {test_data.num_nodes:,}")
    print(f"  Edges: {test_data.num_edges:,}")
    print(f"  Edge features: {test_data.edge_attr.shape}")
    
    # Verify stratification
    print("\n=== Edge Label Distribution ===")
    print(f"Train: {torch.bincount(train_data.y).tolist()}")
    print(f"Valid: {torch.bincount(valid_data.y).tolist()}")
    print(f"Test:  {torch.bincount(test_data.y).tolist()}")
    
    # Memory usage info
    if device.type == 'cuda':
        print(f"\nGPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
