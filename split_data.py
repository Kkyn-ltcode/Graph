import torch
import numpy as np
from torch_geometric.data import Data

def split_graph(data, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split a graph into train/valid/test subgraphs where:
    - Train: only contains train nodes and their edges
    - Valid: contains train + valid nodes and their edges (excluding test)
    - Test: contains all nodes and edges
    
    Args:
        data: PyG Data object with x (node features), edge_index, y (labels)
        train_ratio: proportion of nodes for training
        valid_ratio: proportion of nodes for validation
        test_ratio: proportion of nodes for testing
        seed: random seed for reproducibility
    
    Returns:
        train_data, valid_data, test_data: Three Data objects
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_nodes = data.x.size(0)
    indices = np.random.permutation(num_nodes)
    
    # Calculate split sizes
    train_size = int(num_nodes * train_ratio)
    valid_size = int(num_nodes * valid_ratio)
    
    # Split node indices
    train_nodes = indices[:train_size]
    valid_nodes = indices[train_size:train_size + valid_size]
    test_nodes = indices[train_size + valid_size:]
    
    # Create node sets for efficient lookup
    train_set = set(train_nodes.tolist())
    valid_set = set(valid_nodes.tolist())
    test_set = set(test_nodes.tolist())
    
    # === Create Train Graph ===
    train_node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(train_nodes)}
    
    # Filter edges: both nodes must be in train set
    train_mask = torch.tensor([
        (edge[0].item() in train_set and edge[1].item() in train_set)
        for edge in data.edge_index.t()
    ])
    train_edges = data.edge_index[:, train_mask]
    
    # Remap edge indices to new node indices
    train_edges_remapped = torch.tensor([
        [train_node_mapping[edge[0].item()], train_node_mapping[edge[1].item()]]
        for edge in train_edges.t()
    ]).t()
    
    train_data = Data(
        x=data.x[train_nodes],
        edge_index=train_edges_remapped,
        y=data.y[train_nodes] if data.y is not None else None
    )
    
    # === Create Valid Graph (train + valid nodes) ===
    valid_all_nodes = np.concatenate([train_nodes, valid_nodes])
    valid_all_set = train_set | valid_set
    valid_node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_all_nodes)}
    
    # Filter edges: both nodes must be in train or valid set
    valid_mask = torch.tensor([
        (edge[0].item() in valid_all_set and edge[1].item() in valid_all_set)
        for edge in data.edge_index.t()
    ])
    valid_edges = data.edge_index[:, valid_mask]
    
    # Remap edge indices
    valid_edges_remapped = torch.tensor([
        [valid_node_mapping[edge[0].item()], valid_node_mapping[edge[1].item()]]
        for edge in valid_edges.t()
    ]).t()
    
    valid_data = Data(
        x=data.x[valid_all_nodes],
        edge_index=valid_edges_remapped,
        y=data.y[valid_all_nodes] if data.y is not None else None
    )
    
    # === Create Test Graph (all nodes) ===
    test_data = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        y=data.y.clone() if data.y is not None else None
    )
    
    # Store original indices for reference
    train_data.original_indices = torch.tensor(train_nodes)
    valid_data.original_indices = torch.tensor(valid_all_nodes)
    valid_data.train_mask = torch.tensor([i < len(train_nodes) for i in range(len(valid_all_nodes))])
    
    test_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_data.train_mask[train_nodes] = True
    test_data.valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_data.valid_mask[valid_nodes] = True
    test_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_data.test_mask[test_nodes] = True
    
    return train_data, valid_data, test_data


# Example usage
if __name__ == "__main__":
    # Create a sample graph
    num_nodes = 100
    num_features = 16
    num_edges = 300
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 3, (num_nodes,))  # 3 classes
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Split the graph
    train_data, valid_data, test_data = split_graph(data, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2)
    
    print(f"Original graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"\nTrain graph: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
    print(f"Valid graph: {valid_data.num_nodes} nodes, {valid_data.num_edges} edges")
    print(f"Test graph: {test_data.num_nodes} nodes, {test_data.num_edges} edges")
    
    print(f"\nTrain has only train nodes: {train_data.num_nodes}")
    print(f"Valid has train + valid nodes: {valid_data.num_nodes}")
    print(f"Test has all nodes: {test_data.num_nodes}")
