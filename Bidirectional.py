# Build reverse lookup
flow_dict = {}  # (src_ip, src_port, dst_ip, dst_port, protocol) → flow_id

for flow_id, flow in enumerate(flows):
    key = (flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port, flow.protocol)
    flow_dict[key] = flow_id

# Find bidirectional pairs
bidirectional_edges = []

for flow_id, flow in enumerate(flows):
    reverse_key = (flow.dst_ip, flow.dst_port, flow.src_ip, flow.src_port, flow.protocol)
    
    if reverse_key in flow_dict:
        reverse_id = flow_dict[reverse_key]
        # Avoid duplicate edges (only add if flow_id < reverse_id)
        if flow_id < reverse_id:
            bidirectional_edges.append((flow_id, reverse_id))
