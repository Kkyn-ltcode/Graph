import polars as pl
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def analyze_network_flow_dataset(file_path: str, output_path: str = "dataset_analysis.json"):
    """
    Comprehensive analysis of network flow dataset for graph construction.
    
    Args:
        file_path: Path to the CSV file
        output_path: Path to save the analysis JSON
    """
    
    print("Loading dataset...")
    # Read with Polars for speed
    df = pl.read_csv(file_path)
    
    analysis = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns
        }
    }
    
    # === BASIC STATISTICS ===
    print("Computing basic statistics...")
    analysis["basic_stats"] = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "memory_usage_mb": df.estimated_size("mb"),
        "null_counts": df.null_count().to_dicts()[0],
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    }
    
    # === LABEL ANALYSIS ===
    print("Analyzing labels...")
    label_dist = df.group_by("Label").agg(pl.count()).sort("count", descending=True)
    attack_dist = df.group_by("Attack").agg(pl.count()).sort("count", descending=True)
    
    analysis["label_distribution"] = {
        "binary": label_dist.to_dicts(),
        "multiclass": attack_dist.to_dicts(),
        "class_balance_ratio": {
            "binary": (label_dist["count"][0] / label_dist["count"].sum()).item() if len(label_dist) > 0 else 0,
            "most_common_attack_ratio": (attack_dist["count"][0] / attack_dist["count"].sum()).item() if len(attack_dist) > 0 else 0
        }
    }
    
    # === IP ADDRESS ANALYSIS (for graph construction) ===
    print("Analyzing IP addresses...")
    unique_src_ips = df["IPV4_SRC_ADDR"].n_unique()
    unique_dst_ips = df["IPV4_DST_ADDR"].n_unique()
    unique_ips_total = df.select(
        pl.concat_list(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
    ).explode("IPV4_SRC_ADDR").n_unique()
    
    # Top IPs by flow count
    top_src_ips = df.group_by("IPV4_SRC_ADDR").agg(pl.count().alias("flow_count")).sort("flow_count", descending=True).head(20)
    top_dst_ips = df.group_by("IPV4_DST_ADDR").agg(pl.count().alias("flow_count")).sort("flow_count", descending=True).head(20)
    
    analysis["ip_analysis"] = {
        "unique_source_ips": unique_src_ips,
        "unique_destination_ips": unique_dst_ips,
        "total_unique_ips": unique_ips_total,
        "top_20_source_ips": top_src_ips.to_dicts(),
        "top_20_destination_ips": top_dst_ips.to_dicts(),
        "avg_flows_per_src_ip": len(df) / unique_src_ips,
        "avg_flows_per_dst_ip": len(df) / unique_dst_ips
    }
    
    # === PORT ANALYSIS ===
    print("Analyzing ports...")
    top_src_ports = df.group_by("L4_SRC_PORT").agg(pl.count().alias("count")).sort("count", descending=True).head(20)
    top_dst_ports = df.group_by("L4_DST_PORT").agg(pl.count().alias("count")).sort("count", descending=True).head(20)
    
    analysis["port_analysis"] = {
        "unique_src_ports": df["L4_SRC_PORT"].n_unique(),
        "unique_dst_ports": df["L4_DST_PORT"].n_unique(),
        "top_20_src_ports": top_src_ports.to_dicts(),
        "top_20_dst_ports": top_dst_ports.to_dicts()
    }
    
    # === PROTOCOL ANALYSIS ===
    print("Analyzing protocols...")
    protocol_dist = df.group_by("PROTOCOL").agg(pl.count().alias("count")).sort("count", descending=True)
    l7_proto_dist = df.group_by("L7_PROTO").agg(pl.count().alias("count")).sort("count", descending=True).head(20)
    
    analysis["protocol_analysis"] = {
        "protocol_distribution": protocol_dist.to_dicts(),
        "l7_protocol_distribution": l7_proto_dist.to_dicts()
    }
    
    # === NUMERICAL FEATURES STATISTICS ===
    print("Computing numerical feature statistics...")
    numerical_cols = [
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
        "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
        "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
        "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"
    ]
    
    numerical_stats = {}
    for col in numerical_cols:
        if col in df.columns:
            stats = df.select(col).describe()
            numerical_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
                "q25": df[col].quantile(0.25),
                "q75": df[col].quantile(0.75),
                "null_count": df[col].null_count(),
                "zero_count": (df[col] == 0).sum()
            }
    
    analysis["numerical_features"] = numerical_stats
    
    # === GRAPH CONSTRUCTION INSIGHTS ===
    print("Computing graph construction insights...")
    
    # IP pair analysis (potential edges)
    ip_pairs = df.group_by(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).agg([
        pl.count().alias("flow_count"),
        pl.col("Label").first().alias("sample_label")
    ]).sort("flow_count", descending=True)
    
    analysis["graph_insights"] = {
        "unique_ip_pairs": len(ip_pairs),
        "avg_flows_per_ip_pair": len(df) / len(ip_pairs),
        "top_20_ip_pairs": ip_pairs.head(20).to_dicts(),
        "single_flow_pairs": (ip_pairs["flow_count"] == 1).sum(),
        "multi_flow_pairs": (ip_pairs["flow_count"] > 1).sum()
    }
    
    # Temporal analysis if available
    if "FLOW_DURATION_MILLISECONDS" in df.columns:
        duration_stats = {
            "total_duration_range_ms": (df["FLOW_DURATION_MILLISECONDS"].max() - df["FLOW_DURATION_MILLISECONDS"].min()),
            "avg_duration_ms": df["FLOW_DURATION_MILLISECONDS"].mean(),
            "median_duration_ms": df["FLOW_DURATION_MILLISECONDS"].median(),
            "short_flows_count": (df["FLOW_DURATION_MILLISECONDS"] < 1000).sum(),
            "medium_flows_count": ((df["FLOW_DURATION_MILLISECONDS"] >= 1000) & (df["FLOW_DURATION_MILLISECONDS"] < 10000)).sum(),
            "long_flows_count": (df["FLOW_DURATION_MILLISECONDS"] >= 10000).sum()
        }
        analysis["temporal_analysis"] = duration_stats
    
    # === ATTACK CORRELATION WITH FEATURES ===
    print("Analyzing attack patterns...")
    attack_feature_stats = {}
    for attack in df["Attack"].unique().to_list():
        if attack:
            attack_df = df.filter(pl.col("Attack") == attack)
            attack_feature_stats[str(attack)] = {
                "count": len(attack_df),
                "avg_bytes_in": attack_df["IN_BYTES"].mean(),
                "avg_bytes_out": attack_df["OUT_BYTES"].mean(),
                "avg_duration": attack_df["FLOW_DURATION_MILLISECONDS"].mean(),
                "avg_packets_in": attack_df["IN_PKTS"].mean(),
                "avg_packets_out": attack_df["OUT_PKTS"].mean(),
                "most_common_src_port": attack_df["L4_SRC_PORT"].mode().to_list()[0] if len(attack_df["L4_SRC_PORT"].mode()) > 0 else None,
                "most_common_dst_port": attack_df["L4_DST_PORT"].mode().to_list()[0] if len(attack_df["L4_DST_PORT"].mode()) > 0 else None
            }
    
    analysis["attack_patterns"] = attack_feature_stats
    
    # === FEATURE CORRELATIONS (sample for speed) ===
    print("Computing feature correlations (on sample)...")
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, seed=42)
    
    # Select numerical columns for correlation
    corr_cols = [col for col in numerical_cols if col in df_sample.columns]
    corr_matrix = df_sample.select(corr_cols).to_pandas().corr()
    
    # Find highly correlated features (>0.8 or <-0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": float(corr_val)
                })
    
    analysis["feature_correlations"] = {
        "high_correlation_pairs": high_corr_pairs,
        "sample_size": sample_size
    }
    
    # === RECOMMENDATIONS FOR GRAPH CONSTRUCTION ===
    print("Generating recommendations...")
    recommendations = {
        "edge_construction_options": [
            {
                "method": "IP_PAIR",
                "description": "Each edge represents communication between two IPs",
                "edge_count_estimate": len(ip_pairs),
                "pros": "Natural representation of network communication",
                "cons": "May lose temporal information"
            },
            {
                "method": "FLOW_AS_NODE",
                "description": "Each flow is a node, edges connect sequential/related flows",
                "node_count": len(df),
                "pros": "Preserves all flow information",
                "cons": "Large graph, need similarity metric for edges"
            },
            {
                "method": "HYBRID",
                "description": "IPs as nodes, flows as edge features",
                "estimated_nodes": unique_ips_total,
                "estimated_edges": len(ip_pairs),
                "pros": "Compact representation with rich edge features",
                "cons": "Aggregation needed for multiple flows"
            }
        ],
        "data_preprocessing_needed": [
            "IP address encoding (hash or categorical)",
            "Port number normalization",
            "Feature scaling (values vary widely)",
            "Handle class imbalance" if analysis["label_distribution"]["class_balance_ratio"]["binary"] > 0.9 else "Class balance acceptable",
            "Consider feature selection (high correlations detected)" if len(high_corr_pairs) > 10 else "Feature correlations manageable"
        ]
    }
    
    analysis["recommendations"] = recommendations
    
    # === SAVE RESULTS ===
    print(f"Saving analysis to {output_path}...")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    analysis = convert_to_serializable(analysis)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✓ Analysis complete! Results saved to {output_path}")
    print(f"\nQuick Summary:")
    print(f"  Total flows: {len(df):,}")
    print(f"  Unique IPs: {unique_ips_total:,}")
    print(f"  Unique IP pairs: {len(ip_pairs):,}")
    print(f"  Attack types: {len(attack_dist)}")
    print(f"  Binary class balance: {analysis['label_distribution']['class_balance_ratio']['binary']:.2%}")
    
    return analysis


if __name__ == "__main__":
    # Usage example
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv_file> [output_json_path]")
        print("\nExample: python script.py network_flows.csv analysis_output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dataset_analysis.json"
    
    analyze_network_flow_dataset(input_file, output_file)
