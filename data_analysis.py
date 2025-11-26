import polars as pl
import json
from pathlib import Path
import numpy as np
from datetime import datetime

def analyze_netflow_dataset(file_path, sample_size=100000):
    """
    Comprehensive analysis of NetFlow dataset for graph construction
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of rows to sample for detailed analysis (default 100k)
    """
    
    print(f"Starting analysis at {datetime.now()}")
    print(f"Reading file: {file_path}")
    
    # Read the full dataset to get basic stats
    df = pl.read_csv(file_path)
    
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    results = {
        "metadata": {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "column_names": df.columns,
            "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024)
        },
        "label_distribution": {},
        "attack_distribution": {},
        "column_statistics": {},
        "flow_characteristics": {},
        "graph_construction_insights": {},
        "data_quality": {}
    }
    
    # === LABEL ANALYSIS ===
    print("\n=== Analyzing Labels ===")
    label_counts = df.group_by("Label").agg(pl.count()).sort("Label")
    results["label_distribution"] = {
        "binary_distribution": label_counts.to_dicts(),
        "imbalance_ratio": float(label_counts.filter(pl.col("Label") == 1)["count"][0] / 
                                 label_counts.filter(pl.col("Label") == 0)["count"][0])
    }
    
    # === ATTACK TYPE ANALYSIS ===
    print("\n=== Analyzing Attack Types ===")
    attack_counts = df.group_by("Attack").agg(pl.count()).sort("count", descending=True)
    results["attack_distribution"] = {
        "class_distribution": attack_counts.to_dicts(),
        "num_classes": len(attack_counts),
        "most_common": attack_counts[0, "Attack"],
        "least_common": attack_counts[-1, "Attack"]
    }
    
    # === SAMPLE DETAILED ANALYSIS ===
    print(f"\n=== Sampling {sample_size:,} rows for detailed analysis ===")
    df_sample = df.sample(n=min(sample_size, len(df)), seed=42)
    
    # === COLUMN-WISE STATISTICS ===
    print("\n=== Computing Column Statistics ===")
    numeric_cols = [col for col in df_sample.columns 
                   if df_sample[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32, pl.UInt32, pl.UInt64]]
    
    for col in numeric_cols:
        if col not in ["Label", "Attack"]:
            stats = df_sample[col].describe()
            results["column_statistics"][col] = {
                "mean": float(df_sample[col].mean()) if df_sample[col].mean() is not None else None,
                "std": float(df_sample[col].std()) if df_sample[col].std() is not None else None,
                "min": float(df_sample[col].min()) if df_sample[col].min() is not None else None,
                "max": float(df_sample[col].max()) if df_sample[col].max() is not None else None,
                "median": float(df_sample[col].median()) if df_sample[col].median() is not None else None,
                "null_count": int(df_sample[col].null_count()),
                "null_percentage": float(df_sample[col].null_count() / len(df_sample) * 100),
                "unique_count": int(df_sample[col].n_unique()),
                "zeros_count": int((df_sample[col] == 0).sum())
            }
    
    # === FLOW CHARACTERISTICS ===
    print("\n=== Analyzing Flow Characteristics ===")
    
    # IP Address uniqueness
    results["flow_characteristics"]["unique_src_ips"] = int(df_sample["IPV4_SRC_ADDR"].n_unique())
    results["flow_characteristics"]["unique_dst_ips"] = int(df_sample["IPV4_DST_ADDR"].n_unique())
    
    # Port analysis
    results["flow_characteristics"]["unique_src_ports"] = int(df_sample["L4_SRC_PORT"].n_unique())
    results["flow_characteristics"]["unique_dst_ports"] = int(df_sample["L4_DST_PORT"].n_unique())
    
    # Protocol distribution
    protocol_dist = df_sample.group_by("PROTOCOL").agg(pl.count()).sort("count", descending=True)
    results["flow_characteristics"]["protocol_distribution"] = protocol_dist.to_dicts()
    
    # L7 Protocol distribution
    l7_dist = df_sample.group_by("L7_PROTO").agg(pl.count()).sort("count", descending=True).head(10)
    results["flow_characteristics"]["top_l7_protocols"] = l7_dist.to_dicts()
    
    # Flow duration statistics
    results["flow_characteristics"]["flow_duration"] = {
        "mean_ms": float(df_sample["FLOW_DURATION_MILLISECONDS"].mean()),
        "median_ms": float(df_sample["FLOW_DURATION_MILLISECONDS"].median()),
        "max_ms": float(df_sample["FLOW_DURATION_MILLISECONDS"].max())
    }
    
    # Bytes and packets statistics
    results["flow_characteristics"]["traffic_volume"] = {
        "avg_in_bytes": float(df_sample["IN_BYTES"].mean()),
        "avg_out_bytes": float(df_sample["OUT_BYTES"].mean()),
        "avg_in_pkts": float(df_sample["IN_PKTS"].mean()),
        "avg_out_pkts": float(df_sample["OUT_PKTS"].mean()),
        "total_in_bytes_sample": int(df_sample["IN_BYTES"].sum()),
        "total_out_bytes_sample": int(df_sample["OUT_BYTES"].sum())
    }
    
    # === GRAPH CONSTRUCTION INSIGHTS ===
    print("\n=== Generating Graph Construction Insights ===")
    
    # Potential node definitions
    ip_pairs = df_sample.group_by(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).agg(pl.count())
    results["graph_construction_insights"]["unique_ip_pairs"] = len(ip_pairs)
    results["graph_construction_insights"]["avg_flows_per_ip_pair"] = float(ip_pairs["count"].mean())
    
    # Connection patterns
    src_connections = df_sample.group_by("IPV4_SRC_ADDR").agg(
        pl.col("IPV4_DST_ADDR").n_unique().alias("unique_destinations"),
        pl.count().alias("total_flows")
    )
    
    results["graph_construction_insights"]["source_ip_patterns"] = {
        "avg_destinations_per_src": float(src_connections["unique_destinations"].mean()),
        "max_destinations_per_src": int(src_connections["unique_destinations"].max()),
        "avg_flows_per_src": float(src_connections["total_flows"].mean())
    }
    
    dst_connections = df_sample.group_by("IPV4_DST_ADDR").agg(
        pl.col("IPV4_SRC_ADDR").n_unique().alias("unique_sources"),
        pl.count().alias("total_flows")
    )
    
    results["graph_construction_insights"]["dest_ip_patterns"] = {
        "avg_sources_per_dst": float(dst_connections["unique_sources"].mean()),
        "max_sources_per_dst": int(dst_connections["unique_sources"].max()),
        "avg_flows_per_dst": float(dst_connections["total_flows"].mean())
    }
    
    # Temporal patterns
    if "FLOW_START_MILLISECONDS" in df_sample.columns:
        time_range = df_sample["FLOW_START_MILLISECONDS"].max() - df_sample["FLOW_START_MILLISECONDS"].min()
        results["graph_construction_insights"]["temporal_span_ms"] = int(time_range)
        results["graph_construction_insights"]["temporal_span_hours"] = float(time_range / (1000 * 3600))
    
    # === DATA QUALITY CHECKS ===
    print("\n=== Checking Data Quality ===")
    
    # Missing values per column
    missing_summary = {}
    for col in df_sample.columns:
        null_count = df_sample[col].null_count()
        if null_count > 0:
            missing_summary[col] = {
                "count": int(null_count),
                "percentage": float(null_count / len(df_sample) * 100)
            }
    
    results["data_quality"]["missing_values"] = missing_summary
    
    # Check for suspicious values
    results["data_quality"]["suspicious_patterns"] = {
        "negative_bytes": int((df_sample["IN_BYTES"] < 0).sum() + (df_sample["OUT_BYTES"] < 0).sum()),
        "zero_duration_flows": int((df_sample["FLOW_DURATION_MILLISECONDS"] == 0).sum()),
        "zero_packet_flows": int(((df_sample["IN_PKTS"] == 0) & (df_sample["OUT_PKTS"] == 0)).sum())
    }
    
    # Feature correlations with label (for top features)
    print("\n=== Computing Feature Importance Indicators ===")
    important_features = [
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "L4_DST_PORT", "PROTOCOL"
    ]
    
    feature_label_stats = {}
    for feature in important_features:
        if feature in df_sample.columns:
            normal = df_sample.filter(pl.col("Label") == 0)[feature]
            attack = df_sample.filter(pl.col("Label") == 1)[feature]
            
            feature_label_stats[feature] = {
                "normal_mean": float(normal.mean()) if len(normal) > 0 else None,
                "attack_mean": float(attack.mean()) if len(attack) > 0 else None,
                "normal_std": float(normal.std()) if len(normal) > 0 else None,
                "attack_std": float(attack.std()) if len(attack) > 0 else None
            }
    
    results["graph_construction_insights"]["feature_label_relationship"] = feature_label_stats
    
    # === SAVE RESULTS ===
    output_file = "netflow_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print summary
    print(f"\nQUICK SUMMARY:")
    print(f"  • Total flows: {results['metadata']['total_rows']:,}")
    print(f"  • Features: {results['metadata']['total_columns']}")
    print(f"  • Attack ratio: {results['label_distribution']['imbalance_ratio']:.4f}")
    print(f"  • Unique IP pairs: {results['graph_construction_insights']['unique_ip_pairs']:,}")
    print(f"  • Attack classes: {results['attack_distribution']['num_classes']}")
    
    return results

# Usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "your_netflow_dataset.csv"
    
    results = analyze_netflow_dataset(
        file_path=file_path,
        sample_size=100000  # Adjust based on your memory
    )
