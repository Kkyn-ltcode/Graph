import polars as pl
import json
from pathlib import Path
import numpy as np
from datetime import datetime

def analyze_netflow_dataset(file_path, sample_size=1000000):
    """
    Comprehensive analysis of NetFlow dataset for graph construction
    Focus: Multi-class attack classification with GNN
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of rows to sample for detailed analysis (default 1M)
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
    print("\n=== Analyzing Attack Types (Multi-class Focus) ===")
    attack_counts = df.group_by("Attack").agg(pl.count()).sort("count", descending=True)
    results["attack_distribution"] = {
        "class_distribution": attack_counts.to_dicts(),
        "num_classes": len(attack_counts),
        "most_common": attack_counts[0, "Attack"],
        "least_common": attack_counts[-1, "Attack"],
        "class_balance_metrics": {
            "max_samples": int(attack_counts[0, "count"]),
            "min_samples": int(attack_counts[-1, "count"]),
            "imbalance_ratio": float(attack_counts[0, "count"] / attack_counts[-1, "count"])
        }
    }
    
    # Per-class percentage
    total = df.shape[0]
    attack_percentages = []
    for row in attack_counts.to_dicts():
        attack_percentages.append({
            "attack_type": row["Attack"],
            "count": row["count"],
            "percentage": float(row["count"] / total * 100)
        })
    results["attack_distribution"]["detailed_percentages"] = attack_percentages
    
    # === SAMPLE DETAILED ANALYSIS ===
    print(f"\n=== Sampling {sample_size:,} rows for detailed analysis ===")
    # With 1TB RAM, we can afford a larger sample
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
    
    # Potential node definitions - IP level
    ip_pairs = df_sample.group_by(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).agg(
        pl.count().alias("flow_count"),
        pl.col("Attack").n_unique().alias("unique_attacks")
    )
    results["graph_construction_insights"]["unique_ip_pairs"] = len(ip_pairs)
    results["graph_construction_insights"]["avg_flows_per_ip_pair"] = float(ip_pairs["flow_count"].mean())
    results["graph_construction_insights"]["ip_pairs_with_multiple_attack_types"] = int(
        (ip_pairs["unique_attacks"] > 1).sum()
    )
    
    # Graph density metrics
    num_unique_ips = df_sample.select([
        pl.col("IPV4_SRC_ADDR").unique().count().alias("src"),
        pl.col("IPV4_DST_ADDR").unique().count().alias("dst")
    ]).row(0)
    
    total_possible_edges = num_unique_ips[0] * num_unique_ips[1]
    actual_edges = len(ip_pairs)
    
    results["graph_construction_insights"]["graph_density"] = {
        "unique_src_ips": int(num_unique_ips[0]),
        "unique_dst_ips": int(num_unique_ips[1]),
        "possible_edges": int(total_possible_edges),
        "actual_edges": int(actual_edges),
        "density": float(actual_edges / total_possible_edges) if total_possible_edges > 0 else 0
    }
    
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
    print("\n=== Analyzing Temporal Patterns ===")
    if "FLOW_START_MILLISECONDS" in df_sample.columns:
        # Convert to datetime for better understanding
        min_time = df_sample["FLOW_START_MILLISECONDS"].min()
        max_time = df_sample["FLOW_START_MILLISECONDS"].max()
        time_range = max_time - min_time
        
        results["graph_construction_insights"]["temporal_analysis"] = {
            "time_range_ms": int(time_range),
            "time_range_seconds": float(time_range / 1000),
            "time_range_minutes": float(time_range / (1000 * 60)),
            "time_range_hours": float(time_range / (1000 * 3600)),
            "time_range_days": float(time_range / (1000 * 3600 * 24)),
            "min_timestamp": int(min_time),
            "max_timestamp": int(max_time)
        }
        
        # Temporal distribution of attacks
        df_sample_with_time = df_sample.with_columns([
            (pl.col("FLOW_START_MILLISECONDS") - min_time).alias("relative_time_ms")
        ])
        
        # Divide time into 10 bins and see attack distribution
        time_bins = 10
        df_sample_with_time = df_sample_with_time.with_columns([
            (pl.col("relative_time_ms") / (time_range / time_bins)).cast(pl.Int32).alias("time_bin")
        ])
        
        temporal_attack_dist = df_sample_with_time.group_by(["time_bin", "Attack"]).agg(
            pl.count().alias("count")
        ).sort(["time_bin", "count"], descending=[False, True])
        
        results["graph_construction_insights"]["temporal_attack_patterns"] = temporal_attack_dist.to_dicts()[:50]  # Top 50 patterns
    
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
    print("\n=== Computing Per-Attack-Type Feature Statistics ===")
    important_features = [
        "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS", "L4_DST_PORT", "PROTOCOL",
        "TCP_FLAGS", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
        "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS"
    ]
    
    attack_feature_stats = {}
    for attack_type in df_sample["Attack"].unique().to_list():
        attack_data = df_sample.filter(pl.col("Attack") == attack_type)
        attack_feature_stats[attack_type] = {
            "sample_count": len(attack_data)
        }
        
        for feature in important_features:
            if feature in df_sample.columns:
                try:
                    attack_feature_stats[attack_type][feature] = {
                        "mean": float(attack_data[feature].mean()) if attack_data[feature].mean() is not None else None,
                        "std": float(attack_data[feature].std()) if attack_data[feature].std() is not None else None,
                        "median": float(attack_data[feature].median()) if attack_data[feature].median() is not None else None
                    }
                except:
                    attack_feature_stats[attack_type][feature] = None
    
    results["graph_construction_insights"]["per_attack_feature_profiles"] = attack_feature_stats
    
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
        sample_size=1000000  # 1M samples - adjust based on needs (you have plenty of RAM!)
    )
