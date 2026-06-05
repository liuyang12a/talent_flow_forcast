"""
Generate sample monthly flow network data for demonstration.

This script creates synthetic flow network data that mimics real
employee transition patterns for testing the forecasting framework.
"""

import json
import gzip
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np


def generate_company_names(n_companies: int = 20) -> List[str]:
    """Generate list of company names."""
    prefixes = ["Tech", "Data", "Cloud", "Cyber", "Smart", "Future", "Next", "Digital",
                "Global", "Innovation", "Advanced", "System", "Soft", "Net", "Info"]
    suffixes = ["Corp", "Inc", "Ltd", "Solutions", "Systems", "Tech", "Labs",
                "Group", "Networks", "Software", "Dynamics", "Ventures"]

    companies = set()
    while len(companies) < n_companies:
        name = f"{random.choice(prefixes)}{random.choice(suffixes)}"
        companies.add(name)
    return list(companies)


def generate_seasonal_pattern(month: int, base_value: float = 10.0) -> float:
    """Generate seasonal pattern for flow counts."""
    # Higher flows after year-end (job changes)
    seasonal = {
        1: 1.5,   # January - high turnover
        2: 1.3,
        3: 1.4,   # Spring hiring
        4: 1.2,
        5: 1.1,
        6: 1.0,
        7: 0.9,   # Summer low
        8: 0.9,
        9: 1.2,   # Fall hiring
        10: 1.1,
        11: 1.0,
        12: 1.3,  # Year-end
    }
    return base_value * seasonal.get(month, 1.0)


def generate_trend_component(timestamp_idx: int, trend_type: str = "stable") -> float:
    """Generate trend component."""
    if trend_type == "growing":
        return 1.0 + 0.02 * timestamp_idx
    elif trend_type == "declining":
        return 1.0 - 0.01 * timestamp_idx
    else:
        return 1.0


def generate_flow_network(
    companies: List[str],
    timestamp: str,
    timestamp_idx: int,
    density: float = 0.15
) -> Dict:
    """
    Generate a synthetic flow network for a given timestamp.

    Args:
        companies: List of company names
        timestamp: YYYY-MM format
        timestamp_idx: Index in time series (for trends)
        density: Edge density (0-1)

    Returns:
        Flow network dictionary
    """
    month = int(timestamp.split('-')[1])

    nodes = [{"id": c, "name": c} for c in companies]
    edges = []

    # Generate edges with realistic patterns
    n_possible_edges = len(companies) * (len(companies) - 1)
    n_edges = int(n_possible_edges * density)

    # Create some "hub" companies that attract more flows
    hubs = random.sample(companies, k=min(5, len(companies) // 4))

    generated = 0
    attempts = 0
    while generated < n_edges and attempts < n_edges * 10:
        attempts += 1

        source = random.choice(companies)
        target = random.choice(companies)

        if source == target:
            continue

        # Hub companies have higher probability
        if target in hubs:
            if random.random() > 0.7:
                continue

        # Generate flow with realistic pattern
        base_flow = random.uniform(1, 10)
        seasonal = generate_seasonal_pattern(month, base_flow)

        # Random trend type for this edge
        trend_type = random.choice(["stable", "growing", "declining"])
        trend = generate_trend_component(timestamp_idx, trend_type)

        # Add noise
        noise = np.random.normal(1.0, 0.2)

        flow_count = max(1, int(seasonal * trend * noise))

        edges.append({
            "source": source,
            "target": target,
            "count": flow_count,
            "weight": float(flow_count) / 100.0
        })
        generated += 1

    return {
        "timestamp": timestamp,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "n_companies": len(companies),
            "n_transitions": sum(e["count"] for e in edges),
            "density": density
        }
    }


def generate_dataset(
    start_date: str = "2022-01",
    end_date: str = "2024-06",
    n_companies: int = 20,
    output_dir: str = "data/flow_networks"
):
    """
    Generate complete sample dataset.

    Args:
        start_date: Start month (YYYY-MM)
        end_date: End month (YYYY-MM)
        n_companies: Number of companies
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate company list
    companies = generate_company_names(n_companies)

    # Generate timestamps
    start = datetime.strptime(start_date, "%Y-%m")
    end = datetime.strptime(end_date, "%Y-%m")

    timestamps = []
    current = start
    while current <= end:
        timestamps.append(current.strftime("%Y-%m"))
        # Add one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    print(f"Generating {len(timestamps)} months of data...")
    print(f"Companies: {n_companies}")
    print(f"Output: {output_dir}")
    print()

    # Generate data for each month
    total_transitions = 0
    for idx, timestamp in enumerate(timestamps):
        network = generate_flow_network(companies, timestamp, idx)

        # Save to file
        filename = f"flow_{timestamp}.json.gz"
        filepath = output_path / filename

        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(network, f, indent=2)

        n_trans = network["metadata"]["n_transitions"]
        total_transitions += n_trans
        print(f"  {timestamp}: {len(network['edges'])} edges, {n_trans} transitions")

    # Save company list
    with open(output_path / "companies.json", 'w') as f:
        json.dump(companies, f, indent=2)

    print()
    print(f"Dataset generation complete!")
    print(f"  Total months: {len(timestamps)}")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Avg transitions/month: {total_transitions / len(timestamps):.1f}")


if __name__ == "__main__":
    # Configuration
    CONFIG = {
        "start_date": "2022-01",
        "end_date": "2024-06",
        "n_companies": 25,
        "output_dir": "data/flow_networks"
    }

    print("="*60)
    print("Sample Flow Network Data Generator")
    print("="*60)
    print()

    generate_dataset(**CONFIG)
