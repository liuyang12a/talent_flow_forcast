#!/usr/bin/env python3
"""
Statistics Module for Company Directory and Flow Analysis.

This module provides functionality to build a company directory from job data,
including mappings from company IDs to names, and statistics about talent
flows (in-degree and out-degree) for each company.

Results are cached to avoid recomputation on subsequent runs.
"""

import gzip
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from flow_network import FlowNetwork
from preprocess import extract_flow_networks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CompanyStats:
    """
    Statistics for a single company.

    Attributes:
        company_id: Unique company identifier
        company_name: Human-readable company name
        in_degree: Total incoming talent flow (number of people joining)
        out_degree: Total outgoing talent flow (number of people leaving)
        net_flow: Net flow (in_degree - out_degree)
        unique_sources: Set of companies that employees came from
        unique_targets: Set of companies that employees went to
    """
    company_id: int
    company_name: str
    in_degree: int = 0
    out_degree: int = 0
    net_flow: int = 0
    unique_sources: Set[int] = field(default_factory=set)
    unique_targets: Set[int] = field(default_factory=set)

    def update_in_degree(self, source_company: int, count: int = 1) -> None:
        """Update incoming flow statistics."""
        self.in_degree += count
        self.net_flow = self.in_degree - self.out_degree
        self.unique_sources.add(source_company)

    def update_out_degree(self, target_company: int, count: int = 1) -> None:
        """Update outgoing flow statistics."""
        self.out_degree += count
        self.net_flow = self.in_degree - self.out_degree
        self.unique_targets.add(target_company)


class CompanyDirectory:
    """
    Directory class for managing company information and flow statistics.

    This class maintains:
    - A mapping from company_id to company_name
    - Flow statistics (in-degree, out-degree) for each company
    - Caching mechanism for computed statistics

    Attributes:
        company_names: Dictionary mapping company_id to company_name
        company_stats: Dictionary mapping company_id to CompanyStats
        _cache_dir: Directory path for caching results
    """

    DEFAULT_CACHE_DIR = Path("cache")

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the CompanyDirectory.

        Args:
            cache_dir: Directory for caching results. Defaults to "cache".
        """
        self.company_names: Dict[int, str] = {}
        self.company_stats: Dict[int, CompanyStats] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_company_name(self, company_id: int) -> Optional[str]:
        """
        Get the name of a company by its ID.

        Args:
            company_id: Company ID to look up

        Returns:
            Company name if found, None otherwise
        """
        return self.company_names.get(company_id)

    def get_company_stats(self, company_id: int) -> Optional[CompanyStats]:
        """
        Get statistics for a specific company.

        Args:
            company_id: Company ID to look up

        Returns:
            CompanyStats object if found, None otherwise
        """
        return self.company_stats.get(company_id)

    def add_company_mapping(self, company_id: int, company_name: str) -> None:
        """
        Add or update a company ID to name mapping.

        Args:
            company_id: Company ID
            company_name: Company name
        """
        self.company_names[company_id] = company_name
        if company_id not in self.company_stats:
            self.company_stats[company_id] = CompanyStats(
                company_id=company_id,
                company_name=company_name
            )
        else:
            # Update name if stats already exist
            self.company_stats[company_id].company_name = company_name

    def update_from_flow_network(
        self,
        network: FlowNetwork,
        company_name_lookup: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Update statistics from a flow network.

        Args:
            network: FlowNetwork containing talent flow data
            company_name_lookup: Optional dictionary for company name lookup
        """
        for source, target, weight in network.iter_edges():
            # Ensure source company exists in directory
            if source not in self.company_stats:
                name = company_name_lookup.get(source, "") if company_name_lookup else ""
                self.add_company_mapping(source, name)

            # Ensure target company exists in directory
            if target not in self.company_stats:
                name = company_name_lookup.get(target, "") if company_name_lookup else ""
                self.add_company_mapping(target, name)

            # Update statistics
            self.company_stats[source].update_out_degree(target, weight)
            self.company_stats[target].update_in_degree(source, weight)

    def get_top_companies_by_in_degree(
        self,
        n: int = 10
    ) -> List[Tuple[int, str, int]]:
        """
        Get top companies by in-degree (most employees joining).

        Args:
            n: Number of top companies to return

        Returns:
            List of (company_id, company_name, in_degree) tuples
        """
        sorted_companies = sorted(
            self.company_stats.values(),
            key=lambda x: x.in_degree,
            reverse=True
        )
        return [
            (c.company_id, c.company_name, c.in_degree)
            for c in sorted_companies[:n]
        ]

    def get_top_companies_by_out_degree(
        self,
        n: int = 10
    ) -> List[Tuple[int, str, int]]:
        """
        Get top companies by out-degree (most employees leaving).

        Args:
            n: Number of top companies to return

        Returns:
            List of (company_id, company_name, out_degree) tuples
        """
        sorted_companies = sorted(
            self.company_stats.values(),
            key=lambda x: x.out_degree,
            reverse=True
        )
        return [
            (c.company_id, c.company_name, c.out_degree)
            for c in sorted_companies[:n]
        ]

    def get_top_companies_by_net_flow(
        self,
        n: int = 10
    ) -> List[Tuple[int, str, int]]:
        """
        Get top companies by net flow (in-degree - out-degree).

        Positive values indicate net talent gain, negative values indicate loss.

        Args:
            n: Number of top companies to return

        Returns:
            List of (company_id, company_name, net_flow) tuples
        """
        sorted_companies = sorted(
            self.company_stats.values(),
            key=lambda x: x.net_flow,
            reverse=True
        )
        return [
            (c.company_id, c.company_name, c.net_flow)
            for c in sorted_companies[:n]
        ]

    def get_all_companies(self) -> List[Tuple[int, str]]:
        """
        Get all companies in the directory.

        Returns:
            List of (company_id, company_name) tuples
        """
        return [
            (cid, name)
            for cid, name in self.company_names.items()
        ]

    def save_to_cache(self, cache_name: str = "company_directory.pkl") -> Path:
        """
        Save the company directory to cache.

        Args:
            cache_name: Name of the cache file

        Returns:
            Path to the cached file
        """
        cache_path = self._cache_dir / cache_name
        data = {
            'company_names': self.company_names,
            'company_stats': self.company_stats
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved company directory to {cache_path}")
        return cache_path

    @classmethod
    def load_from_cache(
        cls,
        cache_name: str = "company_directory.pkl",
        cache_dir: Optional[str] = None
    ) -> Optional["CompanyDirectory"]:
        """
        Load a company directory from cache.

        Args:
            cache_name: Name of the cache file
            cache_dir: Directory containing the cache file

        Returns:
            CompanyDirectory instance if cache exists, None otherwise
        """
        cache_path = Path(cache_dir) if cache_dir else cls.DEFAULT_CACHE_DIR
        cache_file = cache_path / cache_name

        if not cache_file.exists():
            logger.warning(f"Cache file not found: {cache_file}")
            return None

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        directory = cls(cache_dir=str(cache_path))
        directory.company_names = data['company_names']
        directory.company_stats = data['company_stats']

        logger.info(
            f"Loaded company directory from cache: "
            f"{len(directory.company_names)} companies"
        )
        return directory

    def __len__(self) -> int:
        """Return the number of companies in the directory."""
        return len(self.company_names)

    def __contains__(self, company_id: int) -> bool:
        """Check if a company ID is in the directory."""
        return company_id in self.company_names

    def __repr__(self) -> str:
        """Return a string representation of the directory."""
        return (
            f"CompanyDirectory("
            f"companies={len(self.company_names)}, "
            f"with_stats={len(self.company_stats)})"
        )


def build_company_directory_from_file(
    file_path: str = "data/profiles_jobs_new.jsonl.gz",
    cache_dir: str = "cache",
    max_records: Optional[int] = None,
    use_cache: bool = True
) -> CompanyDirectory:
    """
    Build a company directory from the job data file.

    This function scans the data file to build a mapping of company IDs to names,
    and optionally computes flow statistics if not using cache.

    Args:
        file_path: Path to the gzipped JSONL file
        cache_dir: Directory for caching results
        max_records: Maximum records to read (None for all)
        use_cache: Whether to use cached results if available

    Returns:
        Populated CompanyDirectory instance
    """
    cache_name = "company_directory.pkl"

    # Try to load from cache first
    if use_cache:
        cached = CompanyDirectory.load_from_cache(cache_name, cache_dir)
        if cached is not None:
            return cached

    logger.info(f"Building company directory from {file_path}...")

    directory = CompanyDirectory(cache_dir=cache_dir)
    record_count = 0
    unique_companies = set()

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if max_records and record_count >= max_records:
                break

            try:
                data = json.loads(line.strip())
                company = data.get("COMPANY")
                company_name = data.get("COMPANY_NAME", "")

                if company is not None:
                    company_id = int(company)
                    if company_id not in unique_companies:
                        unique_companies.add(company_id)
                        directory.add_company_mapping(company_id, company_name)

                record_count += 1

                if record_count % 500000 == 0:
                    logger.info(f"Processed {record_count:,} records, found {len(unique_companies)} companies...")

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error processing record at line {record_count}: {e}")
                continue

    logger.info(f"Built directory with {len(directory)} companies from {record_count:,} records")

    # Save to cache
    directory.save_to_cache(cache_name)

    return directory


def build_company_directory_with_stats(
    file_path: str = "data/profiles_jobs_new.jsonl.gz",
    start_date: str = "2019-01",
    end_date: str = "2021-12",
    interval_months: int = 6,
    cache_dir: str = "cache",
    max_records: Optional[int] = None,
    use_cache: bool = True
) -> CompanyDirectory:
    """
    Build a company directory with flow statistics.

    This function extracts flow networks from the data and computes
    in-degree and out-degree statistics for each company.

    Args:
        file_path: Path to the gzipped JSONL file
        start_date: Start date for flow extraction
        end_date: End date for flow extraction
        interval_months: Time window size in months
        cache_dir: Directory for caching results
        max_records: Maximum records to read (None for all)
        use_cache: Whether to use cached results if available

    Returns:
        CompanyDirectory with flow statistics
    """
    cache_name = f"company_directory_stats_{start_date}_{end_date}_{interval_months}.pkl"

    # Try to load from cache first
    if use_cache:
        cached = CompanyDirectory.load_from_cache(cache_name, cache_dir)
        if cached is not None:
            return cached

    logger.info("Building company directory with flow statistics...")

    # First, build the basic directory to get company names
    directory = build_company_directory_from_file(
        file_path=file_path,
        cache_dir=cache_dir,
        max_records=max_records,
        use_cache=use_cache
    )

    # Extract flow networks
    networks = extract_flow_networks(
        file_path=file_path,
        start_date=start_date,
        end_date=end_date,
        interval_months=interval_months,
        max_records=max_records
    )

    # Update statistics from all networks
    for w_start, w_end, network in networks:
        logger.info(f"Processing network for {w_start} to {w_end}...")
        directory.update_from_flow_network(network, directory.company_names)

    logger.info(
        f"Directory with stats: {len(directory)} companies, "
        f"{sum(s.in_degree for s in directory.company_stats.values())} total in-flow, "
        f"{sum(s.out_degree for s in directory.company_stats.values())} total out-flow"
    )

    # Save to cache
    directory.save_to_cache(cache_name)

    return directory


def print_directory_summary(directory: CompanyDirectory) -> None:
    """
    Print a summary of the company directory.

    Args:
        directory: CompanyDirectory to summarize
    """
    print("\n" + "=" * 70)
    print("Company Directory Summary")
    print("=" * 70)
    print(f"Total companies: {len(directory):,}")

    if directory.company_stats:
        total_in = sum(s.in_degree for s in directory.company_stats.values())
        total_out = sum(s.out_degree for s in directory.company_stats.values())
        print(f"Total incoming flows: {total_in:,}")
        print(f"Total outgoing flows: {total_out:,}")
        print(f"Net flow: {total_in - total_out:+,}")

        print("\n--- Top 10 Companies by In-Degree (Talent Attractors) ---")
        for cid, name, indeg in directory.get_top_companies_by_in_degree(10):
            print(f"  {cid}: {name or '(unknown)'} - {indeg:,} incoming")

        print("\n--- Top 10 Companies by Out-Degree (Talent Sources) ---")
        for cid, name, outdeg in directory.get_top_companies_by_out_degree(10):
            print(f"  {cid}: {name or '(unknown)'} - {outdeg:,} outgoing")

        print("\n--- Top 10 Companies by Net Flow (Net Gainers) ---")
        for cid, name, net in directory.get_top_companies_by_net_flow(10):
            sign = "+" if net > 0 else ""
            print(f"  {cid}: {name or '(unknown)'} - {sign}{net:,} net")

        print("\n--- Bottom 10 Companies by Net Flow (Net Losers) ---")
        sorted_by_net = sorted(
            directory.company_stats.values(),
            key=lambda x: x.net_flow
        )
        for c in sorted_by_net[:10]:
            sign = "+" if c.net_flow > 0 else ""
            print(f"  {c.company_id}: {c.company_name or '(unknown)'} - {sign}{c.net_flow:,} net")

    print("=" * 70)


if __name__ == "__main__":
    # Example usage: Build directory with statistics
    logger.info("Building company directory with statistics...")

    # First, just build the basic directory (company ID -> name mapping)
    directory_basic = build_company_directory_from_file(
        file_path="data/profiles_jobs_new.jsonl.gz",
        max_records=100000,  # Limit for quick testing
        use_cache=True
    )
    print(f"\nBasic directory: {len(directory_basic)} companies")

    # Example: Show some company mappings
    print("\nSample company mappings:")
    for cid, name in list(directory_basic.get_all_companies())[:10]:
        print(f"  {cid}: {name}")

    # Build directory with flow statistics
    # This requires extracting flow networks, so it takes longer
    logger.info("\nBuilding directory with flow statistics...")
    directory_with_stats = build_company_directory_with_stats(
        file_path="data/profiles_jobs_new.jsonl.gz",
        start_date="2019-01",
        end_date="2019-12",
        interval_months=6,
        max_records=100000,
        use_cache=True
    )

    print_directory_summary(directory_with_stats)
