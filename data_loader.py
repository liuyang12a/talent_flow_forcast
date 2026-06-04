#!/usr/bin/env python3
"""
Data loader module for reading and loading job profile data from gzipped JSONL files.

This module provides efficient streaming access to large JSONL datasets,
with support for batch processing and filtering capabilities.
"""

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Callable, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """
    Data class representing a single job record.

    Attributes:
        id: Unique identifier for the profile
        title_raw: Raw job title as extracted from the source
        company_raw: Raw company name as extracted from the source
        job_start_date: Start date of the job in YYYY-MM format
        job_end_date: End date of the job in YYYY-MM format (empty if current)
        company: Normalized company ID (nullable)
        company_name: Normalized company name
        naics6_name: NAICS industry classification name
        naics6: NAICS industry classification code (nullable)
        title: Encoded/normalized title identifier
        title_name: Normalized job title name
        onet: O*NET-SOC code for occupational classification
        onet_name: O*NET-SOC occupation name
        city_raw: City location
        state_raw: State location
        country_raw: Country location
        is_current: Flag indicating if this is the current job
        soc_2021_2: SOC 2-digit code (JSON array string)
        soc_2021_2_name: SOC 2-digit name (JSON array string)
        soc_2021_3: SOC 3-digit code (JSON array string)
        soc_2021_3_name: SOC 3-digit name (JSON array string)
        soc_2021_4: SOC 4-digit code (JSON array string)
        soc_2021_4_name: SOC 4-digit name (JSON array string)
        soc_2021_5: SOC 5-digit code (JSON array string)
        soc_2021_5_name: SOC 5-digit name (JSON array string)
    """
    id: str
    title_raw: str
    company_raw: str
    job_start_date: str
    job_end_date: str
    company: Optional[int]
    company_name: str
    naics6_name: str
    naics6: Optional[int]
    title: str
    title_name: str
    onet: str
    onet_name: str
    city_raw: str
    state_raw: str
    country_raw: str
    is_current: bool
    soc_2021_2: str
    soc_2021_2_name: str
    soc_2021_3: str
    soc_2021_3_name: str
    soc_2021_4: str
    soc_2021_4_name: str
    soc_2021_5: str
    soc_2021_5_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        """
        Create a JobRecord instance from a dictionary.

        Args:
            data: Dictionary containing job record data

        Returns:
            JobRecord instance
        """
        return cls(
            id=data.get("ID", ""),
            title_raw=data.get("TITLE_RAW", ""),
            company_raw=data.get("COMPANY_RAW", ""),
            job_start_date=data.get("JOB_START_DATE", ""),
            job_end_date=data.get("JOB_END_DATE", ""),
            company=data.get("COMPANY"),
            company_name=data.get("COMPANY_NAME", ""),
            naics6_name=data.get("NAICS6_NAME", ""),
            naics6=data.get("NAICS6"),
            title=data.get("TITLE", ""),
            title_name=data.get("TITLE_NAME", ""),
            onet=data.get("ONET", ""),
            onet_name=data.get("ONET_NAME", ""),
            city_raw=data.get("CITY_RAW", ""),
            state_raw=data.get("STATE_RAW", ""),
            country_raw=data.get("COUNTRY_RAW", ""),
            is_current=data.get("IS_CURRENT", False),
            soc_2021_2=data.get("SOC_2021_2", ""),
            soc_2021_2_name=data.get("SOC_2021_2_NAME", ""),
            soc_2021_3=data.get("SOC_2021_3", ""),
            soc_2021_3_name=data.get("SOC_2021_3_NAME", ""),
            soc_2021_4=data.get("SOC_2021_4", ""),
            soc_2021_4_name=data.get("SOC_2021_4_NAME", ""),
            soc_2021_5=data.get("SOC_2021_5", ""),
            soc_2021_5_name=data.get("SOC_2021_5_NAME", ""),
        )


class DataLoader:
    """
    Data loader class for reading and processing gzipped JSONL files.

    This class provides memory-efficient streaming access to large datasets,
    with support for batch processing, filtering, and progress tracking.
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int = 1000,
        skip_errors: bool = True,
    ):
        """
        Initialize the DataLoader.

        Args:
            file_path: Path to the gzipped JSONL file
            batch_size: Number of records to yield per batch
            skip_errors: If True, skip malformed records and log warnings
        """
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.skip_errors = skip_errors
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the file exists and is readable."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

    def _open_file(self) -> gzip.GzipFile:
        """
        Open the gzipped file for reading.

        Returns:
            GzipFile object opened in text mode
        """
        return gzip.open(self.file_path, "rt", encoding="utf-8")

    def iter_records(self) -> Iterator[JobRecord]:
        """
        Iterate over all records in the file.

        Yields:
            JobRecord objects one at a time

        Example:
            >>> loader = DataLoader("data/profiles_jobs_new.jsonl.gz")
            >>> for record in loader.iter_records():
            ...     print(record.title_name)
        """
        with self._open_file() as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    data = json.loads(line.strip())
                    yield JobRecord.from_dict(data)
                except json.JSONDecodeError as e:
                    if self.skip_errors:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                    else:
                        raise ValueError(f"Malformed JSON at line {line_num}: {e}") from e
                except Exception as e:
                    if self.skip_errors:
                        logger.warning(f"Skipping record at line {line_num}: {e}")
                    else:
                        raise

    def iter_batches(self) -> Iterator[List[JobRecord]]:
        """
        Iterate over records in batches.

        Yields:
            Lists of JobRecord objects with size up to batch_size

        Example:
            >>> loader = DataLoader("data/profiles_jobs_new.jsonl.gz", batch_size=1000)
            >>> for batch in loader.iter_batches():
            ...     process_batch(batch)
        """
        batch = []
        for record in self.iter_records():
            batch.append(record)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        # Yield remaining records
        if batch:
            yield batch

    def load_all(self) -> List[JobRecord]:
        """
        Load all records into memory.

        WARNING: This method loads the entire dataset into memory.
        Only use for small datasets. For large files, use iter_records() or iter_batches().

        Returns:
            List of all JobRecord objects
        """
        return list(self.iter_records())

    def filter_records(
        self,
        predicate: Callable[[JobRecord], bool]
    ) -> Iterator[JobRecord]:
        """
        Filter records based on a predicate function.

        Args:
            predicate: Function that takes a JobRecord and returns True if it should be included

        Returns:
            Iterator over filtered JobRecord objects

        Example:
            >>> loader = DataLoader("data/profiles_jobs_new.jsonl.gz")
            >>> current_jobs = loader.filter_records(lambda r: r.is_current)
        """
        for record in self.iter_records():
            if predicate(record):
                yield record

    def count_total(self) -> int:
        """
        Count total number of records in the file.

        Returns:
            Total record count
        """
        count = 0
        for _ in self.iter_records():
            count += 1
        return count

    def get_unique_profiles(self) -> set:
        """
        Get set of unique profile IDs in the dataset.

        Returns:
            Set of unique profile IDs
        """
        unique_ids = set()
        for record in self.iter_records():
            unique_ids.add(record.id)
        return unique_ids

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute basic statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        total_records = 0
        unique_ids = set()
        current_jobs = 0
        companies = set()
        titles = set()

        for record in self.iter_records():
            total_records += 1
            unique_ids.add(record.id)
            if record.is_current:
                current_jobs += 1
            if record.company:
                companies.add(record.company)
            if record.title:
                titles.add(record.title)

        return {
            "total_records": total_records,
            "unique_profiles": len(unique_ids),
            "current_jobs": current_jobs,
            "unique_companies": len(companies),
            "unique_titles": len(titles),
            "average_jobs_per_profile": total_records / len(unique_ids) if unique_ids else 0,
        }


def load_data(
    file_path: str = "data/profiles_jobs_new.jsonl.gz",
    batch_size: int = 1000,
    skip_errors: bool = True,
) -> DataLoader:
    """
    Factory function to create a DataLoader instance.

    Args:
        file_path: Path to the gzipped JSONL file
        batch_size: Number of records to yield per batch
        skip_errors: If True, skip malformed records

    Returns:
        Configured DataLoader instance

    Example:
        >>> loader = load_data()
        >>> for record in loader.iter_records():
        ...     process(record)
    """
    return DataLoader(
        file_path=file_path,
        batch_size=batch_size,
        skip_errors=skip_errors,
    )


if __name__ == "__main__":
    # Example usage and basic statistics
    logger.info("Initializing data loader...")

    loader = load_data()

    logger.info("Computing dataset statistics...")
    stats = loader.get_statistics()

    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total records: {stats['total_records']:,}")
    print(f"Unique profiles: {stats['unique_profiles']:,}")
    print(f"Current jobs: {stats['current_jobs']:,}")
    print(f"Unique companies: {stats['unique_companies']:,}")
    print(f"Unique titles: {stats['unique_titles']:,}")
    print(f"Average jobs per profile: {stats['average_jobs_per_profile']:.2f}")
    print("=" * 50)
