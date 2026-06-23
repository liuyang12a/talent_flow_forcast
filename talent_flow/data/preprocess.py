#!/usr/bin/env python3
"""
Preprocess Module for Extracting Talent Flow Networks from Job Data.

This module provides functions to identify employee transitions between companies
and construct time-windowed flow networks from job profile data.
"""

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from talent_flow.core import FlowNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_year_month(date_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse a year-month string into (year, month) tuple.

    Args:
        date_str: Date string in "YYYY-MM" format, or empty string

    Returns:
        Tuple of (year, month) or None if parsing fails
    """
    if not date_str or not date_str.strip():
        return None

    try:
        year, month = map(int, date_str.split('-'))
        if 1 <= month <= 12 and year > 1900:
            return (year, month)
    except (ValueError, AttributeError):
        pass

    return None


def year_month_to_datetime(year_month: Tuple[int, int]) -> datetime:
    """
    Convert (year, month) tuple to datetime object.

    Args:
        year_month: Tuple of (year, month)

    Returns:
        datetime object representing the first day of that month
    """
    return datetime(year_month[0], year_month[1], 1)


def datetime_to_year_month(dt: datetime) -> str:
    """
    Convert datetime to year-month string.

    Args:
        dt: datetime object

    Returns:
        String in "YYYY-MM" format
    """
    return dt.strftime("%Y-%m")


def add_months(year_month: Tuple[int, int], months: int) -> Tuple[int, int]:
    """
    Add months to a year-month tuple.

    Args:
        year_month: Tuple of (year, month)
        months: Number of months to add (can be negative)

    Returns:
        New (year, month) tuple
    """
    year, month = year_month
    total_months = (year * 12 + month - 1) + months
    new_year = total_months // 12
    new_month = (total_months % 12) + 1
    return (new_year, new_month)


def year_month_diff(
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> int:
    """
    Calculate the difference in months between two year-month tuples.

    Args:
        start: Start (year, month)
        end: End (year, month)

    Returns:
        Number of months difference (end - start)
    """
    return (end[0] * 12 + end[1]) - (start[0] * 12 + start[1])


def generate_time_windows(
    start_date: str,
    end_date: str,
    interval_months: int
) -> List[Tuple[str, str]]:
    """
    Generate a series of consecutive time windows.

    Args:
        start_date: Start date in "YYYY-MM" format (inclusive)
        end_date: End date in "YYYY-MM" format (inclusive)
        interval_months: Length of each time window in months

    Returns:
        List of (window_start, window_end) tuples in "YYYY-MM" format
    """
    start_ym = parse_year_month(start_date)
    end_ym = parse_year_month(end_date)

    if not start_ym or not end_ym:
        raise ValueError("Invalid date format. Expected 'YYYY-MM'")

    windows = []
    current_start = start_ym

    while current_start <= end_ym:
        # Window end is interval_months - 1 months after start
        # (e.g., 3-month interval: Jan-Mar, Apr-Jun, etc.)
        current_end = add_months(current_start, interval_months - 1)

        # Don't extend beyond the overall end date
        if current_end > end_ym:
            current_end = end_ym

        windows.append((
            f"{current_start[0]:04d}-{current_start[1]:02d}",
            f"{current_end[0]:04d}-{current_end[1]:02d}"
        ))

        # Move to next window (start of next interval)
        current_start = add_months(current_start, interval_months)

        if current_start > end_ym:
            break

    return windows


class JobRecord:
    """Internal class to represent a single job record with parsed dates."""

    def __init__(self, data: Dict):
        self.id = data.get("ID", "")
        self.title_raw = data.get("TITLE_RAW", "")
        self.company_raw = data.get("COMPANY_RAW", "")
        self.company = data.get("COMPANY")  # Normalized company ID
        self.company_name = data.get("COMPANY_NAME", "")
        self.job_start_date = parse_year_month(data.get("JOB_START_DATE", ""))
        self.job_end_date = parse_year_month(data.get("JOB_END_DATE", ""))
        self.is_current = data.get("IS_CURRENT", False)
        self.onet = data.get("ONET", "")
        self.onet_name = data.get("ONET_NAME", "")
        self.title_name = data.get("TITLE_NAME", "")

    def is_valid(self) -> bool:
        """Check if the record has necessary information for flow analysis."""
        # Must have a company ID and start date
        return self.company is not None and self.job_start_date is not None

    def get_effective_end_date(self, current_date: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Get the effective end date of the job.

        For current jobs without an end date, use the provided current date
        or a default far-future date.
        """
        if self.job_end_date:
            return self.job_end_date

        if current_date:
            return current_date

        # Default to a far-future date for current jobs
        return (2099, 12)


class CareerTimeline:
    """Represents a person's career timeline with multiple job records."""

    def __init__(self, person_id: str):
        self.person_id = person_id
        self.jobs: List[JobRecord] = []

    def add_job(self, job: JobRecord) -> None:
        """Add a job record to the timeline."""
        if job.is_valid():
            self.jobs.append(job)

    def sort_jobs(self) -> None:
        """Sort jobs by start date."""
        self.jobs.sort(key=lambda j: j.job_start_date or (0, 0))

    def identify_transitions(self) -> List[Dict]:
        """
        Identify job transitions from the sorted timeline.

        Returns:
            List of transition dictionaries, each containing:
            - from_company: Source company ID
            - to_company: Target company ID
            - from_company_name: Source company name
            - to_company_name: Target company name
            - transition_time: Time of transition (year, month)
            - from_job_end: End date of previous job
            - to_job_start: Start date of new job
        """
        transitions = []

        if len(self.jobs) < 2:
            return transitions

        self.sort_jobs()

        for i in range(len(self.jobs) - 1):
            current_job = self.jobs[i]
            next_job = self.jobs[i + 1]

            # Skip if same company (internal transfer, not a transition)
            if current_job.company == next_job.company:
                continue

            # Determine transition time:
            # If current job has an end date, use that
            # Otherwise, use the next job's start date
            if current_job.job_end_date:
                transition_time = current_job.job_end_date
            else:
                transition_time = next_job.job_start_date

            transitions.append({
                "person_id": self.person_id,
                "from_company": current_job.company,
                "to_company": next_job.company,
                "from_company_name": current_job.company_name,
                "to_company_name": next_job.company_name,
                "transition_time": transition_time,
                "from_job_end": current_job.job_end_date,
                "to_job_start": next_job.job_start_date,
                "from_title": current_job.title_name,
                "to_title": next_job.title_name,
            })

        return transitions


def load_and_group_by_person(
    file_path: str,
    max_records: Optional[int] = None
) -> Dict[str, CareerTimeline]:
    """
    Load job data and group by person ID.

    Args:
        file_path: Path to the gzipped JSONL file
        max_records: Maximum number of records to read (None for all)

    Returns:
        Dictionary mapping person IDs to CareerTimeline objects
    """
    timelines: Dict[str, CareerTimeline] = {}
    record_count = 0

    logger.info(f"Loading data from {file_path}...")

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if max_records and record_count >= max_records:
                break

            try:
                data = json.loads(line.strip())
                job = JobRecord(data)

                if not job.is_valid():
                    continue

                person_id = job.id
                if person_id not in timelines:
                    timelines[person_id] = CareerTimeline(person_id)

                timelines[person_id].add_job(job)
                record_count += 1

                if record_count % 100000 == 0:
                    logger.info(f"Processed {record_count:,} records...")

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON at line {record_count}")
                continue

    logger.info(f"Loaded {record_count:,} records for {len(timelines):,} people")

    return timelines


def extract_flow_networks(
    file_path: str,
    start_date: str,
    end_date: str,
    interval_months: int,
    max_records: Optional[int] = None
) -> List[Tuple[str, str, FlowNetwork]]:
    """
    Extract talent flow networks from job data across multiple time windows.

    This function:
    1. Loads job records and groups them by person
    2. Sorts each person's career timeline
    3. Identifies job transitions between companies
    4. Assigns transitions to appropriate time windows
    5. Constructs flow networks for each time window

    Args:
        file_path: Path to the gzipped JSONL file
        start_date: Scan start date in "YYYY-MM" format
        end_date: Scan end date in "YYYY-MM" format
        interval_months: Time window size in months
        max_records: Maximum records to read (None for all)

    Returns:
        List of (window_start, window_end, flow_network) tuples
    """
    # Generate time windows
    windows = generate_time_windows(start_date, end_date, interval_months)
    logger.info(f"Generated {len(windows)} time windows")

    # Initialize flow networks for each window
    window_networks: List[Tuple[str, str, FlowNetwork]] = [
        (w_start, w_end, FlowNetwork.empty()) for w_start, w_end in windows
    ]

    # Parse window boundaries for comparison
    window_boundaries = []
    for w_start, w_end, _ in window_networks:
        start = parse_year_month(w_start)
        end = parse_year_month(w_end)
        window_boundaries.append((start, end))

    # Load and process data
    timelines = load_and_group_by_person(file_path, max_records)

    logger.info("Identifying job transitions...")
    transition_count = 0

    for _, timeline in timelines.items():
        transitions = timeline.identify_transitions()

        for transition in transitions:
            transition_time = transition["transition_time"]
            from_company = transition["from_company"]
            to_company = transition["to_company"]

            # Find which time window this transition belongs to
            for idx, (w_start, w_end) in enumerate(window_boundaries):
                if w_start <= transition_time <= w_end:
                    window_networks[idx][2].add_edge(from_company, to_company, weight=1)
                    transition_count += 1
                    break

    logger.info(f"Identified {transition_count:,} job transitions")

    # Log summary for each window
    for w_start, w_end, network in window_networks:
        logger.info(
            f"Window {w_start} to {w_end}: "
            f"{network.get_node_count()} companies, "
            f"{network.get_edge_count()} transitions"
        )

    return window_networks


def extract_flow_networks_by_gap(
    file_path: str,
    start_date: str,
    end_date: str,
    interval_months: int,
    max_gap_months: int = 6,
    max_records: Optional[int] = None
) -> List[Tuple[str, str, FlowNetwork]]:
    """
    Extract talent flow networks with gap filtering.

    Only considers transitions where the gap between jobs is within max_gap_months.
    This helps filter out unrelated career changes (e.g., someone returning to
    workforce after years of absence).

    Args:
        file_path: Path to the gzipped JSONL file
        start_date: Scan start date in "YYYY-MM" format
        end_date: Scan end date in "YYYY-MM" format
        interval_months: Time window size in months
        max_gap_months: Maximum allowed gap between jobs (in months)
        max_records: Maximum records to read (None for all)

    Returns:
        List of (window_start, window_end, flow_network) tuples
    """
    windows = generate_time_windows(start_date, end_date, interval_months)
    logger.info(f"Generated {len(windows)} time windows")

    window_networks: List[Tuple[str, str, FlowNetwork]] = [
        (w_start, w_end, FlowNetwork.empty()) for w_start, w_end in windows
    ]

    window_boundaries = []
    for w_start, w_end, _ in window_networks:
        start = parse_year_month(w_start)
        end = parse_year_month(w_end)
        window_boundaries.append((start, end))

    timelines = load_and_group_by_person(file_path, max_records)

    logger.info(f"Identifying job transitions with max gap of {max_gap_months} months...")
    transition_count = 0
    filtered_count = 0

    for _, timeline in timelines.items():
        timeline.sort_jobs()

        for i in range(len(timeline.jobs) - 1):
            current_job = timeline.jobs[i]
            next_job = timeline.jobs[i + 1]

            if current_job.company == next_job.company:
                continue

            # Calculate gap between jobs
            if current_job.job_end_date and next_job.job_start_date:
                gap = year_month_diff(current_job.job_end_date, next_job.job_start_date)

                # Filter out transitions with large gaps
                if gap > max_gap_months:
                    filtered_count += 1
                    continue

            if current_job.job_end_date:
                transition_time = current_job.job_end_date
            else:
                transition_time = next_job.job_start_date

            for idx, (w_start, w_end) in enumerate(window_boundaries):
                if w_start <= transition_time <= w_end:
                    window_networks[idx][2].add_edge(
                        current_job.company,
                        next_job.company,
                        weight=1
                    )
                    transition_count += 1
                    break

    logger.info(f"Identified {transition_count:,} job transitions (filtered {filtered_count:,})")

    for w_start, w_end, network in window_networks:
        logger.info(
            f"Window {w_start} to {w_end}: "
            f"{network.get_node_count()} companies, "
            f"{network.get_edge_count()} transitions"
        )

    return window_networks


def save_flow_networks(
    networks: List[Tuple[str, str, FlowNetwork]],
    output_dir: str
) -> None:
    """
    Save flow networks to files.

    Args:
        networks: List of (window_start, window_end, network) tuples
        output_dir: Directory to save the networks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for w_start, w_end, network in networks:
        filename = f"flow_network_{w_start}_to_{w_end}.txt"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            f.write(f"# Flow Network: {w_start} to {w_end}\n")
            f.write(f"# Nodes: {network.get_node_count()}\n")
            f.write(f"# Edges: {network.get_edge_count()}\n")
            f.write(f"# Total Flow: {network.get_total_flow()}\n")
            f.write("# Format: from_company to_company weight\n")

            for source, target, weight in network.iter_edges():
                f.write(f"{source} {target} {weight}\n")

        logger.info(f"Saved network to {filepath}")


def extract_monthly_flow_networks(
    file_path: str,
    max_records: Optional[int] = None
) -> Dict[str, FlowNetwork]:
    """
    Extract talent flow networks grouped by month.

    This function processes all job records, identifies transitions between
    companies, and groups them by the transition month.

    Args:
        file_path: Path to the gzipped JSONL file
        max_records: Maximum records to read (None for all)

    Returns:
        Dictionary mapping "YYYY-MM" month strings to FlowNetwork objects
    """
    logger.info(f"Extracting monthly flow networks from {file_path}...")

    # Dictionary to hold networks for each month
    monthly_networks: Dict[str, FlowNetwork] = {}

    # Load and group data by person
    timelines = load_and_group_by_person(file_path, max_records)

    logger.info("Identifying job transitions by month...")
    transition_count = 0

    for _, timeline in timelines.items():
        transitions = timeline.identify_transitions()

        for transition in transitions:
            transition_time = transition["transition_time"]
            from_company = transition["from_company"]
            to_company = transition["to_company"]

            # Format as YYYY-MM
            month_key = f"{transition_time[0]:04d}-{transition_time[1]:02d}"

            # Create network for this month if not exists
            if month_key not in monthly_networks:
                monthly_networks[month_key] = FlowNetwork.empty()

            # Add the transition
            monthly_networks[month_key].add_edge(from_company, to_company, weight=1)
            transition_count += 1

    logger.info(
        f"Extracted {transition_count:,} transitions across "
        f"{len(monthly_networks)} months"
    )

    return monthly_networks


def save_monthly_flow_networks(
    monthly_networks: Dict[str, FlowNetwork],
    output_dir: str = "datasets/flow_networks",
    format: str = "pickle"
) -> None:
    """
    Save monthly flow networks to files.

    Args:
        monthly_networks: Dictionary mapping month strings to FlowNetwork objects
        output_dir: Directory to save the networks
        format: File format - "pickle" or "txt"
    """
    import pickle

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sort months chronologically
    sorted_months = sorted(monthly_networks.keys())

    logger.info(f"Saving {len(sorted_months)} monthly networks to {output_dir}...")

    for month in sorted_months:
        network = monthly_networks[month]

        if format == "pickle":
            filepath = output_path / f"{month}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(network, f)
        else:
            filepath = output_path / f"{month}.txt"
            with open(filepath, "w") as f:
                f.write(f"# Flow Network: {month}\n")
                f.write(f"# Nodes: {network.get_node_count()}\n")
                f.write(f"# Edges: {network.get_edge_count()}\n")
                f.write(f"# Total Flow: {network.get_total_flow()}\n")
                f.write("# Format: from_company to_company weight\n")

                for source, target, weight in network.iter_edges():
                    f.write(f"{source} {target} {weight}\n")

    logger.info(f"Saved {len(sorted_months)} monthly networks to {output_dir}")


def build_and_save_monthly_flow_networks(
    file_path: str = "datasets/profiles_jobs_new.jsonl.gz",
    output_dir: str = "datasets/flow_networks",
    format: str = "pickle",
    max_records: Optional[int] = None,
    use_tqdm: bool = True,
    batch_size: int = 100000
) -> None:
    """
    Build and save monthly flow networks with memory-efficient streaming.

    This function processes data in batches to minimize memory usage:
    1. Loads all records for a person before processing
    2. Generates transitions for completed persons
    3. Accumulates transitions by month
    4. Saves completed months and frees memory

    Args:
        file_path: Path to the gzipped JSONL file
        output_dir: Directory to save the networks
        format: File format - "pickle" or "txt"
        max_records: Maximum records to read (None for all)
        use_tqdm: Whether to use tqdm progress bars
        batch_size: Number of person records to process before saving
    """
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed, progress bars disabled. "
                      "Install with: pip install tqdm")
        use_tqdm = False

    logger.info(f"Building monthly flow networks from {file_path}...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {batch_size:,} people")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # First pass: count total records and group by person
    total_lines = max_records
    if max_records is None:
        logger.info("Counting total records...")
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            if use_tqdm:
                total_lines = sum(1 for _ in tqdm(f, desc="Counting"))
            else:
                total_lines = sum(1 for _ in f)
        logger.info(f"Total records: {total_lines:,}")

    # Track saved months to avoid re-saving
    saved_months: set = set()

    # Dictionary to hold networks for each month (cleared periodically)
    monthly_networks: Dict[str, FlowNetwork] = {}

    # Process in multiple passes - each pass processes a subset of people
    total_transitions = 0
    batch_count = 0

    # Phase 1: Stream process records and build timelines
    # Key fix: Use pending person to ensure complete timeline
    logger.info("Phase 1: Processing records...")

    record_count = 0
    current_batch_timelines: Dict[str, CareerTimeline] = {}
    last_person_id: Optional[str] = None

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        iterator = tqdm(f, total=total_lines, desc="Loading records") if use_tqdm else f

        for line in iterator:
            if max_records and record_count >= max_records:
                # Process the last person before breaking
                if last_person_id and last_person_id in current_batch_timelines:
                    _process_single_person(
                        last_person_id,
                        current_batch_timelines.pop(last_person_id),
                        monthly_networks
                    )
                break

            try:
                data = json.loads(line.strip())
                job = JobRecord(data)
                current_person_id = job.id if job.is_valid() else None

                # If we've moved to a new person, process the previous one
                if last_person_id and current_person_id != last_person_id:
                    if last_person_id in current_batch_timelines:
                        _process_single_person(
                            last_person_id,
                            current_batch_timelines.pop(last_person_id),
                            monthly_networks
                        )
                        total_transitions += 1  # Will be counted in transitions

                    # Check if we should save and clear memory
                    if len(current_batch_timelines) >= batch_size:
                        # Count actual transitions
                        batch_transition_count = _count_transitions_in_networks(monthly_networks)
                        batch_count += 1
                        logger.info(f"Processed batch {batch_count}: {batch_transition_count:,} transitions, "
                                  f"{len(monthly_networks)} months in memory")
                        _save_large_months(monthly_networks, saved_months, output_path)

                # Add job to timeline
                if job.is_valid():
                    if current_person_id not in current_batch_timelines:
                        current_batch_timelines[current_person_id] = CareerTimeline(current_person_id)
                    current_batch_timelines[current_person_id].add_job(job)
                    last_person_id = current_person_id

                record_count += 1

            except json.JSONDecodeError:
                continue

        # Process final person
        if last_person_id and last_person_id in current_batch_timelines:
            _process_single_person(
                last_person_id,
                current_batch_timelines.pop(last_person_id),
                monthly_networks
            )

    # Count transitions
    total_transitions = _count_transitions_in_networks(monthly_networks)

    logger.info(f"Phase 1 complete: {batch_count} batches, {total_transitions:,} total transitions")

    # Phase 2: Save any remaining months
    logger.info("Phase 2: Saving remaining months...")
    _save_months(monthly_networks, saved_months, output_path, format, use_tqdm)

    # Print summary
    print("\n" + "=" * 70)
    print("Monthly Flow Networks Summary")
    print("=" * 70)
    print(f"Total months: {len(saved_months)}")
    print(f"Total transitions: {total_transitions:,}")
    if len(saved_months) > 0:
        print(f"Average transitions per month: {total_transitions / len(saved_months):,.1f}")
    print("=" * 70)


def _process_single_person(
    person_id: str,
    timeline: CareerTimeline,
    monthly_networks: Dict[str, FlowNetwork]
) -> int:
    """Process a single person's timeline and add transitions to monthly networks."""
    transitions = timeline.identify_transitions()

    for transition in transitions:
        transition_time = transition["transition_time"]
        from_company = transition["from_company"]
        to_company = transition["to_company"]

        # Format as YYYY-MM
        month_key = f"{transition_time[0]:04d}-{transition_time[1]:02d}"

        # Create network for this month if not exists
        if month_key not in monthly_networks:
            monthly_networks[month_key] = FlowNetwork.empty()

        # Add the transition
        monthly_networks[month_key].add_edge(from_company, to_company, weight=1)

    return len(transitions)


def _count_transitions_in_networks(monthly_networks: Dict[str, FlowNetwork]) -> int:
    """Count total transitions across all monthly networks."""
    return sum(net.get_total_flow() for net in monthly_networks.values())


def _save_large_months(
    monthly_networks: Dict[str, FlowNetwork],
    saved_months: set,
    output_path: Path
) -> None:
    """Save large months to free memory."""
    import pickle

    months_saved = 0
    for month_key in list(monthly_networks.keys()):
        if month_key not in saved_months:
            network = monthly_networks[month_key]
            # Save if network is large enough
            if network.get_edge_count() > 1000:
                filepath = output_path / f"{month_key}.pkl"

                # If file exists, merge with existing data
                if filepath.exists():
                    with open(filepath, "rb") as f:
                        existing_network = pickle.load(f)
                    for source, target, weight in network.iter_edges():
                        for _ in range(weight):
                            existing_network.add_edge(source, target, weight=1)
                    network = existing_network

                with open(filepath, "wb") as f:
                    pickle.dump(network, f)

                saved_months.add(month_key)
                del monthly_networks[month_key]
                months_saved += 1

    if months_saved > 0:
        logger.info(f"Saved {months_saved} large months to free memory")


def _save_months(
    monthly_networks: Dict[str, FlowNetwork],
    saved_months: set,
    output_path: Path,
    format: str,
    use_tqdm: bool
) -> None:
    """Save all remaining monthly networks to files."""
    import pickle

    try:
        from tqdm import tqdm
    except ImportError:
        use_tqdm = False

    months_to_process = [m for m in monthly_networks.keys() if m not in saved_months]

    if not months_to_process:
        return

    logger.info(f"Saving {len(months_to_process)} remaining months...")

    month_iter = tqdm(months_to_process, desc="Saving months") if use_tqdm else months_to_process

    for month_key in month_iter:
        network = monthly_networks[month_key]
        filepath = output_path / f"{month_key}.pkl"

        # If file exists, merge with existing data
        if filepath.exists():
            with open(filepath, "rb") as f:
                existing_network = pickle.load(f)
            # Merge edges
            for source, target, weight in network.iter_edges():
                for _ in range(weight):
                    existing_network.add_edge(source, target, weight=1)
            network = existing_network

        # Save the network
        with open(filepath, "wb") as f:
            pickle.dump(network, f)

        saved_months.add(month_key)

    # Clear the dictionary to free memory
    monthly_networks.clear()


if __name__ == "__main__":
    import sys

    # Check if user wants to run the monthly extraction
    if "--monthly" in sys.argv:
        logger.info("Running monthly flow network extraction...")

        build_and_save_monthly_flow_networks(
            file_path="datasets/profiles_jobs_new.jsonl.gz",
            output_dir="datasets/flow_networks",
            format="pickle",  # or "txt"
            max_records=None,  # Process all records (set a number for testing)
            use_tqdm=True,
            batch_size=100000
        )
