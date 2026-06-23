"""Data loading, preprocessing, and the FlowNetwork disk store.

- :mod:`loader` : streaming reader for the raw gzipped JSONL profiles.
- :mod:`preprocess` : builds monthly ``FlowNetwork`` .pkl files from profiles.
- :mod:`flow_network_store` : loads/caches those .pkl files by month.
- :mod:`company_directory` : company attributes (industry/geography) used by
  semantic pooling.
"""

from talent_flow.core import FlowNetwork, merge_networks  # noqa: F401
from .flow_network_store import FlowNetworkStore  # noqa: F401

__all__ = ["FlowNetwork", "merge_networks", "FlowNetworkStore"]
