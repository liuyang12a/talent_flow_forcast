# Talent Flow - Employee Transition Network Analysis Tool

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Extract employee transition networks from job recruitment data and build time-windowed flow network analysis.

## Features

- 📊 **Data Loading**: Efficiently read gzipped JSONL format recruitment data
- 🔄 **Flow Networks**: Identify employee movement paths between companies
- 📈 **Statistical Analysis**: Generate statistical reports for flow networks
- ⏱️ **Time Windows**: Support for time-windowed data segmentation and analysis

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Quick install: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### One-Click Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/talent-flow.git
cd talent-flow

# 2. One-click run with uv (auto-creates venv and installs dependencies)
uv run python preprocess.py

# Or run with specific Python version
uv run --python 3.11 python statistic.py
```

### Common Commands

```bash
# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Run main program
uv run python preprocess.py

# Run statistics module
uv run python statistic.py

# Format code
uv run black *.py

# Lint code
uv run ruff check *.py

# Run tests
uv run pytest
```

## Project Structure

```
talent-flow/
├── data/               # Data directory (large files not committed to Git)
├── cache/              # Cache directory
├── data_loader.py      # Data loading module
├── flow_network.py     # Flow network module
├── preprocess.py       # Preprocessing main program
├── statistic.py        # Statistical analysis module
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # Project documentation
```

## Module Descriptions

### data_loader.py
Data loading module that provides efficient streaming access to gzipped JSONL recruitment data.

### flow_network.py
Core flow network module defining network structures and node/edge operations.

### preprocess.py
Preprocessing main program that extracts employee flow networks from raw data.

### statistic.py
Statistical analysis module that generates reports and visualization data for flow networks.

## Data Format

Input data should be in gzipped JSONL format, with each line containing a JSON object representing a job record:

```json
{
  "company": "Company Name",
  "employee_id": "Employee ID",
  "start_date": "2020-01",
  "end_date": "2023-06",
  "position": "Job Title"
}
```

## Configuration

Project configuration is managed via `pyproject.toml`:

- **Dependency Management**: Pure Python standard library, no third-party dependencies
- **Python Version**: >= 3.9
- **Dev Tools**: pytest, black, ruff

## Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

- Homepage: https://github.com/yourusername/talent-flow
- Issue Tracker: https://github.com/yourusername/talent-flow/issues
