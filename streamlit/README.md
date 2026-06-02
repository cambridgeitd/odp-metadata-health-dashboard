---
title: Cambridge Open Data Health Monitor
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
datasets:
  - spark-dd4g/odp-metadata-health
tags:
  - metadata
  - data-quality
  - open-data
  - dashboard
  - analytics
---

# Cambridge Open Data — Metadata Health Monitor

An interactive dashboard for monitoring the health and quality of metadata in the Cambridge Open Data Portal.

## Features

- **Health Scoring**: Automated scoring of dataset metadata quality
- **AI Suggestions**: LLM-generated descriptions and tags for datasets
- **Human Review Workflow**: Review and approve AI suggestions (session-based)
- **Comprehensive Analytics**: Visualizations of portal health metrics
- **Export Capabilities**: Download filtered views as CSV

## Data Source

This dashboard loads data from the [spark-dd4g/odp-metadata-health](https://huggingface.co/datasets/spark-dd4g/odp-metadata-health) dataset, which is updated daily by an automated pipeline that:

1. Fetches dataset metadata from the Cambridge Open Data Portal
2. Evaluates metadata completeness and quality
3. Generates AI suggestions for improvements
4. Publishes results to HuggingFace

## Usage

### Navigation

The dashboard has 6 tabs:

1. **Overview**: Portal-wide health metrics and distributions
2. **Action Queue**: Datasets needing attention, sorted by health score
3. **AI Descriptions**: Review AI-suggested descriptions (human approval workflow)
4. **AI Tags**: Review AI-suggested tags for datasets missing tags
5. **Spreadsheet View**: Full dataset table with export functionality
6. **Trends**: Historical trends and detailed analytics

### Filters

Use the sidebar to filter datasets by:
- Search query (dataset name)
- Department
- Category
- Health band (Critical, Poor, Fair, Good)
- Stale datasets only
- Missing license only
- Missing tags only

### Human Review Workflow

**⚠️ Important**: Approvals are stored in **session state only** and will be lost when you close the app or refresh the page. To preserve your work:

1. Review AI suggestions in the "AI Descriptions" or "AI Tags" tabs
2. Approve, edit, or reject suggestions
3. **Export your work from the "Spreadsheet View" tab before closing**
4. The exported CSV includes approval status and approved descriptions

## Technical Details

- **Framework**: Streamlit
- **Data Format**: Parquet (from HuggingFace Datasets)
- **Visualization**: Plotly
- **Caching**: 1-hour TTL on data loading
- **State Management**: Streamlit session state (non-persistent)

## Development

Built by the BU Spark! DD4G team in collaboration with the City of Cambridge.

**Project Repository**: [BU-Spark/dd4g-cambridge-meta-health](https://github.com/BU-Spark/dd4g-cambridge-meta-health)

## License

Apache 2.0 - See LICENSE file for details
