"""
Publish pipeline data to a HuggingFace dataset repository.

Exports a comprehensive joined dataset (ODP_datasets + latest evaluations)
as a Parquet file and pushes it to a HuggingFace dataset repository.

The exported dataset includes:
- All dataset metadata from ODP_datasets
- Latest evaluation scores and health metrics
- LLM enrichment results (description/tag suggestions)
- Static health flags and freshness scores

Usage:
    python publish.py                       # uses environment variables
    HF_TOKEN and HF_REPO must be set (or edit the placeholders below)

The script is safe to run repeatedly and handles missing evaluations gracefully.
"""

import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load .env file from project root
# publish.py location: dd4g-cambridge-meta-health/ETL/publish.py
# .env location: dd4g-cambridge-meta-health/.env
project_root = os.path.dirname(os.path.dirname(__file__))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional huggingface_hub import; library must be installed in the environment
try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover
    HfApi = None

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "cambridge_metadata.db")
EXPORT_PATH = os.path.join(os.path.dirname(__file__), "data", "cambridge_odp_health.parquet")

# Fill these via environment or hardcode for your repo
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO = os.getenv("HF_DATASET_REPO") or os.getenv("HF_REPO", "alixepstein/odp-metadata-health")
HF_OWNER = os.getenv("HF_OWNER", "alixepstein")

# Debug: Log configuration status (without exposing full token)
if HF_TOKEN:
    logger.debug(f"HF_TOKEN loaded (length: {len(HF_TOKEN)} chars)")
else:
    logger.debug("HF_TOKEN not set in .env file")
logger.debug(f"HF_REPO: {HF_REPO}")

# ────────────────────────────────────────────────────────────
# EXPORT FUNCTIONS
# ────────────────────────────────────────────────────────────

def export_to_parquet(db_path: str, output_path: str) -> int:
    """
    Export joined dataset (datasets + latest evaluations) to Parquet file.

    Creates a comprehensive dataset with all metadata and evaluation results,
    joining each dataset with its most recent evaluation.

    Args:
        db_path: Path to SQLite database
        output_path: Path for output Parquet file

    Returns:
        Number of rows exported

    Raises:
        FileNotFoundError: If database doesn't exist
        sqlite3.Error: If database query fails
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if evaluations table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='evaluations'
        """)
        evaluations_exists = cursor.fetchone() is not None

        if evaluations_exists:
            logger.info("Evaluations table found - creating joined dataset")

            # Comprehensive join query with all important fields
            query = """
            SELECT
                -- Dataset identification
                d.dataset_id,
                d.title,
                d.description,
                d.category,
                d.department,

                -- Dataset metadata
                d.license,
                d.contact_email,
                d.permalink,
                d.domain,

                -- Timestamps
                d.update_frequency,
                d.data_updated_at,
                d.updated_at,
                d.created_at,

                -- Usage metrics
                d.page_views_total,
                d.page_views_last_month,
                d.download_count,

                -- Tags and columns
                d.tags,
                d.columns_names,

                -- Pipeline state
                d.last_evaluated_at,

                -- Overall health assessment
                e.overall_health_label,
                e.overall_health_score,
                e.freshness_score,

                -- LLM description evaluation
                e.description_score,
                e.description_feedback,
                e.description_suggestion,

                -- LLM tag evaluation
                e.tag_score,
                e.tag_feedback,
                e.tag_suggestion,

                -- Static health flags
                e.description_exists,
                e.tags_count_score,
                e.license_exists,
                e.department_exists,
                e.category_exists,
                e.days_overdue,

                -- Evaluation metadata
                e.evaluated_at,
                e.scored_at

            FROM ODP_datasets d
            LEFT JOIN (
                SELECT dataset_id, MAX(id) as latest_id
                FROM evaluations
                GROUP BY dataset_id
            ) latest ON d.dataset_id = latest.dataset_id
            LEFT JOIN evaluations e ON latest.latest_id = e.id
            ORDER BY d.title
            """

        else:
            logger.warning("Evaluations table not found - exporting datasets only")

            # Fallback: export just datasets if evaluations don't exist yet
            query = """
            SELECT
                dataset_id,
                title,
                description,
                category,
                department,
                license,
                contact_email,
                permalink,
                domain,
                update_frequency,
                data_updated_at,
                updated_at,
                created_at,
                page_views_total,
                page_views_last_month,
                download_count,
                tags,
                columns_names,
                last_evaluated_at
            FROM ODP_datasets
            ORDER BY title
            """

        # Execute query and load into DataFrame
        logger.info("Executing export query...")
        df = pd.read_sql_query(query, conn)

        # Log dataset statistics
        logger.info(f"Loaded {len(df)} datasets")
        if evaluations_exists:
            evaluated_count = df['last_evaluated_at'].notna().sum()
            logger.info(f"  {evaluated_count} datasets with evaluations")
            logger.info(f"  {len(df) - evaluated_count} datasets not yet evaluated")

        # Export to Parquet
        logger.info(f"Writing to Parquet: {output_path}")
        df.to_parquet(output_path, index=False)

        file_size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"Export complete: {len(df)} rows, {file_size_kb:.2f} KB")

        return len(df)

    except sqlite3.Error as e:
        logger.error(f"Database error during export: {e}")
        raise

    finally:
        conn.close()


def publish_to_hf(export_path: str, repo: str, token: str) -> None:
    """
    Push the Parquet file to the specified HuggingFace dataset repo.

    Args:
        export_path: Path to the Parquet file to upload
        repo: HuggingFace dataset repository ID (e.g., "username/dataset-name")
        token: HuggingFace API token for authentication

    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If token is not provided
        Exception: If upload fails
    """
    if HfApi is None:
        raise ImportError(
            "huggingface_hub is not installed. "
            "Install with: pip install huggingface-hub"
        )

    if not token:
        raise ValueError("HF_TOKEN is not set; cannot authenticate to HuggingFace")

    if not os.path.exists(export_path):
        raise FileNotFoundError(f"Export file not found: {export_path}")

    logger.info("=" * 60)
    logger.info("Publishing to HuggingFace")
    logger.info("=" * 60)

    try:
        api = HfApi()

        # Create repository if it doesn't exist (harmless if it does)
        logger.info(f"Ensuring dataset repository exists: {repo}")
        try:
            api.create_repo(
                repo_id=repo,
                repo_type="dataset",
                token=token,
                exist_ok=True
            )
            logger.info(f"✓ Repository ready: {repo}")
        except Exception as e:
            logger.warning(f"Could not create/check repo {repo}: {e}")
            logger.warning("Attempting upload anyway...")

        # Upload file
        file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
        logger.info(f"Uploading {os.path.basename(export_path)} ({file_size_mb:.2f} MB)...")

        api.upload_file(
            path_or_fileobj=export_path,
            path_in_repo=os.path.basename(export_path),
            repo_id=repo,
            repo_type="dataset",
            token=token,
            create_pr=False
        )

        logger.info(f"✓ Published to HuggingFace: https://huggingface.co/datasets/{repo}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to publish to HuggingFace: {e}")
        raise


def validate_hf_repo_owner(repo: str, expected_owner: str) -> None:
    """
    Ensure the configured dataset repo belongs to the expected owner.

    Args:
        repo: Hugging Face repo id in owner/name format
        expected_owner: Expected owner/org name

    Raises:
        ValueError: If repo is malformed or owner does not match
    """
    if "/" not in repo:
        raise ValueError(
            f"HF repo '{repo}' is invalid. Expected format: owner/dataset-name"
        )

    actual_owner = repo.split("/", 1)[0].strip()
    expected_owner = (expected_owner or "").strip()
    if expected_owner and actual_owner.lower() != expected_owner.lower():
        raise ValueError(
            f"HF_REPO owner mismatch: expected '{expected_owner}', got '{actual_owner}'. "
            "Set HF_DATASET_REPO to your repo or update HF_OWNER intentionally."
        )


# ────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────

def main():
    """
    Main publish execution.

    Steps:
    1. Export joined dataset to Parquet file
    2. Publish to HuggingFace (if credentials provided)
    """
    logger.info("=" * 60)
    logger.info("CAMBRIDGE ODP - PUBLISH STEP")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    try:
        # Step 1: Export to Parquet
        logger.info("Step 1: Exporting dataset to Parquet")
        logger.info("-" * 60)
        row_count = export_to_parquet(DB_PATH, EXPORT_PATH)
        logger.info("")

        # Step 2: Publish to HuggingFace (if credentials available)
        if HF_TOKEN and HF_REPO:
            logger.info("Step 2: Publishing to HuggingFace")
            logger.info("-" * 60)
            validate_hf_repo_owner(HF_REPO, HF_OWNER)
            publish_to_hf(EXPORT_PATH, HF_REPO, HF_TOKEN)
        else:
            logger.warning("Step 2: Skipping HuggingFace upload")
            logger.warning("  HF_TOKEN or HF_REPO not configured")
            logger.warning("  Set these environment variables to enable publishing")
            logger.info("")

        # Summary
        logger.info("=" * 60)
        logger.info("PUBLISH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Datasets exported: {row_count}")
        logger.info(f"Output file: {EXPORT_PATH}")
        if HF_TOKEN and HF_REPO:
            logger.info(f"HuggingFace repo: {HF_REPO}")
        logger.info("=" * 60)

        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False
    except Exception as e:
        logger.error(f"Publish failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
