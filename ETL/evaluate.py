"""
Cambridge Open Data Portal - Dataset Health Evaluation

This script evaluates the health and quality of datasets in the Cambridge ODP.
It performs two types of checks:
1. Static Health Checks: Programmatic checks (license, contact, update frequency, column descriptions)
2. AI Metadata Evaluation: LLM-based assessment of description quality, tag relevance, category fit

Database: ETL/data/cambridge_metadata.db
Tables Created: evaluations (stores health check results)
Tables Used: ODP_datasets (reads dataset metadata, updates last_evaluated_at)

Usage:
    python evaluate.py                  # Evaluate all unevaluated datasets
    python evaluate.py --limit 10       # Evaluate at most 10 datasets
    python evaluate.py --dry-run        # Preview without saving results

The script handles:
- Querying datasets that need evaluation (last_evaluated_at IS NULL)
- Running static health checks (fully implemented)
- Running AI evaluation (skeleton with TODOs for developers)
- Calculating overall health status (Healthy/Warning/Fail)
- Storing results and updating timestamps
"""

import sqlite3
import json
import os
import argparse
import logging
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

# Database path (matches ingest.py pattern)
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "cambridge_metadata.db")

# Evaluation thresholds (configurable)
THRESHOLDS = {
    "description_min_score": 0.5,       # AI score below this = Warning
    "description_fail_score": 0.3,      # AI score below this = Fail
    "tag_min_score": 0.5,               # Tag score below this = Warning
    "category_min_score": 0.5,          # Category score below this = Warning
    "update_late_multiplier": 2.0,      # Update >2x late = Fail
    "update_warning_multiplier": 1.2,   # Update >1.2x late = Warning
    "column_desc_min": 0.5,             # Column descriptions <50% = Warning
}

# Logging setup (matches ingest.py pattern)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# DATABASE INITIALIZATION
# ────────────────────────────────────────────────────────────

def init_evaluation_tables() -> None:
    """
    Initialize evaluations table and indexes.

    Creates:
    - evaluations table for storing health check results
    - Indexes for query optimization

    Note: The ODP_datasets table is created by ingest.py

    This function is idempotent - safe to run multiple times.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ─── evaluations Table (following README schema exactly) ───
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            evaluated_at TEXT DEFAULT (datetime('now')),

            -- AI metadata scoring (from llm_enrich.py) - 1-5 scale
            description_score REAL,                   -- LLM rating of description clarity (1-5)
            description_feedback TEXT,                -- LLM qualitative feedback on description
            description_suggestion TEXT,              -- LLM suggested improved description

            tag_score REAL,                           -- LLM rating of tag relevance (1-5)
            tag_feedback TEXT,                        -- LLM qualitative feedback on tags
            tag_suggestion TEXT,                      -- LLM suggested tags (JSON array)

            -- Calculated flags / indicators (from score_and_flag.py)
            description_exists INTEGER,               -- 0/1: Does description exist?
            tags_count_score INTEGER,                 -- 0-100: Tag count-based score
            license_exists INTEGER,                   -- 0/1: Is license populated?
            department_exists INTEGER,                -- 0/1: Is department populated?
            category_exists INTEGER,                  -- 0/1: Is category populated?
            days_overdue INTEGER,                     -- Number of days update is overdue

            freshness_score REAL,                     -- 0.0-1.0: Freshness score
            overall_health_score REAL,                -- 0.0-1.0: Weighted composite score
            overall_health_label TEXT,                -- 'good','fair','poor','critical'

            scored_at TEXT DEFAULT (datetime('now')), -- Timestamp when scoring completed

            FOREIGN KEY (dataset_id) REFERENCES ODP_datasets (dataset_id) ON DELETE CASCADE
        )
    """)

    # ─── Indexes for Query Optimization ───

    # Index for joining evaluations with datasets
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_eval_dataset
        ON evaluations (dataset_id, evaluated_at DESC)
    """)

    conn.commit()
    conn.close()
    logger.debug("Evaluations table initialized")

# ────────────────────────────────────────────────────────────
# DATABASE OPERATIONS
# ────────────────────────────────────────────────────────────

def fetch_unevaluated_datasets(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch datasets that need evaluation.

    Criteria for datasets needing evaluation:
    - last_evaluated_at IS NULL (never evaluated)
    - OR data_updated_at > last_evaluated_at (data refreshed since last eval)
    - OR metadata_updated_at > last_evaluated_at (metadata refreshed since last eval)

    Args:
        limit: Maximum number of datasets to fetch (None = all)

    Returns:
        List of dataset dictionaries with all fields needed for evaluation
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()

    query = """
        SELECT * FROM ODP_datasets
        WHERE last_evaluated_at IS NULL
           OR data_updated_at > last_evaluated_at
           OR metadata_updated_at > last_evaluated_at
        ORDER BY data_updated_at DESC
    """

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    logger.info(f"Found {len(results)} datasets needing evaluation")
    return results


def save_evaluation(dataset_id: str, evaluation_result: Dict[str, Any]) -> None:
    """
    Save evaluation results to database.

    This function performs two operations:
    1. Inserts evaluation results into the evaluations table (README schema)
    2. Updates the last_evaluated_at timestamp in ODP_datasets

    Args:
        dataset_id: The dataset to update
        evaluation_result: Dictionary with all README schema fields:
            - description_score (int 1-5, can be NULL)
            - description_feedback (str, can be NULL)
            - description_suggestion (str, can be NULL)
            - tag_score (int 1-5, can be NULL)
            - tag_feedback (str, can be NULL)
            - tag_suggestion (str JSON array, can be NULL)
            - description_exists (int 0/1)
            - tags_count_score (int 0-100)
            - license_exists (int 0/1)
            - department_exists (int 0/1)
            - category_exists (int 0/1)
            - days_overdue (int)
            - freshness_score (float 0.0-1.0)
            - overall_health_score (float 0.0-1.0)
            - overall_health_label (str: 'good'/'fair'/'poor'/'critical')
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Insert into evaluations table (README schema)
        cursor.execute("""
            INSERT INTO evaluations (
                dataset_id,
                description_score,
                description_feedback,
                description_suggestion,
                tag_score,
                tag_feedback,
                tag_suggestion,
                description_exists,
                tags_count_score,
                license_exists,
                department_exists,
                category_exists,
                days_overdue,
                freshness_score,
                overall_health_score,
                overall_health_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            evaluation_result['description_score'],
            evaluation_result['description_feedback'],
            evaluation_result['description_suggestion'],
            evaluation_result['tag_score'],
            evaluation_result['tag_feedback'],
            evaluation_result['tag_suggestion'],
            evaluation_result['description_exists'],
            evaluation_result['tags_count_score'],
            evaluation_result['license_exists'],
            evaluation_result['department_exists'],
            evaluation_result['category_exists'],
            evaluation_result['days_overdue'],
            evaluation_result['freshness_score'],
            evaluation_result['overall_health_score'],
            evaluation_result['overall_health_label']
        ))

        # Update last_evaluated_at timestamp in ODP_datasets
        # Note: We only reach this point if LLM enrichment succeeded (atomic operation)
        cursor.execute("""
            UPDATE ODP_datasets
            SET last_evaluated_at = ?
            WHERE dataset_id = ?
        """, (datetime.utcnow().isoformat(), dataset_id))

        conn.commit()
        logger.debug(f"Saved evaluation for {dataset_id}")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save evaluation for {dataset_id}: {e}")
        raise
    finally:
        conn.close()

# ────────────────────────────────────────────────────────────
# STATIC HEALTH CHECKS (integrated from evaluate_scripts/score_and_flag.py)
#
# Adapted from contributor file (now archived at evaluate_scripts/archive/score_and_flag.py)
# Refactored to use ODP_datasets table and README evaluations schema.
# ────────────────────────────────────────────────────────────

# Frequency thresholds for staleness calculation
FREQUENCY_THRESHOLDS = {
    "daily": 1,
    "weekly": 7,
    "biweekly": 14,
    "monthly": 30,
    "quarterly": 90,
    "annually": 365,
    "yearly": 365,
    "annual": 365,
    "as needed": None,
    "historical": None,
    "not planned": None,
    "never": None,
}

# Health scoring weights (from score_and_flag.py)
HEALTH_WEIGHTS = {
    "desc": 0.25,      # Description quality: 25%
    "tags": 0.15,      # Tag quality: 15%
    "license": 0.15,   # License presence: 15%
    "dept": 0.10,      # Department presence: 10%
    "category": 0.10,  # Category presence: 10%
    "freshness": 0.20, # Freshness: 20%
    "col_meta": 0.05,  # Column metadata: 5%
}


def compute_staleness(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically evaluate staleness relative to dataset's own update frequency.

    Args:
        dataset: Dataset dictionary from ODP_datasets table

    Returns:
        Dictionary with:
        - is_stale (int): 1 if stale, 0 if fresh
        - days_overdue (int): Number of days overdue (0 if not overdue, -1 if error)
        - freshness_score (float): 0-100 score (100=fresh, 50=1-2x late, 0=>2x late)
    """
    freq = (dataset.get("update_frequency") or "").strip().lower()
    threshold_days = FREQUENCY_THRESHOLDS.get(freq, 730)  # Default: 2 years

    if threshold_days is None:
        # Special frequencies that don't become stale
        return {"is_stale": 0, "days_overdue": 0, "freshness_score": 100.0}

    try:
        last_updated = dataset.get("data_updated_at") or dataset.get("updated_at")
        if not last_updated:
            return {"is_stale": 1, "days_overdue": -1, "freshness_score": 0.0}

        updated_dt = datetime.fromisoformat(str(last_updated).replace("Z", "+00:00"))
        days_since = (datetime.utcnow().replace(tzinfo=updated_dt.tzinfo) - updated_dt).days
        days_overdue = max(0, days_since - threshold_days)
        is_stale = 1 if days_overdue > 0 else 0

        # Calculate freshness score
        overdue_ratio = days_since / threshold_days
        if overdue_ratio <= 1.0:
            freshness_score = 100.0
        elif overdue_ratio <= 2.0:
            freshness_score = 50.0
        else:
            freshness_score = 0.0

        return {"is_stale": is_stale, "days_overdue": days_overdue, "freshness_score": freshness_score}

    except Exception as e:
        logger.debug(f"Could not calculate staleness: {e}")
        return {"is_stale": 1, "days_overdue": -1, "freshness_score": 0.0}


def compute_tag_score(tags: list) -> float:
    """
    Calculate tag score based on count.

    Scoring:
    - 0 tags → 0.0
    - 1-2 tags → 33.0
    - 3-4 tags → 67.0
    - 5+ tags → 100.0

    Args:
        tags: List of tag strings

    Returns:
        Float score 0.0-100.0
    """
    n = len(tags)
    if n == 0:
        return 0.0
    elif n <= 2:
        return 33.0
    elif n <= 4:
        return 67.0
    else:
        return 100.0


def compute_col_metadata_score(col_descriptions: list) -> float:
    """
    Calculate column metadata score.

    Checks if >=50% of columns have descriptions.

    Args:
        col_descriptions: List of column description strings

    Returns:
        100.0 if >= 50% columns have descriptions, else 0.0
    """
    if not col_descriptions:
        return 0.0

    filled = sum(1 for d in col_descriptions if d and str(d).strip())
    return 100.0 if (filled / len(col_descriptions)) >= 0.5 else 0.0


def health_band(score: float) -> str:
    """
    Convert numeric health score to health band label.

    Args:
        score: Health score 0-100

    Returns:
        One of: "Good", "Fair", "Poor", "Critical"
    """
    if score >= 80:
        return "Good"
    elif score >= 60:
        return "Fair"
    elif score >= 40:
        return "Poor"
    else:
        return "Critical"


def calculate_component_scores(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate all static health checks and calculate component scores.

    This is the main static scoring function that:
    1. Checks for missing fields (description, tags, license, department, category)
    2. Calculates staleness/freshness
    3. Scores tag count
    4. Scores column metadata completion
    5. Calculates weighted overall health score
    6. Determines health band

    Args:
        dataset: Dataset dictionary from ODP_datasets table

    Returns:
        Dictionary with all static check results matching README schema:
        {
            'description_exists': int (0/1),
            'tags_count_score': int (0-100),
            'license_exists': int (0/1),
            'department_exists': int (0/1),
            'category_exists': int (0/1),
            'days_overdue': int,
            'freshness_score': float (0-100),
            'overall_health_score': float (0-100),
            'overall_health_label': str ('Good'/'Fair'/'Poor'/'Critical')
        }
    """
    # Parse tags and column descriptions
    try:
        tags = json.loads(dataset.get("tags", "[]")) if dataset.get("tags") else []
    except json.JSONDecodeError:
        tags = []

    try:
        col_desc = json.loads(dataset.get("columns_descriptions", "[]")) if dataset.get("columns_descriptions") else []
    except json.JSONDecodeError:
        col_desc = []

    # Check for missing fields (binary flags)
    description_text = str(dataset.get("description", "")).strip()
    missing_desc = 0 if description_text else 1
    missing_tags = 0 if len(tags) > 0 else 1
    missing_lic = 0 if (dataset.get("license") and str(dataset.get("license")).strip()) else 1
    missing_dept = 0 if (dataset.get("department") and str(dataset.get("department")).strip()) else 1
    missing_cat = 0 if (dataset.get("category") and str(dataset.get("category")).strip()) else 1

    # Calculate description score (heuristic: 0/25/50/75 based on length)
    if len(description_text) == 0:
        desc_score_raw = 0.0
    elif len(description_text) < 30:
        desc_score_raw = 25.0
    elif len(description_text) < 100:
        desc_score_raw = 50.0
    else:
        desc_score_raw = 75.0

    # Calculate other component scores
    tag_score_raw = compute_tag_score(tags)
    license_score_raw = 0.0 if missing_lic else 100.0
    dept_score_raw = 0.0 if missing_dept else 100.0
    category_score_raw = 0.0 if missing_cat else 100.0
    staleness = compute_staleness(dataset)
    freshness_raw = staleness["freshness_score"]
    col_meta_raw = compute_col_metadata_score(col_desc)

    # Calculate weighted overall health score (0-100)
    overall_score = round(
        HEALTH_WEIGHTS["desc"] * desc_score_raw +
        HEALTH_WEIGHTS["tags"] * tag_score_raw +
        HEALTH_WEIGHTS["license"] * license_score_raw +
        HEALTH_WEIGHTS["dept"] * dept_score_raw +
        HEALTH_WEIGHTS["category"] * category_score_raw +
        HEALTH_WEIGHTS["freshness"] * freshness_raw +
        HEALTH_WEIGHTS["col_meta"] * col_meta_raw,
        1
    )

    # Return results mapped to README schema
    return {
        # Binary existence flags (inverted from missing_*)
        "description_exists": 1 - missing_desc,
        "license_exists": 1 - missing_lic,
        "department_exists": 1 - missing_dept,
        "category_exists": 1 - missing_cat,

        # Scores
        "tags_count_score": int(tag_score_raw),          # 0-100
        "days_overdue": staleness["days_overdue"],       # integer
        "freshness_score": freshness_raw,                # 0-100 (will be converted to 0.0-1.0 later)

        # Overall health
        "overall_health_score": overall_score,           # 0-100 (will be converted to 0.0-1.0 later)
        "overall_health_label": health_band(overall_score)  # Good/Fair/Poor/Critical (will be lowercased later)
    }

# ────────────────────────────────────────────────────────────
# AI EVALUATION FUNCTIONS (integrated from evaluate_scripts/llm_enrich.py)
#
# Adapted from contributor file (now archived at evaluate_scripts/archive/llm_enrich.py)
# Refactored to use setup/config.py GeminiClient and README evaluations schema.
# Note: human_approvals table from original file is out of scope (not implemented)
# ────────────────────────────────────────────────────────────

def build_evaluation_prompt_json(dataset: Dict[str, Any]) -> str:
    """
    Build consolidated JSON-structured prompt for all LLM evaluations.

    OPTIMIZED: Requests all 4 evaluations in a single prompt to reduce API calls.
    Uses strict prompts adapted from archived llm_enrich.py to prevent LLM hallucination.

    Requests:
    1. Description clarity scoring (1-5 + feedback)
    2. Description improvement suggestion
    3. Tag relevance scoring (1-5 + feedback)
    4. Tag suggestions (3-5 tags)

    Args:
        dataset: Dataset dictionary from ODP_datasets table

    Returns:
        JSON-formatted prompt string
    """
    # Extract dataset metadata
    title = dataset.get('title', 'N/A')
    description = dataset.get('description', '')
    department = dataset.get('department', 'Unknown')
    category = dataset.get('category', 'Unknown')

    try:
        tags = json.loads(dataset.get('tags', '[]'))
    except json.JSONDecodeError:
        tags = []

    try:
        cols = json.loads(dataset.get('columns_names', '[]'))
    except json.JSONDecodeError:
        cols = []

    # Format context data
    tag_str = ', '.join(tags) if tags else '(no tags)'
    col_str = ', '.join(cols[:10])
    if len(cols) > 10:
        col_str += ', ...'

    # Build comprehensive JSON prompt with strict rules
    prompt = f"""You are auditing metadata quality for a government open data portal.
Evaluate this dataset across 4 dimensions and return a single JSON object.

DATASET METADATA:
- Dataset Name: "{title}"
- Department: {department}
- Category: {category}
- Current Description: {description or "(empty)"}
- Current Tags: {tag_str}
- Column Names: {col_str}

════════════════════════════════════════════════════════════════════════════════
TASK 1: DESCRIPTION CLARITY SCORE
════════════════════════════════════════════════════════════════════════════════

STRICT RULES:
• Rate ONLY what is explicitly stated in the description provided
• Do NOT add information about what the dataset SHOULD contain
• Do NOT assume additional details not mentioned
• Do NOT invent population sizes, dates, or statistics
• If information is missing, that's what you rate - not a problem to fix

SCORING RUBRIC (integer 1-5):
  1 = Missing, empty, or completely uninformative
  2 = Present but extremely vague (e.g., "This dataset contains data")
  3 = Adequate — mentions the topic but lacks what, who, or when
  4 = Good — clearly states what data is collected, by whom, and for what purpose
  5 = Excellent — precise, complete, useful to a data consumer

Provide: Integer score (1-5) and 2-5 word feedback explaining the score.

════════════════════════════════════════════════════════════════════════════════
TASK 2: DESCRIPTION SUGGESTION
════════════════════════════════════════════════════════════════════════════════

You are a government data documentation specialist.

STRICT RULES:
• Use ONLY facts present in the metadata provided
• Do NOT invent statistics, dates, or coverage areas
• If the available metadata is insufficient to write a meaningful description,
  return exactly: "INSUFFICIENT_DATA"

Write a clear, factual single-sentence description (max 25 words) that:
• Clearly explains what data this dataset contains
• Mentions key fields or metrics (if known from columns/name)
• Describes potential use cases or purpose (if inferrable from category/name)

════════════════════════════════════════════════════════════════════════════════
TASK 3: TAG RELEVANCE SCORE
════════════════════════════════════════════════════════════════════════════════

Evaluate how well the current tags match this dataset.

SCORING GUIDELINES (integer 1-5):
  1 = Tags are missing, generic, or mostly unrelated
  2 = Tags are somewhat related but vague or weak
  3 = Tags are broadly correct but not very specific
  4 = Tags match the dataset well and are useful
  5 = Tags are highly accurate, specific, and clearly aligned with the dataset

STRICT RULES:
• Judge only based on dataset name, description, and the provided tags
• Do NOT assume extra context not written above
• PENALIZE generic tags like "data", "government", or overly broad labels
• REWARD tags that would help a user find this dataset

Provide: Integer score (1-5) and one sentence feedback.

════════════════════════════════════════════════════════════════════════════════
TASK 4: TAG SUGGESTIONS
════════════════════════════════════════════════════════════════════════════════

Suggest 3-5 relevant tags for this dataset.

TAG GUIDELINES - Only use tags that:
• Describe what the dataset ACTUALLY CONTAINS (based on name, description, category)
• Are specific, not generic (e.g., "vaccine-records" NOT "health")
• Are based on explicit information given - do NOT invent topics
• Use lowercase, single words or hyphenated phrases
• Are relevant to people searching for similar datasets

INVALID TAGS - DO NOT SUGGEST:
• Generic terms: "data", "information", "government", "dataset"
• Assumptions about content not mentioned
• Tags that duplicate the category already provided
• Made-up topics not in the name/description/category

If insufficient info, return: ["INSUFFICIENT_DATA"]

════════════════════════════════════════════════════════════════════════════════
REQUIRED JSON RESPONSE FORMAT
════════════════════════════════════════════════════════════════════════════════

{{
  "description_score": <integer 1-5>,
  "description_feedback": "<2-5 words explaining the score>",
  "description_suggestion": "<improved 2-3 sentence description OR 'INSUFFICIENT_DATA'>",
  "tag_score": <integer 1-5>,
  "tag_feedback": "<one sentence explaining the score>",
  "tag_suggestions": ["<tag1>", "<tag2>", "<tag3>", ...]
}}

CRITICAL REQUIREMENTS:
• Respond ONLY with the JSON object, no preamble or explanation
• Do NOT wrap output in markdown code fences
• Ensure all 6 fields are present
• Use double quotes for strings
• tag_suggestions must be an array of 3-5 strings
• Scores must be integers between 1 and 5
• Follow all STRICT RULES above to avoid hallucination
"""

    return prompt


def build_evaluation_prompt_json_compact(dataset: Dict[str, Any]) -> str:
    """
    Build a compact fallback prompt to reduce response size and avoid truncation.
    """
    base_prompt = build_evaluation_prompt_json(dataset)
    return base_prompt + """

ADDITIONAL OUTPUT CONSTRAINTS:
• Keep the entire response under 900 characters
• description_suggestion must be <= 25 words
• description_feedback must be <= 5 words
• tag_feedback must be <= 12 words
• tag_suggestions must contain exactly 3 tags
"""


def enrich_with_llm(dataset: Dict[str, Any], llm_client=None) -> Dict[str, Any]:
    """
    Perform LLM-based metadata enrichment using single JSON-structured API call.

    OPTIMIZED: Makes 1 API call instead of 4 separate calls.

    Evaluates:
    1. Description clarity (1-5 score + feedback)
    2. Description improvement suggestion
    3. Tag relevance (1-5 score + feedback)
    4. Tag suggestions (3-5 tags)

    Args:
        dataset: Dataset dictionary from ODP_datasets table
        llm_client: Optional pre-initialized LLM client (avoids re-creation per call)

    Returns:
        Dictionary with all LLM enrichment results matching README schema:
        {
            'description_score': int (1-5),
            'description_feedback': str,
            'description_suggestion': str,
            'tag_score': int (1-5),
            'tag_feedback': str,
            'tag_suggestion': str (JSON array)
        }
    """
    try:
        # Import config and get LLM client
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'setup'))
        from config import get_llm_client

        if llm_client is None:
            llm_client = get_llm_client()

        # Build consolidated JSON prompt
        prompt = build_evaluation_prompt_json(dataset)

        # Make single API call
        logger.debug(f"Calling LLM for dataset {dataset.get('dataset_id')}")
        response = llm_client.call_llm(prompt)

        def parse_llm_json_response(response_text: str) -> Dict[str, Any]:
            if not response_text or not response_text.strip():
                raise ValueError("LLM returned an empty response")

            cleaned_response = response_text.strip()

            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                pass

            fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
            if fenced_match:
                return json.loads(fenced_match.group(1))

            first_brace = cleaned_response.find("{")
            last_brace = cleaned_response.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                return json.loads(cleaned_response[first_brace:last_brace + 1])

            raise ValueError("LLM response did not contain a parseable JSON object")

        # Parse JSON response
        try:
            result = parse_llm_json_response(response)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.warning(f"Raw LLM response preview: {response[:1000]}")

            # Retry once with a compact prompt to avoid truncated responses
            logger.info("Retrying LLM call with compact JSON prompt")
            compact_prompt = build_evaluation_prompt_json_compact(dataset)
            retry_response = llm_client.call_llm(compact_prompt)

            try:
                result = parse_llm_json_response(retry_response)
            except (json.JSONDecodeError, ValueError) as retry_error:
                logger.error(f"Compact retry also failed JSON parsing: {retry_error}")
                logger.warning(f"Compact retry response preview: {retry_response[:1000]}")
                raise ValueError(f"LLM did not return valid JSON after retry: {retry_error}")

        # Validate required fields
        required_fields = [
            'description_score', 'description_feedback', 'description_suggestion',
            'tag_score', 'tag_feedback', 'tag_suggestions'
        ]

        missing_fields = [f for f in required_fields if f not in result]
        if missing_fields:
            raise ValueError(f"LLM response missing fields: {missing_fields}")

        # Validate score ranges
        if not (1 <= result['description_score'] <= 5):
            logger.warning(f"Description score out of range: {result['description_score']}, clamping to 1-5")
            result['description_score'] = max(1, min(5, result['description_score']))

        if not (1 <= result['tag_score'] <= 5):
            logger.warning(f"Tag score out of range: {result['tag_score']}, clamping to 1-5")
            result['tag_score'] = max(1, min(5, result['tag_score']))

        # Validate tag_suggestions is a list
        if not isinstance(result['tag_suggestions'], list):
            logger.warning(f"tag_suggestions is not a list: {type(result['tag_suggestions'])}")
            result['tag_suggestions'] = []

        # Handle INSUFFICIENT_DATA and ERROR responses (from strict prompts)
        # Convert INSUFFICIENT_DATA to None for description_suggestion
        if result['description_suggestion'] == 'INSUFFICIENT_DATA':
            logger.info(f"LLM returned INSUFFICIENT_DATA for description suggestion")
            result['description_suggestion'] = None
        # Convert ERROR: prefix responses to None
        elif isinstance(result['description_suggestion'], str) and result['description_suggestion'].startswith('ERROR:'):
            logger.warning(f"LLM returned error for description: {result['description_suggestion'][:100]}")
            result['description_suggestion'] = None

        # Convert ["INSUFFICIENT_DATA"] to empty list for tag_suggestions
        if result['tag_suggestions'] == ['INSUFFICIENT_DATA']:
            logger.info(f"LLM returned INSUFFICIENT_DATA for tag suggestions")
            result['tag_suggestions'] = []

        # Limit to 5 tags
        tag_suggestions = result['tag_suggestions'][:5]

        # Return in schema format (note: tag_suggestion is singular, not plural)
        return {
            'description_score': result['description_score'],
            'description_feedback': result['description_feedback'],
            'description_suggestion': result['description_suggestion'],
            'tag_score': result['tag_score'],
            'tag_feedback': result['tag_feedback'],
            'tag_suggestion': json.dumps(tag_suggestions)  # Store as JSON array string
        }

    except Exception as e:
        logger.error(f"LLM enrichment failed for {dataset.get('dataset_id')}: {e}")
        # Return NULL values on failure (graceful degradation)
        return {
            'description_score': None,
            'description_feedback': None,
            'description_suggestion': None,
            'tag_score': None,
            'tag_feedback': None,
            'tag_suggestion': None
        }

# ────────────────────────────────────────────────────────────
# RESULT AGGREGATION & STORAGE
# ────────────────────────────────────────────────────────────

def evaluate_dataset(dataset: Dict[str, Any], llm_client=None) -> Dict[str, Any]:
    """
    Perform complete evaluation of a single dataset.

    Execution order (per user requirement):
    1. Static scoring (score_and_flag.py logic) - runs first
    2. LLM enrichment (llm_enrich.py logic) - runs second
    3. Consolidation and mapping to README schema

    Args:
        dataset: Dataset dictionary from ODP_datasets table
        llm_client: Optional pre-initialized LLM client

    Returns:
        Complete evaluation result dictionary matching README schema,
        ready for database insertion into evaluations table
    """
    dataset_id = dataset.get('dataset_id', 'unknown')
    logger.debug(f"Evaluating: {dataset.get('title')} ({dataset_id})")

    # STEP 1: Static scoring (score_and_flag.py logic)
    logger.debug(f"Running static checks for {dataset_id}...")
    static_scores = calculate_component_scores(dataset)

    # STEP 2: LLM enrichment (llm_enrich.py logic)
    # IMPORTANT: This is atomic - if LLM fails, the entire evaluation is aborted
    logger.debug(f"Running LLM enrichment for {dataset_id}...")
    llm_scores = enrich_with_llm(dataset, llm_client=llm_client)

    # Check if LLM enrichment actually succeeded (not all NULL values)
    llm_succeeded = (
        llm_scores['description_score'] is not None or
        llm_scores['tag_score'] is not None
    )

    if not llm_succeeded:
        raise RuntimeError(
            f"LLM enrichment failed for {dataset_id} - both description_score and tag_score are NULL. "
            "Aborting evaluation (atomic operation - all or nothing)."
        )

    # STEP 3: Map to README schema
    evaluation = {
        # LLM fields (1-5 scale, can be NULL)
        'description_score': llm_scores['description_score'],
        'description_feedback': llm_scores['description_feedback'],
        'description_suggestion': llm_scores['description_suggestion'],
        'tag_score': llm_scores['tag_score'],
        'tag_feedback': llm_scores['tag_feedback'],
        'tag_suggestion': llm_scores['tag_suggestion'],

        # Static check fields (always present)
        'description_exists': static_scores['description_exists'],
        'tags_count_score': static_scores['tags_count_score'],
        'license_exists': static_scores['license_exists'],
        'department_exists': static_scores['department_exists'],
        'category_exists': static_scores['category_exists'],
        'days_overdue': static_scores['days_overdue'],

        # Convert freshness_score from 0-100 to 0.0-1.0
        'freshness_score': static_scores['freshness_score'] / 100.0,

        # Convert overall_health_score from 0-100 to 0.0-1.0
        'overall_health_score': static_scores['overall_health_score'] / 100.0,

        # Convert health_label to lowercase (Good → good, Fair → fair, etc.)
        'overall_health_label': static_scores['overall_health_label'].lower()
    }

    logger.debug(f"Evaluation complete for {dataset_id}: health={evaluation['overall_health_label']}, score={evaluation['overall_health_score']:.2f}")
    return evaluation

# ────────────────────────────────────────────────────────────
# COMMAND-LINE INTERFACE
# ────────────────────────────────────────────────────────────

def parse_arguments():
    """
    Parse command-line arguments.

    Supported flags:
    --limit N: Evaluate at most N datasets
    --dry-run: Show what would be evaluated without saving results
    """
    parser = argparse.ArgumentParser(
        description='Evaluate Cambridge ODP dataset health'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of datasets to evaluate (default: all)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview datasets that would be evaluated without saving results'
    )

    return parser.parse_args()

# ────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────

def main():
    """
    Main evaluation pipeline execution.

    Steps:
    1. Parse command-line arguments
    2. Fetch unevaluated datasets from database
    3. Evaluate each dataset (static checks + AI)
    4. Save results to evaluations table
    5. Update last_evaluated_at timestamps
    6. Print summary statistics
    """
    logger.info("=" * 60)
    logger.info("Cambridge Open Data Portal - Evaluation Pipeline")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_arguments()

    # Initialize evaluations table (if not exists)
    init_evaluation_tables()

    # Step 1: Fetch datasets needing evaluation
    logger.info(f"Step 1: Fetching datasets needing evaluation...")
    datasets = fetch_unevaluated_datasets(limit=args.limit)

    if not datasets:
        logger.info("No datasets need evaluation!")
        return

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No results will be saved")
        logger.info("=" * 60)
        logger.info(f"Would evaluate {len(datasets)} datasets:")
        for ds in datasets[:10]:  # Show first 10
            logger.info(f"  - {ds.get('title')}")
        if len(datasets) > 10:
            logger.info(f"  ... and {len(datasets) - 10} more")
        return

    # Step 2: Evaluate each dataset
    logger.info(f"Step 2: Evaluating {len(datasets)} datasets...")

    # Create LLM client once for all evaluations
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'setup'))
    from config import get_llm_client
    llm_client = get_llm_client()

    results_summary = {
        'healthy': 0,
        'warning': 0,
        'fail': 0,
        'errors': 0
    }

    for idx, dataset in enumerate(datasets, 1):
        try:
            title = dataset.get('title', 'Unknown')
            logger.info(f"[{idx}/{len(datasets)}] {title}")

            # Evaluate dataset
            evaluation_result = evaluate_dataset(dataset, llm_client=llm_client)

            # Save to database
            save_evaluation(dataset['dataset_id'], evaluation_result)

            # Rate-limit delay to avoid 429s (especially for GPT-4 with low TPM limits)
            time.sleep(2)

            # Update summary (map health labels to summary categories)
            label = evaluation_result['overall_health_label']  # 'good', 'fair', 'poor', or 'critical'
            if label == 'good':
                results_summary['healthy'] += 1
            elif label in ('fair', 'poor'):
                results_summary['warning'] += 1
            elif label == 'critical':
                results_summary['fail'] += 1

        except Exception as e:
            logger.error(f"Failed to evaluate {dataset.get('dataset_id')}: {e}")
            results_summary['errors'] += 1
            continue

    # Step 3: Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Total evaluated: {len(datasets)}")
    logger.info(f"  Healthy: {results_summary['healthy']}")
    logger.info(f"  Warning: {results_summary['warning']}")
    logger.info(f"  Fail: {results_summary['fail']}")
    logger.info(f"  Errors: {results_summary['errors']}")
    logger.info("=" * 60)
    logger.info(f"Database: {DB_PATH}")
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
