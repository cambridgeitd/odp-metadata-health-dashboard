import streamlit as st
import pandas as pd
import json
import os
import sqlite3
from datetime import datetime, timezone
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
from datasets import load_dataset

st.set_page_config(
    page_title="Cambridge Open Data — Health Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: white; border-radius: 10px; padding: 12px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

BAND_COLORS = {"Critical": "#e74c3c", "Poor": "#e67e22", "Fair": "#f1c40f", "Good": "#2ecc71"}
BAND_EMOJI  = {"Critical": "●", "Poor": "●", "Fair": "●", "Good": "●"}
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "alixepstein/odp-metadata-health")


# ── Data Transformation Helper Functions ──────────────────────────────────────
def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from CSV schema to app-expected schema."""
    rename_map = {
        'dataset_id': 'id',
        'title': 'name',
        'update_frequency': 'updateFrequency',
        'data_updated_at': 'dataUpdatedAt',
        'updated_at': 'updatedAt',
        'created_at': 'createdAt',
        'overall_health_score': 'health_score',
        'overall_health_label': 'health_band',
        'description_score': 'llm_desc_score',
        'description_feedback': 'llm_desc_feedback',
        'description_suggestion': 'llm_suggested_desc',
        'tag_suggestion': 'llm_suggested_tags',
        'tag_feedback': 'llm_tag_alignment_note'
    }

    # Only rename columns that exist and won't create duplicates
    safe_rename = {}
    for old_name, new_name in rename_map.items():
        if old_name in df.columns and new_name not in df.columns:
            safe_rename[old_name] = new_name

    df = df.rename(columns=safe_rename)

    # Handle tag_score specially: use tags_count_score if tag_score doesn't exist
    if 'tag_score' not in df.columns and 'tags_count_score' in df.columns:
        df['tag_score'] = df['tags_count_score']

    return df


def _scale_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Scale scores from 0.0-1.0 range to 0-100 range."""
    if 'health_score' in df.columns:
        # Handle None/NaN values - convert to numeric first
        df['health_score'] = pd.to_numeric(df['health_score'], errors='coerce')
        df['health_score'] = (df['health_score'] * 100).round(1)
    if 'freshness_score' in df.columns:
        df['freshness_score'] = pd.to_numeric(df['freshness_score'], errors='coerce')
        df['freshness_score'] = (df['freshness_score'] * 100).round(1)
    return df


def _is_tags_missing(tags_value) -> str:
    """Check if tags field is empty/null."""
    if pd.isna(tags_value):
        return "Yes"
    if not tags_value or tags_value == "[]" or tags_value == "":
        return "Yes"
    try:
        tags_list = json.loads(tags_value) if isinstance(tags_value, str) else tags_value
        return "No" if tags_list and len(tags_list) > 0 else "Yes"
    except:
        return "Yes"


def _calculate_desc_score(description) -> float:
    """
    Calculate description quality score based on length.
    Heuristic: 0 chars: 0 points, 1-50: 25, 51-150: 50, 151-300: 75, 300+: 100
    """
    if pd.isna(description) or not description:
        return 0.0
    length = len(str(description))
    if length == 0:
        return 0.0
    elif length <= 50:
        return 25.0
    elif length <= 150:
        return 50.0
    elif length <= 300:
        return 75.0
    else:
        return 100.0


def _calculate_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fields that don't exist in CSV but are needed by app."""
    # Binary flag conversions (0/1 → "No"/"Yes")
    if 'description_exists' in df.columns:
        df['missing_description'] = df['description_exists'].apply(
            lambda x: "No" if x == 1 else "Yes"
        )

    if 'license_exists' in df.columns:
        df['missing_license'] = df['license_exists'].apply(
            lambda x: "No" if x == 1 else "Yes"
        )
        df['license_score'] = df['license_exists'].apply(
            lambda x: 100 if x == 1 else 0
        )

    if 'department_exists' in df.columns:
        df['missing_department'] = df['department_exists'].apply(
            lambda x: "No" if x == 1 else "Yes"
        )

    if 'category_exists' in df.columns:
        df['missing_category'] = df['category_exists'].apply(
            lambda x: "No" if x == 1 else "Yes"
        )

    # Tags handling
    if 'tags' in df.columns:
        df['missing_tags'] = df['tags'].apply(_is_tags_missing)

    # Staleness calculation
    if 'days_overdue' in df.columns:
        df['is_stale'] = df['days_overdue'].apply(
            lambda x: "Yes" if pd.notna(x) and x > 0 else "No"
        )

    # Description score heuristic
    if 'description' in df.columns:
        df['desc_score'] = df['description'].apply(_calculate_desc_score)

    # LLM status
    if 'llm_suggested_desc' in df.columns:
        df['llm_status'] = df['llm_suggested_desc'].apply(
            lambda x: "pending_review" if pd.notna(x) and str(x).strip() else "not_evaluated"
        )

    # Capitalize health_band values
    if 'health_band' in df.columns:
        df['health_band'] = df['health_band'].str.capitalize()

    return df


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """
    Load and transform data from local SQLite database.

    Returns:
        Transformed DataFrame matching the expected schema for the app
    """
    try:
        db_path = os.path.join(os.path.dirname(__file__), '..', 'ETL', 'data', 'cambridge_metadata.db')
        db_path = os.path.normpath(db_path)

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            query = """
                SELECT d.*, e.evaluated_at, e.description_score, e.description_feedback,
                       e.description_suggestion, e.tag_score, e.tag_feedback, e.tag_suggestion,
                       e.description_exists, e.tags_count_score, e.license_exists,
                       e.department_exists, e.category_exists, e.days_overdue,
                       e.freshness_score, e.overall_health_score, e.overall_health_label
                FROM ODP_datasets d
                LEFT JOIN evaluations e ON d.dataset_id = e.dataset_id
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            print(f"Loaded {len(df)} rows from local database")
        else:
            # Spaces deployment uses the published dataset instead of local SQLite files.
            hf_dataset = load_dataset(HF_DATASET_REPO, split="train")
            df = hf_dataset.to_pandas()
            print(f"Loaded {len(df)} rows from Hugging Face dataset: {HF_DATASET_REPO}")

        print(f"Available columns: {df.columns.tolist()}")

        # Step 2: Column Renames
        df = _rename_columns(df)
        print(f"Columns after rename: {df.columns.tolist()}")

        # Step 3: Scale Numeric Fields
        df = _scale_scores(df)

        # Step 4: Calculate Missing Fields
        df = _calculate_derived_fields(df)

        # Step 5: Ensure numeric columns are proper numeric types (only if they exist)
        numeric_cols = ['health_score', 'freshness_score', 'llm_desc_score',
                       'tag_score', 'desc_score', 'license_score', 'days_overdue']
        for col in numeric_cols:
            if col in df.columns and not df[col].empty:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass  # Skip columns that can't be converted

        return df

    except Exception as e:
        st.error(f"Failed to load data from database: {e}")

        import traceback
        with st.expander("Show full error details"):
            st.code(traceback.format_exc())

        return pd.DataFrame()


def save_approval(dataset_id: str, approved_desc: Optional[str],
                 approved_tags: Optional[str], status: str,
                 reviewer: str, note: str = "") -> None:
    """
    Save approval decision to session state (replaces SQLite writes).

    Args:
        dataset_id: Dataset ID
        approved_desc: Approved description text (or None if rejected)
        approved_tags: Approved tags JSON (or None if rejected)
        status: One of 'approved', 'edited', 'rejected'
        reviewer: Name of reviewer
        note: Optional reviewer note
    """
    st.session_state.human_approvals[dataset_id] = {
        'approved_description': approved_desc,
        'approved_tags': approved_tags,
        'human_status': status,
        'reviewed_by': reviewer,
        'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'edit_note': note
    }

    # Increment counter to force re-render
    st.session_state.approval_counter += 1


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# Initialize session state for human approvals
if 'human_approvals' not in st.session_state:
    st.session_state.human_approvals = {}
    # Structure: {dataset_id: {
    #   'approved_description': str,
    #   'approved_tags': str,
    #   'human_status': str,  # 'approved', 'edited', 'rejected'
    #   'reviewed_by': str,
    #   'reviewed_at': str (ISO format),
    #   'edit_note': str
    # }}

if 'approval_counter' not in st.session_state:
    st.session_state.approval_counter = 0
    # Used to force re-renders when approvals change

df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("⚠️ No data available. The dataset may be temporarily unavailable.")
    st.info("Please try refreshing the page in a few minutes.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Cambridge Open Data\n**Metadata Health Monitor**")
    st.markdown("---")
    st.markdown("#### Filters")

    search_query = st.text_input("Search dataset name", "")

    dept_options   = sorted(df["department"].dropna().unique().tolist())
    selected_depts = st.multiselect("Department", dept_options, default=[])

    cat_options   = sorted(df["category"].dropna().unique().tolist())
    selected_cats = st.multiselect("Category", cat_options, default=[])

    selected_bands = st.multiselect(
        "Health Band", ["Critical", "Poor", "Fair", "Good"],
        default=["Critical", "Poor", "Fair", "Good"]
    )

    stale_only   = st.toggle("Stale datasets only",  value=False)
    no_lic_only  = st.toggle("Missing license only",    value=False)
    no_tags_only = st.toggle("Missing tags only", value=False)

    st.markdown("---")
    st.caption("Cambridge Open Data Portal\nReinhard Engels, Open Data Program\n\nData refreshed daily from pipeline")

# Apply filters
filtered = df.copy()
if search_query:
    filtered = filtered[filtered["name"].str.contains(search_query, case=False, na=False)]
if selected_depts:
    filtered = filtered[filtered["department"].isin(selected_depts)]
if selected_cats:
    filtered = filtered[filtered["category"].isin(selected_cats)]
if selected_bands:
    filtered = filtered[filtered["health_band"].isin(selected_bands)]
# Only filter stale if updateFrequency requires updates
if stale_only:
    active_freqs_temp = filtered[~filtered["updateFrequency"].isin(["historical", "as needed", "never", "not planned"])]
    filtered = active_freqs_temp[active_freqs_temp["is_stale"] == "Yes"]
if no_lic_only:
    filtered = filtered[filtered["missing_license"] == "Yes"]
if no_tags_only:
    filtered = filtered[filtered["missing_tags"] == "Yes"]

# ── Header & KPIs ─────────────────────────────────────────────────────────────
st.markdown("## Cambridge Open Data — Metadata Health Dashboard")
st.caption(f"Showing **{len(filtered)}** of **{len(df)}** datasets after filters")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Datasets",    len(filtered))
k2.metric("Avg Health Score",  f"{filtered['health_score'].mean():.1f}" if not filtered.empty else "—")
k3.metric("Critical",          int((filtered["health_band"] == "Critical").sum()))
# Only count stale for datasets that need regular updates
stale_count_filtered = filtered[~filtered["updateFrequency"].isin(["historical", "as needed", "never", "not planned"])]
k4.metric("Stale",          int((stale_count_filtered["is_stale"] == "Yes").sum()))
k5.metric("Missing License",   int((filtered["missing_license"] == "Yes").sum()))
k6.metric("Missing Tags", int((filtered["missing_tags"] == "Yes").sum()))

st.markdown("---")

pending_count = len(filtered[filtered["llm_status"] == "pending_review"])
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Action Queue",
    f"AI Descriptions ({pending_count} pending)",
    "AI Tags",
    "Spreadsheet View",
    "Trends"
])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Portal Health Overview")
    c1, c2 = st.columns(2)

    band_counts = filtered["health_band"].value_counts().reset_index()
    band_counts.columns = ["Band", "Count"]
    fig_donut = px.pie(band_counts, names="Band", values="Count",
                       color="Band", color_discrete_map=BAND_COLORS,
                       hole=0.5, title="Health Band Distribution")
    fig_donut.update_traces(textposition="outside", textinfo="percent+label")
    c1.plotly_chart(fig_donut, use_container_width=True)

    missing = {
        "Tags":        int((filtered["missing_tags"] == "Yes").sum()),
        "License":     int((filtered["missing_license"] == "Yes").sum()),
        "Description": int((filtered["missing_description"] == "Yes").sum()),
        "Department":  int((filtered["missing_department"] == "Yes").sum()),
        "Category":    int((filtered["missing_category"] == "Yes").sum()),
    }
    fig_missing = px.bar(x=list(missing.keys()), y=list(missing.values()),
                         labels={"x": "Field", "y": "# Datasets Missing"},
                         title="Missing Metadata Fields",
                         color=list(missing.values()), color_continuous_scale="Reds")
    fig_missing.update_layout(coloraxis_showscale=False)
    c2.plotly_chart(fig_missing, use_container_width=True)

    # Filter out NaN values for histogram
    filtered_health = filtered[filtered["health_score"].notna()].copy()
    if not filtered_health.empty:
        fig_hist = px.histogram(filtered_health, x="health_score", nbins=20,
                                title="Health Score Distribution",
                                labels={"health_score": "Health Score"},
                                color_discrete_sequence=["#3498db"])
        fig_hist.add_vline(x=60, line_dash="dash", line_color="orange", annotation_text="Fair threshold")
        fig_hist.add_vline(x=80, line_dash="dash", line_color="green",  annotation_text="Good threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No health score data available for histogram.")

    if filtered["department"].notna().any():
        dept_health = (filtered.groupby("department")["health_score"]
                       .mean().reset_index()
                       .rename(columns={"health_score": "Avg Health Score"})
                       .sort_values("Avg Health Score").head(20))
        fig_dept = px.bar(dept_health, x="Avg Health Score", y="department",
                          orientation="h",
                          title="Average Health Score by Department (Bottom 20)",
                          color="Avg Health Score",
                          color_continuous_scale="RdYlGn", range_color=[0, 100])
        st.plotly_chart(fig_dept, use_container_width=True)

    # Only show stale status for datasets that need regular updates
    # Exclude historical, as-needed, never, not planned (these don't expire)
    active_freqs = filtered[~filtered["updateFrequency"].isin(["historical", "as needed", "never", "not planned"])]
    
    stale_freq = active_freqs.groupby("updateFrequency")["is_stale"].apply(
                  lambda x: (x == "Yes").sum()
                  ).reset_index()
    stale_total = active_freqs.groupby("updateFrequency").size().reset_index(name="Total")
    stale_freq = stale_freq.merge(stale_total, on="updateFrequency")
    stale_freq.columns = ["Frequency", "Stale", "Total"]
    stale_freq = stale_freq.reset_index(drop=True)
    stale_freq["Pct Stale"] = (stale_freq["Stale"] / stale_freq["Total"] * 100).round(1)
    stale_freq = stale_freq.dropna(subset=["Frequency"]).sort_values("Pct Stale", ascending=False)
    fig_stale = px.bar(stale_freq, x="Frequency", y="Pct Stale",
                       title="% Stale Datasets by Update Frequency",
                       labels={"Pct Stale": "% Stale"},
                       color="Pct Stale", color_continuous_scale="Reds")
    st.plotly_chart(fig_stale, use_container_width=True)

# ── TAB 2: ACTION QUEUE ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Datasets Needing Attention")
    st.caption("Sorted by health score (worst first). Expand a row for details.")
    action_df = filtered.sort_values("health_score").head(50)

    if action_df.empty:
        st.success("No datasets match the current filters.")
    else:
        for _, row in action_df.iterrows():
            emoji = BAND_EMOJI.get(row["health_band"], "\u26aa")
            label = f"{emoji} {row['name']}  —  Score: {row['health_score']}  |  {row['health_band']}"
            with st.expander(label):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Department:** {row['department'] or 'Missing'}")
                c1.markdown(f"**Category:** {row['category'] or 'Missing'}")
                c1.markdown(f"**License:** {row['license'] or 'Missing'}")
                c2.markdown(f"**Update Frequency:** {row['updateFrequency'] or 'Unknown'}")
                c2.markdown(f"**Last Updated:** {str(row['dataUpdatedAt'] or row['updatedAt'] or 'Unknown')[:10]}")
                # Only show stale status for datasets that need regular updates
                is_active_freq = row["updateFrequency"] not in ["historical", "as needed", "never", "not planned"]
                stale_label = f"Yes ({row['days_overdue']} days overdue)" if (row["is_stale"] == "Yes" and is_active_freq) else "No"
                c2.markdown(f"**Stale:** {stale_label}")
                c3.markdown(f"**AI Desc Score:** {row['llm_desc_score'] or '—'}/5")
                tag_count = len(json.loads(row["tags"])) if row["tags"] else 0
                c3.markdown(f"**Tag Count:** {tag_count}")

                flags = []
                if row["missing_description"] == "Yes": flags.append("No description")
                if row["missing_tags"] == "Yes":        flags.append("No tags")
                if row["missing_license"] == "Yes":     flags.append("No license")
                # Only flag as stale if it's a dataset that needs regular updates
                is_active_freq = row["updateFrequency"] not in ["historical", "as needed", "never", "not planned"]
                if row["is_stale"] == "Yes" and is_active_freq:            flags.append("Stale")
                if flags:
                    st.markdown("**Issues:** " + "  |  ".join(flags))
                st.markdown(f"**Description:** {row['description'] or '*(empty)*'}")
                if row["llm_desc_feedback"]:
                    st.info(f"\U0001f4ac AI Feedback: {row['llm_desc_feedback']}")

# ── TAB 3: AI DESCRIPTIONS ────────────────────────────────────────────────────
with tab3:
    st.subheader("AI-Suggested Descriptions — Human Review")
    st.caption("Review AI suggestions carefully. Edit before approving if needed.")
    reviewer = "Automated"

    needs_review = filtered[filtered["llm_status"] == "pending_review"].sort_values("health_score")

    if needs_review.empty:
        st.info("No descriptions pending review. Run pipeline.py to generate suggestions.")
    else:
        for _, row in needs_review.iterrows():
            emoji = BAND_EMOJI.get(row["health_band"], "\u26aa")
            with st.expander(f"{emoji} {row['name']}  |  AI Score: {row['llm_desc_score']}/5"):
                c1, c2 = st.columns(2)
                c1.markdown("**Original Description:**")
                c1.write(row["description"] or "*(empty)*")
                c2.markdown("**AI Suggested Description:**")
                edited_desc = c2.text_area("Edit before approving:",
                                           value=row["llm_suggested_desc"] or "",
                                           key=f"desc_{row['id']}", height=120)
                if row["llm_desc_feedback"]:
                    st.info(f"\U0001f4ac AI Feedback: {row['llm_desc_feedback']}")
                edit_note = st.text_input("Reviewer note (optional):", key=f"note_{row['id']}")

                ca, cb, cc = st.columns(3)
                if ca.button("Approve", key=f"app_{row['id']}"): 
                    save_approval(row["id"], edited_desc, row["llm_suggested_tags"], "approved", reviewer, edit_note)
                    ca.success("Approved!")
                if cb.button("Approve Edited", key=f"edit_{row['id']}"):
                    save_approval(row["id"], edited_desc, row["llm_suggested_tags"], "edited", reviewer, edit_note)
                    cb.success("Saved as edited!")
                if cc.button("Reject", key=f"rej_{row['id']}"):
                    save_approval(row["id"], None, None, "rejected", reviewer, edit_note)
# ── TAB 4: AI TAGS ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("AI-Suggested Tags — Datasets with Missing Tags")
    reviewer_tags = "Automated"
    tag_df = filtered[filtered["missing_tags"] == "Yes"].sort_values("health_score")

    if tag_df.empty:
        st.success("No datasets with missing tags in current filter.")
    else:
        for _, row in tag_df.iterrows():
            emoji = BAND_EMOJI.get(row["health_band"], "\u26aa")
            with st.expander(f"{emoji} {row['name']}"):
                st.markdown(f"**Category:** {row['category'] or 'Unknown'}  |  **Dept:** {row['department'] or 'Unknown'}")
                st.markdown("**Current Tags:** *(none)*")

                raw_tags = row["llm_suggested_tags"]
                try:
                    tags_list = json.loads(raw_tags) if raw_tags else []
                except Exception:
                    tags_list = [t.strip() for t in str(raw_tags).split(",") if t.strip()]

                if tags_list:
                    st.markdown("**AI Suggested Tags:**")
                    st.code(", ".join(tags_list))

                    if pd.notna(row.get("llm_tag_alignment_note")) and row["llm_tag_alignment_note"]:
                        st.info(f"🏷️ Tag Alignment Note: {row['llm_tag_alignment_note']}")

                    ca, cb = st.columns(2)

                    if ca.button("Approve Tags", key=f"tappr_{row['id']}"):
                        save_approval(row["id"], None, raw_tags, "approved", reviewer_tags)
                        ca.success("Tags approved!")
                    if cb.button("Reject Tags", key=f"trej_{row['id']}"):
                        save_approval(row["id"], None, None, "rejected", reviewer_tags)
                        cb.warning("Rejected.")
                else:
                    st.info("No AI tag suggestions yet. Run pipeline.py to generate.")

# ── TAB 5: SPREADSHEET VIEW ───────────────────────────────────────────────────
with tab5:
    st.subheader("Spreadsheet View — Full Dataset Table")
    st.caption("Reflects your current sidebar filters. Export includes all visible rows.")

    export_cols = [
        "id", "name", "department", "category", "license",
        "tags", "updatedAt", "dataUpdatedAt", "updateFrequency",
        "health_score", "health_band",
        "missing_description", "missing_tags", "missing_license",
        "missing_department", "missing_category",
        "is_stale", "days_overdue",
        "llm_desc_score", "llm_desc_feedback",
        "llm_suggested_desc", "llm_suggested_tags", "llm_status"
    ]
    available_cols = [c for c in export_cols if c in filtered.columns]
    export_df = filtered[available_cols].sort_values("health_score")

    # Add human approval status from session state
    if st.session_state.get('human_approvals'):
        # Convert session state to DataFrame
        approvals_data = []
        for dataset_id, approval in st.session_state.human_approvals.items():
            approvals_data.append({
                'id': dataset_id,
                'approval_status': approval['human_status'],
                'approved_description': approval['approved_description']
            })

        approvals_df = pd.DataFrame(approvals_data)

        # Merge with export data
        export_df = export_df.merge(approvals_df, on='id', how='left')
    else:
        # No approvals in session state
        export_df['approval_status'] = None
        export_df['approved_description'] = None

    # Fill pending approvals
    export_df['approval_status'] = export_df['approval_status'].fillna("pending")

    # Use approved description if exists, otherwise use AI suggestion
    export_df['llm_suggested_desc'] = export_df['approved_description'].fillna(
        export_df['llm_suggested_desc']
    )

    # Drop the approved_description column since we merged it into llm_suggested_desc
    export_df = export_df.drop(columns=['approved_description'], errors='ignore')
    
    # Rename columns for clarity
    export_df = export_df.rename(columns={
        "llm_status": "AI Review Status",
        "llm_desc_score": "AI Score",
        "llm_desc_feedback": "AI Feedback",
        "llm_suggested_desc": "AI Suggested Description",
        "llm_suggested_tags": "AI Suggested Tags",
        "approval_status": "Approved"
    })


    def style_band(val):
        colors = {"Critical": "background-color:#fde8e8",
                  "Poor":     "background-color:#fef0e0",
                  "Fair":     "background-color:#fefde0",
                  "Good":     "background-color:#e8f8ee"}
        return colors.get(val, "")

    st.dataframe(
        export_df.style.map(style_band, subset=["health_band"]),
        use_container_width=True, height=500
    )

    st.download_button(
        label="Export Filtered View as CSV (ODP Format)",
        data=to_csv_bytes(export_df),
        file_name=f"cambridge_metadata_health_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        type="primary"
    )

# ── TAB 6: TRENDS ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Metadata Health Trends & Analysis")
    c1, c2 = st.columns(2)

    lic_counts = filtered["missing_license"].value_counts().reset_index()
    lic_counts.columns = ["Status", "Count"]
    lic_counts["Status"] = lic_counts["Status"].map({"No": "Has License", "Yes": "Missing License"})
    fig_lic = px.pie(lic_counts, names="Status", values="Count",
                     title="License Coverage (~25% expected missing)",
                     color="Status",
                     color_discrete_map={"Has License": "#2ecc71", "Missing License": "#e74c3c"})
    c1.plotly_chart(fig_lic, use_container_width=True)

    filtered["tag_count"] = filtered["tags"].apply(lambda x: len(json.loads(x)) if x else 0)
    fig_tags = px.histogram(filtered, x="tag_count", nbins=15,
                            title="Tag Count Distribution",
                            labels={"tag_count": "Number of Tags"},
                            color_discrete_sequence=["#9b59b6"])
    fig_tags.add_vline(x=3, line_dash="dash", line_color="green", annotation_text="3-tag target")
    c2.plotly_chart(fig_tags, use_container_width=True)

    dim_scores = {
        "Description":   filtered["desc_score"].mean(),
        "Tags":          filtered["tag_score"].mean(),
        "License":       filtered["license_score"].mean(),
        "Freshness":     filtered["freshness_score"].mean(),
    }
    fig_dims = go.Figure(go.Bar(
        x=list(dim_scores.keys()),
        y=[round(v, 1) if not pd.isna(v) else 0 for v in dim_scores.values()],
        marker_color=["#3498db","#9b59b6","#2ecc71","#e67e22"]
    ))
    fig_dims.update_layout(title="Average Sub-Score by Dimension (0-100)",
                           yaxis=dict(range=[0, 100]),
                           xaxis_title="Dimension", yaxis_title="Avg Score")
    st.plotly_chart(fig_dims, use_container_width=True)

    st.markdown("#### Datasets Overdue Relative to Update Frequency")
    two_yr_df = filtered[filtered["days_overdue"] > 0].sort_values("days_overdue", ascending=False)
    if two_yr_df.empty:
        st.success("No datasets are overdue relative to their update frequency.")
    else:
        display_cols = ["name","department","updateFrequency","dataUpdatedAt","days_overdue","health_band"]
        st.dataframe(two_yr_df[[c for c in display_cols if c in two_yr_df.columns]].head(30),
                     use_container_width=True)
