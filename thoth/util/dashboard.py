import os
from typing import List, Optional

import yarl

PROFILING_VIEW = "👤 Profiling"
OPTIMIZATION_VIEW = "📈 Optimization"
SCORING_VIEW = "💯 Scoring"


def build_dashboard_link(
    dataset_uri: str, view: str, instances: Optional[List[str]] = None
) -> str:
    """Dashboard link with filters for a specif dataset and target instances."""
    base_url = yarl.URL(os.environ.get("DASHBOARD_URL", "http://localhost:8501"))
    query_url = (
        base_url
        % {"dataset_uri": dataset_uri, "view": view}
        % ({"instances": instances} if instances else {})
    )
    return str(query_url)
