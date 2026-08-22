import os
import json
import logging
from datetime import datetime
from src.core.state import ProjectState
from src.research.search_client import search_web

logger = logging.getLogger(__name__)

def research_node(state: ProjectState) -> dict:
    """
    StateGraph node that executes real-time web search and stores results
    in a local JSON file memory cache.
    """
    payload = state.get("generated_prompt_payload") or {}
    objective = payload.get("objective", "")

    if not objective:
        logger.warning("Research Node: No search query objective found.")
        return {"retrieved_web_docs": []}

    logger.info(f"Research Node: Querying web for: '{objective}'")
    results = search_web(objective, limit=3)

    # Cache research results persistently in docs/research_history.json
    history_dir = os.path.join(state.get("project_root", os.getcwd()), "docs")
    history_file = os.path.join(history_dir, "research_history.json")

    # Ensure target folder exists
    os.makedirs(history_dir, exist_ok=True)

    history_log = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_log = json.load(f)
                if not isinstance(history_log, list):
                    history_log = []
        except Exception as e:
            logger.warning(f"Failed to read existing research history: {e}")

    # Append new search entry
    history_log.append({
        "query": objective,
        "timestamp": datetime.utcnow().isoformat(),
        "results": results
    })

    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_log, f, indent=4, ensure_ascii=False)
        logger.info(f"Research Node: Cached query results in {history_file}")
    except Exception as e:
        logger.error(f"Failed to write research history: {e}")

    return {
        "retrieved_web_docs": results
    }
