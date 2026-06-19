"""
core/cleanup.py
Background scheduler that wipes the ChromaDB collection every 12 hours
to prevent unbounded local storage growth.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.ingest import reset_collection

logger = logging.getLogger("auron.cleanup")


def _wipe_job() -> None:
    logger.info("Scheduled cleanup: wiping ChromaDB collection")
    reset_collection()


def start_cleanup_scheduler() -> BackgroundScheduler:
    """
    Starts a daemon background scheduler that wipes the vector store every
    12 hours. Must be called through st.cache_resource in app.py --
    Streamlit reruns the whole script on every user interaction, and
    without caching this would spawn a duplicate scheduler thread on every
    rerun instead of one singleton per server process.
    """
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        _wipe_job,
        trigger=IntervalTrigger(hours=12),
        id="wipe_chroma_collection",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Cleanup scheduler started: wiping every 12 hours")
    return scheduler
