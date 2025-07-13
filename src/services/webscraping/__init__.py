"""
Web Scraping as a Service module.

Zero-cost bootstrap revenue generation through professional web scraping services.
Target: $900/month revenue in 7 days.
"""

from .scraping_service import (
    WebScrapingService,
    ScrapingJob,
    ScrapingResult,
    ScrapingJobManager,
    JobStatus,
    PricingTier,
    create_web_scraping_service,
    create_job_manager
)

__all__ = [
    "WebScrapingService",
    "ScrapingJob", 
    "ScrapingResult",
    "ScrapingJobManager",
    "JobStatus",
    "PricingTier",
    "create_web_scraping_service",
    "create_job_manager"
]