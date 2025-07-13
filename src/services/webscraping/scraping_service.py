"""
Web Scraping as a Service - Revenue Generation Implementation.

Zero-cost bootstrap service targeting $900/month revenue in 7 days.
Uses free tools and platforms to provide professional scraping services.
"""

import asyncio
import aiohttp
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup4 not available - using basic parsing")

try:
    import lxml
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of scraping jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PricingTier(str, Enum):
    """Pricing tiers for the service."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class ScrapingJob:
    """Represents a web scraping job."""
    
    job_id: str
    url: str
    selectors: Dict[str, str]
    customer_email: str
    pricing_tier: PricingTier
    
    # Job configuration
    max_pages: int = 1
    delay_between_requests: float = 1.0
    timeout: int = 30
    
    # Status tracking
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Pricing
    estimated_price: float = 0.0
    actual_price: float = 0.0
    
    # Results
    pages_scraped: int = 0
    items_extracted: int = 0
    error_message: Optional[str] = None


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    
    job_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    pages_processed: int = 0
    items_found: int = 0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    user_agent: str = "WebScrapingService/1.0"


class WebScrapingService:
    """
    Core web scraping service for revenue generation.
    
    Provides freemium web scraping with automated pricing,
    customer management, and revenue tracking.
    """
    
    def __init__(self):
        # Job management
        self.active_jobs: Dict[str, ScrapingJob] = {}
        self.completed_jobs: List[ScrapingJob] = []
        self.job_queue: List[ScrapingJob] = []
        
        # Revenue tracking
        self.total_revenue: float = 0.0
        self.revenue_history: List[Dict[str, Any]] = []
        self.customer_stats: Dict[str, Dict[str, Any]] = {}
        
        # Service configuration
        self.pricing_config = self._load_pricing_config()
        self.tier_limits = self._load_tier_limits()
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Data storage
        self.data_dir = Path("data/webscraping_service")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🕷️ Web Scraping Service initialized")
    
    def _load_pricing_config(self) -> Dict[str, Any]:
        """Load pricing configuration."""
        return {
            PricingTier.FREE: {
                "base_price": 0.0,
                "price_per_page": 0.0,
                "price_per_item": 0.0
            },
            PricingTier.BASIC: {
                "base_price": 5.0,
                "price_per_page": 2.0,
                "price_per_item": 0.10
            },
            PricingTier.PREMIUM: {
                "base_price": 15.0,
                "price_per_page": 1.5,
                "price_per_item": 0.08
            },
            PricingTier.ENTERPRISE: {
                "base_price": 50.0,
                "price_per_page": 1.0,
                "price_per_item": 0.05
            }
        }
    
    def _load_tier_limits(self) -> Dict[PricingTier, Dict[str, int]]:
        """Load tier-specific limits."""
        return {
            PricingTier.FREE: {
                "max_pages": 5,
                "max_jobs_per_day": 3,
                "max_selectors": 3,
                "max_items": 50
            },
            PricingTier.BASIC: {
                "max_pages": 100,
                "max_jobs_per_day": 50,
                "max_selectors": 10,
                "max_items": 1000
            },
            PricingTier.PREMIUM: {
                "max_pages": 500,
                "max_jobs_per_day": 200,
                "max_selectors": 25,
                "max_items": 10000
            },
            PricingTier.ENTERPRISE: {
                "max_pages": 10000,
                "max_jobs_per_day": 1000,
                "max_selectors": 100,
                "max_items": 100000
            }
        }
    
    def create_job(self, url: str, selectors: Dict[str, str], 
                   customer_email: str, pricing_tier: str, **kwargs) -> ScrapingJob:
        """Create a new scraping job."""
        
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        tier = PricingTier(pricing_tier.lower())
        
        job = ScrapingJob(
            job_id=job_id,
            url=url,
            selectors=selectors,
            customer_email=customer_email,
            pricing_tier=tier,
            **kwargs
        )
        
        # Calculate pricing
        job.estimated_price = self.calculate_price(tier, job.max_pages, len(selectors))
        
        # Add to active jobs
        self.active_jobs[job_id] = job
        self.job_queue.append(job)
        
        # Track customer
        self._track_customer_activity(customer_email, job)
        
        logger.info(f"📝 Created job {job_id} for {customer_email} (${job.estimated_price:.2f})")
        
        return job
    
    def calculate_price(self, tier: PricingTier, pages: int = 1, selectors: int = 1) -> float:
        """Calculate price for scraping job."""
        
        if tier == PricingTier.FREE:
            return 0.0
        
        config = self.pricing_config[tier]
        
        base_price = config["base_price"]
        page_cost = config["price_per_page"] * pages
        selector_cost = config["price_per_item"] * selectors
        
        total_price = base_price + page_cost + selector_cost
        
        # Volume discounts
        if pages > 100:
            total_price *= 0.8  # 20% discount for bulk
        elif pages > 50:
            total_price *= 0.9  # 10% discount
        
        return round(total_price, 2)
    
    def get_tier_limits(self, tier: PricingTier) -> Dict[str, int]:
        """Get limits for pricing tier."""
        return self.tier_limits.get(tier, {})
    
    async def execute_job(self, job: ScrapingJob) -> ScrapingResult:
        """Execute a scraping job."""
        
        start_time = time.time()
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        logger.info(f"🚀 Executing job {job.job_id} for {job.url}")
        
        try:
            # Initialize session if needed
            if self.session is None:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=job.timeout),
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; WebScrapingService/1.0)"
                    }
                )
            
            # Scrape the URL
            scraped_data = await self._scrape_url(job.url, job.selectors)
            
            # Update job status
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.pages_scraped = 1  # For now, single page
            job.items_extracted = len(scraped_data) if scraped_data else 0
            
            # Create result
            execution_time = time.time() - start_time
            result = ScrapingResult(
                job_id=job.job_id,
                success=True,
                data=scraped_data,
                execution_time=execution_time,
                pages_processed=1,
                items_found=job.items_extracted
            )
            
            # Record revenue if paid tier
            if job.pricing_tier != PricingTier.FREE:
                self.record_revenue(job.job_id, job.estimated_price)
            
            logger.info(f"✅ Job {job.job_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Handle errors
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            execution_time = time.time() - start_time
            result = ScrapingResult(
                job_id=job.job_id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
            
            logger.error(f"❌ Job {job.job_id} failed: {e}")
            return result
        
        finally:
            # Move from active to completed
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            self.completed_jobs.append(job)
    
    async def _scrape_url(self, url: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        """Scrape data from URL using provided selectors."""
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                html_content = await response.text()
                
                # Parse HTML
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    scraped_data = {}
                    
                    for field_name, selector in selectors.items():
                        elements = soup.select(selector)
                        
                        if elements:
                            if len(elements) == 1:
                                scraped_data[field_name] = elements[0].get_text().strip()
                            else:
                                scraped_data[field_name] = [el.get_text().strip() for el in elements]
                        else:
                            scraped_data[field_name] = None
                    
                    return scraped_data
                else:
                    # Fallback: basic text extraction
                    return {"raw_content": html_content[:1000]}  # First 1000 chars
                    
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            raise
    
    async def process_job_queue(self) -> None:
        """Process all jobs in the queue."""
        
        while self.job_queue:
            job = self.job_queue.pop(0)
            
            # Check tier limits
            if not self._check_tier_limits(job):
                job.status = JobStatus.FAILED
                job.error_message = "Tier limits exceeded"
                continue
            
            # Execute job
            await self.execute_job(job)
            
            # Add delay between jobs
            await asyncio.sleep(1.0)
    
    def _check_tier_limits(self, job: ScrapingJob) -> bool:
        """Check if job is within tier limits."""
        
        limits = self.tier_limits.get(job.pricing_tier, {})
        
        # Check page limits
        if job.max_pages > limits.get("max_pages", float('inf')):
            return False
        
        # Check selector limits
        if len(job.selectors) > limits.get("max_selectors", float('inf')):
            return False
        
        # Check daily job limits for customer
        customer_jobs_today = self._get_customer_jobs_today(job.customer_email)
        if len(customer_jobs_today) >= limits.get("max_jobs_per_day", float('inf')):
            return False
        
        return True
    
    def _get_customer_jobs_today(self, customer_email: str) -> List[ScrapingJob]:
        """Get customer's jobs from today."""
        today = datetime.now().date()
        
        customer_jobs = []
        for job in self.completed_jobs:
            if (job.customer_email == customer_email and 
                job.created_at.date() == today):
                customer_jobs.append(job)
        
        return customer_jobs
    
    def record_revenue(self, job_id: str, amount: float) -> None:
        """Record revenue from completed job."""
        
        self.total_revenue += amount
        
        revenue_record = {
            "job_id": job_id,
            "amount": amount,
            "timestamp": datetime.now().isoformat(),
            "cumulative_revenue": self.total_revenue
        }
        
        self.revenue_history.append(revenue_record)
        
        logger.info(f"💰 Revenue recorded: ${amount:.2f} (Total: ${self.total_revenue:.2f})")
    
    def _track_customer_activity(self, customer_email: str, job: ScrapingJob) -> None:
        """Track customer activity for analytics."""
        
        if customer_email not in self.customer_stats:
            self.customer_stats[customer_email] = {
                "first_job": job.created_at.isoformat(),
                "total_jobs": 0,
                "total_spent": 0.0,
                "current_tier": job.pricing_tier.value,
                "last_activity": job.created_at.isoformat()
            }
        
        stats = self.customer_stats[customer_email]
        stats["total_jobs"] += 1
        stats["last_activity"] = job.created_at.isoformat()
        
        if job.pricing_tier.value != stats["current_tier"]:
            stats["current_tier"] = job.pricing_tier.value
    
    def get_upselling_recommendations(self, customer_usage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate upselling recommendations for customers."""
        
        recommendations = []
        current_tier = PricingTier(customer_usage.get("current_tier", "free"))
        jobs_this_month = customer_usage.get("jobs_this_month", 0)
        
        # Recommend upgrade based on usage
        if current_tier == PricingTier.FREE and jobs_this_month >= 10:
            monthly_cost_basic = self.calculate_price(PricingTier.BASIC, 10, 5)
            
            recommendations.append({
                "recommended_tier": PricingTier.BASIC,
                "reason": "Heavy usage - save money with Basic plan",
                "monthly_cost": monthly_cost_basic,
                "savings": f"${monthly_cost_basic * 0.3:.2f}/month",
                "benefits": ["More pages", "Faster processing", "Priority support"]
            })
        
        elif current_tier == PricingTier.BASIC and jobs_this_month >= 50:
            monthly_cost_premium = self.calculate_price(PricingTier.PREMIUM, 50, 10)
            
            recommendations.append({
                "recommended_tier": PricingTier.PREMIUM,
                "reason": "Power user - unlock premium features",
                "monthly_cost": monthly_cost_premium,
                "savings": f"${monthly_cost_premium * 0.2:.2f}/month",
                "benefits": ["Bulk discounts", "Advanced selectors", "Custom scheduling"]
            })
        
        return recommendations
    
    def calculate_monthly_revenue_projection(self, customer_data: List[Dict[str, Any]]) -> float:
        """Calculate projected monthly revenue."""
        
        total_projection = 0.0
        
        for segment in customer_data:
            tier = segment["tier"]
            customer_count = segment["count"]
            
            # Estimate average monthly revenue per customer by tier
            if tier == PricingTier.FREE:
                avg_monthly = 0.0  # Free tier converts ~5% to paid
                conversion_revenue = customer_count * 0.05 * 15.0  # 5% convert to $15/month
                total_projection += conversion_revenue
            elif tier == PricingTier.BASIC:
                avg_monthly = 25.0  # Average basic customer
                total_projection += customer_count * avg_monthly
            elif tier == PricingTier.PREMIUM:
                avg_monthly = 75.0  # Average premium customer
                total_projection += customer_count * avg_monthly
            elif tier == PricingTier.ENTERPRISE:
                avg_monthly = 300.0  # Average enterprise customer
                total_projection += customer_count * avg_monthly
        
        return total_projection
    
    def calculate_customer_lifetime_value(self, customer_history: Dict[str, Any]) -> float:
        """Calculate customer lifetime value."""
        
        total_spent = customer_history.get("total_spent", 0.0)
        months_active = len(customer_history.get("tier_history", []))
        
        if months_active == 0:
            return 0.0
        
        # Calculate average monthly spend
        avg_monthly_spend = total_spent / months_active
        
        # Project based on tier progression (customers tend to upgrade over time)
        tier_multiplier = 1.5  # Assumption: customers increase spend by 50% over time
        projected_lifetime_months = 12  # Assume 12-month lifetime
        
        clv = avg_monthly_spend * tier_multiplier * projected_lifetime_months
        
        return clv
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        
        logger.info("🧹 Web Scraping Service cleaned up")


class ScrapingJobManager:
    """
    High-level job management and customer interaction.
    Handles pricing, invoicing, and customer lifecycle.
    """
    
    def __init__(self):
        self.scraping_service = WebScrapingService()
        self.pricing_calculator = self.scraping_service
        
        # Customer management
        self.customer_limits: Dict[str, Dict[str, Any]] = {}
        
        logger.info("📋 Scraping Job Manager initialized")
    
    def create_and_price_job(self, **job_data) -> ScrapingJob:
        """Create job with automatic pricing."""
        
        # Validate customer can create job
        customer_email = job_data["customer_email"]
        pricing_tier = PricingTier(job_data["pricing_tier"].lower())
        
        if not self.can_customer_create_job(customer_email, pricing_tier):
            raise ValueError("Customer has exceeded tier limits")
        
        # Create job
        job = self.scraping_service.create_job(**job_data)
        
        # Track customer job creation
        self.track_customer_job(customer_email, pricing_tier)
        
        return job
    
    def can_customer_create_job(self, customer_email: str, tier: PricingTier) -> bool:
        """Check if customer can create another job."""
        
        if customer_email not in self.customer_limits:
            self.customer_limits[customer_email] = {
                "jobs_today": 0,
                "last_reset": datetime.now().date()
            }
        
        customer_data = self.customer_limits[customer_email]
        
        # Reset daily limits if new day
        if customer_data["last_reset"] != datetime.now().date():
            customer_data["jobs_today"] = 0
            customer_data["last_reset"] = datetime.now().date()
        
        # Check tier limits
        tier_limits = self.scraping_service.get_tier_limits(tier)
        max_jobs_per_day = tier_limits.get("max_jobs_per_day", float('inf'))
        
        return customer_data["jobs_today"] < max_jobs_per_day
    
    def track_customer_job(self, customer_email: str, tier: PricingTier) -> None:
        """Track customer job creation."""
        
        if customer_email not in self.customer_limits:
            self.customer_limits[customer_email] = {
                "jobs_today": 0,
                "last_reset": datetime.now().date()
            }
        
        self.customer_limits[customer_email]["jobs_today"] += 1
    
    def generate_invoice(self, job: ScrapingJob) -> Dict[str, Any]:
        """Generate invoice for completed job."""
        
        invoice = {
            "invoice_id": f"inv_{uuid.uuid4().hex[:8]}",
            "job_id": job.job_id,
            "customer_email": job.customer_email,
            "amount": job.estimated_price,
            "currency": "USD",
            "created_at": datetime.now().isoformat(),
            "due_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "payment_link": f"https://pay.webscraping-service.com/invoice/{job.job_id}",
            "items": [
                {
                    "description": f"Web scraping for {job.url}",
                    "quantity": job.max_pages,
                    "unit_price": job.estimated_price / job.max_pages,
                    "total": job.estimated_price
                }
            ]
        }
        
        return invoice


# Factory functions
def create_web_scraping_service() -> WebScrapingService:
    """Create web scraping service instance."""
    return WebScrapingService()


def create_job_manager() -> ScrapingJobManager:
    """Create job manager instance."""
    return ScrapingJobManager()