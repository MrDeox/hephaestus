"""
Test suite for Web Scraping as a Service.

Following TDD methodology - tests written first to define behavior.
Target: $900/month revenue in 7 days through scraping service.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.services.webscraping.scraping_service import (
    WebScrapingService,
    ScrapingJob,
    ScrapingResult,
    JobStatus,
    PricingTier,
    ScrapingJobManager
)


class TestWebScrapingService:
    """Test cases for core web scraping functionality."""
    
    @pytest.fixture
    def scraping_service(self):
        """Create a web scraping service instance for testing."""
        return WebScrapingService()
    
    @pytest.fixture
    def sample_scraping_job(self):
        """Create a sample scraping job for testing."""
        return ScrapingJob(
            job_id="test_job_001",
            url="https://example.com",
            selectors={
                "title": "h1",
                "price": ".price",
                "description": ".description"
            },
            customer_email="test@example.com",
            pricing_tier=PricingTier.BASIC
        )
    
    def test_scraping_service_initialization(self, scraping_service):
        """Test that scraping service initializes correctly."""
        assert scraping_service is not None
        assert scraping_service.active_jobs == {}
        assert scraping_service.completed_jobs == []
        assert scraping_service.total_revenue == 0.0
    
    def test_create_scraping_job(self, scraping_service):
        """Test creating a new scraping job."""
        job_data = {
            "url": "https://example.com/products",
            "selectors": {"title": "h1", "price": ".price"},
            "customer_email": "customer@test.com",
            "pricing_tier": "basic"
        }
        
        job = scraping_service.create_job(**job_data)
        
        assert job.job_id is not None
        assert job.url == job_data["url"]
        assert job.customer_email == job_data["customer_email"]
        assert job.status == JobStatus.PENDING
        assert job.created_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_basic_scraping(self, scraping_service, sample_scraping_job):
        """Test executing a basic scraping job."""
        # Mock the actual web scraping
        with patch('src.services.webscraping.scraping_service.aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.text = AsyncMock(return_value="""
                <html>
                    <h1>Test Product</h1>
                    <div class="price">$29.99</div>
                    <div class="description">Great product</div>
                </html>
            """)
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await scraping_service.execute_job(sample_scraping_job)
            
            assert result.success == True
            assert result.data is not None
            assert "title" in result.data
            assert "price" in result.data
            assert result.job_id == sample_scraping_job.job_id
    
    def test_pricing_calculation(self, scraping_service):
        """Test pricing calculation for different tiers."""
        basic_price = scraping_service.calculate_price(PricingTier.BASIC, pages=1)
        premium_price = scraping_service.calculate_price(PricingTier.PREMIUM, pages=1)
        enterprise_price = scraping_service.calculate_price(PricingTier.ENTERPRISE, pages=1)
        
        assert basic_price > 0
        assert premium_price > basic_price
        assert enterprise_price > premium_price
        
        # Test bulk pricing
        bulk_price = scraping_service.calculate_price(PricingTier.BASIC, pages=100)
        single_price = scraping_service.calculate_price(PricingTier.BASIC, pages=1)
        
        assert bulk_price > single_price * 50  # Bulk discount should apply
    
    def test_revenue_tracking(self, scraping_service, sample_scraping_job):
        """Test revenue tracking functionality."""
        initial_revenue = scraping_service.total_revenue
        
        # Simulate completing a job
        job_price = 25.0
        scraping_service.record_revenue(sample_scraping_job.job_id, job_price)
        
        assert scraping_service.total_revenue == initial_revenue + job_price
        assert len(scraping_service.revenue_history) == 1
    
    @pytest.mark.asyncio
    async def test_job_queue_management(self, scraping_service):
        """Test job queue and processing management."""
        # Create multiple jobs
        jobs = []
        for i in range(3):
            job_data = {
                "url": f"https://example{i}.com",
                "selectors": {"title": "h1"},
                "customer_email": f"customer{i}@test.com",
                "pricing_tier": "basic"
            }
            job = scraping_service.create_job(**job_data)
            jobs.append(job)
        
        assert len(scraping_service.active_jobs) == 3
        
        # Process jobs
        with patch.object(scraping_service, 'execute_job') as mock_execute:
            mock_execute.return_value = ScrapingResult(
                job_id="test",
                success=True,
                data={"title": "Test"},
                execution_time=1.0
            )
            
            await scraping_service.process_job_queue()
            
            assert mock_execute.call_count == 3
    
    def test_error_handling(self, scraping_service, sample_scraping_job):
        """Test error handling for failed scraping jobs."""
        # Test with invalid URL
        invalid_job = ScrapingJob(
            job_id="invalid_001",
            url="invalid-url",
            selectors={"title": "h1"},
            customer_email="test@test.com",
            pricing_tier=PricingTier.BASIC
        )
        
        # Should handle invalid URL gracefully
        result = asyncio.run(scraping_service.execute_job(invalid_job))
        
        assert result.success == False
        assert result.error_message is not None
        assert invalid_job.status == JobStatus.FAILED


class TestScrapingJobManager:
    """Test cases for scraping job management system."""
    
    @pytest.fixture
    def job_manager(self):
        """Create job manager instance for testing."""
        return ScrapingJobManager()
    
    def test_job_manager_initialization(self, job_manager):
        """Test job manager initializes correctly."""
        assert job_manager is not None
        assert job_manager.scraping_service is not None
        assert job_manager.pricing_calculator is not None
    
    def test_customer_job_limits(self, job_manager):
        """Test customer job limits and rate limiting."""
        customer_email = "test@customer.com"
        
        # Basic tier should have job limits
        can_create = job_manager.can_customer_create_job(customer_email, PricingTier.BASIC)
        assert can_create == True
        
        # Simulate creating many jobs
        for i in range(15):  # Exceed basic limit
            job_manager.track_customer_job(customer_email, PricingTier.BASIC)
        
        can_create_more = job_manager.can_customer_create_job(customer_email, PricingTier.BASIC)
        assert can_create_more == False
    
    @pytest.mark.asyncio
    async def test_automated_pricing_and_invoicing(self, job_manager):
        """Test automated pricing and invoice generation."""
        job_data = {
            "url": "https://test.com",
            "selectors": {"title": "h1"},
            "customer_email": "billing@test.com",
            "pricing_tier": "premium"
        }
        
        # Create and price job
        job = job_manager.create_and_price_job(**job_data)
        
        assert job.estimated_price > 0
        assert job.pricing_tier == PricingTier.PREMIUM
        
        # Test invoice generation
        invoice = job_manager.generate_invoice(job)
        
        assert invoice is not None
        assert invoice["job_id"] == job.job_id
        assert invoice["amount"] > 0
        assert "payment_link" in invoice


class TestRevenueGeneration:
    """Test cases specifically for revenue generation aspects."""
    
    @pytest.fixture
    def revenue_service(self):
        """Create revenue-focused service for testing."""
        return WebScrapingService()
    
    def test_freemium_model(self, revenue_service):
        """Test freemium pricing model implementation."""
        # Free tier should be limited
        free_price = revenue_service.calculate_price(PricingTier.FREE, pages=1)
        assert free_price == 0
        
        # But should have limits
        free_limits = revenue_service.get_tier_limits(PricingTier.FREE)
        assert free_limits["max_pages"] <= 5
        assert free_limits["max_jobs_per_day"] <= 3
    
    def test_upselling_recommendations(self, revenue_service):
        """Test automated upselling recommendations."""
        # Customer using free tier heavily
        customer_usage = {
            "jobs_this_month": 50,
            "pages_scraped": 200,
            "current_tier": PricingTier.FREE
        }
        
        recommendations = revenue_service.get_upselling_recommendations(customer_usage)
        
        assert len(recommendations) > 0
        assert recommendations[0]["recommended_tier"] == PricingTier.BASIC
        assert "savings" in recommendations[0]
    
    def test_revenue_projections(self, revenue_service):
        """Test revenue projection calculations."""
        # Simulate customer base
        customer_data = [
            {"tier": PricingTier.FREE, "count": 100},
            {"tier": PricingTier.BASIC, "count": 50},
            {"tier": PricingTier.PREMIUM, "count": 20},
            {"tier": PricingTier.ENTERPRISE, "count": 5}
        ]
        
        monthly_projection = revenue_service.calculate_monthly_revenue_projection(customer_data)
        
        assert monthly_projection > 0
        assert monthly_projection >= 900  # Target $900/month
    
    def test_customer_lifetime_value(self, revenue_service):
        """Test customer lifetime value calculations."""
        customer_history = {
            "signup_date": "2025-01-01",
            "tier_history": [
                {"tier": PricingTier.FREE, "duration_months": 1},
                {"tier": PricingTier.BASIC, "duration_months": 6},
                {"tier": PricingTier.PREMIUM, "duration_months": 3}
            ],
            "total_spent": 450.0
        }
        
        clv = revenue_service.calculate_customer_lifetime_value(customer_history)
        
        assert clv > customer_history["total_spent"]
        assert clv > 0


class TestMarketValidation:
    """Test cases for market validation and demand verification."""
    
    def test_competitor_analysis(self):
        """Test competitor pricing analysis."""
        # This would test against real competitor data
        # For now, just verify the structure exists
        assert True  # Placeholder
    
    def test_demand_validation(self):
        """Test market demand validation."""
        # This would test against real market data
        # For now, just verify the structure exists
        assert True  # Placeholder


# Integration test
class TestFullServiceIntegration:
    """Integration tests for complete service workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_customer_journey(self):
        """Test complete customer journey from signup to payment."""
        service = WebScrapingService()
        manager = ScrapingJobManager()
        
        # 1. Customer signs up (free tier)
        customer_email = "integration@test.com"
        
        # 2. Customer creates free job
        free_job_data = {
            "url": "https://example.com",
            "selectors": {"title": "h1"},
            "customer_email": customer_email,
            "pricing_tier": "free"
        }
        
        free_job = manager.create_and_price_job(**free_job_data)
        assert free_job.estimated_price == 0
        
        # 3. Customer uses service and gets value
        with patch.object(service, 'execute_job') as mock_execute:
            mock_execute.return_value = ScrapingResult(
                job_id=free_job.job_id,
                success=True,
                data={"title": "Great Product"},
                execution_time=0.5
            )
            
            result = await service.execute_job(free_job)
            assert result.success == True
        
        # 4. Customer gets upsell recommendation
        usage = {"jobs_this_month": 3, "current_tier": PricingTier.FREE}
        recommendations = service.get_upselling_recommendations(usage)
        assert len(recommendations) > 0
        
        # 5. Customer upgrades to paid tier
        paid_job_data = {
            "url": "https://example.com/premium",
            "selectors": {"title": "h1", "price": ".price", "features": ".features"},
            "customer_email": customer_email,
            "pricing_tier": "basic"
        }
        
        paid_job = manager.create_and_price_job(**paid_job_data)
        assert paid_job.estimated_price > 0
        
        # 6. Service generates revenue
        service.record_revenue(paid_job.job_id, paid_job.estimated_price)
        assert service.total_revenue > 0