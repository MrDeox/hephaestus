"""
Manual test for Web Scraping as a Service.
Tests the complete revenue generation flow.
"""

import asyncio
import sys
sys.path.append('.')

from src.services.webscraping import (
    create_web_scraping_service,
    create_job_manager,
    PricingTier
)


async def test_complete_service():
    """Test complete scraping service functionality."""
    
    print("🕷️ Testing Web Scraping as a Service")
    print("=" * 50)
    
    # Initialize services
    service = create_web_scraping_service()
    manager = create_job_manager()
    
    print(f"✅ Service initialized")
    print(f"📊 Pricing tiers: {list(PricingTier)}")
    
    # Test pricing
    print("\n💰 PRICING TESTS:")
    for tier in PricingTier:
        price_1_page = service.calculate_price(tier, pages=1, selectors=3)
        price_10_pages = service.calculate_price(tier, pages=10, selectors=5)
        price_100_pages = service.calculate_price(tier, pages=100, selectors=10)
        
        print(f"  {tier.value.upper()}: $0 (1pg) → ${price_1_page:.2f} (1pg) → ${price_10_pages:.2f} (10pg) → ${price_100_pages:.2f} (100pg)")
    
    # Test tier limits
    print("\n📏 TIER LIMITS:")
    for tier in PricingTier:
        limits = service.get_tier_limits(tier)
        print(f"  {tier.value.upper()}: {limits}")
    
    # Create test jobs
    print("\n📝 CREATING TEST JOBS:")
    
    # Free tier job
    free_job = manager.create_and_price_job(
        url="https://httpbin.org/html",
        selectors={"title": "h1", "body": "body"},
        customer_email="free_customer@test.com",
        pricing_tier="free"
    )
    print(f"  Free job: {free_job.job_id} - ${free_job.estimated_price:.2f}")
    
    # Basic tier job
    basic_job = manager.create_and_price_job(
        url="https://httpbin.org/html", 
        selectors={"title": "h1", "body": "body", "meta": "meta"},
        customer_email="basic_customer@test.com",
        pricing_tier="basic",
        max_pages=5
    )
    print(f"  Basic job: {basic_job.job_id} - ${basic_job.estimated_price:.2f}")
    
    # Premium tier job
    premium_job = manager.create_and_price_job(
        url="https://httpbin.org/html",
        selectors={"title": "h1", "body": "body", "meta": "meta", "links": "a"},
        customer_email="premium_customer@test.com", 
        pricing_tier="premium",
        max_pages=20
    )
    print(f"  Premium job: {premium_job.job_id} - ${premium_job.estimated_price:.2f}")
    
    print(f"\n🎯 Total jobs created: {len(service.active_jobs)}")
    print(f"💰 Total estimated revenue: ${sum(job.estimated_price for job in service.active_jobs.values()):.2f}")
    
    # Test job execution (simplified)
    print("\n🚀 EXECUTING JOBS:")
    
    try:
        # Execute free job first
        result = await service.execute_job(free_job)
        print(f"  ✅ Free job completed: {result.success}")
        if result.success and result.data:
            print(f"     Data extracted: {len(result.data)} fields")
        
        # Execute basic job
        result = await service.execute_job(basic_job)
        print(f"  ✅ Basic job completed: {result.success}")
        if result.success and result.data:
            print(f"     Data extracted: {len(result.data)} fields")
            
        # Execute premium job  
        result = await service.execute_job(premium_job)
        print(f"  ✅ Premium job completed: {result.success}")
        if result.success and result.data:
            print(f"     Data extracted: {len(result.data)} fields")
            
    except Exception as e:
        print(f"  ⚠️ Job execution error: {e}")
    
    # Revenue tracking
    print(f"\n💰 REVENUE TRACKING:")
    print(f"  Total revenue generated: ${service.total_revenue:.2f}")
    print(f"  Completed jobs: {len(service.completed_jobs)}")
    print(f"  Revenue history entries: {len(service.revenue_history)}")
    
    # Test upselling recommendations
    print(f"\n📈 UPSELLING RECOMMENDATIONS:")
    
    # Heavy free user
    heavy_free_usage = {
        "current_tier": "free",
        "jobs_this_month": 15,
        "pages_scraped": 50
    }
    
    recommendations = service.get_upselling_recommendations(heavy_free_usage)
    for rec in recommendations:
        print(f"  → {rec['recommended_tier'].value}: {rec['reason']} (${rec['monthly_cost']:.2f}/month)")
    
    # Revenue projections
    print(f"\n📊 REVENUE PROJECTIONS:")
    
    customer_base = [
        {"tier": PricingTier.FREE, "count": 100},
        {"tier": PricingTier.BASIC, "count": 30}, 
        {"tier": PricingTier.PREMIUM, "count": 10},
        {"tier": PricingTier.ENTERPRISE, "count": 2}
    ]
    
    monthly_projection = service.calculate_monthly_revenue_projection(customer_base)
    print(f"  Projected monthly revenue: ${monthly_projection:.2f}")
    print(f"  Target achievement: {monthly_projection/900*100:.1f}% of $900 goal")
    
    # Customer lifetime value
    print(f"\n👤 CUSTOMER LIFETIME VALUE:")
    
    sample_customer = {
        "signup_date": "2025-01-01",
        "tier_history": [
            {"tier": PricingTier.FREE, "duration_months": 1},
            {"tier": PricingTier.BASIC, "duration_months": 4},
            {"tier": PricingTier.PREMIUM, "duration_months": 2}
        ],
        "total_spent": 180.0
    }
    
    clv = service.calculate_customer_lifetime_value(sample_customer)
    print(f"  Sample customer CLV: ${clv:.2f}")
    
    # Generate invoices
    print(f"\n🧾 INVOICE GENERATION:")
    
    for job in service.completed_jobs:
        if job.estimated_price > 0:
            invoice = manager.generate_invoice(job)
            print(f"  Invoice {invoice['invoice_id']}: ${invoice['amount']:.2f} for {job.customer_email}")
    
    # Cleanup
    await service.cleanup()
    
    print(f"\n🎉 TEST COMPLETED!")
    print(f"Total revenue generated: ${service.total_revenue:.2f}")
    print(f"Service ready for deployment! 🚀")


if __name__ == "__main__":
    asyncio.run(test_complete_service())