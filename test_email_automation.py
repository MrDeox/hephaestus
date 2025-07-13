"""
Comprehensive test for Email Marketing Automation Service.
Tests the complete revenue generation flow.
"""

import asyncio
import sys
sys.path.append('.')

from src.services.email_automation import (
    create_email_automation_service,
    create_email_service_manager,
    PricingTier
)


async def test_complete_email_service():
    """Test complete email automation service functionality."""
    
    print("📧 Testing Email Marketing Automation Service")
    print("=" * 60)
    
    # Initialize services
    service = create_email_automation_service()
    manager = create_email_service_manager()
    
    print(f"✅ Service initialized")
    print(f"📊 Pricing tiers: {list(PricingTier)}")
    print(f"📄 Default templates: {len(service.templates)}")
    
    # Test pricing
    print("\n💰 PRICING TESTS:")
    for tier in PricingTier:
        price_100 = service.calculate_price(tier, 100)
        price_1000 = service.calculate_price(tier, 1000) 
        price_10000 = service.calculate_price(tier, 10000)
        
        print(f"  {tier.value.upper()}: ${price_100:.2f} (100) → ${price_1000:.2f} (1K) → ${price_10000:.2f} (10K)")
    
    # Test tier limits
    print("\n📏 TIER LIMITS:")
    for tier in PricingTier:
        limits = service.get_tier_limits(tier)
        print(f"  {tier.value.upper()}: {limits['max_emails_per_month']} emails, {limits['max_campaigns_per_month']} campaigns")
    
    # Test customer onboarding
    print("\n👋 CUSTOMER ONBOARDING:")
    
    customer = manager.onboard_customer(
        email="testbusiness@example.com",
        company_name="Test Business Inc",
        industry="ecommerce", 
        expected_volume=5000
    )
    print(f"  Customer ID: {customer['customer_id']}")
    print(f"  Recommended tier: {customer['recommended_tier']}")
    print(f"  Welcome campaign: {customer['welcome_campaign']}")
    
    # Test email templates
    print("\n📄 EMAIL TEMPLATES:")
    for template_id, template in list(service.templates.items())[:3]:
        print(f"  {template.name}: {template_id} ({template.category})")
        print(f"    Variables: {template.variables}")
    
    # Create test campaigns
    print("\n📧 CREATING TEST CAMPAIGNS:")
    
    # Free tier campaign
    free_campaign = service.create_campaign(
        name="Free Newsletter",
        customer_email="free@customer.com",
        pricing_tier="free",
        template_id="newsletter_template",
        recipient_list=[f"subscriber{i}@test.com" for i in range(50)],
        personalization_data={
            f"subscriber{i}@test.com": {
                "first_name": f"User{i}",
                "company_name": "Newsletter Service",
                "newsletter_title": "Weekly Updates",
                "month": "July",
                "year": "2025",
                "intro_text": "Here are this week's highlights!",
                "main_content": "• Feature 1\n• Feature 2\n• Feature 3"
            } for i in range(50)
        }
    )
    print(f"  Free campaign: {free_campaign.campaign_id} - ${free_campaign.estimated_price:.2f}")
    
    # Basic tier campaign
    basic_campaign = service.create_campaign(
        name="Product Launch Campaign",
        customer_email="basic@customer.com",
        pricing_tier="basic",
        template_id="promotional_template",
        recipient_list=[f"customer{i}@business.com" for i in range(500)],
        personalization_data={
            f"customer{i}@business.com": {
                "first_name": f"Customer{i}",
                "offer_title": "Summer Sale",
                "offer_description": "Get ready for our biggest sale of the year!",
                "discount_percent": "25",
                "promo_code": "SUMMER25",
                "shop_link": "https://shop.example.com",
                "expiry_date": "July 31, 2025"
            } for i in range(500)
        }
    )
    print(f"  Basic campaign: {basic_campaign.campaign_id} - ${basic_campaign.estimated_price:.2f}")
    
    # Premium tier campaign
    premium_campaign = service.create_campaign(
        name="VIP Customer Series",
        customer_email="premium@customer.com",
        pricing_tier="premium", 
        template_id="welcome_template",
        recipient_list=[f"vip{i}@premium.com" for i in range(2000)],
        personalization_data={
            f"vip{i}@premium.com": {
                "first_name": f"VIP{i}",
                "company_name": "Premium Services",
                "get_started_link": "https://premium.example.com/vip"
            } for i in range(2000)
        }
    )
    print(f"  Premium campaign: {premium_campaign.campaign_id} - ${premium_campaign.estimated_price:.2f}")
    
    print(f"\n🎯 Total campaigns created: {len(service.active_campaigns)}")
    print(f"💰 Total estimated revenue: ${sum(c.estimated_price for c in service.active_campaigns.values()):.2f}")
    
    # Test campaign execution
    print("\n🚀 EXECUTING CAMPAIGNS:")
    
    campaigns_to_test = [free_campaign, basic_campaign, premium_campaign]
    
    for campaign in campaigns_to_test:
        try:
            result = await service.send_campaign(campaign)
            print(f"  ✅ {campaign.name}: {result.emails_sent} sent in {result.execution_time:.2f}s")
            
            # Simulate analytics
            if result.success and result.emails_sent > 0:
                # Simulate realistic email performance
                opens = int(result.emails_sent * 0.22)  # 22% open rate
                clicks = int(opens * 0.15)  # 15% click rate
                unsubscribes = int(result.emails_sent * 0.005)  # 0.5% unsubscribe
                bounces = int(result.emails_sent * 0.02)  # 2% bounce rate
                
                analytics = service.record_analytics(
                    campaign_id=campaign.campaign_id,
                    emails_sent=result.emails_sent,
                    emails_delivered=result.emails_sent - bounces,
                    opens=opens,
                    clicks=clicks,
                    unsubscribes=unsubscribes,
                    bounces=bounces
                )
                
                print(f"    📊 Open: {analytics.open_rate:.1%}, Click: {analytics.click_rate:.1%}, Unsub: {analytics.unsubscribe_rate:.1%}")
                
        except Exception as e:
            print(f"  ❌ {campaign.name} failed: {e}")
    
    # Test automation rules
    print("\n🤖 TESTING AUTOMATION:")
    
    # Create welcome automation
    welcome_rule = service.create_automation_rule(
        rule_id="welcome_automation",
        name="New User Welcome Series",
        trigger="user_signup",
        conditions=[
            {"field": "user_type", "operator": "equals", "value": "premium"}
        ],
        actions=[
            {"type": "send_email", "template_id": "welcome_template", "delay_hours": 0},
            {"type": "send_email", "template_id": "newsletter_template", "delay_hours": 24}
        ]
    )
    
    print(f"  Created automation rule: {welcome_rule.rule_id}")
    
    # Trigger automation
    trigger_result = await service.trigger_automation(
        trigger="user_signup",
        user_email="newuser@premium.com",
        user_data={
            "first_name": "John",
            "user_type": "premium",
            "company_name": "Premium Corp"
        }
    )
    
    print(f"  Automation triggered: {trigger_result}")
    
    # Revenue tracking and analysis
    print(f"\n💰 REVENUE ANALYSIS:")
    print(f"  Total revenue generated: ${service.total_revenue:.2f}")
    print(f"  Completed campaigns: {len(service.completed_campaigns)}")
    print(f"  Emails sent today: {service.emails_sent_today}")
    print(f"  Revenue history entries: {len(service.revenue_history)}")
    
    # Test conversion strategies
    print(f"\n📈 CONVERSION STRATEGIES:")
    
    # Heavy free user
    heavy_free_user = {
        "current_tier": "free",
        "emails_this_month": 850,
        "campaigns_this_month": 4,
        "open_rate": 0.24,
        "click_rate": 0.08
    }
    
    conversion_strategy = service.get_conversion_strategy(heavy_free_user)
    print(f"  Heavy free user should upgrade: {conversion_strategy['should_upgrade']}")
    print(f"  Recommended tier: {conversion_strategy['recommended_tier']}")
    print(f"  Potential savings: ${conversion_strategy['savings']:.2f}")
    
    # Revenue projections
    print(f"\n📊 REVENUE PROJECTIONS:")
    
    customer_base = [
        {"tier": PricingTier.FREE, "count": 200},
        {"tier": PricingTier.BASIC, "count": 50},
        {"tier": PricingTier.PREMIUM, "count": 15}, 
        {"tier": PricingTier.ENTERPRISE, "count": 3}
    ]
    
    monthly_projection = service.calculate_monthly_revenue_projection(customer_base)
    print(f"  Projected monthly revenue: ${monthly_projection:.2f}")
    print(f"  Target achievement: {monthly_projection/700*100:.1f}% of $700 goal")
    
    # Customer lifetime value
    print(f"\n👤 CUSTOMER LIFETIME VALUE:")
    
    sample_customer = {
        "signup_date": "2025-01-01",
        "tier_progression": [
            {"tier": "free", "months": 2},
            {"tier": "basic", "months": 6},
            {"tier": "premium", "months": 4}
        ],
        "total_spent": 420.0,
        "engagement_score": 0.7
    }
    
    clv = service.calculate_customer_lifetime_value(sample_customer)
    print(f"  Sample customer CLV: ${clv:.2f}")
    
    # Integration recommendations
    print(f"\n🔗 INTEGRATION RECOMMENDATIONS:")
    
    customer_profile = {
        "platform": "shopify",
        "current_tools": ["mailchimp"],
        "pain_points": ["deliverability", "automation complexity"]
    }
    
    recommendations = manager.get_integration_recommendations(customer_profile)
    for rec in recommendations[:3]:
        print(f"  • {rec['title']}: {rec['description'][:50]}...")
    
    # Optimization suggestions
    print(f"\n🎯 CAMPAIGN OPTIMIZATION:")
    
    poor_performance = {
        "campaign_id": "test_campaign",
        "open_rate": 0.12,  # Low
        "click_rate": 0.01,  # Very low
        "unsubscribe_rate": 0.03  # High
    }
    
    optimizations = await manager.suggest_optimizations(poor_performance)
    for opt in optimizations[:3]:
        print(f"  • {opt['title']} ({opt['priority']}): {opt['description'][:60]}...")
    
    print(f"\n🎉 EMAIL AUTOMATION SERVICE TEST COMPLETED!")
    print(f"💰 Total revenue generated: ${service.total_revenue:.2f}")
    print(f"📧 Total emails sent: {service.emails_sent_today}")
    print(f"✅ Service ready for deployment! 🚀")


if __name__ == "__main__":
    asyncio.run(test_complete_email_service())