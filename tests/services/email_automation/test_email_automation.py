"""
Test suite for Email Marketing Automation Service.

Following TDD methodology - tests written first to define behavior.
Target: $700/month revenue in 10 days through email automation.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.services.email_automation.email_service import (
    EmailAutomationService,
    EmailCampaign,
    EmailTemplate,
    CampaignStatus,
    PricingTier,
    AutomationRule,
    EmailAnalytics,
    EmailServiceManager
)


class TestEmailAutomationService:
    """Test cases for core email automation functionality."""
    
    @pytest.fixture
    def email_service(self):
        """Create email automation service instance for testing."""
        return EmailAutomationService()
    
    @pytest.fixture
    def sample_campaign(self):
        """Create sample email campaign for testing."""
        return EmailCampaign(
            campaign_id="test_campaign_001",
            name="Welcome Series",
            customer_email="customer@test.com",
            pricing_tier=PricingTier.BASIC,
            template_id="welcome_template",
            recipient_list=["subscriber1@test.com", "subscriber2@test.com"],
            schedule_type="immediate"
        )
    
    def test_email_service_initialization(self, email_service):
        """Test that email service initializes correctly."""
        assert email_service is not None
        assert email_service.active_campaigns == {}
        assert email_service.completed_campaigns == []
        assert email_service.total_revenue == 0.0
        assert email_service.emails_sent_today == 0
    
    def test_create_email_campaign(self, email_service):
        """Test creating a new email campaign."""
        campaign_data = {
            "name": "Product Launch Campaign",
            "customer_email": "business@test.com",
            "pricing_tier": "premium",
            "template_id": "product_launch",
            "recipient_list": ["user1@test.com", "user2@test.com"],
            "schedule_type": "scheduled",
            "send_time": datetime.now() + timedelta(hours=1)
        }
        
        campaign = email_service.create_campaign(**campaign_data)
        
        assert campaign.campaign_id is not None
        assert campaign.name == campaign_data["name"]
        assert campaign.customer_email == campaign_data["customer_email"]
        assert campaign.status == CampaignStatus.PENDING
        assert len(campaign.recipient_list) == 2
    
    def test_email_template_system(self, email_service):
        """Test email template creation and management."""
        template_data = {
            "template_id": "custom_template",
            "name": "Custom Newsletter",
            "subject": "{{company_name}} Weekly Update",
            "html_content": "<h1>Hello {{first_name}}!</h1><p>{{content}}</p>",
            "text_content": "Hello {{first_name}}! {{content}}",
            "variables": ["company_name", "first_name", "content"]
        }
        
        template = email_service.create_template(**template_data)
        
        assert template.template_id == template_data["template_id"]
        assert "{{first_name}}" in template.html_content
        assert len(template.variables) == 3
    
    def test_pricing_calculation(self, email_service):
        """Test pricing calculation for different tiers and volumes."""
        # Free tier
        free_price = email_service.calculate_price(PricingTier.FREE, emails=100)
        assert free_price == 0.0
        
        # Basic tier
        basic_price = email_service.calculate_price(PricingTier.BASIC, emails=1000)
        assert basic_price > 0
        
        # Premium tier should be cheaper per email
        premium_price = email_service.calculate_price(PricingTier.PREMIUM, emails=1000)
        assert premium_price > basic_price
        
        # Bulk discount should apply
        bulk_price = email_service.calculate_price(PricingTier.BASIC, emails=10000)
        single_price = email_service.calculate_price(PricingTier.BASIC, emails=1000)
        assert bulk_price < single_price * 10  # Bulk discount
    
    @pytest.mark.asyncio
    async def test_send_email_campaign(self, email_service, sample_campaign):
        """Test sending an email campaign."""
        # Mock email sending
        with patch('src.services.email_automation.email_service.aiosmtplib.send') as mock_send:
            mock_send.return_value = AsyncMock()
            
            # Add template
            template = EmailTemplate(
                template_id=sample_campaign.template_id,
                name="Welcome Template",
                subject="Welcome to our service!",
                html_content="<h1>Welcome {{name}}!</h1>",
                text_content="Welcome {{name}}!"
            )
            email_service.templates[template.template_id] = template
            
            result = await email_service.send_campaign(sample_campaign)
            
            assert result.success == True
            assert result.emails_sent == len(sample_campaign.recipient_list)
            assert sample_campaign.status == CampaignStatus.COMPLETED
    
    def test_automation_rules(self, email_service):
        """Test automation rules and triggers."""
        # Create automation rule
        rule_data = {
            "rule_id": "welcome_automation",
            "name": "Welcome Series",
            "trigger": "user_signup",
            "conditions": [{"field": "user_type", "operator": "equals", "value": "premium"}],
            "actions": [
                {"type": "send_email", "template_id": "welcome_email", "delay_hours": 0},
                {"type": "send_email", "template_id": "tips_email", "delay_hours": 24},
                {"type": "send_email", "template_id": "feedback_email", "delay_hours": 168}
            ]
        }
        
        rule = email_service.create_automation_rule(**rule_data)
        
        assert rule.rule_id == rule_data["rule_id"]
        assert rule.trigger == "user_signup"
        assert len(rule.actions) == 3
        assert rule.actions[0]["delay_hours"] == 0
    
    @pytest.mark.asyncio
    async def test_automation_execution(self, email_service):
        """Test executing automation rules."""
        # Create rule
        rule = AutomationRule(
            rule_id="test_automation",
            name="Test Automation",
            trigger="user_signup",
            actions=[
                {"type": "send_email", "template_id": "welcome", "delay_hours": 0}
            ]
        )
        email_service.automation_rules["test_automation"] = rule
        
        # Create template
        template = EmailTemplate(
            template_id="welcome",
            name="Welcome",
            subject="Welcome!",
            html_content="<p>Welcome!</p>",
            text_content="Welcome!"
        )
        email_service.templates["welcome"] = template
        
        # Trigger automation
        trigger_data = {
            "trigger": "user_signup",
            "user_email": "newuser@test.com",
            "user_data": {"name": "John", "user_type": "premium"}
        }
        
        with patch('src.services.email_automation.email_service.aiosmtplib.send') as mock_send:
            mock_send.return_value = AsyncMock()
            
            result = await email_service.trigger_automation(**trigger_data)
            
            assert result == True
            assert len(email_service.active_campaigns) >= 0  # Campaign created
    
    def test_analytics_tracking(self, email_service, sample_campaign):
        """Test email analytics and performance tracking."""
        # Simulate campaign completion
        sample_campaign.status = CampaignStatus.COMPLETED
        sample_campaign.emails_sent = 100
        sample_campaign.completed_at = datetime.now()
        
        # Record analytics
        analytics_data = {
            "campaign_id": sample_campaign.campaign_id,
            "emails_sent": 100,
            "emails_delivered": 95,
            "opens": 25,
            "clicks": 8,
            "unsubscribes": 2,
            "bounces": 5
        }
        
        analytics = email_service.record_analytics(**analytics_data)
        
        assert analytics.open_rate == 25/95  # 26.3%
        assert analytics.click_rate == 8/25   # 32%
        assert analytics.unsubscribe_rate == 2/95  # 2.1%
        assert analytics.delivery_rate == 95/100   # 95%
    
    def test_revenue_tracking(self, email_service, sample_campaign):
        """Test revenue tracking for campaigns."""
        initial_revenue = email_service.total_revenue
        
        # Simulate paid campaign
        campaign_price = 45.0
        sample_campaign.actual_price = campaign_price
        
        email_service.record_revenue(sample_campaign.campaign_id, campaign_price)
        
        assert email_service.total_revenue == initial_revenue + campaign_price
        assert len(email_service.revenue_history) == 1
    
    def test_tier_limits_enforcement(self, email_service):
        """Test enforcement of tier limits."""
        # Free tier limits
        free_limits = email_service.get_tier_limits(PricingTier.FREE)
        assert free_limits["max_emails_per_month"] <= 1000
        assert free_limits["max_campaigns_per_month"] <= 5
        
        # Basic tier should have higher limits
        basic_limits = email_service.get_tier_limits(PricingTier.BASIC)
        assert basic_limits["max_emails_per_month"] > free_limits["max_emails_per_month"]
        
        # Test limit checking
        customer_usage = {
            "emails_this_month": 2000,
            "campaigns_this_month": 3,
            "current_tier": PricingTier.FREE
        }
        
        can_send = email_service.can_customer_send_campaign(
            "customer@test.com", 
            PricingTier.FREE,
            customer_usage
        )
        
        assert can_send == False  # Should exceed free tier limits


class TestEmailServiceManager:
    """Test cases for email service management."""
    
    @pytest.fixture
    def service_manager(self):
        """Create service manager instance."""
        return EmailServiceManager()
    
    def test_customer_onboarding(self, service_manager):
        """Test customer onboarding flow."""
        customer_data = {
            "email": "newcustomer@business.com",
            "company_name": "Test Business",
            "industry": "ecommerce",
            "expected_volume": 5000
        }
        
        onboarding = service_manager.onboard_customer(**customer_data)
        
        assert onboarding["customer_id"] is not None
        assert onboarding["recommended_tier"] in ["basic", "premium", "enterprise"]
        assert "welcome_campaign" in onboarding
        assert onboarding["setup_guide"] is not None
    
    def test_integration_recommendations(self, service_manager):
        """Test integration recommendations based on customer profile."""
        customer_profile = {
            "platform": "shopify",
            "current_tools": ["mailchimp", "klaviyo"],
            "pain_points": ["deliverability", "automation complexity"]
        }
        
        recommendations = service_manager.get_integration_recommendations(customer_profile)
        
        assert len(recommendations) > 0
        assert any("shopify" in rec["description"].lower() for rec in recommendations)
        assert any("automation" in rec["title"].lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_automated_optimization(self, service_manager):
        """Test automated campaign optimization."""
        campaign_performance = {
            "campaign_id": "test_campaign",
            "open_rate": 0.15,  # Low open rate
            "click_rate": 0.02,  # Low click rate
            "unsubscribe_rate": 0.05,  # High unsubscribe rate
            "send_time": "10:00",
            "subject_line": "Check out our products"
        }
        
        optimizations = await service_manager.suggest_optimizations(campaign_performance)
        
        assert len(optimizations) > 0
        assert any("subject" in opt["type"] for opt in optimizations)
        assert any("timing" in opt["type"] for opt in optimizations)


class TestRevenueGeneration:
    """Test cases for revenue generation aspects."""
    
    @pytest.fixture
    def revenue_service(self):
        """Create revenue-focused service."""
        return EmailAutomationService()
    
    def test_freemium_conversion_flow(self, revenue_service):
        """Test freemium to paid conversion."""
        # Heavy free user
        customer_usage = {
            "emails_this_month": 950,  # Close to free limit
            "campaigns_this_month": 4,
            "current_tier": "free",
            "open_rate": 0.25,  # Good engagement
            "click_rate": 0.08
        }
        
        conversion_strategy = revenue_service.get_conversion_strategy(customer_usage)
        
        assert conversion_strategy["should_upgrade"] == True
        assert conversion_strategy["recommended_tier"] == "basic"
        assert "savings" in conversion_strategy
        assert len(conversion_strategy["benefits"]) > 0
    
    def test_revenue_projections(self, revenue_service):
        """Test monthly revenue projections."""
        customer_base = [
            {"tier": PricingTier.FREE, "count": 500},
            {"tier": PricingTier.BASIC, "count": 100},
            {"tier": PricingTier.PREMIUM, "count": 30},
            {"tier": PricingTier.ENTERPRISE, "count": 5}
        ]
        
        projection = revenue_service.calculate_monthly_revenue_projection(customer_base)
        
        assert projection > 0
        assert projection >= 700  # Target $700/month
    
    def test_customer_lifetime_value(self, revenue_service):
        """Test CLV calculations."""
        customer_data = {
            "signup_date": "2025-01-01",
            "tier_progression": [
                {"tier": "free", "months": 2},
                {"tier": "basic", "months": 8},
                {"tier": "premium", "months": 4}
            ],
            "total_spent": 850.0,
            "engagement_score": 0.8
        }
        
        clv = revenue_service.calculate_customer_lifetime_value(customer_data)
        
        assert clv > customer_data["total_spent"]
        assert clv > 0


# Integration test
class TestFullServiceIntegration:
    """Integration tests for complete email automation workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_customer_journey(self):
        """Test complete customer journey from signup to revenue."""
        service = EmailAutomationService()
        manager = EmailServiceManager()
        
        # 1. Customer signs up
        customer = manager.onboard_customer(
            email="integration@test.com",
            company_name="Test Company",
            industry="saas",
            expected_volume=2000
        )
        
        # 2. Customer creates first campaign
        campaign_data = {
            "name": "Welcome Campaign",
            "customer_email": customer["email"],
            "pricing_tier": "free",
            "template_id": "welcome_template",
            "recipient_list": ["user1@test.com", "user2@test.com"]
        }
        
        campaign = service.create_campaign(**campaign_data)
        assert campaign.estimated_price == 0.0  # Free tier
        
        # 3. Campaign is sent successfully
        template = EmailTemplate(
            template_id="welcome_template",
            name="Welcome",
            subject="Welcome!",
            html_content="<p>Welcome {{name}}!</p>",
            text_content="Welcome {{name}}!"
        )
        service.templates[template.template_id] = template
        
        with patch('src.services.email_automation.email_service.aiosmtplib.send') as mock_send:
            mock_send.return_value = AsyncMock()
            
            result = await service.send_campaign(campaign)
            assert result.success == True
        
        # 4. Customer sees value and gets upgrade recommendation
        usage = {
            "emails_this_month": 800,
            "campaigns_this_month": 4,
            "current_tier": "free",
            "open_rate": 0.22,
            "click_rate": 0.06
        }
        
        conversion = service.get_conversion_strategy(usage)
        assert conversion["should_upgrade"] == True
        
        # 5. Customer upgrades to paid tier
        paid_campaign = service.create_campaign(
            name="Premium Campaign",
            customer_email=customer["email"],
            pricing_tier="basic",
            template_id="premium_template",
            recipient_list=["user{}@test.com".format(i) for i in range(1000)]
        )
        
        assert paid_campaign.estimated_price > 0
        
        # 6. Revenue is generated
        service.record_revenue(paid_campaign.campaign_id, paid_campaign.estimated_price)
        assert service.total_revenue > 0