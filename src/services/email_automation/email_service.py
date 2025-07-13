"""
Email Marketing Automation Service - Revenue Generation Implementation.

Zero-cost bootstrap service targeting $700/month revenue in 10 days.
Provides comprehensive email automation with freemium pricing model.
"""

import asyncio
import aiosmtplib
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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available - using basic templating")

logger = logging.getLogger(__name__)


class CampaignStatus(str, Enum):
    """Status of email campaigns."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    SCHEDULED = "scheduled"


class PricingTier(str, Enum):
    """Pricing tiers for email automation service."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class AutomationType(str, Enum):
    """Types of email automation."""
    WELCOME_SERIES = "welcome_series"
    DRIP_CAMPAIGN = "drip_campaign" 
    TRIGGERED_EMAIL = "triggered_email"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    TRANSACTIONAL = "transactional"


@dataclass
class EmailTemplate:
    """Email template for campaigns."""
    
    template_id: str
    name: str
    subject: str
    html_content: str
    text_content: str
    
    # Template metadata
    category: str = "general"
    variables: List[str] = field(default_factory=list)
    preview_text: Optional[str] = None
    
    # Performance tracking
    usage_count: int = 0
    average_open_rate: float = 0.0
    average_click_rate: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AutomationRule:
    """Email automation rule definition."""
    
    rule_id: str
    name: str
    trigger: str  # Event that triggers the automation
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Rule configuration
    is_active: bool = True
    priority: int = 1
    max_executions: Optional[int] = None
    
    # Performance tracking
    execution_count: int = 0
    success_rate: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EmailCampaign:
    """Represents an email marketing campaign."""
    
    campaign_id: str
    name: str
    customer_email: str
    pricing_tier: PricingTier
    
    # Campaign configuration
    template_id: str
    recipient_list: List[str]
    schedule_type: str = "immediate"  # immediate, scheduled, recurring
    send_time: Optional[datetime] = None
    
    # Personalization
    personalization_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    segmentation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Status tracking
    status: CampaignStatus = CampaignStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Performance metrics
    emails_sent: int = 0
    emails_delivered: int = 0
    emails_opened: int = 0
    emails_clicked: int = 0
    unsubscribes: int = 0
    bounces: int = 0
    
    # Pricing
    estimated_price: float = 0.0
    actual_price: float = 0.0
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class EmailAnalytics:
    """Email campaign analytics."""
    
    campaign_id: str
    emails_sent: int
    emails_delivered: int
    opens: int
    clicks: int
    unsubscribes: int
    bounces: int
    
    # Calculated rates
    delivery_rate: float = 0.0
    open_rate: float = 0.0
    click_rate: float = 0.0
    click_through_rate: float = 0.0
    unsubscribe_rate: float = 0.0
    bounce_rate: float = 0.0
    
    # Revenue tracking
    revenue_generated: float = 0.0
    cost_per_acquisition: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SendResult:
    """Result of sending an email campaign."""
    
    campaign_id: str
    success: bool
    emails_sent: int = 0
    emails_failed: int = 0
    execution_time: float = 0.0
    
    # Detailed results
    successful_recipients: List[str] = field(default_factory=list)
    failed_recipients: List[Dict[str, str]] = field(default_factory=list)
    
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class EmailAutomationService:
    """
    Core email marketing automation service for revenue generation.
    
    Provides freemium email automation with comprehensive features:
    - Campaign management
    - Template system
    - Automation rules
    - Analytics tracking
    - Revenue optimization
    """
    
    def __init__(self):
        # Campaign management
        self.active_campaigns: Dict[str, EmailCampaign] = {}
        self.completed_campaigns: List[EmailCampaign] = []
        self.campaign_queue: List[EmailCampaign] = []
        
        # Template and automation management
        self.templates: Dict[str, EmailTemplate] = {}
        self.automation_rules: Dict[str, AutomationRule] = {}
        
        # Analytics and revenue tracking
        self.analytics_data: List[EmailAnalytics] = []
        self.total_revenue: float = 0.0
        self.revenue_history: List[Dict[str, Any]] = []
        
        # Service metrics
        self.emails_sent_today: int = 0
        self.emails_sent_this_month: int = 0
        self.customer_stats: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.pricing_config = self._load_pricing_config()
        self.tier_limits = self._load_tier_limits()
        
        # SMTP configuration (using free tier initially)
        self.smtp_config = {
            "hostname": "smtp.gmail.com",  # Will use free Gmail SMTP
            "port": 587,
            "use_tls": True,
            "username": None,  # Will be configured by customer
            "password": None   # Will use app passwords
        }
        
        # Data storage
        self.data_dir = Path("data/email_automation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load default templates
        self._load_default_templates()
        
        logger.info("📧 Email Automation Service initialized")
    
    def _load_pricing_config(self) -> Dict[str, Any]:
        """Load pricing configuration for email automation."""
        return {
            PricingTier.FREE: {
                "monthly_fee": 0.0,
                "price_per_email": 0.0,
                "setup_fee": 0.0
            },
            PricingTier.BASIC: {
                "monthly_fee": 25.0,
                "price_per_email": 0.02,  # $0.02 per email
                "setup_fee": 0.0
            },
            PricingTier.PREMIUM: {
                "monthly_fee": 75.0,
                "price_per_email": 0.015,  # $0.015 per email
                "setup_fee": 0.0
            },
            PricingTier.ENTERPRISE: {
                "monthly_fee": 250.0,
                "price_per_email": 0.01,  # $0.01 per email
                "setup_fee": 100.0
            }
        }
    
    def _load_tier_limits(self) -> Dict[PricingTier, Dict[str, int]]:
        """Load tier-specific limits."""
        return {
            PricingTier.FREE: {
                "max_emails_per_month": 1000,
                "max_campaigns_per_month": 5,
                "max_automation_rules": 2,
                "max_templates": 5,
                "max_recipients_per_campaign": 100
            },
            PricingTier.BASIC: {
                "max_emails_per_month": 10000,
                "max_campaigns_per_month": 50,
                "max_automation_rules": 10,
                "max_templates": 25,
                "max_recipients_per_campaign": 2000
            },
            PricingTier.PREMIUM: {
                "max_emails_per_month": 50000,
                "max_campaigns_per_month": 200,
                "max_automation_rules": 50,
                "max_templates": 100,
                "max_recipients_per_campaign": 10000
            },
            PricingTier.ENTERPRISE: {
                "max_emails_per_month": 500000,
                "max_campaigns_per_month": 1000,
                "max_automation_rules": 200,
                "max_templates": 500,
                "max_recipients_per_campaign": 100000
            }
        }
    
    def _load_default_templates(self) -> None:
        """Load default email templates."""
        
        default_templates = [
            {
                "template_id": "welcome_template",
                "name": "Welcome Email",
                "subject": "Welcome to {{company_name}}!",
                "html_content": """
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h1 style="color: #333;">Welcome {{first_name}}!</h1>
                    <p>Thank you for joining {{company_name}}. We're excited to have you on board!</p>
                    <p>Here's what you can expect:</p>
                    <ul>
                        <li>Exclusive updates and offers</li>
                        <li>Helpful tips and resources</li>
                        <li>Priority customer support</li>
                    </ul>
                    <a href="{{get_started_link}}" style="background: #007cba; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px;">Get Started</a>
                </div>
                """,
                "text_content": "Welcome {{first_name}}! Thank you for joining {{company_name}}. Get started: {{get_started_link}}",
                "variables": ["company_name", "first_name", "get_started_link"],
                "category": "onboarding"
            },
            {
                "template_id": "newsletter_template",
                "name": "Newsletter Template", 
                "subject": "{{company_name}} Newsletter - {{month}} {{year}}",
                "html_content": """
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h1 style="color: #333;">{{newsletter_title}}</h1>
                    <p>Hi {{first_name}},</p>
                    <p>{{intro_text}}</p>
                    <div style="margin: 20px 0;">
                        {{main_content}}
                    </div>
                    <p>Best regards,<br>The {{company_name}} Team</p>
                </div>
                """,
                "text_content": "{{newsletter_title}}\n\nHi {{first_name}},\n\n{{intro_text}}\n\n{{main_content}}\n\nBest regards,\nThe {{company_name}} Team",
                "variables": ["company_name", "first_name", "newsletter_title", "month", "year", "intro_text", "main_content"],
                "category": "newsletter"
            },
            {
                "template_id": "promotional_template",
                "name": "Promotional Email",
                "subject": "🎉 {{offer_title}} - Limited Time!",
                "html_content": """
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h1 style="color: #e74c3c;">{{offer_title}}</h1>
                    <p>Hi {{first_name}},</p>
                    <p>{{offer_description}}</p>
                    <div style="background: #f8f9fa; padding: 20px; margin: 20px 0; text-align: center;">
                        <h2 style="color: #e74c3c; margin: 0;">{{discount_percent}}% OFF</h2>
                        <p style="margin: 10px 0;">Use code: <strong>{{promo_code}}</strong></p>
                    </div>
                    <a href="{{shop_link}}" style="background: #e74c3c; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px;">Shop Now</a>
                    <p><small>Offer expires {{expiry_date}}</small></p>
                </div>
                """,
                "text_content": "{{offer_title}}\n\nHi {{first_name}},\n\n{{offer_description}}\n\n{{discount_percent}}% OFF - Use code: {{promo_code}}\n\nShop: {{shop_link}}\n\nExpires: {{expiry_date}}",
                "variables": ["first_name", "offer_title", "offer_description", "discount_percent", "promo_code", "shop_link", "expiry_date"],
                "category": "promotional"
            }
        ]
        
        for template_data in default_templates:
            template = EmailTemplate(**template_data)
            self.templates[template.template_id] = template
        
        logger.info(f"📄 Loaded {len(default_templates)} default templates")
    
    def create_campaign(self, name: str, customer_email: str, pricing_tier: str,
                       template_id: str, recipient_list: List[str], **kwargs) -> EmailCampaign:
        """Create a new email campaign."""
        
        campaign_id = f"campaign_{uuid.uuid4().hex[:8]}"
        tier = PricingTier(pricing_tier.lower())
        
        campaign = EmailCampaign(
            campaign_id=campaign_id,
            name=name,
            customer_email=customer_email,
            pricing_tier=tier,
            template_id=template_id,
            recipient_list=recipient_list,
            **kwargs
        )
        
        # Calculate pricing
        campaign.estimated_price = self.calculate_price(tier, len(recipient_list))
        
        # Add to active campaigns
        self.active_campaigns[campaign_id] = campaign
        self.campaign_queue.append(campaign)
        
        # Track customer activity
        self._track_customer_activity(customer_email, campaign)
        
        logger.info(f"📧 Created campaign {campaign_id} for {customer_email} (${campaign.estimated_price:.2f})")
        
        return campaign
    
    def create_template(self, template_id: str, name: str, subject: str,
                       html_content: str, text_content: str, **kwargs) -> EmailTemplate:
        """Create a new email template."""
        
        template = EmailTemplate(
            template_id=template_id,
            name=name,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            **kwargs
        )
        
        self.templates[template_id] = template
        
        logger.info(f"📄 Created template {template_id}")
        
        return template
    
    def create_automation_rule(self, rule_id: str, name: str, trigger: str,
                              conditions: List[Dict[str, Any]], actions: List[Dict[str, Any]],
                              **kwargs) -> AutomationRule:
        """Create an automation rule."""
        
        rule = AutomationRule(
            rule_id=rule_id,
            name=name,
            trigger=trigger,
            conditions=conditions,
            actions=actions,
            **kwargs
        )
        
        self.automation_rules[rule_id] = rule
        
        logger.info(f"🤖 Created automation rule {rule_id}")
        
        return rule
    
    def calculate_price(self, tier: PricingTier, email_count: int) -> float:
        """Calculate price for email campaign."""
        
        if tier == PricingTier.FREE:
            return 0.0
        
        config = self.pricing_config[tier]
        
        monthly_fee_portion = config["monthly_fee"] / 30  # Daily portion
        email_cost = config["price_per_email"] * email_count
        
        total_price = monthly_fee_portion + email_cost
        
        # Volume discounts
        if email_count > 10000:
            total_price *= 0.8  # 20% discount
        elif email_count > 5000:
            total_price *= 0.9  # 10% discount
        
        return round(total_price, 2)
    
    def get_tier_limits(self, tier: PricingTier) -> Dict[str, int]:
        """Get limits for pricing tier."""
        return self.tier_limits.get(tier, {})
    
    async def send_campaign(self, campaign: EmailCampaign) -> SendResult:
        """Send an email campaign."""
        
        start_time = time.time()
        campaign.status = CampaignStatus.RUNNING
        campaign.started_at = datetime.now()
        
        logger.info(f"🚀 Sending campaign {campaign.campaign_id} to {len(campaign.recipient_list)} recipients")
        
        # Get template
        template = self.templates.get(campaign.template_id)
        if not template:
            raise ValueError(f"Template {campaign.template_id} not found")
        
        successful_recipients = []
        failed_recipients = []
        
        try:
            # Send emails to all recipients
            for recipient_email in campaign.recipient_list:
                try:
                    # Get personalization data for recipient
                    personal_data = campaign.personalization_data.get(recipient_email, {})
                    personal_data.setdefault("first_name", recipient_email.split("@")[0])
                    
                    # Render template
                    rendered_subject = self._render_template(template.subject, personal_data)
                    rendered_html = self._render_template(template.html_content, personal_data)
                    rendered_text = self._render_template(template.text_content, personal_data)
                    
                    # Create email message
                    message = MIMEMultipart("alternative")
                    message["Subject"] = rendered_subject
                    message["From"] = "noreply@emailautomation.service"  # Would be customer's domain
                    message["To"] = recipient_email
                    
                    # Add text and HTML parts
                    text_part = MIMEText(rendered_text, "plain")
                    html_part = MIMEText(rendered_html, "html")
                    
                    message.attach(text_part)
                    message.attach(html_part)
                    
                    # Send email (simulated for testing)
                    await self._send_email_message(message, recipient_email)
                    
                    successful_recipients.append(recipient_email)
                    
                except Exception as e:
                    failed_recipients.append({
                        "email": recipient_email,
                        "error": str(e)
                    })
                    logger.warning(f"Failed to send to {recipient_email}: {e}")
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            # Update campaign status
            campaign.status = CampaignStatus.COMPLETED
            campaign.completed_at = datetime.now()
            campaign.emails_sent = len(successful_recipients)
            campaign.emails_delivered = len(successful_recipients)  # Assume all sent emails are delivered
            
            # Record revenue if paid tier
            if campaign.pricing_tier != PricingTier.FREE:
                self.record_revenue(campaign.campaign_id, campaign.estimated_price)
            
            # Update service metrics
            self.emails_sent_today += len(successful_recipients)
            self.emails_sent_this_month += len(successful_recipients)
            
            execution_time = time.time() - start_time
            
            result = SendResult(
                campaign_id=campaign.campaign_id,
                success=True,
                emails_sent=len(successful_recipients),
                emails_failed=len(failed_recipients),
                successful_recipients=successful_recipients,
                failed_recipients=failed_recipients,
                execution_time=execution_time
            )
            
            logger.info(f"✅ Campaign {campaign.campaign_id} completed: {len(successful_recipients)} sent, {len(failed_recipients)} failed")
            
            return result
            
        except Exception as e:
            campaign.status = CampaignStatus.FAILED
            campaign.error_message = str(e)
            campaign.completed_at = datetime.now()
            
            execution_time = time.time() - start_time
            
            result = SendResult(
                campaign_id=campaign.campaign_id,
                success=False,
                emails_failed=len(campaign.recipient_list),
                error_message=str(e),
                execution_time=execution_time
            )
            
            logger.error(f"❌ Campaign {campaign.campaign_id} failed: {e}")
            
            return result
        
        finally:
            # Move from active to completed
            if campaign.campaign_id in self.active_campaigns:
                del self.active_campaigns[campaign.campaign_id]
            self.completed_campaigns.append(campaign)
    
    async def _send_email_message(self, message: MIMEMultipart, recipient: str) -> None:
        """Send individual email message."""
        
        # For testing/development, we'll simulate sending
        # In production, this would use actual SMTP
        
        # Simulate email sending delay
        await asyncio.sleep(0.01)
        
        # In real implementation:
        # await aiosmtplib.send(
        #     message,
        #     hostname=self.smtp_config["hostname"],
        #     port=self.smtp_config["port"],
        #     use_tls=self.smtp_config["use_tls"],
        #     username=self.smtp_config["username"],
        #     password=self.smtp_config["password"]
        # )
        
        logger.debug(f"📤 Email sent to {recipient}")
    
    def _render_template(self, template_content: str, data: Dict[str, Any]) -> str:
        """Render template with personalization data."""
        
        if JINJA2_AVAILABLE:
            template = Template(template_content)
            return template.render(**data)
        else:
            # Basic template rendering
            rendered = template_content
            for key, value in data.items():
                rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
            return rendered
    
    async def trigger_automation(self, trigger: str, user_email: str, user_data: Dict[str, Any]) -> bool:
        """Trigger automation rules based on events."""
        
        triggered_rules = []
        
        for rule in self.automation_rules.values():
            if rule.trigger == trigger and rule.is_active:
                # Check conditions
                if self._check_automation_conditions(rule, user_data):
                    triggered_rules.append(rule)
        
        # Execute triggered rules
        for rule in triggered_rules:
            try:
                await self._execute_automation_rule(rule, user_email, user_data)
                rule.execution_count += 1
                logger.info(f"🤖 Executed automation rule {rule.rule_id} for {user_email}")
            except Exception as e:
                logger.error(f"Failed to execute automation rule {rule.rule_id}: {e}")
        
        return len(triggered_rules) > 0
    
    def _check_automation_conditions(self, rule: AutomationRule, user_data: Dict[str, Any]) -> bool:
        """Check if automation conditions are met."""
        
        for condition in rule.conditions:
            field = condition["field"]
            operator = condition["operator"]
            expected_value = condition["value"]
            
            actual_value = user_data.get(field)
            
            if operator == "equals" and actual_value != expected_value:
                return False
            elif operator == "not_equals" and actual_value == expected_value:
                return False
            elif operator == "contains" and expected_value not in str(actual_value):
                return False
            # Add more operators as needed
        
        return True
    
    async def _execute_automation_rule(self, rule: AutomationRule, user_email: str, user_data: Dict[str, Any]) -> None:
        """Execute automation rule actions."""
        
        for action in rule.actions:
            action_type = action["type"]
            
            if action_type == "send_email":
                template_id = action["template_id"]
                delay_hours = action.get("delay_hours", 0)
                
                # Create automated campaign
                campaign = self.create_campaign(
                    name=f"Automation: {rule.name}",
                    customer_email="automation@system.com",  # System-generated
                    pricing_tier="free",  # Automation emails are typically free
                    template_id=template_id,
                    recipient_list=[user_email],
                    personalization_data={user_email: user_data}
                )
                
                # Send immediately or schedule
                if delay_hours == 0:
                    await self.send_campaign(campaign)
                else:
                    campaign.schedule_type = "scheduled"
                    campaign.send_time = datetime.now() + timedelta(hours=delay_hours)
                    # In production, would add to scheduler
    
    def record_analytics(self, campaign_id: str, emails_sent: int, emails_delivered: int,
                        opens: int, clicks: int, unsubscribes: int, bounces: int,
                        **kwargs) -> EmailAnalytics:
        """Record email campaign analytics."""
        
        analytics = EmailAnalytics(
            campaign_id=campaign_id,
            emails_sent=emails_sent,
            emails_delivered=emails_delivered,
            opens=opens,
            clicks=clicks,
            unsubscribes=unsubscribes,
            bounces=bounces
        )
        
        # Calculate rates
        if emails_delivered > 0:
            analytics.delivery_rate = emails_delivered / emails_sent
            analytics.open_rate = opens / emails_delivered
            analytics.unsubscribe_rate = unsubscribes / emails_delivered
            analytics.bounce_rate = bounces / emails_sent
            
        if opens > 0:
            analytics.click_rate = clicks / opens
            
        if emails_sent > 0:
            analytics.click_through_rate = clicks / emails_sent
        
        self.analytics_data.append(analytics)
        
        logger.info(f"📊 Recorded analytics for campaign {campaign_id}")
        
        return analytics
    
    def record_revenue(self, campaign_id: str, amount: float) -> None:
        """Record revenue from campaign."""
        
        self.total_revenue += amount
        
        revenue_record = {
            "campaign_id": campaign_id,
            "amount": amount,
            "timestamp": datetime.now().isoformat(),
            "cumulative_revenue": self.total_revenue
        }
        
        self.revenue_history.append(revenue_record)
        
        logger.info(f"💰 Revenue recorded: ${amount:.2f} (Total: ${self.total_revenue:.2f})")
    
    def _track_customer_activity(self, customer_email: str, campaign: EmailCampaign) -> None:
        """Track customer activity for analytics."""
        
        if customer_email not in self.customer_stats:
            self.customer_stats[customer_email] = {
                "first_campaign": campaign.created_at.isoformat(),
                "total_campaigns": 0,
                "total_spent": 0.0,
                "current_tier": campaign.pricing_tier.value,
                "last_activity": campaign.created_at.isoformat()
            }
        
        stats = self.customer_stats[customer_email]
        stats["total_campaigns"] += 1
        stats["last_activity"] = campaign.created_at.isoformat()
        
        if campaign.pricing_tier.value != stats["current_tier"]:
            stats["current_tier"] = campaign.pricing_tier.value
    
    def can_customer_send_campaign(self, customer_email: str, tier: PricingTier,
                                  customer_usage: Dict[str, Any]) -> bool:
        """Check if customer can send campaign within tier limits."""
        
        limits = self.get_tier_limits(tier)
        
        emails_this_month = customer_usage.get("emails_this_month", 0)
        campaigns_this_month = customer_usage.get("campaigns_this_month", 0)
        
        if emails_this_month >= limits.get("max_emails_per_month", float('inf')):
            return False
        
        if campaigns_this_month >= limits.get("max_campaigns_per_month", float('inf')):
            return False
        
        return True
    
    def get_conversion_strategy(self, customer_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversion strategy for freemium users."""
        
        current_tier = customer_usage.get("current_tier", "free")
        emails_this_month = customer_usage.get("emails_this_month", 0)
        campaigns_this_month = customer_usage.get("campaigns_this_month", 0)
        
        strategy = {
            "should_upgrade": False,
            "recommended_tier": current_tier,
            "reasons": [],
            "benefits": [],
            "savings": 0.0
        }
        
        if current_tier == "free":
            free_limits = self.get_tier_limits(PricingTier.FREE)
            
            # Check if approaching limits
            email_usage_percent = emails_this_month / free_limits["max_emails_per_month"]
            campaign_usage_percent = campaigns_this_month / free_limits["max_campaigns_per_month"]
            
            if email_usage_percent > 0.8 or campaign_usage_percent > 0.8:
                strategy["should_upgrade"] = True
                strategy["recommended_tier"] = "basic"
                strategy["reasons"].append("Approaching free tier limits")
                strategy["benefits"] = [
                    "10x more emails per month",
                    "Advanced automation features",
                    "Priority support",
                    "Custom templates"
                ]
                
                # Calculate potential savings
                overage_emails = max(0, emails_this_month - free_limits["max_emails_per_month"])
                basic_monthly_cost = self.pricing_config[PricingTier.BASIC]["monthly_fee"]
                pay_per_email_cost = overage_emails * 0.05  # Hypothetical overage rate
                
                if pay_per_email_cost > basic_monthly_cost:
                    strategy["savings"] = pay_per_email_cost - basic_monthly_cost
        
        return strategy
    
    def calculate_monthly_revenue_projection(self, customer_base: List[Dict[str, Any]]) -> float:
        """Calculate projected monthly revenue."""
        
        total_projection = 0.0
        
        for segment in customer_base:
            tier = segment["tier"]
            customer_count = segment["count"]
            
            if tier == PricingTier.FREE:
                # Free tier converts ~8% to paid
                conversion_revenue = customer_count * 0.08 * 25.0  # 8% convert to $25/month
                total_projection += conversion_revenue
            elif tier == PricingTier.BASIC:
                avg_monthly = 45.0  # Average basic customer with email costs
                total_projection += customer_count * avg_monthly
            elif tier == PricingTier.PREMIUM:
                avg_monthly = 125.0  # Average premium customer
                total_projection += customer_count * avg_monthly
            elif tier == PricingTier.ENTERPRISE:
                avg_monthly = 400.0  # Average enterprise customer
                total_projection += customer_count * avg_monthly
        
        return total_projection
    
    def calculate_customer_lifetime_value(self, customer_data: Dict[str, Any]) -> float:
        """Calculate customer lifetime value."""
        
        total_spent = customer_data.get("total_spent", 0.0)
        tier_progression = customer_data.get("tier_progression", [])
        
        if not tier_progression:
            return 0.0
        
        # Calculate average monthly spend
        total_months = sum(tier["months"] for tier in tier_progression)
        if total_months == 0:
            return 0.0
        
        avg_monthly_spend = total_spent / total_months
        
        # Project based on engagement and tier progression
        engagement_multiplier = customer_data.get("engagement_score", 0.5) + 0.5
        tier_growth_multiplier = 1.3  # Customers tend to upgrade over time
        projected_lifetime_months = 18  # Assume 18-month lifetime
        
        clv = avg_monthly_spend * engagement_multiplier * tier_growth_multiplier * projected_lifetime_months
        
        return clv


class EmailServiceManager:
    """
    High-level email service management and customer lifecycle.
    Handles onboarding, integrations, and optimization.
    """
    
    def __init__(self):
        self.email_service = EmailAutomationService()
        
        logger.info("📋 Email Service Manager initialized")
    
    def onboard_customer(self, email: str, company_name: str, industry: str,
                        expected_volume: int, **kwargs) -> Dict[str, Any]:
        """Onboard new customer with personalized setup."""
        
        customer_id = f"customer_{uuid.uuid4().hex[:8]}"
        
        # Recommend tier based on expected volume
        if expected_volume <= 1000:
            recommended_tier = "free"
        elif expected_volume <= 10000:
            recommended_tier = "basic"
        elif expected_volume <= 50000:
            recommended_tier = "premium"
        else:
            recommended_tier = "enterprise"
        
        # Create welcome campaign
        welcome_campaign = self.email_service.create_campaign(
            name="Welcome to Email Automation",
            customer_email="system@emailautomation.service",
            pricing_tier="free",
            template_id="welcome_template",
            recipient_list=[email],
            personalization_data={
                email: {
                    "first_name": email.split("@")[0],
                    "company_name": company_name,
                    "get_started_link": f"https://app.emailautomation.service/setup/{customer_id}"
                }
            }
        )
        
        onboarding_data = {
            "customer_id": customer_id,
            "email": email,
            "company_name": company_name,
            "industry": industry,
            "recommended_tier": recommended_tier,
            "expected_volume": expected_volume,
            "welcome_campaign": welcome_campaign.campaign_id,
            "setup_guide": f"https://docs.emailautomation.service/setup/{industry}",
            "onboarded_at": datetime.now().isoformat()
        }
        
        logger.info(f"👋 Onboarded customer {customer_id} ({email}) - recommended tier: {recommended_tier}")
        
        return onboarding_data
    
    def get_integration_recommendations(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get integration recommendations based on customer profile."""
        
        platform = customer_profile.get("platform", "").lower()
        current_tools = customer_profile.get("current_tools", [])
        pain_points = customer_profile.get("pain_points", [])
        
        recommendations = []
        
        # Platform-specific integrations
        if "shopify" in platform:
            recommendations.append({
                "title": "Shopify Integration",
                "description": "Sync customer data and trigger automated emails based on purchase behavior",
                "setup_time": "15 minutes",
                "benefits": ["Automated abandoned cart recovery", "Post-purchase follow-ups", "Customer segmentation"]
            })
        
        if "wordpress" in platform:
            recommendations.append({
                "title": "WordPress Plugin",
                "description": "Integrate with your WordPress site for seamless subscriber management",
                "setup_time": "10 minutes",
                "benefits": ["Automatic subscriber sync", "Comment-based triggers", "Content-based automation"]
            })
        
        # Tool migration recommendations
        if "mailchimp" in [tool.lower() for tool in current_tools]:
            recommendations.append({
                "title": "Mailchimp Migration Assistant",
                "description": "Easily migrate your existing campaigns and subscriber lists",
                "setup_time": "30 minutes",
                "benefits": ["Preserve subscriber data", "Import existing templates", "Maintain automation workflows"]
            })
        
        # Pain point solutions
        if "deliverability" in [p.lower() for p in pain_points]:
            recommendations.append({
                "title": "Deliverability Optimization",
                "description": "Advanced deliverability features to improve inbox placement",
                "setup_time": "20 minutes",
                "benefits": ["Spam score checking", "Domain authentication", "Reputation monitoring"]
            })
        
        return recommendations
    
    async def suggest_optimizations(self, campaign_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest campaign optimizations based on performance."""
        
        optimizations = []
        
        open_rate = campaign_performance.get("open_rate", 0)
        click_rate = campaign_performance.get("click_rate", 0)
        unsubscribe_rate = campaign_performance.get("unsubscribe_rate", 0)
        
        # Subject line optimization
        if open_rate < 0.20:  # Below average open rate
            optimizations.append({
                "type": "subject",
                "priority": "high",
                "title": "Improve Subject Lines",
                "description": "Your open rate is below average. Try more compelling subject lines.",
                "suggestions": [
                    "Add personalization (use recipient's name)",
                    "Create urgency with time-sensitive language",
                    "Use emojis sparingly to stand out",
                    "A/B test different subject line styles"
                ]
            })
        
        # Send time optimization
        if open_rate < 0.18:
            optimizations.append({
                "type": "timing",
                "priority": "medium",
                "title": "Optimize Send Times",
                "description": "Consider testing different send times for better engagement.",
                "suggestions": [
                    "Test Tuesday-Thursday 10am-2pm",
                    "Avoid Monday mornings and Friday afternoons",
                    "Consider your audience's time zone",
                    "Test weekend sends for B2C audiences"
                ]
            })
        
        # Content optimization
        if click_rate < 0.03:  # Below average click rate
            optimizations.append({
                "type": "content",
                "priority": "high",
                "title": "Improve Email Content",
                "description": "Low click rates suggest content needs improvement.",
                "suggestions": [
                    "Add clear call-to-action buttons",
                    "Simplify your message and focus on one goal",
                    "Use more visual elements (images, GIFs)",
                    "Segment your audience for more relevant content"
                ]
            })
        
        # List health optimization
        if unsubscribe_rate > 0.02:  # High unsubscribe rate
            optimizations.append({
                "type": "list_health",
                "priority": "critical",
                "title": "Reduce Unsubscribe Rate",
                "description": "High unsubscribe rate indicates content or frequency issues.",
                "suggestions": [
                    "Reduce email frequency",
                    "Better segment your audience",
                    "Provide more value in each email",
                    "Survey unsubscribers for feedback"
                ]
            })
        
        return optimizations
    
    def get_customer_metrics(self) -> Dict[str, Any]:
        """Get comprehensive customer and service metrics."""
        
        # Calculate total customers from campaigns
        total_customers = len(set(
            campaign.customer_email 
            for campaign in list(self.active_campaigns.values()) + self.completed_campaigns
        ))
        
        # Calculate active campaigns
        active_campaigns_count = len(self.active_campaigns)
        
        # Calculate total emails sent from all campaigns
        total_emails_sent = sum(
            getattr(campaign, 'emails_sent', len(campaign.recipient_list))
            for campaign in list(self.active_campaigns.values()) + self.completed_campaigns
        )
        
        # Calculate conversion rate based on completed campaigns
        total_recipients = sum(
            len(campaign.recipient_list) 
            for campaign in self.completed_campaigns
        )
        
        # Estimate conversion rate (this would be real data in production)
        estimated_conversions = total_recipients * 0.05  # 5% conversion rate
        conversion_rate = estimated_conversions / max(total_recipients, 1)
        
        # Calculate CLV (simplified estimation)
        average_revenue_per_customer = 25.0  # From pricing tiers
        average_customer_lifetime_months = 12
        customer_lifetime_value = average_revenue_per_customer * average_customer_lifetime_months
        
        return {
            'total_revenue': self.total_revenue,
            'total_customers': total_customers,
            'active_campaigns': active_campaigns_count,
            'emails_sent_today': self.emails_sent_today,
            'conversion_rate': conversion_rate,
            'customer_lifetime_value': customer_lifetime_value,
            'total_emails_sent': total_emails_sent,
            'completed_campaigns': len(self.completed_campaigns),
            'revenue_per_customer': average_revenue_per_customer,
            'estimated_conversions': estimated_conversions
        }


# Factory functions
def create_email_automation_service() -> EmailAutomationService:
    """Create email automation service instance."""
    return EmailAutomationService()


def create_email_service_manager() -> EmailServiceManager:
    """Create email service manager instance.""" 
    return EmailServiceManager()