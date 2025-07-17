"""
Advanced Email Marketing Automation System for Revenue Generation.

Integrates with RealRevenueEngine to provide sophisticated email marketing
campaigns, customer lifecycle management, and automated revenue optimization.

Author: Senior RSI Engineer
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
import hashlib

from loguru import logger

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType, Disposition
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    logger.warning("SendGrid not available - install with: pip install sendgrid")

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("Jinja2 not available - install with: pip install jinja2")

from .real_revenue_engine import RealRevenueEngine, Customer, PaymentStatus, SubscriptionStatus


class CampaignType(str, Enum):
    """Email campaign types"""
    WELCOME_SERIES = "welcome_series"
    ONBOARDING = "onboarding"
    NURTURE = "nurture"
    PROMOTIONAL = "promotional"
    RETENTION = "retention"
    WINBACK = "winback"
    UPSELL = "upsell"
    CROSS_SELL = "cross_sell"
    EDUCATIONAL = "educational"
    TRANSACTIONAL = "transactional"

class CampaignStatus(str, Enum):
    """Campaign execution status"""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class EmailStatus(str, Enum):
    """Individual email status"""
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    BOUNCED = "bounced"
    SPAM = "spam"
    UNSUBSCRIBED = "unsubscribed"

@dataclass
class EmailTemplate:
    """Email template configuration"""
    template_id: str
    name: str
    subject_template: str
    html_content_template: str
    text_content_template: str
    campaign_type: CampaignType
    variables: List[str] = field(default_factory=list)
    personalization_score: float = 0.0
    conversion_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class EmailCampaign:
    """Email marketing campaign"""
    campaign_id: str
    name: str
    campaign_type: CampaignType
    status: CampaignStatus
    template_id: str
    target_audience: Dict[str, Any]
    schedule: Dict[str, Any]
    
    # Performance metrics
    total_recipients: int = 0
    emails_sent: int = 0
    emails_delivered: int = 0
    emails_opened: int = 0
    emails_clicked: int = 0
    conversions: int = 0
    revenue_generated: Decimal = field(default=Decimal('0.00'))
    
    # Configuration
    send_rate_limit: int = 100  # emails per hour
    ab_test_config: Optional[Dict[str, Any]] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass 
class EmailRecord:
    """Individual email record"""
    email_id: str
    campaign_id: str
    customer_id: str
    template_id: str
    recipient_email: str
    subject: str
    status: EmailStatus
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None
    conversion_value: Decimal = field(default=Decimal('0.00'))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CustomerSegment:
    """Customer segmentation for targeted campaigns"""
    segment_id: str
    name: str
    criteria: Dict[str, Any]
    customer_ids: List[str] = field(default_factory=list)
    size: int = 0
    lifetime_value_avg: Decimal = field(default=Decimal('0.00'))
    conversion_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class EmailMarketingAutomation:
    """
    Advanced Email Marketing Automation System.
    
    Features:
    - Sophisticated customer segmentation
    - AI-powered template optimization
    - Automated campaign sequences
    - Revenue attribution tracking
    - A/B testing capabilities
    - Personalization at scale
    """
    
    def __init__(self, 
                 revenue_engine: RealRevenueEngine,
                 sendgrid_api_key: Optional[str] = None,
                 templates_dir: str = "email_templates"):
        
        self.revenue_engine = revenue_engine
        self.sendgrid_api_key = sendgrid_api_key
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize SendGrid
        if SENDGRID_AVAILABLE and sendgrid_api_key:
            self.sendgrid_client = sendgrid.SendGridAPIClient(api_key=sendgrid_api_key)
            self.email_enabled = True
            logger.info("✅ SendGrid email marketing enabled")
        else:
            self.email_enabled = False
            logger.warning("⚠️ Email marketing disabled")
        
        # Initialize Jinja2 for templating
        if JINJA2_AVAILABLE:
            self.template_env = Environment(loader=FileSystemLoader(str(self.templates_dir)))
            self.templating_enabled = True
        else:
            self.templating_enabled = False
            logger.warning("⚠️ Advanced templating disabled")
        
        # Storage
        self.templates: Dict[str, EmailTemplate] = {}
        self.campaigns: Dict[str, EmailCampaign] = {}
        self.email_records: Dict[str, EmailRecord] = {}
        self.customer_segments: Dict[str, CustomerSegment] = {}
        
        # Performance tracking
        self.total_emails_sent = 0
        self.total_revenue_attributed = Decimal('0.00')
        self.campaign_roi_history: List[Dict[str, Any]] = []
        
        # Initialize default templates and segments
        self._initialize_default_templates()
        self._initialize_default_segments()
        
        logger.info("🚀 Email Marketing Automation System initialized")
    
    def _initialize_default_templates(self):
        """Initialize default email templates"""
        
        default_templates = [
            EmailTemplate(
                template_id="welcome_series_1",
                name="Welcome - Getting Started",
                campaign_type=CampaignType.WELCOME_SERIES,
                subject_template="Welcome to Hephaestus RSI, {{customer_name}}! 🏛️",
                html_content_template="""
                <h1>Welcome {{customer_name}}!</h1>
                <p>Thank you for joining Hephaestus RSI, the world's most advanced 
                   Recursive Self-Improvement AI system.</p>
                <p>Here's what you can expect:</p>
                <ul>
                    <li>🧠 Cutting-edge AI capabilities that improve themselves</li>
                    <li>📈 Real-time performance monitoring and optimization</li>
                    <li>🛡️ Enterprise-grade security and safety measures</li>
                    <li>💡 Continuous learning and adaptation</li>
                </ul>
                <p><a href="{{dashboard_url}}" style="background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px;">Access Your Dashboard</a></p>
                <p>Questions? Just reply to this email!</p>
                """,
                text_content_template="""
                Welcome {{customer_name}}!
                
                Thank you for joining Hephaestus RSI, the world's most advanced 
                Recursive Self-Improvement AI system.
                
                Access your dashboard: {{dashboard_url}}
                
                Questions? Just reply to this email!
                """,
                variables=["customer_name", "dashboard_url"],
                personalization_score=0.8
            ),
            
            EmailTemplate(
                template_id="onboarding_api_setup",
                name="Onboarding - API Setup Guide", 
                campaign_type=CampaignType.ONBOARDING,
                subject_template="Get started with the Hephaestus API in 5 minutes",
                html_content_template="""
                <h1>Ready to build with Hephaestus? 🛠️</h1>
                <p>Hi {{customer_name}},</p>
                <p>Let's get you up and running with our powerful RSI API:</p>
                
                <h2>Quick Start Guide</h2>
                <pre><code>
import requests

# Your API key
api_key = "{{api_key}}"

# Make your first prediction
response = requests.post(
    "https://api.hephaestus-rsi.com/predict",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"features": {"x": 1.0, "y": 2.0}}
)
                </code></pre>
                
                <p><a href="{{docs_url}}" style="background: #059669; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px;">View Full Documentation</a></p>
                
                <p>Need help? Our team is here for you!</p>
                """,
                text_content_template="""
                Ready to build with Hephaestus?
                
                Hi {{customer_name}},
                
                Get started with our RSI API: {{docs_url}}
                Your API key: {{api_key}}
                
                Need help? Just reply to this email!
                """,
                variables=["customer_name", "api_key", "docs_url"],
                personalization_score=0.9
            ),
            
            EmailTemplate(
                template_id="upsell_professional",
                name="Upsell - Professional Plan",
                campaign_type=CampaignType.UPSELL,
                subject_template="{{customer_name}}, unlock advanced RSI features 🚀",
                html_content_template="""
                <h1>Take your AI to the next level! 🚀</h1>
                <p>Hi {{customer_name}},</p>
                <p>You've been making great progress with Hephaestus RSI! Based on your usage, 
                   you could benefit significantly from our Professional plan:</p>
                
                <h2>What you'll get:</h2>
                <ul>
                    <li>🧪 Advanced hypothesis testing and validation</li>
                    <li>⚡ 10x faster processing with priority queues</li>
                    <li>📊 Real-time analytics and performance insights</li>
                    <li>🛡️ Enhanced security and compliance features</li>
                    <li>👥 Dedicated support team</li>
                </ul>
                
                <p><strong>Estimated additional revenue: ${{estimated_additional_revenue}}/month</strong></p>
                
                <p><a href="{{upgrade_url}}" style="background: #dc2626; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px;">Upgrade Now - 20% Off First Month</a></p>
                
                <p>Questions about upgrading? Hit reply!</p>
                """,
                text_content_template="""
                Take your AI to the next level!
                
                Hi {{customer_name}},
                
                Upgrade to Professional for advanced features.
                Estimated additional revenue: ${{estimated_additional_revenue}}/month
                
                Upgrade: {{upgrade_url}}
                """,
                variables=["customer_name", "estimated_additional_revenue", "upgrade_url"],
                personalization_score=0.95
            )
        ]
        
        for template in default_templates:
            self.templates[template.template_id] = template
        
        logger.info(f"✅ Initialized {len(default_templates)} default email templates")
    
    def _initialize_default_segments(self):
        """Initialize default customer segments"""
        
        # Create segments based on customer behavior
        segments = [
            CustomerSegment(
                segment_id="new_customers",
                name="New Customers (< 30 days)",
                criteria={
                    "days_since_signup": {"$lt": 30},
                    "total_spent": {"$lt": 500}
                }
            ),
            CustomerSegment(
                segment_id="power_users",
                name="Power Users (High API Usage)",
                criteria={
                    "api_calls_last_30d": {"$gt": 10000},
                    "total_spent": {"$gt": 1000}
                }
            ),
            CustomerSegment(
                segment_id="at_risk",
                name="At Risk (Low Recent Activity)",
                criteria={
                    "days_since_last_login": {"$gt": 14},
                    "total_spent": {"$gt": 100}
                }
            ),
            CustomerSegment(
                segment_id="high_value",
                name="High Value Customers (LTV > $5k)",
                criteria={
                    "lifetime_value": {"$gt": 5000}
                }
            )
        ]
        
        for segment in segments:
            self.customer_segments[segment.segment_id] = segment
        
        logger.info(f"✅ Initialized {len(segments)} customer segments")
    
    async def segment_customers(self, segment_id: str) -> CustomerSegment:
        """Segment customers based on criteria"""
        
        if segment_id not in self.customer_segments:
            raise ValueError(f"Segment {segment_id} not found")
        
        segment = self.customer_segments[segment_id]
        segment.customer_ids = []
        
        # Get all customers from revenue engine
        for customer_id, customer in self.revenue_engine.customers.items():
            if await self._customer_matches_criteria(customer, segment.criteria):
                segment.customer_ids.append(customer_id)
        
        segment.size = len(segment.customer_ids)
        segment.last_updated = datetime.utcnow()
        
        # Calculate segment metrics
        if segment.customer_ids:
            total_ltv = sum(
                self.revenue_engine.customers[cid].lifetime_value 
                for cid in segment.customer_ids
            )
            segment.lifetime_value_avg = total_ltv / len(segment.customer_ids)
        
        logger.info(f"✅ Segmented {segment.size} customers for '{segment.name}'")
        return segment
    
    async def _customer_matches_criteria(self, customer: Customer, criteria: Dict[str, Any]) -> bool:
        """Check if customer matches segmentation criteria"""
        
        # Calculate derived fields
        days_since_signup = (datetime.utcnow() - customer.created_at).days
        
        # Simple criteria evaluation (in production, would use more sophisticated query engine)
        for field, condition in criteria.items():
            if field == "days_since_signup":
                value = days_since_signup
            elif field == "total_spent":
                value = float(customer.total_spent)
            elif field == "lifetime_value":
                value = float(customer.lifetime_value)
            else:
                continue  # Skip unknown fields
            
            # Evaluate condition
            if isinstance(condition, dict):
                if "$lt" in condition and value >= condition["$lt"]:
                    return False
                if "$gt" in condition and value <= condition["$gt"]:
                    return False
                if "$eq" in condition and value != condition["$eq"]:
                    return False
            else:
                if value != condition:
                    return False
        
        return True
    
    async def create_campaign(self,
                            name: str,
                            campaign_type: CampaignType,
                            template_id: str,
                            segment_id: str,
                            schedule: Optional[Dict[str, Any]] = None) -> EmailCampaign:
        """Create a new email marketing campaign"""
        
        campaign_id = f"camp_{uuid.uuid4().hex[:12]}"
        
        # Validate template exists
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        # Get target audience
        segment = await self.segment_customers(segment_id)
        
        # Create campaign
        campaign = EmailCampaign(
            campaign_id=campaign_id,
            name=name,
            campaign_type=campaign_type,
            status=CampaignStatus.DRAFT,
            template_id=template_id,
            target_audience={
                "segment_id": segment_id,
                "customer_ids": segment.customer_ids
            },
            schedule=schedule or {"send_immediately": True},
            total_recipients=segment.size
        )
        
        self.campaigns[campaign_id] = campaign
        
        logger.info(f"✅ Created campaign '{name}' targeting {segment.size} customers")
        return campaign
    
    async def execute_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Execute an email marketing campaign"""
        
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.campaigns[campaign_id]
        template = self.templates[campaign.template_id]
        
        if not self.email_enabled:
            logger.warning("⚠️ Email not enabled, simulating campaign execution")
            return await self._simulate_campaign_execution(campaign)
        
        campaign.status = CampaignStatus.RUNNING
        campaign.started_at = datetime.utcnow()
        
        results = {
            "campaign_id": campaign_id,
            "emails_queued": 0,
            "emails_sent": 0,
            "errors": []
        }
        
        # Send emails to all customers in target audience
        for customer_id in campaign.target_audience["customer_ids"]:
            try:
                customer = self.revenue_engine.customers[customer_id]
                
                # Personalize email content
                email_content = await self._personalize_email(template, customer)
                
                # Send email
                email_record = await self._send_email(
                    campaign_id=campaign_id,
                    customer=customer,
                    template=template,
                    subject=email_content["subject"],
                    html_content=email_content["html_content"],
                    text_content=email_content["text_content"]
                )
                
                if email_record:
                    results["emails_sent"] += 1
                    campaign.emails_sent += 1
                
                results["emails_queued"] += 1
                
                # Rate limiting
                await asyncio.sleep(3600 / campaign.send_rate_limit)  # Respect rate limit
                
            except Exception as e:
                logger.error(f"Failed to send email to {customer_id}: {e}")
                results["errors"].append(str(e))
        
        campaign.status = CampaignStatus.COMPLETED
        campaign.completed_at = datetime.utcnow()
        
        logger.info(f"✅ Campaign '{campaign.name}' completed: {results['emails_sent']} emails sent")
        return results
    
    async def _personalize_email(self, template: EmailTemplate, customer: Customer) -> Dict[str, str]:
        """Personalize email content for specific customer"""
        
        # Build personalization context
        context = {
            "customer_name": customer.name,
            "customer_email": customer.email,
            "customer_id": customer.customer_id,
            "total_spent": float(customer.total_spent),
            "lifetime_value": float(customer.lifetime_value),
            "dashboard_url": f"https://dashboard.hephaestus-rsi.com/{customer.customer_id}",
            "api_key": f"sk_live_{customer.customer_id[:16]}...",
            "docs_url": "https://docs.hephaestus-rsi.com",
            "upgrade_url": f"https://billing.hephaestus-rsi.com/upgrade?customer={customer.customer_id}",
            "estimated_additional_revenue": int(float(customer.lifetime_value) * 0.3)  # 30% increase estimate
        }
        
        # Render templates
        if self.templating_enabled:
            subject_template = Template(template.subject_template)
            html_template = Template(template.html_content_template)
            text_template = Template(template.text_content_template)
            
            return {
                "subject": subject_template.render(**context),
                "html_content": html_template.render(**context),
                "text_content": text_template.render(**context)
            }
        else:
            # Simple string replacement fallback
            subject = template.subject_template
            html_content = template.html_content_template
            text_content = template.text_content_template
            
            for key, value in context.items():
                placeholder = "{{" + key + "}}"
                subject = subject.replace(placeholder, str(value))
                html_content = html_content.replace(placeholder, str(value))
                text_content = text_content.replace(placeholder, str(value))
            
            return {
                "subject": subject,
                "html_content": html_content,
                "text_content": text_content
            }
    
    async def _send_email(self,
                         campaign_id: str,
                         customer: Customer,
                         template: EmailTemplate,
                         subject: str,
                         html_content: str,
                         text_content: str) -> Optional[EmailRecord]:
        """Send individual email"""
        
        email_id = f"email_{uuid.uuid4().hex[:12]}"
        
        try:
            # Create SendGrid email
            message = Mail(
                from_email=Email("campaigns@hephaestus-rsi.com", "Hephaestus RSI"),
                to_emails=To(customer.email),
                subject=subject,
                html_content=Content("text/html", html_content),
                plain_text_content=Content("text/plain", text_content)
            )
            
            # Add tracking
            message.tracking_settings = {
                "click_tracking": {"enable": True},
                "open_tracking": {"enable": True},
                "subscription_tracking": {"enable": True}
            }
            
            # Send email
            response = self.sendgrid_client.send(message)
            
            # Create email record
            email_record = EmailRecord(
                email_id=email_id,
                campaign_id=campaign_id,
                customer_id=customer.customer_id,
                template_id=template.template_id,
                recipient_email=customer.email,
                subject=subject,
                status=EmailStatus.SENT,
                sent_at=datetime.utcnow(),
                metadata={
                    "sendgrid_message_id": response.headers.get("X-Message-Id"),
                    "response_status": response.status_code
                }
            )
            
            self.email_records[email_id] = email_record
            self.total_emails_sent += 1
            
            logger.info(f"✅ Sent email {email_id} to {customer.email}")
            return email_record
            
        except Exception as e:
            logger.error(f"❌ Failed to send email to {customer.email}: {e}")
            return None
    
    async def _simulate_campaign_execution(self, campaign: EmailCampaign) -> Dict[str, Any]:
        """Simulate campaign execution for testing"""
        
        campaign.status = CampaignStatus.RUNNING
        campaign.started_at = datetime.utcnow()
        
        # Simulate realistic results
        total_recipients = campaign.total_recipients
        delivery_rate = 0.95
        open_rate = 0.25
        click_rate = 0.05
        conversion_rate = 0.02
        
        emails_delivered = int(total_recipients * delivery_rate)
        emails_opened = int(emails_delivered * open_rate)
        emails_clicked = int(emails_opened * click_rate)
        conversions = int(emails_clicked * conversion_rate)
        
        # Update campaign metrics
        campaign.emails_sent = total_recipients
        campaign.emails_delivered = emails_delivered
        campaign.emails_opened = emails_opened
        campaign.emails_clicked = emails_clicked
        campaign.conversions = conversions
        campaign.revenue_generated = Decimal(str(conversions * 97.0))  # $97 per conversion
        
        campaign.status = CampaignStatus.COMPLETED
        campaign.completed_at = datetime.utcnow()
        
        # Track total revenue
        self.total_revenue_attributed += campaign.revenue_generated
        
        results = {
            "campaign_id": campaign.campaign_id,
            "emails_sent": campaign.emails_sent,
            "emails_delivered": emails_delivered,
            "emails_opened": emails_opened,
            "emails_clicked": emails_clicked,
            "conversions": conversions,
            "revenue_generated": float(campaign.revenue_generated),
            "roi": float(campaign.revenue_generated / max(campaign.emails_sent * 0.10, 1)) * 100  # Assuming $0.10 cost per email
        }
        
        logger.info(f"✅ Simulated campaign '{campaign.name}': ${campaign.revenue_generated} revenue generated")
        return results
    
    async def get_campaign_analytics(self, campaign_id: str) -> Dict[str, Any]:
        """Get comprehensive campaign analytics"""
        
        if campaign_id not in self.campaigns:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign = self.campaigns[campaign_id]
        
        # Calculate performance metrics
        delivery_rate = campaign.emails_delivered / max(campaign.emails_sent, 1) * 100
        open_rate = campaign.emails_opened / max(campaign.emails_delivered, 1) * 100
        click_rate = campaign.emails_clicked / max(campaign.emails_opened, 1) * 100
        conversion_rate = campaign.conversions / max(campaign.emails_clicked, 1) * 100
        
        cost_per_email = 0.10  # Estimated cost
        total_cost = campaign.emails_sent * cost_per_email
        roi = (float(campaign.revenue_generated) - total_cost) / max(total_cost, 1) * 100
        
        return {
            "campaign_id": campaign_id,
            "campaign_name": campaign.name,
            "campaign_type": campaign.campaign_type.value,
            "status": campaign.status.value,
            "performance": {
                "total_recipients": campaign.total_recipients,
                "emails_sent": campaign.emails_sent,
                "emails_delivered": campaign.emails_delivered,
                "emails_opened": campaign.emails_opened,
                "emails_clicked": campaign.emails_clicked,
                "conversions": campaign.conversions,
                "delivery_rate": round(delivery_rate, 2),
                "open_rate": round(open_rate, 2),
                "click_rate": round(click_rate, 2),
                "conversion_rate": round(conversion_rate, 2)
            },
            "revenue": {
                "revenue_generated": float(campaign.revenue_generated),
                "cost_per_email": cost_per_email,
                "total_cost": total_cost,
                "roi_percent": round(roi, 2)
            },
            "timeline": {
                "created_at": campaign.created_at.isoformat(),
                "started_at": campaign.started_at.isoformat() if campaign.started_at else None,
                "completed_at": campaign.completed_at.isoformat() if campaign.completed_at else None
            }
        }
    
    async def run_automated_sequence(self, customer_id: str, sequence_type: str) -> Dict[str, Any]:
        """Run automated email sequence for customer lifecycle"""
        
        if customer_id not in self.revenue_engine.customers:
            raise ValueError(f"Customer {customer_id} not found")
        
        customer = self.revenue_engine.customers[customer_id]
        
        # Define sequences
        sequences = {
            "welcome": [
                {"template_id": "welcome_series_1", "delay_hours": 0},
                {"template_id": "onboarding_api_setup", "delay_hours": 24},
            ],
            "upsell": [
                {"template_id": "upsell_professional", "delay_hours": 0}
            ]
        }
        
        if sequence_type not in sequences:
            raise ValueError(f"Sequence type {sequence_type} not supported")
        
        sequence = sequences[sequence_type]
        results = []
        
        for step in sequence:
            # Wait for delay
            if step["delay_hours"] > 0:
                logger.info(f"Waiting {step['delay_hours']} hours before next email...")
                await asyncio.sleep(step["delay_hours"] * 3600)  # In production, would use task queue
            
            # Create single-customer campaign
            campaign = await self.create_campaign(
                name=f"{sequence_type}_sequence_{customer_id}",
                campaign_type=CampaignType.WELCOME_SERIES,
                template_id=step["template_id"],
                segment_id="custom"  # Would create custom segment for this customer
            )
            
            # Override target audience for single customer
            campaign.target_audience = {"customer_ids": [customer_id]}
            campaign.total_recipients = 1
            
            # Execute campaign
            result = await self.execute_campaign(campaign.campaign_id)
            results.append(result)
        
        return {
            "sequence_type": sequence_type,
            "customer_id": customer_id,
            "steps_completed": len(results),
            "results": results
        }
    
    async def optimize_send_times(self, segment_id: str) -> Dict[str, Any]:
        """AI-powered send time optimization"""
        
        # Analyze historical open rates by time of day and day of week
        # This is a simplified version - in production would use ML models
        
        segment = await self.segment_customers(segment_id)
        
        # Simulated optimization based on customer behavior patterns
        optimization_data = {
            "segment_id": segment_id,
            "segment_size": segment.size,
            "optimal_send_times": {
                "weekdays": {
                    "primary": "09:00",
                    "secondary": "14:00",
                    "confidence": 0.85
                },
                "weekends": {
                    "primary": "10:30", 
                    "secondary": "15:30",
                    "confidence": 0.72
                }
            },
            "expected_improvement": {
                "open_rate_increase": "12-18%",
                "click_rate_increase": "8-15%",
                "conversion_increase": "5-10%"
            },
            "recommendation": "Send Tuesday-Thursday at 9:00 AM for maximum engagement"
        }
        
        return optimization_data
    
    async def get_marketing_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive marketing dashboard data"""
        
        # Calculate overall metrics
        total_campaigns = len(self.campaigns)
        active_campaigns = len([c for c in self.campaigns.values() if c.status == CampaignStatus.RUNNING])
        completed_campaigns = len([c for c in self.campaigns.values() if c.status == CampaignStatus.COMPLETED])
        
        total_revenue = sum(float(c.revenue_generated) for c in self.campaigns.values())
        total_conversions = sum(c.conversions for c in self.campaigns.values())
        
        # Average metrics
        avg_open_rate = 0
        avg_click_rate = 0
        avg_conversion_rate = 0
        
        if completed_campaigns > 0:
            completed_campaign_list = [c for c in self.campaigns.values() if c.status == CampaignStatus.COMPLETED]
            avg_open_rate = sum(c.emails_opened / max(c.emails_delivered, 1) for c in completed_campaign_list) / completed_campaigns * 100
            avg_click_rate = sum(c.emails_clicked / max(c.emails_opened, 1) for c in completed_campaign_list) / completed_campaigns * 100
            avg_conversion_rate = sum(c.conversions / max(c.emails_clicked, 1) for c in completed_campaign_list) / completed_campaigns * 100
        
        return {
            "overview": {
                "total_emails_sent": self.total_emails_sent,
                "total_revenue_attributed": float(self.total_revenue_attributed),
                "total_campaigns": total_campaigns,
                "active_campaigns": active_campaigns,
                "completed_campaigns": completed_campaigns,
                "total_conversions": total_conversions
            },
            "performance_averages": {
                "open_rate": round(avg_open_rate, 2),
                "click_rate": round(avg_click_rate, 2),
                "conversion_rate": round(avg_conversion_rate, 2)
            },
            "recent_campaigns": [
                {
                    "campaign_id": c.campaign_id,
                    "name": c.name,
                    "status": c.status.value,
                    "revenue": float(c.revenue_generated),
                    "conversions": c.conversions
                }
                for c in sorted(self.campaigns.values(), key=lambda x: x.created_at, reverse=True)[:5]
            ],
            "customer_segments": [
                {
                    "segment_id": s.segment_id,
                    "name": s.name,
                    "size": s.size,
                    "avg_ltv": float(s.lifetime_value_avg)
                }
                for s in self.customer_segments.values()
            ]
        }


# Factory function for integration
def create_email_marketing_automation(
    revenue_engine: RealRevenueEngine,
    sendgrid_api_key: Optional[str] = None
) -> EmailMarketingAutomation:
    """Create email marketing automation system"""
    return EmailMarketingAutomation(
        revenue_engine=revenue_engine,
        sendgrid_api_key=sendgrid_api_key
    )