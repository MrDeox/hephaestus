"""
Email Marketing Automation Service module.

Zero-cost bootstrap revenue generation through professional email automation.
Target: $700/month revenue in 10 days.
"""

from .email_service import (
    EmailAutomationService,
    EmailCampaign,
    EmailTemplate,
    AutomationRule,
    EmailAnalytics,
    SendResult,
    EmailServiceManager,
    CampaignStatus,
    PricingTier,
    AutomationType,
    create_email_automation_service,
    create_email_service_manager
)

__all__ = [
    "EmailAutomationService",
    "EmailCampaign",
    "EmailTemplate", 
    "AutomationRule",
    "EmailAnalytics",
    "SendResult",
    "EmailServiceManager",
    "CampaignStatus",
    "PricingTier",
    "AutomationType",
    "create_email_automation_service",
    "create_email_service_manager"
]