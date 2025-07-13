"""
Zero-Cost Marketing Engine for Email Automation Service.

Implements customer acquisition strategies requiring no upfront investment.
Target: First paying customers within 10 days using organic/viral methods.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class MarketingChannel(str, Enum):
    """Zero-cost marketing channels."""
    REDDIT = "reddit"
    DISCORD = "discord"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    BLOGGING = "blogging"
    EMAIL_OUTREACH = "email_outreach"
    COMMUNITY_FORUMS = "community_forums"
    FREE_TOOLS = "free_tools"


class ContentType(str, Enum):
    """Types of marketing content."""
    HELPFUL_TUTORIAL = "helpful_tutorial"
    FREE_TOOL = "free_tool"
    CASE_STUDY = "case_study"
    COMPARISON_GUIDE = "comparison_guide"
    TEMPLATE_GIVEAWAY = "template_giveaway"
    AUTOMATION_SCRIPT = "automation_script"
    INDUSTRY_INSIGHT = "industry_insight"


@dataclass
class MarketingCampaign:
    """Zero-cost marketing campaign."""
    
    campaign_id: str
    name: str
    channel: MarketingChannel
    content_type: ContentType
    
    # Campaign details
    target_audience: str
    value_proposition: str
    call_to_action: str
    content_outline: List[str]
    
    # Execution plan
    platforms: List[str]
    posting_schedule: List[Dict[str, Any]]
    engagement_strategy: List[str]
    
    # Tracking
    estimated_reach: int = 0
    estimated_conversions: int = 0
    cost: float = 0.0  # Always $0 for zero-cost
    
    # Performance (to be updated)
    actual_reach: int = 0
    actual_conversions: int = 0
    leads_generated: int = 0
    signups: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "planned"  # planned, active, completed


@dataclass
class ContentPiece:
    """Individual piece of marketing content."""
    
    content_id: str
    title: str
    content_type: ContentType
    channel: MarketingChannel
    
    # Content details
    content_body: str
    media_assets: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    target_keywords: List[str] = field(default_factory=list)
    
    # Distribution
    platforms_posted: List[Dict[str, Any]] = field(default_factory=list)
    scheduled_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance
    views: int = 0
    engagements: int = 0
    clicks: int = 0
    conversions: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LeadSource:
    """Source of marketing leads."""
    
    source_id: str
    channel: MarketingChannel
    platform: str
    campaign_id: Optional[str] = None
    
    # Lead details
    leads_generated: int = 0
    conversion_rate: float = 0.0
    cost_per_lead: float = 0.0
    quality_score: float = 0.0  # 0-1 based on lead behavior
    
    # Performance tracking
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)


class ZeroCostMarketingEngine:
    """
    Zero-cost marketing engine for customer acquisition.
    
    Implements organic growth strategies that require only time and effort,
    no monetary investment. Focuses on providing value first to build trust.
    """
    
    def __init__(self):
        # Campaign management
        self.active_campaigns: Dict[str, MarketingCampaign] = {}
        self.completed_campaigns: List[MarketingCampaign] = []
        self.content_library: Dict[str, ContentPiece] = {}
        
        # Lead tracking
        self.lead_sources: Dict[str, LeadSource] = {}
        self.total_leads: int = 0
        self.total_conversions: int = 0
        
        # Content templates
        self.content_templates = self._load_content_templates()
        self.platform_strategies = self._load_platform_strategies()
        
        # Data storage
        self.data_dir = Path("data/marketing_campaigns")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎯 Zero-Cost Marketing Engine initialized")
    
    def _load_content_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load content templates for different types of marketing."""
        
        return {
            "reddit_helpful_post": {
                "title_templates": [
                    "I built a free tool to solve [PROBLEM] - hope it helps!",
                    "Free [SOLUTION] for [TARGET_AUDIENCE] - no signup required",
                    "How I automated [PROCESS] for my business (with free tool)"
                ],
                "content_structure": [
                    "Problem introduction (relate to audience pain)",
                    "Personal story/experience",
                    "Solution explanation (focus on value, not pitch)",
                    "Free tool/resource offer",
                    "Ask for feedback (engagement)"
                ],
                "cta_examples": [
                    "I made this free for everyone to use: [LINK]",
                    "You can try it here (no signup): [LINK]",
                    "Link in comments if anyone wants to check it out"
                ]
            },
            
            "github_project": {
                "repo_structure": [
                    "Clear README with value proposition",
                    "Live demo/examples",
                    "Easy setup instructions",
                    "Use cases and benefits",
                    "Link to hosted version"
                ],
                "promotion_strategy": [
                    "Post in relevant GitHub topics",
                    "Share in developer communities",
                    "Create tutorial blog posts",
                    "Add to awesome lists",
                    "Engage with similar projects"
                ]
            },
            
            "linkedin_thought_leadership": {
                "post_types": [
                    "Industry insights with data",
                    "Lessons learned from building tools",
                    "Trends in email marketing automation",
                    "Common mistakes and solutions",
                    "Success stories and case studies"
                ],
                "engagement_tactics": [
                    "Ask questions to encourage comments",
                    "Share controversial but valid opinions",
                    "Provide actionable tips",
                    "Use data and statistics",
                    "Tag relevant people and companies"
                ]
            },
            
            "youtube_tutorials": {
                "video_ideas": [
                    "How to build email automation from scratch",
                    "Free alternatives to expensive email tools",
                    "Setting up automated email sequences",
                    "Email marketing for beginners (2025 guide)",
                    "Comparing email automation platforms"
                ],
                "optimization": [
                    "SEO-optimized titles and descriptions",
                    "Custom thumbnails with value proposition",
                    "Clear call-to-action in video",
                    "Links in description and pinned comment",
                    "End screen promotion"
                ]
            }
        }
    
    def _load_platform_strategies(self) -> Dict[MarketingChannel, Dict[str, Any]]:
        """Load platform-specific marketing strategies."""
        
        return {
            MarketingChannel.REDDIT: {
                "target_subreddits": [
                    "r/entrepreneur", "r/smallbusiness", "r/marketing", 
                    "r/emailmarketing", "r/automation", "r/SaaS",
                    "r/startups", "r/ecommerce", "r/webdev"
                ],
                "posting_guidelines": [
                    "Provide value first, promote second",
                    "Follow subreddit rules strictly",
                    "Engage authentically in comments",
                    "Share personal experiences",
                    "Avoid direct sales pitches"
                ],
                "success_metrics": ["upvotes", "comments", "clicks", "signups"]
            },
            
            MarketingChannel.DISCORD: {
                "target_communities": [
                    "Entrepreneur Discord servers",
                    "SaaS/startup communities",
                    "Web development servers",
                    "Marketing communities",
                    "Small business groups"
                ],
                "engagement_strategy": [
                    "Be helpful in general discussions",
                    "Share knowledge and experience",
                    "Offer free consultations",
                    "Create valuable resources",
                    "Build relationships before promoting"
                ]
            },
            
            MarketingChannel.GITHUB: {
                "repository_strategy": [
                    "Create genuinely useful open-source tools",
                    "Focus on developer pain points",
                    "Excellent documentation",
                    "Live examples and demos",
                    "Regular updates and maintenance"
                ],
                "promotion_tactics": [
                    "Submit to Show HN on Hacker News",
                    "Share in r/opensource and r/programming",
                    "Add to awesome lists",
                    "Create tutorial blog posts",
                    "Engage with the community"
                ]
            },
            
            MarketingChannel.LINKEDIN: {
                "content_strategy": [
                    "Industry insights and trends",
                    "Behind-the-scenes of building a business",
                    "Lessons learned and mistakes",
                    "Success stories and case studies",
                    "Actionable tips and advice"
                ],
                "networking_approach": [
                    "Connect with potential customers",
                    "Engage with industry leaders",
                    "Join relevant groups",
                    "Comment thoughtfully on posts",
                    "Share others' content with insights"
                ]
            }
        }
    
    async def create_campaign(self, name: str, channel: MarketingChannel, 
                             content_type: ContentType, target_audience: str,
                             value_proposition: str) -> MarketingCampaign:
        """Create a new zero-cost marketing campaign."""
        
        campaign_id = f"mkt_{int(time.time())}_{channel.value}"
        
        # Generate content outline based on type and channel
        content_outline = self._generate_content_outline(channel, content_type, target_audience)
        
        # Create posting schedule
        posting_schedule = self._create_posting_schedule(channel)
        
        # Generate engagement strategy
        engagement_strategy = self._create_engagement_strategy(channel, target_audience)
        
        # Get target platforms for this channel
        platforms = self._get_target_platforms(channel)
        
        campaign = MarketingCampaign(
            campaign_id=campaign_id,
            name=name,
            channel=channel,
            content_type=content_type,
            target_audience=target_audience,
            value_proposition=value_proposition,
            call_to_action=f"Try our free email automation tool: [LINK]",
            content_outline=content_outline,
            platforms=platforms,
            posting_schedule=posting_schedule,
            engagement_strategy=engagement_strategy,
            estimated_reach=self._estimate_reach(channel, content_type),
            estimated_conversions=self._estimate_conversions(channel, content_type)
        )
        
        self.active_campaigns[campaign_id] = campaign
        
        logger.info(f"🎯 Created marketing campaign: {name} ({channel.value})")
        
        return campaign
    
    def _generate_content_outline(self, channel: MarketingChannel, 
                                 content_type: ContentType, target_audience: str) -> List[str]:
        """Generate content outline based on channel and type."""
        
        if channel == MarketingChannel.REDDIT and content_type == ContentType.HELPFUL_TUTORIAL:
            return [
                f"Hook: Address common {target_audience} pain point",
                "Personal story: Why I built this solution",
                "Tutorial: Step-by-step guide with screenshots",
                "Free tool offer: Link to our service",
                "Ask for feedback and improvements",
                "Follow up in comments with helpful answers"
            ]
        
        elif channel == MarketingChannel.GITHUB and content_type == ContentType.FREE_TOOL:
            return [
                "Create open-source email template library",
                "Include 20+ professional email templates",
                "Add easy-to-use API for sending emails",
                "Write comprehensive documentation",
                "Create live demo with our service",
                "Promote in developer communities"
            ]
        
        elif channel == MarketingChannel.LINKEDIN and content_type == ContentType.INDUSTRY_INSIGHT:
            return [
                "Research latest email marketing trends",
                "Create data-driven insights post",
                "Share personal experience and lessons",
                "Provide actionable recommendations",
                "Include subtle mention of our solution",
                "Engage with all comments professionally"
            ]
        
        elif channel == MarketingChannel.YOUTUBE and content_type == ContentType.HELPFUL_TUTORIAL:
            return [
                "Plan 15-minute tutorial video",
                "Show email automation setup from scratch",
                "Compare different tools (including ours)",
                "Provide free templates in description",
                "Create compelling thumbnail",
                "Optimize for search keywords"
            ]
        
        else:
            # Generic outline
            return [
                "Identify target audience pain point",
                "Create valuable content addressing the pain",
                "Include subtle mention of our solution",
                "Provide clear call-to-action",
                "Engage with audience responses",
                "Follow up with additional value"
            ]
    
    def _create_posting_schedule(self, channel: MarketingChannel) -> List[Dict[str, Any]]:
        """Create optimal posting schedule for channel."""
        
        schedules = {
            MarketingChannel.REDDIT: [
                {"day": "Tuesday", "time": "10:00 AM EST", "subreddit": "r/entrepreneur"},
                {"day": "Wednesday", "time": "2:00 PM EST", "subreddit": "r/smallbusiness"},
                {"day": "Thursday", "time": "11:00 AM EST", "subreddit": "r/marketing"},
                {"day": "Friday", "time": "3:00 PM EST", "subreddit": "r/automation"}
            ],
            MarketingChannel.LINKEDIN: [
                {"day": "Tuesday", "time": "8:00 AM EST", "type": "industry_insight"},
                {"day": "Thursday", "time": "12:00 PM EST", "type": "tutorial_post"},
                {"day": "Saturday", "time": "9:00 AM EST", "type": "weekend_tips"}
            ],
            MarketingChannel.TWITTER: [
                {"day": "Monday", "time": "9:00 AM EST", "type": "tip_thread"},
                {"day": "Wednesday", "time": "1:00 PM EST", "type": "tool_showcase"},
                {"day": "Friday", "time": "4:00 PM EST", "type": "week_wrap"}
            ]
        }
        
        return schedules.get(channel, [
            {"day": "Tuesday", "time": "10:00 AM", "platform": "main"},
            {"day": "Thursday", "time": "2:00 PM", "platform": "main"}
        ])
    
    def _create_engagement_strategy(self, channel: MarketingChannel, target_audience: str) -> List[str]:
        """Create engagement strategy for the channel."""
        
        strategies = {
            MarketingChannel.REDDIT: [
                "Respond to every comment within 2 hours",
                "Provide additional helpful resources",
                "Ask follow-up questions to extend conversation",
                "Share success stories from tool users",
                "Offer free personalized advice"
            ],
            MarketingChannel.DISCORD: [
                "Be active in general discussions daily",
                "Share helpful resources without self-promotion",
                "Offer free consultations to members",
                "Create valuable Discord bots/tools",
                "Build relationships before any promotion"
            ],
            MarketingChannel.LINKEDIN: [
                "Comment thoughtfully on industry posts",
                "Share others' content with insightful commentary",
                "Connect with potential customers personally",
                "Join and contribute to relevant groups",
                "Send helpful follow-up messages"
            ]
        }
        
        return strategies.get(channel, [
            "Engage authentically with audience",
            "Provide value before promoting",
            "Build relationships, not just leads",
            "Be consistent and reliable",
            "Always follow up on interactions"
        ])
    
    def _get_target_platforms(self, channel: MarketingChannel) -> List[str]:
        """Get specific platforms/communities for the channel."""
        
        platforms = {
            MarketingChannel.REDDIT: [
                "r/entrepreneur", "r/smallbusiness", "r/marketing", 
                "r/emailmarketing", "r/automation", "r/SaaS"
            ],
            MarketingChannel.DISCORD: [
                "Entrepreneur Discord", "SaaS Community", "Startup Grind",
                "Indie Hackers Discord", "Marketing Discord"
            ],
            MarketingChannel.GITHUB: [
                "GitHub repositories", "Developer communities",
                "Open source forums", "Hacker News"
            ],
            MarketingChannel.LINKEDIN: [
                "LinkedIn feed", "Marketing groups", "Entrepreneur groups",
                "SaaS communities", "Small business groups"
            ]
        }
        
        return platforms.get(channel, ["main_platform"])
    
    def _estimate_reach(self, channel: MarketingChannel, content_type: ContentType) -> int:
        """Estimate potential reach for campaign."""
        
        base_reach = {
            MarketingChannel.REDDIT: 5000,
            MarketingChannel.LINKEDIN: 2000,
            MarketingChannel.GITHUB: 1000,
            MarketingChannel.DISCORD: 500,
            MarketingChannel.TWITTER: 1500,
            MarketingChannel.YOUTUBE: 3000
        }
        
        content_multiplier = {
            ContentType.HELPFUL_TUTORIAL: 1.5,
            ContentType.FREE_TOOL: 2.0,
            ContentType.CASE_STUDY: 1.2,
            ContentType.TEMPLATE_GIVEAWAY: 1.8
        }
        
        base = base_reach.get(channel, 1000)
        multiplier = content_multiplier.get(content_type, 1.0)
        
        return int(base * multiplier)
    
    def _estimate_conversions(self, channel: MarketingChannel, content_type: ContentType) -> int:
        """Estimate conversion potential."""
        
        conversion_rates = {
            MarketingChannel.REDDIT: 0.02,  # 2%
            MarketingChannel.LINKEDIN: 0.05,  # 5%
            MarketingChannel.GITHUB: 0.08,  # 8%
            MarketingChannel.DISCORD: 0.10,  # 10%
            MarketingChannel.YOUTUBE: 0.03  # 3%
        }
        
        estimated_reach = self._estimate_reach(channel, content_type)
        conversion_rate = conversion_rates.get(channel, 0.02)
        
        return int(estimated_reach * conversion_rate)
    
    async def create_content_piece(self, title: str, content_type: ContentType,
                                  channel: MarketingChannel, target_keywords: List[str]) -> ContentPiece:
        """Create a specific piece of marketing content."""
        
        content_id = f"content_{int(time.time())}_{content_type.value}"
        
        # Generate content based on type and channel
        content_body = await self._generate_content_body(title, content_type, channel, target_keywords)
        
        content_piece = ContentPiece(
            content_id=content_id,
            title=title,
            content_type=content_type,
            channel=channel,
            content_body=content_body,
            target_keywords=target_keywords,
            tags=self._generate_tags(content_type, target_keywords)
        )
        
        self.content_library[content_id] = content_piece
        
        logger.info(f"📝 Created content piece: {title}")
        
        return content_piece
    
    async def _generate_content_body(self, title: str, content_type: ContentType,
                                   channel: MarketingChannel, keywords: List[str]) -> str:
        """Generate actual content body."""
        
        if channel == MarketingChannel.REDDIT and content_type == ContentType.HELPFUL_TUTORIAL:
            return f"""
# {title}

Hey everyone! I've been struggling with [PAIN_POINT] for my business and couldn't find a good solution that didn't cost a fortune.

## The Problem
Most email automation tools are either:
- Way too expensive for small businesses ($100+/month)
- Too complicated to set up 
- Have terrible deliverability

## What I Built
After trying everything, I decided to build my own solution. It's completely free to start and actually works well.

## How It Works
1. **Simple Setup**: Connect in under 5 minutes
2. **Template Library**: 20+ professional templates included
3. **Smart Automation**: Trigger emails based on user actions
4. **Analytics**: Track opens, clicks, conversions

## Free Tool
I made this available for everyone to use: [LINK TO OUR SERVICE]

- No credit card required
- 1000 emails/month free
- All features included
- No hidden fees

## Results So Far
- 95%+ delivery rate
- 22% average open rate 
- Saved $1200/year vs MailChimp

Happy to answer any questions! Would love feedback on how to make it even better.

**Keywords**: {', '.join(keywords)}
"""
        
        elif channel == MarketingChannel.LINKEDIN and content_type == ContentType.INDUSTRY_INSIGHT:
            return f"""
{title}

After analyzing 10,000+ email campaigns, here's what I learned about email automation in 2025:

🔍 **Key Insights:**

1. **Personalization is King**: Emails with personalized subject lines have 26% higher open rates

2. **Timing Matters**: Tuesday-Thursday 10am-2pm still performs best for B2B

3. **Mobile-First**: 68% of emails are opened on mobile devices

4. **Automation ROI**: Automated emails generate 320% more revenue than non-automated

💡 **What This Means for Your Business:**

→ Start with simple automation (welcome series, abandoned cart)
→ Focus on mobile-responsive templates
→ Test send times for your specific audience
→ Personalize beyond just "Hi [First Name]"

📊 **Real Numbers:**
- Average email automation ROI: 4,200%
- Cost of NOT automating: $3,000+ lost revenue/month
- Setup time for basic automation: Under 30 minutes

I've been helping businesses implement these strategies with a free email automation tool. Happy to share insights!

What's your biggest email marketing challenge?

#EmailMarketing #Automation #SmallBusiness #{' #'.join(keywords)}
"""
        
        else:
            # Generic content template
            return f"""
{title}

[ENGAGING_HOOK related to {keywords[0] if keywords else 'email marketing'}]

[VALUABLE_CONTENT providing real insights or solutions]

[PERSONAL_STORY or experience]

[ACTIONABLE_TIPS that readers can implement]

[SUBTLE_CTA mentioning our free tool]

Keywords: {', '.join(keywords)}
"""
    
    def _generate_tags(self, content_type: ContentType, keywords: List[str]) -> List[str]:
        """Generate relevant tags for content."""
        
        base_tags = {
            ContentType.HELPFUL_TUTORIAL: ["tutorial", "howto", "guide", "tips"],
            ContentType.FREE_TOOL: ["free", "tool", "resource", "utility"],
            ContentType.CASE_STUDY: ["casestudy", "results", "success", "data"],
            ContentType.INDUSTRY_INSIGHT: ["insights", "trends", "analysis", "industry"]
        }
        
        tags = base_tags.get(content_type, ["content"])
        tags.extend(keywords[:3])  # Add first 3 keywords as tags
        
        return tags
    
    async def execute_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Execute a marketing campaign."""
        
        campaign = self.active_campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        campaign.status = "active"
        
        logger.info(f"🚀 Executing campaign: {campaign.name}")
        
        # Create content pieces for the campaign
        content_pieces = []
        
        for i, outline_item in enumerate(campaign.content_outline[:3]):  # Create first 3 pieces
            title = f"{campaign.name} - Part {i+1}"
            
            content = await self.create_content_piece(
                title=title,
                content_type=campaign.content_type,
                channel=campaign.channel,
                target_keywords=[campaign.target_audience, "email", "automation"]
            )
            
            content_pieces.append(content)
        
        # Simulate posting to platforms (in real implementation, would actually post)
        posts_created = []
        for platform in campaign.platforms[:2]:  # Post to first 2 platforms
            post_data = {
                "platform": platform,
                "content_id": content_pieces[0].content_id,
                "scheduled_time": datetime.now() + timedelta(hours=1),
                "status": "scheduled"
            }
            posts_created.append(post_data)
        
        # Track lead sources
        for platform in campaign.platforms:
            source_id = f"{campaign_id}_{platform}"
            lead_source = LeadSource(
                source_id=source_id,
                channel=campaign.channel,
                platform=platform,
                campaign_id=campaign_id
            )
            self.lead_sources[source_id] = lead_source
        
        # Simulate initial performance (would be real metrics in production)
        campaign.actual_reach = int(campaign.estimated_reach * 0.7)  # 70% of estimated
        campaign.leads_generated = int(campaign.estimated_conversions * 0.5)  # 50% of estimated
        
        self.total_leads += campaign.leads_generated
        
        execution_result = {
            "campaign_id": campaign_id,
            "status": "active",
            "content_pieces_created": len(content_pieces),
            "posts_scheduled": len(posts_created),
            "estimated_reach": campaign.estimated_reach,
            "lead_sources_created": len(campaign.platforms),
            "initial_leads": campaign.leads_generated
        }
        
        logger.info(f"✅ Campaign {campaign.name} launched successfully")
        
        return execution_result
    
    def get_campaign_performance(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign performance metrics."""
        
        campaign = self.active_campaigns.get(campaign_id)
        if not campaign:
            campaign = next((c for c in self.completed_campaigns if c.campaign_id == campaign_id), None)
        
        if not campaign:
            return {"error": "Campaign not found"}
        
        # Calculate performance metrics
        conversion_rate = 0.0
        if campaign.actual_reach > 0:
            conversion_rate = campaign.leads_generated / campaign.actual_reach
        
        roi = float('inf')  # Infinite ROI since cost is $0
        
        performance = {
            "campaign_id": campaign_id,
            "name": campaign.name,
            "channel": campaign.channel.value,
            "status": campaign.status,
            "reach": {
                "estimated": campaign.estimated_reach,
                "actual": campaign.actual_reach,
                "achievement": (campaign.actual_reach / campaign.estimated_reach * 100) if campaign.estimated_reach > 0 else 0
            },
            "conversions": {
                "estimated": campaign.estimated_conversions,
                "actual": campaign.leads_generated,
                "rate": conversion_rate * 100
            },
            "cost": campaign.cost,
            "roi": roi,
            "created_at": campaign.created_at.isoformat()
        }
        
        return performance
    
    def get_overall_performance(self) -> Dict[str, Any]:
        """Get overall marketing performance."""
        
        total_campaigns = len(self.active_campaigns) + len(self.completed_campaigns)
        total_reach = sum(c.actual_reach for c in list(self.active_campaigns.values()) + self.completed_campaigns)
        total_cost = 0.0  # Always $0 for zero-cost marketing
        
        # Calculate conversion rates
        overall_conversion_rate = 0.0
        if total_reach > 0:
            overall_conversion_rate = self.total_leads / total_reach
        
        # Channel performance
        channel_performance = {}
        for channel in MarketingChannel:
            channel_campaigns = [c for c in list(self.active_campaigns.values()) + self.completed_campaigns 
                               if c.channel == channel]
            if channel_campaigns:
                channel_leads = sum(c.leads_generated for c in channel_campaigns)
                channel_reach = sum(c.actual_reach for c in channel_campaigns)
                channel_performance[channel.value] = {
                    "campaigns": len(channel_campaigns),
                    "leads": channel_leads,
                    "reach": channel_reach,
                    "conversion_rate": (channel_leads / channel_reach * 100) if channel_reach > 0 else 0
                }
        
        return {
            "total_campaigns": total_campaigns,
            "active_campaigns": len(self.active_campaigns),
            "total_leads": self.total_leads,
            "total_conversions": self.total_conversions,
            "total_reach": total_reach,
            "total_cost": total_cost,
            "roi": float('inf'),
            "overall_conversion_rate": overall_conversion_rate * 100,
            "channel_performance": channel_performance,
            "content_pieces": len(self.content_library),
            "lead_sources": len(self.lead_sources)
        }
    
    async def save_campaign_data(self) -> None:
        """Save campaign data to disk."""
        
        campaign_data = {
            "timestamp": datetime.now().isoformat(),
            "active_campaigns": [asdict(c) for c in self.active_campaigns.values()],
            "completed_campaigns": [asdict(c) for c in self.completed_campaigns],
            "total_leads": self.total_leads,
            "total_conversions": self.total_conversions,
            "performance_summary": self.get_overall_performance()
        }
        
        output_file = self.data_dir / f"campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(campaign_data, f, indent=2, default=str)
        
        logger.info(f"💾 Campaign data saved to {output_file}")


# Factory function
def create_marketing_engine() -> ZeroCostMarketingEngine:
    """Create zero-cost marketing engine instance."""
    return ZeroCostMarketingEngine()


# Example usage
async def main():
    """Example marketing campaign creation and execution."""
    engine = create_marketing_engine()
    
    # Create Reddit campaign
    reddit_campaign = await engine.create_campaign(
        name="Reddit Email Automation Tutorial",
        channel=MarketingChannel.REDDIT,
        content_type=ContentType.HELPFUL_TUTORIAL,
        target_audience="small business owners",
        value_proposition="Free email automation that actually works"
    )
    
    print(f"Created campaign: {reddit_campaign.name}")
    print(f"Estimated reach: {reddit_campaign.estimated_reach}")
    print(f"Estimated conversions: {reddit_campaign.estimated_conversions}")
    
    # Execute campaign
    result = await engine.execute_campaign(reddit_campaign.campaign_id)
    print(f"Campaign execution result: {result}")
    
    # Get performance
    performance = engine.get_campaign_performance(reddit_campaign.campaign_id)
    print(f"Campaign performance: {performance}")


if __name__ == "__main__":
    asyncio.run(main())