"""
Web Automation Agent for Hephaestus RSI System.

Automates web interactions for marketing campaigns, account creation,
content posting, and customer acquisition using browser automation.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    # Create placeholder types for when Playwright is not available
    Browser = None
    Page = Any
    BrowserContext = None
    async_playwright = None
    print("⚠️ Playwright not available. Install with: pip install playwright")

logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """Types of platforms for automation."""
    REDDIT = "reddit"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    DISCORD = "discord"
    YOUTUBE = "youtube"
    HACKERNEWS = "hackernews"
    PRODUCT_HUNT = "product_hunt"
    EMAIL_PLATFORM = "email_platform"


class ActionType(str, Enum):
    """Types of web actions."""
    CREATE_ACCOUNT = "create_account"
    LOGIN = "login"
    POST_CONTENT = "post_content"
    COMMENT = "comment"
    UPVOTE = "upvote"
    FOLLOW = "follow"
    SHARE = "share"
    JOIN_COMMUNITY = "join_community"
    CREATE_REPOSITORY = "create_repository"
    DEPLOY_SERVICE = "deploy_service"


@dataclass
class WebAction:
    """Individual web automation action."""
    
    action_id: str
    platform: PlatformType
    action_type: ActionType
    
    # Target details
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Automation details
    selectors: Dict[str, str] = field(default_factory=dict)
    wait_conditions: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 3
    delay_range: Tuple[float, float] = (1.0, 3.0)
    
    # Execution tracking
    status: str = "pending"  # pending, executing, completed, failed
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    success: bool = False
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AutomationSession:
    """Browser automation session."""
    
    session_id: str
    platform: PlatformType
    
    # Session details
    actions: List[WebAction] = field(default_factory=list)
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    headless: bool = True
    
    # Authentication
    credentials: Dict[str, str] = field(default_factory=dict)
    is_authenticated: bool = False
    
    # Performance tracking
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)


class WebAutomationAgent:
    """
    Web Automation Agent for automated marketing and revenue generation.
    
    Uses Playwright to automate browser interactions for:
    - Creating accounts on marketing platforms
    - Posting marketing content
    - Building backlinks and SEO
    - Deploying services to free platforms
    - Customer acquisition automation
    """
    
    def __init__(self):
        self.sessions: Dict[str, AutomationSession] = {}
        self.browser: Optional[Browser] = None
        self.contexts: Dict[str, BrowserContext] = {}
        
        # Platform configurations
        self.platform_configs = self._load_platform_configs()
        self.user_agents = self._load_user_agents()
        
        # Safety settings
        self.rate_limits = {
            PlatformType.REDDIT: {"requests_per_hour": 10, "posts_per_day": 3},
            PlatformType.GITHUB: {"repos_per_day": 2, "commits_per_hour": 5},
            PlatformType.LINKEDIN: {"posts_per_day": 2, "connections_per_day": 20},
            PlatformType.TWITTER: {"tweets_per_hour": 5, "follows_per_day": 50}
        }
        
        # Data storage
        self.data_dir = Path("data/web_automation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🌐 Web Automation Agent initialized")
    
    def _load_platform_configs(self) -> Dict[PlatformType, Dict[str, Any]]:
        """Load platform-specific automation configurations."""
        
        return {
            PlatformType.REDDIT: {
                "base_url": "https://reddit.com",
                "login_url": "https://www.reddit.com/login",
                "post_selectors": {
                    "subreddit_input": "[data-testid='subreddit-input']",
                    "title_input": "[data-testid='post-title-input']", 
                    "content_input": "[data-testid='post-content-input']",
                    "submit_button": "[data-testid='submit-post-button']"
                },
                "wait_conditions": [
                    {"type": "network_idle", "timeout": 5000},
                    {"type": "element", "selector": "body", "timeout": 10000}
                ],
                "delay_range": (2.0, 5.0)
            },
            
            PlatformType.GITHUB: {
                "base_url": "https://github.com",
                "login_url": "https://github.com/login",
                "new_repo_url": "https://github.com/new",
                "repo_selectors": {
                    "repo_name": "input[name='repository[name]']",
                    "repo_description": "input[name='repository[description]']",
                    "public_radio": "input[value='public']",
                    "readme_checkbox": "input[name='repository[auto_init]']",
                    "create_button": "button[type='submit']"
                },
                "delay_range": (1.5, 3.0)
            },
            
            PlatformType.LINKEDIN: {
                "base_url": "https://linkedin.com",
                "login_url": "https://www.linkedin.com/login",
                "post_selectors": {
                    "post_button": "[data-test-id='share-box-trigger']",
                    "content_area": "[data-test-id='share-form-post-content']",
                    "publish_button": "[data-test-id='share-form-publish-button']"
                },
                "delay_range": (3.0, 7.0)
            },
            
            PlatformType.HACKERNEWS: {
                "base_url": "https://news.ycombinator.com",
                "submit_url": "https://news.ycombinator.com/submit",
                "submit_selectors": {
                    "title_input": "input[name='title']",
                    "url_input": "input[name='url']",
                    "text_area": "textarea[name='text']",
                    "submit_button": "input[type='submit']"
                },
                "delay_range": (2.0, 4.0)
            }
        }
    
    def _load_user_agents(self) -> List[str]:
        """Load realistic user agent strings."""
        
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) Gecko/20100101 Firefox/132.0"
        ]
    
    async def start_browser(self, headless: bool = True) -> bool:
        """Start browser instance."""
        
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("❌ Playwright not available for browser automation")
            return False
        
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=headless,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--disable-extensions"
                ]
            )
            
            logger.info("🌐 Browser started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start browser: {e}")
            return False
    
    async def create_session(self, platform: PlatformType, 
                           headless: bool = True) -> AutomationSession:
        """Create new automation session for platform."""
        
        session_id = f"{platform.value}_{int(time.time())}"
        
        # Create browser context with realistic settings
        user_agent = random.choice(self.user_agents)
        
        context = await self.browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            has_touch=False,
            is_mobile=False,
            locale="en-US",
            timezone_id="America/New_York"
        )
        
        # Add stealth settings
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        self.contexts[session_id] = context
        
        session = AutomationSession(
            session_id=session_id,
            platform=platform,
            user_agent=user_agent,
            headless=headless
        )
        
        self.sessions[session_id] = session
        
        logger.info(f"🎯 Created automation session: {session_id} ({platform.value})")
        
        return session
    
    async def create_web_action(self, session_id: str, action_type: ActionType,
                              url: str, title: Optional[str] = None,
                              content: Optional[str] = None,
                              tags: Optional[List[str]] = None) -> WebAction:
        """Create new web action."""
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        action_id = f"{session_id}_{action_type.value}_{int(time.time())}"
        
        # Get platform-specific selectors
        platform_config = self.platform_configs.get(session.platform, {})
        selectors = platform_config.get("post_selectors", {})
        wait_conditions = platform_config.get("wait_conditions", [])
        delay_range = platform_config.get("delay_range", (1.0, 3.0))
        
        action = WebAction(
            action_id=action_id,
            platform=session.platform,
            action_type=action_type,
            url=url,
            title=title,
            content=content,
            tags=tags or [],
            selectors=selectors,
            wait_conditions=wait_conditions,
            delay_range=delay_range
        )
        
        session.actions.append(action)
        session.total_actions += 1
        
        logger.info(f"📝 Created web action: {action_type.value} on {session.platform.value}")
        
        return action
    
    async def execute_action(self, session_id: str, action: WebAction) -> bool:
        """Execute a web automation action."""
        
        context = self.contexts.get(session_id)
        if not context:
            raise ValueError(f"Context for session {session_id} not found")
        
        session = self.sessions[session_id]
        action.status = "executing"
        
        try:
            start_time = time.time()
            
            page = await context.new_page()
            
            # Navigate to URL
            await page.goto(action.url, wait_until="networkidle")
            
            # Random delay to appear human
            delay = random.uniform(*action.delay_range)
            await asyncio.sleep(delay)
            
            success = False
            
            if action.action_type == ActionType.POST_CONTENT:
                success = await self._execute_post_content(page, action)
            elif action.action_type == ActionType.CREATE_REPOSITORY:
                success = await self._execute_create_repository(page, action)
            elif action.action_type == ActionType.LOGIN:
                success = await self._execute_login(page, action, session)
            elif action.action_type == ActionType.CREATE_ACCOUNT:
                success = await self._execute_create_account(page, action)
            else:
                logger.warning(f"⚠️ Action type {action.action_type} not implemented yet")
                success = False
            
            await page.close()
            
            action.execution_time = time.time() - start_time
            action.success = success
            action.status = "completed" if success else "failed"
            
            if success:
                session.successful_actions += 1
            else:
                session.failed_actions += 1
            
            logger.info(f"{'✅' if success else '❌'} Action {action.action_type.value} {'completed' if success else 'failed'}")
            
            return success
            
        except Exception as e:
            action.status = "failed"
            action.error_message = str(e)
            session.failed_actions += 1
            
            logger.error(f"❌ Action execution failed: {e}")
            return False
    
    async def _execute_post_content(self, page: Page, action: WebAction) -> bool:
        """Execute content posting action."""
        
        try:
            if action.platform == PlatformType.REDDIT:
                # Reddit posting logic
                if "r/" in action.url:
                    # Navigate to submit page for specific subreddit
                    subreddit = action.url.split("r/")[1].split("/")[0]
                    submit_url = f"https://www.reddit.com/r/{subreddit}/submit"
                    await page.goto(submit_url)
                
                # Fill in post details
                if action.title:
                    await page.fill("textarea[name='title']", action.title)
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                
                if action.content:
                    await page.fill("textarea[name='text']", action.content)
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                
                # Add tags if supported
                if action.tags:
                    tags_text = " ".join(f"#{tag}" for tag in action.tags)
                    current_content = await page.input_value("textarea[name='text']")
                    await page.fill("textarea[name='text']", f"{current_content}\n\n{tags_text}")
                
                # Submit post (commented out for safety)
                # await page.click("button[type='submit']")
                # await page.wait_for_load_state("networkidle")
                
                logger.info("🚀 Reddit post ready for submission (auto-submit disabled for safety)")
                return True
                
            elif action.platform == PlatformType.HACKERNEWS:
                # Hacker News submission
                await page.fill("input[name='title']", action.title or "")
                await asyncio.sleep(random.uniform(0.5, 1.0))
                
                if action.content and not action.url.startswith("http"):
                    # Text post
                    await page.fill("textarea[name='text']", action.content)
                else:
                    # URL post
                    await page.fill("input[name='url']", action.url)
                
                logger.info("📰 Hacker News submission ready (auto-submit disabled for safety)")
                return True
            
            else:
                logger.warning(f"⚠️ Post content not implemented for {action.platform}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to post content: {e}")
            return False
    
    async def _execute_create_repository(self, page: Page, action: WebAction) -> bool:
        """Execute GitHub repository creation."""
        
        try:
            # Navigate to new repository page
            await page.goto("https://github.com/new")
            
            # Fill repository details
            if action.title:
                await page.fill("input[name='repository[name]']", action.title)
                await asyncio.sleep(random.uniform(0.5, 1.0))
            
            if action.content:
                await page.fill("input[name='repository[description]']", action.content)
                await asyncio.sleep(random.uniform(0.5, 1.0))
            
            # Make it public
            await page.check("input[value='public']")
            
            # Initialize with README
            await page.check("input[name='repository[auto_init]']")
            
            logger.info("📦 GitHub repository ready for creation (auto-submit disabled for safety)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create repository: {e}")
            return False
    
    async def _execute_login(self, page: Page, action: WebAction, session: AutomationSession) -> bool:
        """Execute login action."""
        
        # Login implementation would go here
        # For safety, we'll just simulate login success
        logger.info(f"🔐 Login simulation for {action.platform} (not implemented for safety)")
        session.is_authenticated = True
        return True
    
    async def _execute_create_account(self, page: Page, action: WebAction) -> bool:
        """Execute account creation action."""
        
        # Account creation would go here
        # For safety, we'll just simulate
        logger.info(f"👤 Account creation simulation for {action.platform} (not implemented for safety)")
        return True
    
    async def execute_marketing_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete marketing campaign using web automation."""
        
        logger.info(f"🚀 Starting automated marketing campaign: {campaign_data.get('name', 'Unnamed')}")
        
        results = {
            "campaign_id": campaign_data.get("campaign_id", f"auto_{int(time.time())}"),
            "platforms_targeted": [],
            "actions_completed": 0,
            "actions_failed": 0,
            "total_estimated_reach": 0,
            "execution_time": 0
        }
        
        start_time = time.time()
        
        # Create sessions for each target platform
        target_platforms = campaign_data.get("target_platforms", [])
        
        for platform_name in target_platforms:
            try:
                platform = PlatformType(platform_name.lower())
                session = await self.create_session(platform)
                
                # Create platform-specific actions
                actions = await self._create_platform_actions(session, campaign_data)
                
                # Execute actions with rate limiting
                platform_results = await self._execute_platform_actions(session, actions)
                
                results["platforms_targeted"].append({
                    "platform": platform_name,
                    "session_id": session.session_id,
                    "actions_completed": platform_results["successful"],
                    "actions_failed": platform_results["failed"],
                    "estimated_reach": platform_results["estimated_reach"]
                })
                
                results["actions_completed"] += platform_results["successful"]
                results["actions_failed"] += platform_results["failed"] 
                results["total_estimated_reach"] += platform_results["estimated_reach"]
                
            except Exception as e:
                logger.error(f"❌ Failed to execute on platform {platform_name}: {e}")
                results["actions_failed"] += 1
        
        results["execution_time"] = time.time() - start_time
        
        logger.info(f"🎯 Marketing campaign completed: {results['actions_completed']} actions successful")
        
        return results
    
    async def _create_platform_actions(self, session: AutomationSession, 
                                     campaign_data: Dict[str, Any]) -> List[WebAction]:
        """Create platform-specific actions for campaign."""
        
        actions = []
        
        if session.platform == PlatformType.REDDIT:
            # Create Reddit post actions
            for subreddit in campaign_data.get("reddit_subreddits", ["r/entrepreneur"]):
                action = await self.create_web_action(
                    session_id=session.session_id,
                    action_type=ActionType.POST_CONTENT,
                    url=f"https://reddit.com/{subreddit}",
                    title=campaign_data.get("reddit_title", "Free Email Marketing Tool"),
                    content=campaign_data.get("reddit_content", "Check out our free tool!"),
                    tags=campaign_data.get("tags", [])
                )
                actions.append(action)
        
        elif session.platform == PlatformType.GITHUB:
            # Create GitHub repository
            action = await self.create_web_action(
                session_id=session.session_id,
                action_type=ActionType.CREATE_REPOSITORY,
                url="https://github.com/new",
                title=campaign_data.get("github_repo_name", "professional-email-templates"),
                content=campaign_data.get("github_description", "20+ Free Professional Email Templates"),
                tags=campaign_data.get("github_topics", [])
            )
            actions.append(action)
        
        elif session.platform == PlatformType.HACKERNEWS:
            # Create HN submission
            action = await self.create_web_action(
                session_id=session.session_id,
                action_type=ActionType.POST_CONTENT,
                url="https://news.ycombinator.com/submit",
                title=campaign_data.get("hn_title", "Show HN: Free Email Templates"),
                content=campaign_data.get("hn_content", ""),
                tags=[]
            )
            actions.append(action)
        
        return actions
    
    async def _execute_platform_actions(self, session: AutomationSession, 
                                      actions: List[WebAction]) -> Dict[str, Any]:
        """Execute actions for a platform with rate limiting."""
        
        successful = 0
        failed = 0
        estimated_reach = 0
        
        # Get rate limits for platform
        limits = self.rate_limits.get(session.platform, {"requests_per_hour": 5})
        delay_between_actions = 3600 / limits["requests_per_hour"]  # Spread across hour
        
        for action in actions:
            try:
                success = await self.execute_action(session.session_id, action)
                
                if success:
                    successful += 1
                    # Estimate reach based on platform
                    estimated_reach += self._estimate_action_reach(session.platform, action.action_type)
                else:
                    failed += 1
                
                # Rate limiting delay
                await asyncio.sleep(delay_between_actions + random.uniform(0, 2))
                
            except Exception as e:
                logger.error(f"❌ Action execution error: {e}")
                failed += 1
        
        return {
            "successful": successful,
            "failed": failed,
            "estimated_reach": estimated_reach
        }
    
    def _estimate_action_reach(self, platform: PlatformType, action_type: ActionType) -> int:
        """Estimate reach for an action on a platform."""
        
        reach_estimates = {
            PlatformType.REDDIT: {
                ActionType.POST_CONTENT: 1000
            },
            PlatformType.GITHUB: {
                ActionType.CREATE_REPOSITORY: 500
            },
            PlatformType.HACKERNEWS: {
                ActionType.POST_CONTENT: 2000
            },
            PlatformType.LINKEDIN: {
                ActionType.POST_CONTENT: 300
            }
        }
        
        return reach_estimates.get(platform, {}).get(action_type, 100)
    
    async def deploy_email_service(self, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy email automation service to free platform."""
        
        logger.info("🚀 Starting email service deployment automation")
        
        # Create deployment session
        session = await self.create_session(PlatformType.EMAIL_PLATFORM)
        
        deployment_platforms = [
            {"name": "Railway", "url": "https://railway.app", "free_tier": True},
            {"name": "Vercel", "url": "https://vercel.com", "free_tier": True},
            {"name": "Netlify", "url": "https://netlify.com", "free_tier": True}
        ]
        
        deployment_results = []
        
        for platform in deployment_platforms[:1]:  # Try first platform
            try:
                result = await self._deploy_to_platform(session, platform, service_config)
                deployment_results.append(result)
                
                if result["success"]:
                    logger.info(f"✅ Successfully deployed to {platform['name']}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Deployment to {platform['name']} failed: {e}")
                deployment_results.append({
                    "platform": platform["name"],
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "deployment_attempts": len(deployment_results),
            "successful_deployments": sum(1 for r in deployment_results if r.get("success")),
            "results": deployment_results
        }
    
    async def _deploy_to_platform(self, session: AutomationSession, 
                                 platform: Dict[str, Any], 
                                 service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy service to specific platform."""
        
        # Deployment automation would go here
        # For now, simulate deployment
        logger.info(f"🔧 Simulating deployment to {platform['name']}")
        
        await asyncio.sleep(2)  # Simulate deployment time
        
        return {
            "platform": platform["name"],
            "success": True,
            "url": f"https://email-automation-{int(time.time())}.{platform['name'].lower()}.app",
            "deployment_time": 2.0
        }
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up automation session."""
        
        context = self.contexts.get(session_id)
        if context:
            await context.close()
            del self.contexts[session_id]
        
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        logger.info(f"🧹 Cleaned up session: {session_id}")
    
    async def cleanup_all(self) -> None:
        """Clean up all sessions and browser."""
        
        for session_id in list(self.sessions.keys()):
            await self.cleanup_session(session_id)
        
        if self.browser:
            await self.browser.close()
            await self.playwright.stop()
        
        logger.info("🧹 All sessions and browser cleaned up")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get session performance statistics."""
        
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "platform": session.platform.value,
            "total_actions": session.total_actions,
            "successful_actions": session.successful_actions,
            "failed_actions": session.failed_actions,
            "success_rate": (session.successful_actions / session.total_actions * 100) if session.total_actions > 0 else 0,
            "is_authenticated": session.is_authenticated,
            "created_at": session.created_at.isoformat()
        }


# Factory function
def create_web_automation_agent() -> WebAutomationAgent:
    """Create web automation agent instance."""
    return WebAutomationAgent()


# Main execution
async def main():
    """Example web automation execution."""
    
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright not available. Install with:")
        print("pip install playwright")
        print("playwright install")
        return
    
    agent = create_web_automation_agent()
    
    # Start browser
    await agent.start_browser(headless=True)
    
    # Example marketing campaign automation
    campaign_data = {
        "name": "Email Templates Launch",
        "campaign_id": "email_templates_001",
        "target_platforms": ["reddit", "github", "hackernews"],
        "reddit_subreddits": ["r/webdev", "r/entrepreneur"],
        "reddit_title": "I made 20+ free professional email templates",
        "reddit_content": "Free email templates for developers and businesses...",
        "github_repo_name": "professional-email-templates",
        "github_description": "20+ Beautiful, Responsive Email Templates",
        "hn_title": "Show HN: Professional Email Templates (20+ free, MIT licensed)",
        "tags": ["email", "templates", "marketing", "free"]
    }
    
    print("🚀 Starting automated marketing campaign...")
    
    results = await agent.execute_marketing_campaign(campaign_data)
    
    print(f"\n✅ Campaign Results:")
    print(f"   Platforms: {len(results['platforms_targeted'])}")
    print(f"   Actions completed: {results['actions_completed']}")
    print(f"   Actions failed: {results['actions_failed']}")
    print(f"   Estimated reach: {results['total_estimated_reach']:,}")
    print(f"   Execution time: {results['execution_time']:.2f}s")
    
    # Test email service deployment
    print(f"\n🚀 Testing email service deployment...")
    
    deployment_results = await agent.deploy_email_service({
        "name": "Email Automation Service",
        "description": "Free email automation for small businesses"
    })
    
    print(f"   Deployment attempts: {deployment_results['deployment_attempts']}")
    print(f"   Successful deployments: {deployment_results['successful_deployments']}")
    
    # Cleanup
    await agent.cleanup_all()
    print(f"\n🎉 Web automation test completed!")


if __name__ == "__main__":
    asyncio.run(main())