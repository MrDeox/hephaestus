"""
Production Deployment Script for Hephaestus Revenue Infrastructure.

Deploys the complete real revenue generation system including:
- Stripe payment processing
- SendGrid email automation  
- Customer management APIs
- Revenue analytics dashboard
- RSI integration

Author: Senior RSI Engineer
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import app, RSIOrchestrator

class RevenueSystemDeployment:
    """
    Production deployment manager for revenue infrastructure.
    """
    
    def __init__(self):
        self.deployment_config = self._load_deployment_config()
        self.health_checks_passed = False
        
    def _load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        
        # Production configuration
        config = {
            "server": {
                "host": os.getenv("HOST", "0.0.0.0"),
                "port": int(os.getenv("PORT", 8000)),
                "workers": int(os.getenv("WORKERS", 1)),
                "reload": False,
                "log_level": "info"
            },
            "revenue": {
                "stripe_secret_key": os.getenv("STRIPE_SECRET_KEY"),
                "sendgrid_api_key": os.getenv("SENDGRID_API_KEY"),
                "webhook_secret": os.getenv("STRIPE_WEBHOOK_SECRET"),
                "api_key": os.getenv("REVENUE_API_KEY", "prod-revenue-api-key-secure-123")
            },
            "database": {
                "url": os.getenv("DATABASE_URL", "sqlite:///revenue_production.db")
            },
            "monitoring": {
                "enable_telemetry": os.getenv("ENABLE_TELEMETRY", "true").lower() == "true",
                "log_level": os.getenv("LOG_LEVEL", "INFO")
            },
            "security": {
                "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
                "rate_limiting": True,
                "api_key_required": True
            }
        }
        
        return config
    
    async def run_deployment_checks(self) -> bool:
        """Run comprehensive deployment readiness checks"""
        
        logger.info("🔍 Running deployment readiness checks...")
        
        checks = [
            self._check_environment_variables(),
            self._check_dependencies(),
            self._check_database_connectivity(),
            self._check_external_services(),
            self._check_security_configuration(),
            await self._check_revenue_system_health()
        ]
        
        if all(checks):
            self.health_checks_passed = True
            logger.info("✅ All deployment checks passed!")
            return True
        else:
            logger.error("❌ Some deployment checks failed")
            return False
    
    def _check_environment_variables(self) -> bool:
        """Check required environment variables"""
        
        logger.info("Checking environment variables...")
        
        required_vars = {
            "STRIPE_SECRET_KEY": "Stripe payment processing",
            "SENDGRID_API_KEY": "Email automation",
            "STRIPE_WEBHOOK_SECRET": "Webhook security"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing_vars.append(f"{var} ({description})")
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
            logger.info("💡 System will run in simulation mode without these variables")
            return True  # Allow deployment in simulation mode
        
        logger.info("✅ Environment variables configured")
        return True
    
    def _check_dependencies(self) -> bool:
        """Check required Python dependencies"""
        
        logger.info("Checking dependencies...")
        
        required_packages = [
            "fastapi",
            "uvicorn", 
            "pydantic",
            "loguru"
        ]
        
        optional_packages = {
            "stripe": "Payment processing",
            "sendgrid": "Email automation",
            "jinja2": "Email templating"
        }
        
        missing_required = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)
        
        if missing_required:
            logger.error(f"❌ Missing required packages: {', '.join(missing_required)}")
            return False
        
        missing_optional = []
        for package, description in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(f"{package} ({description})")
        
        if missing_optional:
            logger.warning(f"⚠️ Missing optional packages: {', '.join(missing_optional)}")
            logger.info("💡 Some features will be disabled")
        
        logger.info("✅ Core dependencies available")
        return True
    
    def _check_database_connectivity(self) -> bool:
        """Check database connectivity"""
        
        logger.info("Checking database connectivity...")
        
        try:
            # For SQLite, just check if directory is writable
            db_path = Path("revenue_production.db")
            test_file = db_path.parent / "test_write.tmp"
            
            # Test write access
            test_file.write_text("test")
            test_file.unlink()
            
            logger.info("✅ Database directory writable")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connectivity issue: {e}")
            return False
    
    def _check_external_services(self) -> bool:
        """Check external service connectivity"""
        
        logger.info("Checking external services...")
        
        # Check Stripe connectivity (if configured)
        stripe_key = os.getenv("STRIPE_SECRET_KEY")
        if stripe_key:
            try:
                import stripe
                stripe.api_key = stripe_key
                # Test API call
                stripe.Account.retrieve()
                logger.info("✅ Stripe connectivity verified")
            except Exception as e:
                logger.warning(f"⚠️ Stripe connectivity issue: {e}")
        
        # Check SendGrid connectivity (if configured)
        sendgrid_key = os.getenv("SENDGRID_API_KEY")
        if sendgrid_key:
            try:
                import sendgrid
                sg = sendgrid.SendGridAPIClient(api_key=sendgrid_key)
                # Test API call (get API key info)
                response = sg.client.api_keys.get()
                logger.info("✅ SendGrid connectivity verified")
            except Exception as e:
                logger.warning(f"⚠️ SendGrid connectivity issue: {e}")
        
        return True
    
    def _check_security_configuration(self) -> bool:
        """Check security configuration"""
        
        logger.info("Checking security configuration...")
        
        # Check API key strength
        api_key = self.deployment_config["revenue"]["api_key"]
        if len(api_key) < 32:
            logger.warning("⚠️ API key should be at least 32 characters")
        
        # Check CORS configuration
        cors_origins = self.deployment_config["security"]["cors_origins"]
        if "*" in cors_origins:
            logger.warning("⚠️ CORS configured to allow all origins (not recommended for production)")
        
        logger.info("✅ Security configuration reviewed")
        return True
    
    async def _check_revenue_system_health(self) -> bool:
        """Check revenue system health"""
        
        logger.info("Checking revenue system health...")
        
        try:
            # Initialize orchestrator to test system
            orchestrator = RSIOrchestrator(environment="production")
            await orchestrator.start()
            
            # Test basic functionality
            if hasattr(orchestrator, 'real_revenue_engine') and orchestrator.real_revenue_engine:
                logger.info("✅ Real Revenue Engine initialized")
            
            if hasattr(orchestrator, 'revenue_generator') and orchestrator.revenue_generator:
                logger.info("✅ Autonomous Revenue Generator initialized")
            
            await orchestrator.stop()
            logger.info("✅ Revenue system health check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Revenue system health check failed: {e}")
            return False
    
    def _setup_production_logging(self):
        """Setup production logging configuration"""
        
        logger.remove()  # Remove default logger
        
        # Add structured logging for production
        logger.add(
            "logs/revenue_system_{time:YYYY-MM-DD}.log",
            level=self.deployment_config["monitoring"]["log_level"],
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation="daily",
            retention="30 days",
            compression="gzip"
        )
        
        # Console logging for immediate feedback
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>"
        )
        
        logger.info("✅ Production logging configured")
    
    def _display_deployment_summary(self):
        """Display deployment summary"""
        
        logger.info("=" * 70)
        logger.info("🚀 HEPHAESTUS REVENUE SYSTEM DEPLOYMENT SUMMARY")
        logger.info("=" * 70)
        
        config = self.deployment_config
        
        logger.info(f"🌐 Server: http://{config['server']['host']}:{config['server']['port']}")
        logger.info(f"🔧 Environment: Production")
        logger.info(f"💳 Stripe: {'✅ Configured' if config['revenue']['stripe_secret_key'] else '⚠️ Simulation Mode'}")
        logger.info(f"📧 SendGrid: {'✅ Configured' if config['revenue']['sendgrid_api_key'] else '⚠️ Simulation Mode'}")
        logger.info(f"🛡️ Security: {'✅ Enabled' if config['security']['api_key_required'] else '⚠️ Disabled'}")
        
        logger.info("\n📊 Available Endpoints:")
        logger.info("  • /api/v1/revenue/* - Payment & subscription management")
        logger.info("  • /api/v1/dashboard/* - Revenue analytics dashboard")
        logger.info("  • /health - System health check")
        logger.info("  • /docs - API documentation")
        
        logger.info("\n🎯 Revenue Capabilities:")
        logger.info("  • Real Stripe payment processing")
        logger.info("  • Automated email marketing campaigns")
        logger.info("  • Customer lifecycle management")
        logger.info("  • Revenue analytics and reporting")
        logger.info("  • RSI-driven optimization")
        logger.info("  • Webhook handling for real-time events")
        
        logger.info("=" * 70)
    
    async def deploy(self):
        """Execute production deployment"""
        
        logger.info("🚀 Starting Hephaestus Revenue System deployment...")
        
        # Setup logging
        self._setup_production_logging()
        
        # Run deployment checks
        if not await self.run_deployment_checks():
            logger.error("❌ Deployment checks failed. Aborting deployment.")
            return False
        
        # Display deployment summary
        self._display_deployment_summary()
        
        # Start server
        logger.info("🌟 Starting production server...")
        
        try:
            uvicorn.run(
                "src.main:app",
                host=self.deployment_config["server"]["host"],
                port=self.deployment_config["server"]["port"],
                workers=self.deployment_config["server"]["workers"],
                reload=self.deployment_config["server"]["reload"],
                log_level=self.deployment_config["server"]["log_level"],
                access_log=True
            )
        except KeyboardInterrupt:
            logger.info("👋 Shutting down gracefully...")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
            return False
        
        return True

async def main():
    """Main deployment function"""
    
    print("\n" + "=" * 70)
    print("🏛️  HEPHAESTUS RSI - REAL REVENUE SYSTEM DEPLOYMENT")
    print("   Senior RSI Engineer - Production Revenue Infrastructure")
    print("=" * 70 + "\n")
    
    deployment = RevenueSystemDeployment()
    success = await deployment.deploy()
    
    if success:
        logger.info("✅ Deployment completed successfully!")
    else:
        logger.error("❌ Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("data/revenue_generation").mkdir(exist_ok=True)
    
    # Run deployment
    asyncio.run(main())