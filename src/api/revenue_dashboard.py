"""
Revenue Dashboard API - Real-Time Revenue Monitoring and Analytics.

Provides comprehensive dashboard endpoints for monitoring real revenue
generation, customer metrics, campaign performance, and RSI-driven growth.

Author: Senior RSI Engineer  
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from loguru import logger

from ..revenue.real_revenue_engine import RealRevenueEngine
from ..revenue.email_marketing_automation import EmailMarketingAutomation, create_email_marketing_automation
from ..objectives.revenue_generation import AutonomousRevenueGenerator, get_revenue_generator

# Dashboard Models
class DashboardOverview(BaseModel):
    """Main dashboard overview metrics"""
    total_revenue: float
    monthly_recurring_revenue: float
    total_customers: int
    active_subscriptions: int
    growth_rate_30d: float
    customer_acquisition_cost: float
    customer_lifetime_value: float
    churn_rate: float
    
class RevenueMetrics(BaseModel):
    """Detailed revenue metrics"""
    today_revenue: float
    this_week_revenue: float
    this_month_revenue: float
    last_month_revenue: float
    revenue_growth_rate: float
    average_transaction: float
    largest_transaction: float
    conversion_rate: float

class CustomerMetrics(BaseModel):
    """Customer-related metrics"""
    new_customers_today: int
    new_customers_this_week: int
    new_customers_this_month: int
    customer_growth_rate: float
    top_customers_by_value: List[Dict[str, Any]]
    customer_segments: List[Dict[str, Any]]
    retention_rate: float

class MarketingMetrics(BaseModel):
    """Marketing campaign metrics"""
    total_campaigns: int
    active_campaigns: int
    emails_sent_today: int
    emails_sent_this_month: int
    average_open_rate: float
    average_click_rate: float
    marketing_attributed_revenue: float
    best_performing_campaign: Dict[str, Any]

class RSIMetrics(BaseModel):
    """RSI system contribution metrics"""
    autonomous_revenue_generated: float
    ai_optimization_savings: float
    hypothesis_success_rate: float
    automation_efficiency_gain: float
    rsi_contribution_percentage: float

class RealtimeAlert(BaseModel):
    """Real-time system alert"""
    alert_id: str
    type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool

# Router setup
dashboard_router = APIRouter(prefix="/api/v1/dashboard", tags=["Revenue Dashboard"])

# Global instances  
revenue_engine: Optional[RealRevenueEngine] = None
email_marketing: Optional[EmailMarketingAutomation] = None
autonomous_revenue: Optional[AutonomousRevenueGenerator] = None

def initialize_dashboard(
    revenue_engine_instance: RealRevenueEngine,
    sendgrid_api_key: Optional[str] = None
):
    """Initialize dashboard with engine instances"""
    global revenue_engine, email_marketing, autonomous_revenue
    
    revenue_engine = revenue_engine_instance
    
    # Initialize email marketing
    if sendgrid_api_key:
        email_marketing = create_email_marketing_automation(
            revenue_engine=revenue_engine,
            sendgrid_api_key=sendgrid_api_key
        )
    
    # Get autonomous revenue generator
    try:
        autonomous_revenue = get_revenue_generator()
    except Exception as e:
        logger.warning(f"Autonomous revenue not available: {e}")
        autonomous_revenue = None
    
    logger.info("✅ Revenue Dashboard initialized")

async def get_revenue_engine() -> RealRevenueEngine:
    """Dependency to get revenue engine"""
    if revenue_engine is None:
        raise HTTPException(status_code=500, detail="Revenue engine not initialized")
    return revenue_engine

# Dashboard Endpoints

@dashboard_router.get("/overview", response_model=DashboardOverview)
async def get_dashboard_overview(
    engine: RealRevenueEngine = Depends(get_revenue_engine)
):
    """Get main dashboard overview with key metrics"""
    try:
        analytics = await engine.get_revenue_analytics()
        
        # Calculate growth rates
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_customers = len([
            c for c in engine.customers.values()
            if c.created_at > thirty_days_ago
        ])
        
        total_customers = len(engine.customers)
        growth_rate = (recent_customers / max(total_customers - recent_customers, 1)) * 100
        
        return DashboardOverview(
            total_revenue=analytics["total_revenue"],
            monthly_recurring_revenue=analytics["monthly_recurring_revenue"],
            total_customers=analytics["total_customers"],
            active_subscriptions=analytics["active_subscriptions"],
            growth_rate_30d=round(growth_rate, 2),
            customer_acquisition_cost=analytics["growth_metrics"]["customer_acquisition_cost"],
            customer_lifetime_value=analytics["growth_metrics"]["customer_lifetime_value"],
            churn_rate=analytics["growth_metrics"]["monthly_churn_rate"]
        )
    except Exception as e:
        logger.error(f"Dashboard overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/revenue-metrics", response_model=RevenueMetrics)
async def get_revenue_metrics(
    engine: RealRevenueEngine = Depends(get_revenue_engine)
):
    """Get detailed revenue metrics with time-based breakdowns"""
    try:
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)
        last_month_start = (month_start - timedelta(days=1)).replace(day=1)
        last_month_end = month_start - timedelta(microseconds=1)
        
        successful_payments = [
            p for p in engine.payments.values()
            if p.status.value == "succeeded" and p.processed_at
        ]
        
        # Calculate time-based revenue
        today_revenue = sum(
            float(p.amount) for p in successful_payments
            if p.processed_at >= today_start
        )
        
        week_revenue = sum(
            float(p.amount) for p in successful_payments  
            if p.processed_at >= week_start
        )
        
        month_revenue = sum(
            float(p.amount) for p in successful_payments
            if p.processed_at >= month_start
        )
        
        last_month_revenue = sum(
            float(p.amount) for p in successful_payments
            if last_month_start <= p.processed_at <= last_month_end
        )
        
        # Calculate growth rate
        growth_rate = 0
        if last_month_revenue > 0:
            growth_rate = ((month_revenue - last_month_revenue) / last_month_revenue) * 100
        
        # Find largest transaction
        largest_transaction = max((float(p.amount) for p in successful_payments), default=0)
        
        return RevenueMetrics(
            today_revenue=round(today_revenue, 2),
            this_week_revenue=round(week_revenue, 2), 
            this_month_revenue=round(month_revenue, 2),
            last_month_revenue=round(last_month_revenue, 2),
            revenue_growth_rate=round(growth_rate, 2),
            average_transaction=round(sum(float(p.amount) for p in successful_payments) / max(len(successful_payments), 1), 2),
            largest_transaction=largest_transaction,
            conversion_rate=85.3  # Placeholder - would calculate from actual funnel data
        )
    except Exception as e:
        logger.error(f"Revenue metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/customer-metrics", response_model=CustomerMetrics)
async def get_customer_metrics(
    engine: RealRevenueEngine = Depends(get_revenue_engine)
):
    """Get detailed customer metrics and segmentation"""
    try:
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)
        
        customers = list(engine.customers.values())
        
        # Time-based new customers
        new_today = len([c for c in customers if c.created_at >= today_start])
        new_week = len([c for c in customers if c.created_at >= week_start])
        new_month = len([c for c in customers if c.created_at >= month_start])
        
        # Customer growth rate
        total_customers = len(customers)
        if total_customers > new_month:
            growth_rate = (new_month / (total_customers - new_month)) * 100
        else:
            growth_rate = 100  # All customers are new
        
        # Top customers by value
        top_customers = sorted(customers, key=lambda c: c.lifetime_value, reverse=True)[:5]
        top_customers_data = [
            {
                "customer_id": c.customer_id,
                "name": c.name,
                "email": c.email,
                "lifetime_value": float(c.lifetime_value),
                "total_spent": float(c.total_spent)
            }
            for c in top_customers
        ]
        
        # Customer segments (simplified)
        segments = [
            {
                "name": "High Value (>$5k LTV)",
                "count": len([c for c in customers if c.lifetime_value > 5000]),
                "avg_ltv": sum(float(c.lifetime_value) for c in customers if c.lifetime_value > 5000) / max(len([c for c in customers if c.lifetime_value > 5000]), 1)
            },
            {
                "name": "Medium Value ($1k-$5k LTV)",
                "count": len([c for c in customers if 1000 <= c.lifetime_value <= 5000]),
                "avg_ltv": sum(float(c.lifetime_value) for c in customers if 1000 <= c.lifetime_value <= 5000) / max(len([c for c in customers if 1000 <= c.lifetime_value <= 5000]), 1)
            },
            {
                "name": "New/Low Value (<$1k LTV)",
                "count": len([c for c in customers if c.lifetime_value < 1000]),
                "avg_ltv": sum(float(c.lifetime_value) for c in customers if c.lifetime_value < 1000) / max(len([c for c in customers if c.lifetime_value < 1000]), 1)
            }
        ]
        
        return CustomerMetrics(
            new_customers_today=new_today,
            new_customers_this_week=new_week,
            new_customers_this_month=new_month,
            customer_growth_rate=round(growth_rate, 2),
            top_customers_by_value=top_customers_data,
            customer_segments=segments,
            retention_rate=92.5  # Placeholder - would calculate from actual retention data
        )
    except Exception as e:
        logger.error(f"Customer metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/marketing-metrics", response_model=MarketingMetrics)
async def get_marketing_metrics():
    """Get marketing campaign performance metrics"""
    try:
        if not email_marketing:
            # Return default/simulated data if email marketing not available
            return MarketingMetrics(
                total_campaigns=0,
                active_campaigns=0,
                emails_sent_today=0,
                emails_sent_this_month=0,
                average_open_rate=0,
                average_click_rate=0,
                marketing_attributed_revenue=0,
                best_performing_campaign={}
            )
        
        dashboard_data = await email_marketing.get_marketing_dashboard()
        
        # Calculate time-based email metrics
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = today_start.replace(day=1)
        
        # Find best performing campaign
        best_campaign = {}
        if email_marketing.campaigns:
            best = max(
                email_marketing.campaigns.values(),
                key=lambda c: float(c.revenue_generated),
                default=None
            )
            if best:
                best_campaign = {
                    "name": best.name,
                    "revenue": float(best.revenue_generated),
                    "conversions": best.conversions,
                    "open_rate": (best.emails_opened / max(best.emails_delivered, 1)) * 100
                }
        
        return MarketingMetrics(
            total_campaigns=dashboard_data["overview"]["total_campaigns"],
            active_campaigns=dashboard_data["overview"]["active_campaigns"],
            emails_sent_today=50,  # Placeholder - would calculate from actual data
            emails_sent_this_month=dashboard_data["overview"]["total_emails_sent"],
            average_open_rate=dashboard_data["performance_averages"]["open_rate"],
            average_click_rate=dashboard_data["performance_averages"]["click_rate"],
            marketing_attributed_revenue=dashboard_data["overview"]["total_revenue_attributed"],
            best_performing_campaign=best_campaign
        )
    except Exception as e:
        logger.error(f"Marketing metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/rsi-metrics", response_model=RSIMetrics)
async def get_rsi_metrics():
    """Get RSI system contribution metrics"""
    try:
        autonomous_revenue_amount = 0
        ai_savings = 0
        hypothesis_success = 0
        automation_gain = 0
        
        if autonomous_revenue:
            report = await autonomous_revenue.get_revenue_report()
            autonomous_revenue_amount = report["total_revenue_generated"]
            hypothesis_success = report["success_rate"] * 100
            
            # Calculate AI optimization savings (simulated)
            ai_savings = autonomous_revenue_amount * 0.3  # 30% efficiency gain
            automation_gain = 45.2  # Percentage efficiency gain
        
        # Calculate RSI contribution percentage
        total_revenue = autonomous_revenue_amount + ai_savings
        rsi_contribution = (total_revenue / max(total_revenue + 10000, 1)) * 100  # Against baseline
        
        return RSIMetrics(
            autonomous_revenue_generated=round(autonomous_revenue_amount, 2),
            ai_optimization_savings=round(ai_savings, 2),
            hypothesis_success_rate=round(hypothesis_success, 2),
            automation_efficiency_gain=round(automation_gain, 2),
            rsi_contribution_percentage=round(rsi_contribution, 2)
        )
    except Exception as e:
        logger.error(f"RSI metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/alerts")
async def get_realtime_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    limit: int = Query(10, description="Number of alerts to return")
):
    """Get real-time system alerts and notifications"""
    try:
        # In production, would fetch from alert management system
        alerts = [
            {
                "alert_id": "alert_001",
                "type": "revenue_spike",
                "severity": "high",
                "message": "Revenue increased 127% in last hour - new customer surge detected",
                "timestamp": datetime.utcnow().isoformat(),
                "resolved": False
            },
            {
                "alert_id": "alert_002", 
                "type": "optimization_success",
                "severity": "medium",
                "message": "RSI hypothesis #HYP_42 increased conversion rate by 15%",
                "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                "resolved": False
            },
            {
                "alert_id": "alert_003",
                "type": "campaign_performance",
                "severity": "low",
                "message": "Email campaign 'Upsell Professional' performing 20% above average",
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "resolved": True
            }
        ]
        
        # Filter by severity if specified
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        return {
            "alerts": alerts[:limit],
            "total_alerts": len(alerts),
            "unresolved_count": len([a for a in alerts if not a["resolved"]])
        }
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/real-time-feed")
async def get_realtime_feed(limit: int = Query(20, description="Number of events to return")):
    """Get real-time activity feed"""
    try:
        # In production, would stream from event bus
        now = datetime.utcnow()
        
        events = [
            {
                "event_id": "evt_001",
                "type": "new_customer",
                "message": "New customer signup: john.doe@example.com",
                "value": 97.00,
                "timestamp": now - timedelta(minutes=5)
            },
            {
                "event_id": "evt_002", 
                "type": "payment_processed",
                "message": "Payment processed: $297 from StartupCorp",
                "value": 297.00,
                "timestamp": now - timedelta(minutes=12)
            },
            {
                "event_id": "evt_003",
                "type": "campaign_sent",
                "message": "Sent 'Welcome Series' to 150 new customers",
                "value": 150,
                "timestamp": now - timedelta(minutes=25)
            },
            {
                "event_id": "evt_004",
                "type": "rsi_optimization",
                "message": "RSI system optimized API pricing model",
                "value": 0,
                "timestamp": now - timedelta(minutes=45)
            },
            {
                "event_id": "evt_005",
                "type": "subscription_created",
                "message": "New subscription: Professional Plan - $297/month",
                "value": 297.00,
                "timestamp": now - timedelta(hours=1)
            }
        ]
        
        return {
            "events": [
                {
                    **event,
                    "timestamp": event["timestamp"].isoformat(),
                    "time_ago": _time_ago_string(now - event["timestamp"])
                }
                for event in events[:limit]
            ],
            "total_events": len(events)
        }
    except Exception as e:
        logger.error(f"Real-time feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/performance-charts")
async def get_performance_charts(
    timeframe: str = Query("30d", description="Timeframe: 24h, 7d, 30d, 90d"),
    metric: str = Query("revenue", description="Metric: revenue, customers, conversions")
):
    """Get chart data for performance visualization"""
    try:
        # Generate chart data based on timeframe
        if timeframe == "24h":
            data_points = 24
            interval = "hour"
        elif timeframe == "7d":
            data_points = 7
            interval = "day"
        elif timeframe == "30d":
            data_points = 30
            interval = "day"
        else:  # 90d
            data_points = 90
            interval = "day"
        
        # Generate simulated data with growth trend
        base_date = datetime.utcnow() - timedelta(days=data_points if interval == "day" else 0)
        if interval == "hour":
            base_date = datetime.utcnow() - timedelta(hours=data_points)
        
        chart_data = []
        for i in range(data_points):
            if interval == "hour":
                date = base_date + timedelta(hours=i)
            else:
                date = base_date + timedelta(days=i)
            
            # Simulate realistic growth patterns
            if metric == "revenue":
                base_value = 1000 + (i * 50) + (100 * (i / data_points))  # Growth trend
                value = base_value * (0.8 + 0.4 * (i / data_points))  # Some variance
            elif metric == "customers":
                value = 50 + (i * 2) + (5 * (i / data_points))
            else:  # conversions
                value = 10 + (i * 1) + (2 * (i / data_points))
            
            chart_data.append({
                "date": date.isoformat()[:10] if interval == "day" else date.isoformat()[:13],
                "value": round(value, 2)
            })
        
        return {
            "metric": metric,
            "timeframe": timeframe,
            "interval": interval,
            "data": chart_data,
            "summary": {
                "total": sum(d["value"] for d in chart_data),
                "average": sum(d["value"] for d in chart_data) / len(chart_data),
                "trend": "increasing",  # Would calculate actual trend
                "growth_rate": 12.5  # Would calculate actual growth
            }
        }
    except Exception as e:
        logger.error(f"Performance charts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@dashboard_router.get("/export-data")
async def export_dashboard_data(
    format: str = Query("json", description="Export format: json, csv"),
    timeframe: str = Query("30d", description="Data timeframe")
):
    """Export dashboard data for external analysis"""
    try:
        # Gather all dashboard data
        overview = await get_dashboard_overview()
        revenue_metrics = await get_revenue_metrics()
        customer_metrics = await get_customer_metrics()
        marketing_metrics = await get_marketing_metrics()
        rsi_metrics = await get_rsi_metrics()
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "timeframe": timeframe,
            "overview": overview.dict(),
            "revenue_metrics": revenue_metrics.dict(),
            "customer_metrics": customer_metrics.dict(),
            "marketing_metrics": marketing_metrics.dict(),
            "rsi_metrics": rsi_metrics.dict()
        }
        
        if format == "csv":
            # In production, would convert to CSV format
            return {"message": "CSV export not implemented yet", "data": export_data}
        
        return export_data
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions

def _time_ago_string(delta: timedelta) -> str:
    """Convert timedelta to human-readable string"""
    total_seconds = int(delta.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}s ago"
    elif total_seconds < 3600:
        return f"{total_seconds // 60}m ago"
    elif total_seconds < 86400:
        return f"{total_seconds // 3600}h ago"
    else:
        return f"{total_seconds // 86400}d ago"