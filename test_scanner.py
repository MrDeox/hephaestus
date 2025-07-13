import asyncio
import sys
sys.path.append('.')
from src.bootstrap.zero_cost_scanner import create_zero_cost_scanner

async def test_scanner():
    scanner = create_zero_cost_scanner()
    opportunities = await scanner.scan_all_opportunities()
    
    print(f'🎯 FOUND {len(opportunities)} ZERO-COST OPPORTUNITIES!')
    print()
    
    # Show top 5 opportunities
    top_5 = scanner.get_top_opportunities(5)
    for i, opp in enumerate(top_5, 1):
        print(f'{i}. {opp.title}')
        print(f'   💰 Revenue: ${opp.estimated_revenue_potential:.0f}/month')
        print(f'   ⏱️ Time to $1: {opp.time_to_first_dollar} days')
        print(f'   🎯 Priority: {opp.priority_score:.2f}')
        print(f'   📈 Category: {opp.category.value}')
        print()
    
    # Quick wins (<=7 days)
    quick_wins = scanner.get_quick_win_opportunities(7)
    print(f'⚡ QUICK WINS ({len(quick_wins)} opportunities ≤7 days):')
    for qw in quick_wins[:3]:
        print(f'- {qw.title} ({qw.time_to_first_dollar} days, ${qw.estimated_revenue_potential:.0f}/month)')

if __name__ == "__main__":
    asyncio.run(test_scanner())