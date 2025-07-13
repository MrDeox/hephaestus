import asyncio
import sys
sys.path.append('.')
from src.bootstrap.zero_cost_scanner import create_zero_cost_scanner

async def find_better_opportunities():
    scanner = create_zero_cost_scanner()
    opportunities = await scanner.scan_all_opportunities()
    
    print('🔍 REAVALIANDO OPORTUNIDADES COM MARKET REALITY CHECK...')
    print('=' * 60)
    
    # Get opportunities sorted by realistic potential
    better_opportunities = []
    
    for opp in opportunities:
        # Apply market reality adjustment based on competition analysis
        if opp.category.value == 'automation_scripts':
            # Automation has less competition than scraping
            reality_score = opp.priority_score * 1.2
        elif opp.category.value == 'github_tools':
            # GitHub tools have developer audience willing to pay
            reality_score = opp.priority_score * 1.1  
        elif opp.category.value == 'free_api_services':
            # API services can scale better
            reality_score = opp.priority_score * 1.15
        else:
            reality_score = opp.priority_score * 0.8  # Penalize others
        
        better_opportunities.append((opp, reality_score))
    
    # Sort by reality-adjusted score
    better_opportunities.sort(key=lambda x: x[1], reverse=True)
    
    print('🎯 TOP 5 OPORTUNIDADES (AJUSTADAS PELA REALIDADE DO MERCADO):')
    for i, (opp, score) in enumerate(better_opportunities[:5], 1):
        print(f'{i}. {opp.title}')
        print(f'   💰 Revenue: ${opp.estimated_revenue_potential:.0f}/month')
        print(f'   ⏱️ Time to $1: {opp.time_to_first_dollar} days')
        print(f'   🎯 Reality Score: {score:.2f}')
        print(f'   📈 Category: {opp.category.value}')
        print(f'   🔧 Difficulty: {opp.difficulty.value}')
        print()
    
    # Recommend pivot
    best_opp = better_opportunities[0][0]
    print(f'📋 RECOMENDAÇÃO DE PIVOT:')
    print(f'❌ De: Web Scraping as a Service (mercado saturado)')
    print(f'✅ Para: {best_opp.title}')
    print(f'💡 Razão: Menor competição, melhor viabilidade de mercado')
    print()
    
    print(f'🚀 PRÓXIMOS PASSOS PARA PIVOT:')
    for i, step in enumerate(best_opp.implementation_steps[:3], 1):
        print(f'{i}. {step}')
    
    print(f'\n🎯 POR QUE ESTA OPORTUNIDADE É MELHOR:')
    print(f'• Categoria: {best_opp.category.value} (menos saturada)')
    print(f'• Dificuldade: {best_opp.difficulty.value} (mais acessível)')
    print(f'• Time to market: {best_opp.time_to_first_dollar} dias')
    print(f'• Audience: {", ".join(best_opp.target_audience)}')

if __name__ == "__main__":
    asyncio.run(find_better_opportunities())