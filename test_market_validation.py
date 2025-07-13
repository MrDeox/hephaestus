import asyncio
import sys
sys.path.append('.')
from src.bootstrap.market_validator import create_market_validator

async def test_validation():
    validator = create_market_validator()
    
    print('🔍 VALIDANDO MERCADO REAL DE WEB SCRAPING...')
    print('=' * 60)
    
    result = await validator.validate_market()
    
    print(f'📊 RESULTADOS DA PESQUISA DE MERCADO:')
    print(f'Pontos de dados analisados: {result.total_data_points}')
    print(f'Score de viabilidade: {result.market_viability_score:.2f}/1.0')
    
    print(f'\n💰 PREÇOS REAIS DO MERCADO:')
    for complexity, price_range in result.price_ranges.items():
        avg_price = result.average_prices.get(complexity, 0)
        print(f'  {complexity.upper()}: ${price_range[0]:.0f}-${price_range[1]:.0f} (média: ${avg_price:.0f})')
    
    print(f'\n🎯 PREÇOS RECOMENDADOS (baseado na pesquisa):')
    for tier, price in result.recommended_pricing.items():
        print(f'  {tier.upper()}: ${price:.2f}')
    
    print(f'\n🔍 LACUNAS DO MERCADO IDENTIFICADAS:')
    for gap in result.market_gaps:
        print(f'  • {gap}')
    
    print(f'\n🏢 COMPETIDORES ANALISADOS:')
    for comp in result.competitors:
        print(f'  • {comp.name}: ${comp.price_range[0]:.0f}-${comp.price_range[1]:.0f} ({comp.pricing_model})')
    
    print(f'\n🚀 ESTRATÉGIA DE ENTRADA NO MERCADO:')
    for strategy in result.go_to_market_strategy:
        print(f'  • {strategy}')
    
    print(f'\n🎯 OPORTUNIDADES IDENTIFICADAS:')
    for opp in result.opportunities:
        print(f'  • {opp}')
    
    print(f'\n📈 CONCLUSÃO:')
    if result.market_viability_score > 0.7:
        print('✅ MERCADO VIÁVEL - Prosseguir com implementação')
    elif result.market_viability_score > 0.5:
        print('⚠️ MERCADO MODERADO - Ajustar estratégia')
    else:
        print('❌ MERCADO DIFÍCIL - Considerar mudança de abordagem')

if __name__ == "__main__":
    asyncio.run(test_validation())