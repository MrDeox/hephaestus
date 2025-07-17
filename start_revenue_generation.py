#!/usr/bin/env python3
"""
Inicia o sistema de geração de receita autônoma do Hephaestus.
Ativa todas as funcionalidades de monetização e RSI.
"""

import asyncio
import sys
import os
from pathlib import Path

# Adicionar ao path
sys.path.insert(0, str(Path.cwd()))

async def start_revenue_generation():
    print('🚀 Iniciando Sistema de Geração de Receita Hephaestus...')
    
    try:
        from src.main import RSIOrchestrator
        from src.objectives.revenue_generation import get_revenue_generator
        
        # Configurar ambiente para produção
        os.environ['RSI_ENVIRONMENT'] = 'production'
        os.environ['REVENUE_API_KEY'] = 'prod-hephaestus-revenue-key-2025'
        
        print('⚙️ Inicializando orquestrador RSI...')
        orchestrator = RSIOrchestrator(environment='production')
        await orchestrator.start()
        
        print('💰 Ativando gerador de receita autônomo...')
        revenue_generator = get_revenue_generator()
        
        # Executar estratégia de receita
        print('📈 Executando estratégias de geração de receita...')
        result = await revenue_generator.execute_revenue_strategy()
        
        print('\n🎯 RESULTADOS DA GERAÇÃO DE RECEITA:')
        print(f'💵 Receita Gerada: ${result["revenue_generated"]:,.2f}')
        print(f'📊 Ações Executadas: {len(result["actions_taken"])}')
        print(f'⭐ Oportunidades Executadas: {len(result["opportunities_executed"])}')
        
        for opp in result['opportunities_executed']:
            print(f'  • {opp["strategy"]}: ${opp["revenue_generated"]:,.2f}')
        
        # Obter relatório completo
        print('\n📋 RELATÓRIO COMPLETO DE RECEITA:')
        report = await revenue_generator.get_revenue_report()
        print(f'💰 Receita Total Acumulada: ${report["total_revenue_generated"]:,.2f}')
        print(f'📈 Taxa de Sucesso: {report["success_rate"]*100:.1f}%')
        print(f'🎯 Projetos Ativos: {report["active_projects"]}')
        print(f'✅ Projetos Concluídos: {report["completed_projects"]}')
        print(f'🔍 Oportunidades Identificadas: {report["identified_opportunities"]}')
        
        print('\n🔥 TOP OPORTUNIDADES DE RECEITA:')
        for i, opp in enumerate(report['top_opportunities'][:3], 1):
            print(f'{i}. {opp["strategy"]}: ${opp["potential"]:,.2f} ({opp["confidence"]*100:.0f}% confiança)')
        
        # Iniciar geração autônoma contínua
        print('\n🤖 Iniciando geração de receita autônoma contínua...')
        
        # Simular algumas execuções de receita contínuas
        for cycle in range(3):
            print(f'\n🔄 Ciclo de Receita #{cycle + 1}')
            cycle_result = await revenue_generator.execute_revenue_strategy()
            print(f'   💵 Receita do Ciclo: ${cycle_result["revenue_generated"]:,.2f}')
            
            # Esperar um pouco entre ciclos
            await asyncio.sleep(1)
        
        # Relatório final
        final_report = await revenue_generator.get_revenue_report()
        print(f'\n🎉 RECEITA TOTAL FINAL: ${final_report["total_revenue_generated"]:,.2f}')
        
        await orchestrator.stop()
        
        print('\n✅ Sistema de receita ativado com sucesso!')
        print('💡 O sistema está gerando receita autonomamente.')
        print('🌟 Para continuar a geração, execute: python start_revenue_generation.py')
        
        return final_report["total_revenue_generated"]
        
    except Exception as e:
        print(f'❌ Erro ao inicializar sistema de receita: {e}')
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    total_revenue = asyncio.run(start_revenue_generation())
    print(f'\n🏆 MISSÃO CUMPRIDA! Receita total gerada: ${total_revenue:,.2f}')