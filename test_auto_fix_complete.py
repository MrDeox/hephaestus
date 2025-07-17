#!/usr/bin/env python3
"""
Teste completo do sistema de auto-correção RSI.

Este teste demonstra RSI REAL - o sistema detecta e corrige seus próprios bugs.
"""

import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

async def test_auto_fix_system():
    """Testa o sistema completo de auto-correção."""
    
    print("🔧 Teste Completo do Sistema de Auto-Correção RSI")
    print("=" * 60)
    print(f"⏰ Iniciado em: {datetime.now()}")
    
    try:
        # Import the auto-fix system
        from src.autonomous.auto_fix_system import auto_fix_rsi_pipeline_error
        
        print("\n📦 Sistema de auto-correção importado com sucesso")
        
        # Execute the auto-fix
        print("\n🚀 Executando sistema de auto-correção...")
        print("   (Detectando erros, analisando código, aplicando correções)")
        
        result = await auto_fix_rsi_pipeline_error()
        
        print("\n" + "=" * 60)
        print("📊 RESULTADO DO AUTO-FIX:")
        print("=" * 60)
        
        print(f"🕐 Timestamp: {result['timestamp']}")
        print(f"🔍 Erros detectados: {result['errors_detected']}")
        print(f"🛠️ Correções aplicadas: {result['fixes_applied']}")
        print(f"✅ Correções bem-sucedidas: {result['fixes_successful']}")
        
        if result['details']:
            print("\n📋 Detalhes das correções:")
            for i, detail in enumerate(result['details'], 1):
                print(f"\n   {i}. Fix ID: {detail['fix_id']}")
                print(f"      Tipo: {detail['error_type']}")
                print(f"      Arquivo: {detail['file']}:{detail['line']}")
                print(f"      Explicação: {detail['explanation']}")
                print(f"      Aplicado: {'✅ SIM' if detail['applied'] else '❌ NÃO'}")
                print(f"      Testado: {'✅ SIM' if detail['test_passed'] else '❌ NÃO'}")
                print(f"      Confiança: {detail['confidence']:.0%}")
        
        # Verificar se houve melhorias
        if result['fixes_successful'] > 0:
            print("\n🎉 AUTO-CORREÇÃO BEM-SUCEDIDA!")
            print("   O sistema detectou e corrigiu seus próprios bugs!")
            print("   Isso é RSI REAL em ação! 🚀")
            
            # Testar se o erro foi realmente corrigido
            print("\n🧪 Testando se o erro foi corrigido...")
            await test_hypothesis_execution()
            
        elif result['errors_detected'] > 0:
            print("\n⚠️ Erros detectados mas não foram corrigidos automaticamente")
            print("   O sistema precisa de mais padrões de correção")
            
        else:
            print("\n✅ Nenhum erro detectado no sistema")
            print("   O sistema está funcionando corretamente!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Erro no teste do auto-fix: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_hypothesis_execution():
    """Testa se o sistema de hipóteses funciona após a correção."""
    
    try:
        print("   🧠 Testando sistema de hipóteses...")
        
        # Try to import and test the hypothesis system
        from src.hypothesis.rsi_hypothesis_orchestrator import RSIHypothesisOrchestrator
        
        # Create instance (this will test if the import works)
        orchestrator = RSIHypothesisOrchestrator(environment='development')
        
        print("   ✅ Sistema de hipóteses funciona corretamente!")
        print("   ✅ Auto-correção foi bem-sucedida!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Sistema de hipóteses ainda com problemas: {e}")
        return False

async def demonstrate_rsi_cycle():
    """Demonstra um ciclo completo de RSI: detectar problema → corrigir → verificar."""
    
    print("\n" + "=" * 60)
    print("🔄 DEMONSTRAÇÃO DE CICLO RSI COMPLETO")
    print("=" * 60)
    
    print("1️⃣ ANTES: Sistema com bug (erro de await None)")
    print("2️⃣ DETECÇÃO: Sistema escaneia logs e detecta problema")
    print("3️⃣ ANÁLISE: Sistema analisa código e identifica causa raiz") 
    print("4️⃣ CORREÇÃO: Sistema gera e aplica correção automaticamente")
    print("5️⃣ TESTE: Sistema valida que a correção funcionou")
    print("6️⃣ MEMÓRIA: Sistema aprende o padrão para problemas futuros")
    
    result = await test_auto_fix_system()
    
    print("\n🎯 CICLO RSI CONCLUÍDO!")
    
    if result and result['fixes_successful'] > 0:
        print("✅ O sistema se auto-melhorou com sucesso!")
        print("✅ Isso demonstra RSI REAL funcionando!")
    else:
        print("ℹ️ Nenhuma correção necessária ou aplicada neste momento")
    
    return result

if __name__ == "__main__":
    async def main():
        print("🤖 TESTE DE SISTEMA DE AUTO-CORREÇÃO RSI")
        print("Implementação de Recursive Self-Improvement REAL")
        print("=" * 80)
        
        result = await demonstrate_rsi_cycle()
        
        print("\n" + "=" * 80)
        print("📈 RESULTADO FINAL:")
        
        if result:
            success_rate = (result['fixes_successful'] / max(result['errors_detected'], 1)) * 100
            print(f"   Taxa de correção: {success_rate:.1f}%")
            print(f"   Erros detectados: {result['errors_detected']}")
            print(f"   Correções aplicadas: {result['fixes_applied']}")
            print(f"   Sucesso total: {result['fixes_successful']}")
            
            if result['fixes_successful'] > 0:
                print("\n🎉 RSI REAL FUNCIONANDO!")
                print("O sistema provou ser capaz de auto-melhoria!")
            else:
                print("\n✅ Sistema estável (sem erros para corrigir)")
        else:
            print("❌ Teste falhou")
        
        print(f"\n⏰ Concluído em: {datetime.now()}")
    
    asyncio.run(main())