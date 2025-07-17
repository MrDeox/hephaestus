#!/usr/bin/env python3
"""
Teste do sistema de auto-correção com erro simulado.
Vamos criar um erro real para ver o sistema corrigir.
"""

import asyncio
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

async def create_test_error():
    """Cria um erro real no código para testar a auto-correção."""
    
    print("🎭 Criando erro simulado para testar auto-correção...")
    
    # Lê o arquivo atual
    target_file = "/home/arthur/projects/hephaestus/src/objectives/revenue_generation.py"
    
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Verifica se já tem o erro
    if "generate_hypotheses" in content:
        print("   ✅ Erro 'generate_hypotheses' já existe no código")
        return True
    else:
        print("   ⚠️ Erro já foi corrigido anteriormente ou não existe")
        print("   Vamos verificar os logs para erros históricos...")
        return False

async def test_auto_fix_with_real_error():
    """Testa o auto-fix com um erro real."""
    
    print("🔧 Teste de Auto-Correção com Erro Real")
    print("=" * 50)
    
    try:
        # Verificar se existe erro para corrigir
        has_error = await create_test_error()
        
        # Importar o sistema
        from src.autonomous.auto_fix_system import auto_fix_rsi_pipeline_error
        
        print("\n🚀 Executando sistema de auto-correção...")
        
        # Executar auto-fix
        result = await auto_fix_rsi_pipeline_error()
        
        print("\n" + "=" * 50)
        print("📊 RESULTADO:")
        print(f"🔍 Erros detectados: {result['errors_detected']}")
        print(f"🛠️ Correções aplicadas: {result['fixes_applied']}")
        print(f"✅ Correções bem-sucedidas: {result['fixes_successful']}")
        
        if result['details']:
            print("\n📋 Detalhes:")
            for detail in result['details']:
                print(f"   • {detail['explanation']}")
                print(f"     Arquivo: {detail['file']}:{detail['line']}")
                print(f"     Aplicado: {'✅' if detail['applied'] else '❌'}")
                print(f"     Testado: {'✅' if detail['test_passed'] else '❌'}")
        
        # Teste final
        if result['fixes_successful'] > 0:
            print("\n🎉 AUTO-CORREÇÃO BEM-SUCEDIDA!")
            print("🚀 RSI REAL em ação!")
        elif result['errors_detected'] > 0:
            print("\n⚠️ Erros detectados mas não corrigidos")
        else:
            print("\n✅ Sistema estável")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None

async def manual_error_injection():
    """Injeta um erro manualmente para testar a correção."""
    
    print("\n🧪 Teste com Injeção Manual de Erro")
    print("=" * 40)
    
    try:
        # Cria um arquivo de teste com erro
        test_file = "/tmp/test_rsi_error.py"
        
        error_code = '''
async def test_function():
    # Este código tem um erro de await None
    result = await hypothesis_orchestrator.generate_hypotheses(
        targets={"improvement": 0.1}
    )
    return result
'''
        
        with open(test_file, 'w') as f:
            f.write(error_code)
        
        print(f"   📝 Arquivo de teste criado: {test_file}")
        
        # Cria um log simulado com o erro
        log_content = f'''
2025-07-16T20:00:00.000000-0300 | ERROR | ❌ Erro no pipeline RSI: object NoneType can't be used in 'await' expression
File "{test_file}", line 4, in test_function
    result = await hypothesis_orchestrator.generate_hypotheses(
'''
        
        test_log = "/tmp/test_error.log"
        with open(test_log, 'w') as f:
            f.write(log_content)
        
        print(f"   📋 Log de erro criado: {test_log}")
        
        # Configura o sistema para usar nosso log de teste
        from src.autonomous.auto_fix_system import AutoFixSystem
        
        system = AutoFixSystem()
        system.log_paths = [test_log]  # Usar nosso log de teste
        
        print("\n🚀 Executando auto-fix no arquivo de teste...")
        result = await system.auto_fix_rsi_pipeline_error()
        
        print(f"\n📊 Resultado:")
        print(f"   Erros detectados: {result['errors_detected']}")
        print(f"   Correções aplicadas: {result['fixes_applied']}")
        
        if result['errors_detected'] > 0:
            print("\n✅ Sistema detectou erro injetado!")
            
            if result['fixes_applied'] > 0:
                print("✅ Sistema aplicou correções!")
                
                # Verificar se o arquivo foi corrigido
                with open(test_file, 'r') as f:
                    corrected_content = f.read()
                
                print("\n📄 Código corrigido:")
                print(corrected_content)
                
                if "orchestrate_hypothesis_lifecycle" in corrected_content:
                    print("🎉 CORREÇÃO BEM-SUCEDIDA!")
                    print("O sistema substituiu 'generate_hypotheses' pelo método correto!")
            else:
                print("⚠️ Sistema detectou mas não corrigiu")
        else:
            print("❌ Sistema não detectou o erro injetado")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        Path(test_log).unlink(missing_ok=True)
        
        return result
        
    except Exception as e:
        print(f"❌ Erro no teste manual: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    async def main():
        print("🤖 TESTE AVANÇADO DE AUTO-CORREÇÃO RSI")
        print("Testando capacidade real de auto-melhoria")
        print("=" * 60)
        
        # Teste 1: Verificar erros reais no sistema
        print("\n1️⃣ TESTE COM ERROS REAIS DO SISTEMA:")
        result1 = await test_auto_fix_with_real_error()
        
        # Teste 2: Injeção manual de erro
        print("\n2️⃣ TESTE COM INJEÇÃO MANUAL DE ERRO:")
        result2 = await manual_error_injection()
        
        print("\n" + "=" * 60)
        print("🎯 RESUMO FINAL:")
        
        if result1 and result1['fixes_successful'] > 0:
            print("✅ Teste 1: Sistema corrigiu erros reais!")
        elif result1 and result1['errors_detected'] == 0:
            print("ℹ️ Teste 1: Sistema está limpo (sem erros)")
        else:
            print("⚠️ Teste 1: Erros detectados mas não corrigidos")
        
        if result2 and result2['fixes_successful'] > 0:
            print("✅ Teste 2: Sistema corrigiu erro injetado!")
            print("🚀 RSI REAL COMPROVADO!")
        elif result2 and result2['errors_detected'] > 0:
            print("⚠️ Teste 2: Detectou mas não corrigiu erro injetado")
        else:
            print("❌ Teste 2: Falhou na detecção")
        
        # Conclusão
        total_fixes = 0
        if result1:
            total_fixes += result1['fixes_successful']
        if result2:
            total_fixes += result2['fixes_successful']
        
        if total_fixes > 0:
            print(f"\n🎉 SUCESSO TOTAL: {total_fixes} correções aplicadas!")
            print("O sistema demonstrou capacidade de RSI REAL!")
        else:
            print("\n📊 Sistema funcionando mas sem correções necessárias")
        
        print(f"\n⏰ Concluído em: {datetime.now()}")
    
    asyncio.run(main())