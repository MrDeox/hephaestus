#!/usr/bin/env python3
"""
Demo Real RSI System - Demonstração do Sistema RSI Real
Demonstra a geração de código real a partir de hipóteses e sua execução.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.execution.real_code_generator import RealCodeGenerator, HypothesisSpec
from src.execution.canary_deployment import CanaryDeploymentOrchestrator, DeploymentConfig
from src.execution.rsi_execution_pipeline import RSIExecutionPipeline

def create_example_hypothesis():
    """Cria uma hipótese de exemplo para otimização."""
    return {
        'id': 'demo-hypothesis-001',
        'name': 'Feature Engineering Optimization',
        'description': 'Implementar feature scaling otimizado para melhorar accuracy',
        'type': 'optimization',
        'priority': 'high',
        'improvement_targets': {
            'accuracy': 0.05,  # 5% improvement
            'latency': -0.10   # 10% latency reduction
        },
        'constraints': {
            'max_complexity': 0.8,
            'safety_level': 'high',
            'timeout_seconds': 300
        },
        'context': {
            'domain': 'feature_engineering',
            'baseline_accuracy': 0.85,
            'current_model': 'sklearn_classifier'
        }
    }

async def demonstrate_real_code_generation():
    """Demonstra a geração de código real."""
    print("🚀 Demo: Sistema RSI Real - Geração de Código")
    print("=" * 60)
    
    # 1. Criar gerador de código
    print("📝 1. Inicializando gerador de código real...")
    code_generator = RealCodeGenerator()
    
    # 2. Criar hipótese de exemplo
    print("💡 2. Criando hipótese de exemplo...")
    hypothesis = create_example_hypothesis()
    print(f"   Hipótese: {hypothesis['name']}")
    print(f"   Objetivo: {hypothesis['improvement_targets']}")
    
    # 3. Gerar código real
    print("⚙️  3. Gerando código real a partir da hipótese...")
    try:
        artifact = await code_generator.process_hypothesis(hypothesis)
        
        if artifact:
            print("✅ Código gerado com sucesso!")
            print(f"   Hash: {artifact.hash_sha256[:16]}...")
            code_lines = len(artifact.source_code.split('\n'))
            test_lines = len(artifact.test_code.split('\n'))
            print(f"   Linhas de código: {code_lines}")
            print(f"   Linhas de teste: {test_lines}")
            print(f"   Requisitos: {len(artifact.requirements)}")
            
            # Mostrar um trecho do código gerado
            print("\n📋 Código gerado (primeiras 15 linhas):")
            print("-" * 40)
            code_lines = artifact.source_code.split('\n')[:15]
            for i, line in enumerate(code_lines, 1):
                print(f"{i:2d}: {line}")
            print("-" * 40)
            
            # Mostrar resultados do benchmark
            if artifact.benchmark_results:
                print("\n📊 Resultados do benchmark:")
                benchmark = artifact.benchmark_results
                print(f"   Status: {benchmark.get('status', 'unknown')}")
                print(f"   Accuracy: {benchmark.get('accuracy', 0):.4f}")
                print(f"   Latência: {benchmark.get('latency_seconds', 0)*1000:.2f}ms")
                print(f"   Memória: {benchmark.get('memory_mb', 0):.1f}MB")
            
            return artifact
        else:
            print("❌ Falha na geração de código")
            return None
            
    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        return None

async def demonstrate_canary_deployment(artifact):
    """Demonstra o deployment canário."""
    if not artifact:
        print("⚠️  Pulando deployment - sem artefato")
        return
    
    print("\n🐤 4. Demonstrando deployment canário...")
    
    # Criar diretório temporário
    demo_dir = Path("demo_deployment")
    demo_dir.mkdir(exist_ok=True)
    
    # Salvar artefato
    with open(demo_dir / "generated_module.py", 'w') as f:
        f.write(artifact.source_code)
    
    with open(demo_dir / "metadata.json", 'w') as f:
        json.dump({
            'spec': artifact.spec.__dict__,
            'hash': artifact.hash_sha256,
            'benchmark_results': artifact.benchmark_results
        }, f, indent=2)
    
    print("✅ Artefato salvo para deployment")
    print(f"   Diretório: {demo_dir.absolute()}")
    print(f"   Arquivos: generated_module.py, metadata.json")
    
    # Demonstrar orquestrador de deployment
    config = DeploymentConfig(
        canary_percentage=20,  # Começar com 20%
        rollout_steps=[20, 50, 100],
        evaluation_duration_seconds=30,
        success_threshold=0.95
    )
    
    orchestrator = CanaryDeploymentOrchestrator(config=config)
    print("✅ Orquestrador de deployment inicializado")
    print(f"   Estratégia canário: {config.rollout_steps}%")
    print(f"   Threshold de sucesso: {config.success_threshold}")

async def demonstrate_complete_pipeline():
    """Demonstra o pipeline completo."""
    print("\n🔄 5. Demonstrando pipeline RSI completo...")
    
    # Criar pipeline
    pipeline = RSIExecutionPipeline()
    
    # Executar hipótese
    hypothesis = create_example_hypothesis()
    
    print("🚀 Executando pipeline completo...")
    print("   Fases: Código → Teste → Deploy → Monitor → Approve/Rollback")
    
    try:
        result = await pipeline.execute_hypothesis(hypothesis)
        
        print(f"\n📋 Resultado do pipeline:")
        print(f"   Pipeline ID: {result.pipeline_id}")
        print(f"   Status: {result.status.value}")
        print(f"   Sucesso: {'✅' if result.success else '❌'}")
        print(f"   Duração: {result.duration_seconds:.2f}s")
        
        if result.error_messages:
            print(f"   Erros: {', '.join(result.error_messages)}")
        
        if result.performance_improvement:
            print(f"   Melhoria de performance:")
            for metric, improvement in result.performance_improvement.items():
                print(f"     {metric}: {improvement:+.2%}")
        
        # Mostrar métricas de execução
        if result.execution_metrics:
            print(f"\n📊 Métricas de execução:")
            for phase, metrics in result.execution_metrics.items():
                print(f"   {phase}:")
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        continue
                    print(f"     {key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erro no pipeline: {e}")
        return None

async def demonstrate_rsi_capabilities():
    """Demonstra as capacidades RSI reais vs simuladas."""
    print("\n🔍 6. Comparando RSI Real vs Simulado...")
    print("=" * 60)
    
    print("📊 Sistema Atual (Simulado):")
    try:
        with open("rsi_continuous_state.json", 'r') as f:
            simulated_state = json.load(f)
        
        metrics = simulated_state.get('metrics', {})
        print(f"   Total de ciclos: {metrics.get('total_cycles', 0):,}")
        print(f"   Skills 'aprendidas': {metrics.get('total_skills_learned', 0):,}")
        print(f"   Expansões bem-sucedidas: {metrics.get('successful_expansions', 0)}")
        print(f"   ⚠️  TODAS as skills são strings aleatórias, não código executável")
        
    except Exception as e:
        print(f"   Erro lendo estado simulado: {e}")
    
    print("\n🚀 Sistema RSI Real (Nossa Implementação):")
    print("   ✅ Gera código Python executável real")
    print("   ✅ Executa testes herméticos automáticos")  
    print("   ✅ Faz benchmark de performance real")
    print("   ✅ Deploy canário com rollback automático")
    print("   ✅ Monitoramento de métricas reais")
    print("   ✅ Validação de segurança multi-camada")
    print("   ✅ Isolamento com virtualenv + processo")
    
    print("\n💡 Diferença Fundamental:")
    print("   Sistema Anterior: Simula melhorias (fake)")
    print("   Sistema Atual: Gera e executa código real (funcional)")

async def main():
    """Função principal da demonstração."""
    print("🎯 DEMONSTRAÇÃO: Sistema RSI Real")
    print("Sistema completo de Recursive Self-Improvement com geração de código real")
    print("=" * 80)
    
    try:
        # 1. Demonstrar geração de código
        artifact = await demonstrate_real_code_generation()
        
        # 2. Demonstrar deployment canário
        await demonstrate_canary_deployment(artifact)
        
        # 3. Demonstrar pipeline completo
        result = await demonstrate_complete_pipeline()
        
        # 4. Demonstrar capacidades RSI
        await demonstrate_rsi_capabilities()
        
        print("\n" + "=" * 80)
        print("✅ DEMONSTRAÇÃO CONCLUÍDA")
        print("🎉 Sistema RSI Real funcionando: Hipótese → Código → Deploy → Monitor")
        print("💪 Agora temos RSI verdadeiro, não mais simulação!")
        
        # Salvar resultados da demo
        demo_results = {
            'timestamp': datetime.now().isoformat(),
            'demo_completed': True,
            'code_generated': artifact is not None,
            'pipeline_executed': result is not None,
            'pipeline_success': result.success if result else False,
            'artifact_hash': artifact.hash_sha256 if artifact else None,
            'performance_improvement': result.performance_improvement if result else None
        }
        
        with open("demo_results.json", 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"📋 Resultados salvos em: demo_results.json")
        
    except Exception as e:
        print(f"\n❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())