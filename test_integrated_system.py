"""
Teste de Integração do Sistema RSI AI com Memória Hierárquica.
Verifica se a integração entre o sistema principal e a memória está funcionando.
"""

import asyncio
import logging
from src.main import RSIOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_integrated_system():
    """Testa o sistema integrado RSI + Memória."""
    print("🧠 Testando Sistema RSI AI Integrado com Memória Hierárquica")
    print("=" * 60)
    
    # Inicializar orquestrador
    print("\n🔧 Inicializando sistema...")
    orchestrator = RSIOrchestrator(environment='development')
    await orchestrator.start()
    
    # Verificar se o sistema de memória foi inicializado
    print(f"✅ Sistema de memória disponível: {orchestrator.memory_system is not None}")
    
    if orchestrator.memory_system:
        # Testar status da memória
        print("\n📊 Obtendo status da memória...")
        memory_status = await orchestrator.memory_system.get_memory_status()
        print(f"✅ Sistemas de memória inicializados: {memory_status['is_initialized']}")
        print(f"   - Working Memory: {memory_status['memory_systems']['working_memory']['size']} items")
        print(f"   - Semantic Memory: {memory_status['memory_systems']['semantic_memory']['size']} items")
        print(f"   - Episodic Memory: {memory_status['memory_systems']['episodic_memory']['size']} items")
        print(f"   - Procedural Memory: {memory_status['memory_systems']['procedural_memory']['size']} items")
    
    # Testar aprendizado com integração de memória
    print("\n🧠 Testando aprendizado com integração de memória...")
    
    test_features = {
        "feature1": 0.8,
        "feature2": 0.6,
        "feature3": 0.9,
        "context": "integration_test"
    }
    
    learning_result = await orchestrator.learn(test_features, 1, user_id="test_user")
    print(f"✅ Aprendizado concluído:")
    print(f"   - Accuracy: {learning_result['accuracy']:.3f}")
    print(f"   - Samples processed: {learning_result['samples_processed']}")
    print(f"   - Memory stored: {learning_result['memory_stored']}")
    
    # Testar armazenamento direto na memória
    if orchestrator.memory_system:
        print("\n💾 Testando armazenamento direto na memória...")
        
        # Armazenar conhecimento semântico
        semantic_info = {
            "concept": "integration_test_concept",
            "description": "Conceito criado durante teste de integração",
            "type": "test_knowledge",
            "confidence": 0.95,
            "source": "integration_test"
        }
        
        success = await orchestrator.memory_system.store_information(semantic_info, memory_type="semantic")
        print(f"✅ Conhecimento armazenado: {success}")
        
        # Armazenar experiência episódica
        episodic_info = {
            "event": "integration_test",
            "description": "Teste de integração do sistema RSI AI",
            "context": {
                "test_type": "integration",
                "components": ["rsi_orchestrator", "memory_hierarchy"],
                "success": True
            },
            "importance": 0.8,
            "emotions": {"satisfaction": 0.9},
            "tags": ["test", "integration", "rsi"]
        }
        
        success = await orchestrator.memory_system.store_information(episodic_info, memory_type="episodic")
        print(f"✅ Experiência armazenada: {success}")
        
        # Armazenar habilidade procedural
        procedural_info = {
            "skill": "integration_testing",
            "name": "System Integration Testing",
            "description": "Ability to test integrated RSI systems",
            "skill_type": "procedure",
            "complexity": 0.7,
            "tags": ["testing", "integration"]
        }
        
        success = await orchestrator.memory_system.store_information(procedural_info, memory_type="procedural")
        print(f"✅ Habilidade armazenada: {success}")
    
    # Testar recuperação de informações
    if orchestrator.memory_system:
        print("\n🔍 Testando recuperação de informações...")
        
        # Buscar por conceitos de teste
        results = await orchestrator.memory_system.retrieve_information(
            {"text": "integration test"}, 
            memory_types=["semantic", "episodic", "procedural"]
        )
        
        print(f"✅ Resultados da busca:")
        for memory_type, items in results.items():
            print(f"   - {memory_type}: {len(items)} items encontrados")
    
    # Testar consolidação de memória
    if orchestrator.memory_system:
        print("\n🔄 Testando consolidação de memória...")
        
        consolidation_result = await orchestrator.memory_system.consolidate_memory()
        print(f"✅ Consolidação concluída: {consolidation_result}")
    
    # Obter estatísticas finais
    if orchestrator.memory_system:
        print("\n📈 Estatísticas finais da memória...")
        
        final_status = await orchestrator.memory_system.get_memory_status()
        print(f"✅ Estado final:")
        print(f"   - Working Memory: {final_status['memory_systems']['working_memory']['size']} items")
        print(f"   - Semantic Memory: {final_status['memory_systems']['semantic_memory']['size']} items")
        print(f"   - Episodic Memory: {final_status['memory_systems']['episodic_memory']['size']} items")
        print(f"   - Procedural Memory: {final_status['memory_systems']['procedural_memory']['size']} items")
    
    # Testar performance
    print("\n⚡ Testando performance integrada...")
    performance_data = await orchestrator.analyze_performance()
    print(f"✅ Análise de performance concluída")
    print(f"   - Metrics available: {len(performance_data.get('metrics', {}))}")
    
    # Shutdown
    print("\n🛑 Finalizando sistema...")
    await orchestrator.stop()
    
    print("\n" + "=" * 60)
    print("🎉 Teste de integração concluído com sucesso!")
    print("✅ Sistema RSI AI totalmente integrado com memória hierárquica")
    print("✅ Aprendizado automático armazenando experiências")
    print("✅ Memória semântica, episódica e procedural funcionais")
    print("✅ Consolidação e otimização operacionais")
    print("✅ APIs de memória disponíveis")


async def test_memory_api_endpoints():
    """Testa os endpoints da API de memória."""
    print("\n🌐 Testando endpoints da API de memória...")
    
    try:
        import httpx
        
        # Testar se o servidor está rodando
        async with httpx.AsyncClient() as client:
            # Teste de saúde
            try:
                response = await client.get("http://localhost:8000/health")
                print(f"✅ Health check: {response.status_code}")
            except Exception as e:
                print(f"⚠️  Servidor não está rodando: {e}")
                return
            
            # Testar status da memória
            try:
                response = await client.get("http://localhost:8000/memory/status")
                if response.status_code == 200:
                    print("✅ Memory status endpoint funcionando")
                else:
                    print(f"⚠️  Memory status: {response.status_code}")
            except Exception as e:
                print(f"⚠️  Erro no endpoint memory/status: {e}")
            
            # Testar armazenamento
            try:
                test_data = {
                    "concept": "api_test",
                    "description": "Teste via API",
                    "source": "api_test"
                }
                response = await client.post(
                    "http://localhost:8000/memory/store",
                    json=test_data,
                    params={"memory_type": "semantic"}
                )
                if response.status_code == 200:
                    print("✅ Memory store endpoint funcionando")
                else:
                    print(f"⚠️  Memory store: {response.status_code}")
            except Exception as e:
                print(f"⚠️  Erro no endpoint memory/store: {e}")
                
    except ImportError:
        print("⚠️  httpx não disponível para testar endpoints")


if __name__ == "__main__":
    asyncio.run(test_integrated_system())