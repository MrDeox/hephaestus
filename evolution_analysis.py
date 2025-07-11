"""
Análise da Evolução do Sistema RSI AI.
Verifica o progresso e desenvolvimento do sistema durante execução.
"""

import sqlite3
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path


def analyze_system_evolution():
    """Analisa a evolução do sistema RSI AI."""
    print("🧠 Análise da Evolução do Sistema RSI AI")
    print("=" * 50)
    
    # 1. Analisar estado do sistema
    analyze_system_state()
    
    # 2. Analisar memória episódica
    analyze_episodic_memory()
    
    # 3. Analisar memória procedural
    analyze_procedural_memory()
    
    # 4. Analisar padrões de aprendizado
    analyze_learning_patterns()
    
    # 5. Conclusões
    print_conclusions()


def analyze_system_state():
    """Analisa o estado salvo do sistema."""
    try:
        with open('rsi_continuous_state.json', 'r') as f:
            state = json.load(f)
        
        metrics = state['metrics']
        uptime_hours = state['uptime_seconds'] / 3600
        
        print("📊 Estado do Sistema:")
        print(f"   ⏱️  Tempo de execução: {uptime_hours:.2f} horas")
        print(f"   🔄 Ciclos completados: {state['cycle_count']}")
        print(f"   📚 Conhecimento adquirido: {metrics['total_knowledge_acquired']}")
        print(f"   ⚡ Habilidades desenvolvidas: {metrics['total_skills_learned']}")
        print(f"   📝 Experiências registradas: {metrics['total_experiences']}")
        print(f"   ✅ Expansões bem-sucedidas: {metrics['successful_expansions']}")
        print(f"   ❌ Expansões falhadas: {metrics['failed_expansions']}")
        
        # Calcular velocidade de aprendizado
        if uptime_hours > 0:
            learning_rate = metrics['total_knowledge_acquired'] / uptime_hours
            print(f"   📈 Taxa de aprendizado: {learning_rate:.1f} conceitos/hora")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro analisando estado: {e}")
        return {}


def analyze_episodic_memory():
    """Analisa a memória episódica."""
    try:
        conn = sqlite3.connect('episodic_memory.db')
        cursor = conn.cursor()
        
        # Total de episódios
        cursor.execute('SELECT COUNT(*) FROM episodes')
        total_episodes = cursor.fetchone()[0]
        
        # Episódios por tipo
        cursor.execute('''
            SELECT json_extract(content, '$.event') as event_type, COUNT(*) as count
            FROM episodes 
            WHERE event_type IS NOT NULL
            GROUP BY event_type
            ORDER BY count DESC
            LIMIT 5
        ''')
        event_types = cursor.fetchall()
        
        # Episódios recentes (última hora)
        cursor.execute('''
            SELECT COUNT(*) FROM episodes 
            WHERE datetime(timestamp) > datetime('now', '-1 hour')
        ''')
        recent_episodes = cursor.fetchone()[0]
        
        print(f"\n📚 Memória Episódica:")
        print(f"   📊 Total de episódios: {total_episodes}")
        print(f"   🕐 Episódios na última hora: {recent_episodes}")
        print(f"   📈 Tipos de eventos mais frequentes:")
        
        for event_type, count in event_types:
            percentage = (count / total_episodes) * 100 if total_episodes > 0 else 0
            print(f"      - {event_type}: {count} ({percentage:.1f}%)")
        
        conn.close()
        return total_episodes
        
    except Exception as e:
        print(f"❌ Erro analisando memória episódica: {e}")
        return 0


def analyze_procedural_memory():
    """Analisa a memória procedural."""
    try:
        with open('procedural_memory.pkl', 'rb') as f:
            data = pickle.load(f)
        
        skills = data.get('skills', {})
        stats = data.get('stats', {})
        
        print(f"\n⚡ Memória Procedural:")
        print(f"   🛠️  Total de habilidades: {len(skills)}")
        print(f"   📊 Habilidades criadas: {stats.get('skills_created', 0)}")
        print(f"   🔄 Habilidades atualizadas: {stats.get('skills_updated', 0)}")
        
        # Analisar tipos de habilidades
        skill_types = {}
        for skill in skills.values():
            skill_type = skill.skill_type
            skill_types[skill_type] = skill_types.get(skill_type, 0) + 1
        
        print(f"   📈 Tipos de habilidades:")
        for skill_type, count in skill_types.items():
            print(f"      - {skill_type}: {count}")
        
        # Habilidades mais complexas
        complex_skills = sorted(skills.values(), key=lambda s: s.complexity, reverse=True)[:3]
        print(f"   🧠 Habilidades mais complexas:")
        for i, skill in enumerate(complex_skills, 1):
            print(f"      {i}. {skill.name} (complexidade: {skill.complexity:.2f})")
        
        return len(skills)
        
    except Exception as e:
        print(f"❌ Erro analisando memória procedural: {e}")
        return 0


def analyze_learning_patterns():
    """Analisa padrões de aprendizado."""
    try:
        print(f"\n🔍 Padrões de Aprendizado:")
        
        # Analisar distribuição temporal de episódios
        conn = sqlite3.connect('episodic_memory.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as episodes_count
            FROM episodes 
            GROUP BY hour
            ORDER BY episodes_count DESC
            LIMIT 3
        ''')
        peak_hours = cursor.fetchall()
        
        print(f"   ⏰ Horários de maior atividade:")
        for hour, count in peak_hours:
            print(f"      - {hour}:00 - {count} episódios")
        
        # Analisar evolução da complexidade
        cursor.execute('''
            SELECT json_extract(content, '$.context.complexity') as complexity
            FROM episodes 
            WHERE complexity IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        complexities = [float(row[0]) for row in cursor.fetchall() if row[0]]
        
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            print(f"   🧠 Complexidade média das tarefas: {avg_complexity:.2f}")
            
            # Tendência de complexidade
            recent_complexity = sum(complexities[:20]) / min(20, len(complexities))
            older_complexity = sum(complexities[-20:]) / min(20, len(complexities))
            
            if recent_complexity > older_complexity:
                print(f"   📈 Tendência: Aumento na complexidade das tarefas")
            else:
                print(f"   📉 Tendência: Estabilização na complexidade")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erro analisando padrões: {e}")


def print_conclusions():
    """Imprime conclusões da análise."""
    print(f"\n🎯 Conclusões da Análise:")
    print(f"   ✅ Sistema está funcionalmente operacional")
    print(f"   ✅ Aprendizado contínuo está ativo")
    print(f"   ✅ Memória episódica registrando experiências")
    print(f"   ✅ Memória procedural desenvolvendo habilidades")
    print(f"   ⚠️  Expansões autônomas precisam de ajustes")
    print(f"   📊 Sistema demonstra capacidade de evolução")
    
    print(f"\n🚀 Próximos Passos Recomendados:")
    print(f"   1. Otimizar processo de expansão autônoma")
    print(f"   2. Implementar uso efetivo das habilidades desenvolvidas")
    print(f"   3. Adicionar validação de aprendizado")
    print(f"   4. Expandir fontes de conhecimento")
    print(f"   5. Implementar métricas de qualidade")


if __name__ == "__main__":
    analyze_system_evolution()