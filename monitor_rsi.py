#!/usr/bin/env python3
"""
Monitor RSI System - Script para monitorar o sistema RSI real em execução
"""

import time
import requests
import json
from datetime import datetime
import subprocess
import os

def check_process_running():
    """Verifica se o processo RSI está rodando"""
    try:
        with open('rsi_system.pid', 'r') as f:
            pid = f.read().strip()
        
        result = subprocess.run(['ps', '-p', pid], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def get_system_health():
    """Obtém status de health do sistema"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        return response.json()
    except:
        return {"status": "unreachable"}

def count_rsi_cycles():
    """Conta quantos ciclos RSI foram executados"""
    try:
        result = subprocess.run(['grep', '-c', 'Real RSI Cycle', 'rsi_system.log'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 0

def count_improvements():
    """Conta quantas melhorias foram aplicadas"""
    try:
        result = subprocess.run(['grep', '-c', 'improvement applied', 'rsi_system.log'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 0

def get_latest_activity():
    """Obtém as últimas 3 linhas de atividade RSI"""
    try:
        result = subprocess.run(['grep', 'Real RSI Cycle\\|improvement.*applied\\|🔧\\|✅.*Cycle.*completed', 'rsi_system.log'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        return lines[-3:] if lines else []
    except:
        return []

def main():
    print("🔍 RSI System Monitor - Iniciando monitoramento...")
    print("=" * 80)
    
    start_time = datetime.now()
    
    while True:
        current_time = datetime.now()
        uptime = current_time - start_time
        
        # Status do processo
        process_running = check_process_running()
        
        # Health do sistema
        health = get_system_health()
        
        # Contadores
        cycles = count_rsi_cycles()
        improvements = count_improvements()
        
        # Atividade recente
        recent_activity = get_latest_activity()
        
        # Clear screen e mostrar status
        os.system('clear')
        
        print("🚀 RSI SYSTEM MONITOR")
        print("=" * 80)
        print(f"⏰ Tempo de monitoramento: {uptime}")
        print(f"📊 Status do processo: {'🟢 RODANDO' if process_running else '🔴 PARADO'}")
        
        if health.get('status') == 'healthy':
            print(f"💚 Sistema: SAUDÁVEL")
            print(f"🧠 Metacognitive awareness: {health.get('metacognitive_status', {}).get('metacognitive_awareness', 0)*100:.1f}%")
            print(f"⚡ Learning efficiency: {health.get('metacognitive_status', {}).get('learning_efficiency', 0)*100:.1f}%")
            print(f"🛡️ Safety score: {health.get('metacognitive_status', {}).get('safety_score', 0)*100:.1f}%")
        else:
            print(f"❌ Sistema: {health.get('status', 'UNKNOWN')}")
        
        print(f"🔄 Total de ciclos RSI: {cycles}")
        print(f"✅ Total de melhorias aplicadas: {improvements}")
        print()
        
        if recent_activity:
            print("📝 Atividade recente:")
            for activity in recent_activity:
                if activity.strip():
                    # Extrair timestamp e mensagem
                    parts = activity.split(' | ')
                    if len(parts) >= 3:
                        timestamp = parts[0]
                        level = parts[1]
                        message = ' | '.join(parts[2:])
                        print(f"  {timestamp} - {message}")
        else:
            print("📝 Nenhuma atividade RSI detectada ainda")
        
        print()
        print("💡 Dica: Os ciclos RSI executam a cada 5 minutos (300s)")
        print("🔄 Próxima atualização em 30 segundos...")
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n👋 Monitor interrompido pelo usuário")
            break

if __name__ == "__main__":
    main()