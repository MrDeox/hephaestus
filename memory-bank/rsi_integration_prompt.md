# PROMPT DE INTEGRAÇÃO RSI PARA CLAUDE CODE

## CONTEXTO FUNDAMENTAL
Você está trabalhando com o sistema **Hephaestus RSI (Recursive Self-Improvement)** - um sistema de IA de produção com medidas de segurança abrangentes. TODAS as funcionalidades devem ser integradas ao ciclo RSI existente, nunca criadas como peças isoladas.

## ARQUITETURA RSI OBRIGATÓRIA
O sistema segue este pipeline: **Input → Validation → Prediction → Learning → State Update → Monitoring**

### Componentes Core que DEVEM ser respeitados:
- **State Management**: Imutável com pyrsistent
- **Validation**: Pydantic para integridade
- **Safety**: Circuit breakers e sandboxing
- **Learning**: Online learning com River
- **Monitoring**: Detecção de anomalias e telemetria
- **Security**: Múltiplas camadas de sandbox
- **Audit**: Logs criptografados completos

## REGRAS DE INTEGRAÇÃO OBRIGATÓRIAS

### 1. VERIFICAÇÃO PRE-IMPLEMENTAÇÃO
Antes de criar QUALQUER funcionalidade, você DEVE:
```
❓ CHECKLIST DE INTEGRAÇÃO RSI:
□ Como esta funcionalidade se integra ao pipeline RSI?
□ Qual etapa do ciclo (Validation/Prediction/Learning/Monitoring) ela afeta?
□ Como ela contribui para o self-improvement do sistema?
□ Quais validações de segurança ela precisa?
□ Como ela será monitorada para anomalias?
□ Como ela atualizará o estado imutável?
□ Que dados ela gerará para aprendizado?
```

### 2. INTEGRAÇÃO OBRIGATÓRIA AO CICLO RSI
TODA funcionalidade DEVE:
- **Usar o RSIOrchestrator** como ponto central
- **Implementar validação** com SafetyConstraints
- **Gerar métricas** para o sistema de monitoramento
- **Contribuir para aprendizado** online via River
- **Respeitar estado imutável** com pyrsistent
- **Registrar auditoria** completa

### 3. PADRÃO DE IMPLEMENTAÇÃO OBRIGATÓRIO
```python
# EXEMPLO DE INTEGRAÇÃO CORRETA
async def nova_funcionalidade_rsi(orchestrator: RSIOrchestrator, data):
    # 1. VALIDAÇÃO (obrigatória)
    validated_data = await orchestrator.validate_input(data)
    
    # 2. PREDIÇÃO/PROCESSAMENTO (com monitoramento)
    with orchestrator.monitor_operation("nova_funcionalidade"):
        result = await orchestrator.process_with_learning(validated_data)
    
    # 3. APRENDIZADO (atualização do modelo)
    await orchestrator.learn_from_operation(validated_data, result)
    
    # 4. ATUALIZAÇÃO DE ESTADO (imutável)
    new_state = await orchestrator.update_state_immutably(result)
    
    # 5. AUDITORIA (obrigatória)
    await orchestrator.log_operation("nova_funcionalidade", data, result)
    
    return result
```

### 4. PROIBIÇÕES ABSOLUTAS
❌ **NUNCA CRIE:**
- Funcionalidades standalone fora do RSI
- Loops de processamento separados
- Estados mutáveis diretos
- Validações customizadas que bypassem o sistema
- Logs fora do sistema de auditoria
- Execução de código sem sandbox

## DIRETRIZES DE GERAÇÃO DE RENDA INTEGRADA AO RSI

### Funcionalidades de Renda DEVEM:
1. **Usar predições RSI** para otimização
2. **Aprender continuamente** com resultados de receita
3. **Monitorar performance** financeira como métrica RSI
4. **Validar transações** através do sistema de segurança
5. **Evoluir estratégias** baseado no feedback do sistema

### Exemplo de Integração Financeira:
```python
# Geração de renda integrada ao RSI
async def optimize_revenue_strategy(orchestrator):
    # Usa aprendizado RSI para prever melhor estratégia
    market_prediction = await orchestrator.predict({
        "market_features": current_market_data
    })
    
    # Aprende com resultados anteriores
    await orchestrator.learn_from_revenue_data(
        strategy_features, actual_revenue
    )
    
    # Sistema evolui baseado no sucesso financeiro
    return optimized_strategy
```

## VERIFICAÇÃO CONTÍNUA
Após implementar, SEMPRE pergunte:
1. "Esta funcionalidade está realmente integrada ao ciclo RSI?"
2. "O sistema está aprendendo e melhorando com esta funcionalidade?"
3. "Todas as camadas de segurança estão sendo respeitadas?"
4. "A funcionalidade contribui para o self-improvement geral?"

## COMANDO DE VERIFICAÇÃO
Sempre execute esta verificação antes de finalizar:
```bash
# Verifique se a integração está correta
python -c "
from src.main import RSIOrchestrator
# Teste se a nova funcionalidade está no pipeline RSI
# Verifique logs de auditoria
# Confirme aprendizado contínuo
"
```

## LEMBRETE CRÍTICO
🚨 **O objetivo é um SISTEMA COESO que melhora recursivamente, não uma coleção de ferramentas separadas!** 

Cada linha de código deve contribuir para que o sistema RSI se torne mais inteligente, mais seguro e mais eficaz na geração de renda ao longo do tempo.