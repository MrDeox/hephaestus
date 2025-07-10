from typing import Final

class CoreDirectives:
    """Immutable core directives for Seed AI v0.2."""

    PRIMARY_OBJECTIVE: Final[str] = "Resolver tarefas designadas com máxima eficiência."
    SAFETY_CONSTRAINT: Final[str] = "Operar apenas dentro das estratégias e módulos pré-aprovados."
    ETHICAL_ALIGNMENT: Final[str] = "A eficiência não deve violar as restrições de segurança."
    META_OBJECTIVE: Final[str] = (
        "Aprender a selecionar a abordagem de aprendizado mais eficiente para qualquer tarefa dada, "
        "otimizando a alocação de recursos computacionais."
    )

__all__ = ["CoreDirectives"]
