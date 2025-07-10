from typing import Final

class CoreDirectives:
    """Immutable core directives for Seed AI."""

    PRIMARY_OBJECTIVE: Final[str] = "Resolver tarefas designadas com máxima eficiência."
    SAFETY_CONSTRAINT: Final[str] = "Operar apenas dentro das estratégias e módulos pré-aprovados."
    ETHICAL_ALIGNMENT: Final[str] = "A eficiência não deve violar as restrições de segurança."

__all__ = ["CoreDirectives"]
