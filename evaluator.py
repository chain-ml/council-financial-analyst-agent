from typing import List

from council.contexts import (
    AgentContext,
    ScoredChatMessage,
    ChatMessage,
)
from council.runners.budget import Budget
from council.evaluators import EvaluatorBase


class Evaluator(EvaluatorBase):
    """
    A BasicEvaluator that filters for chains that executed successfully and carries along the chain name in the source of the returned ScoredChatMessage.
    """
    def execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        result = []
        for chain_name, chain_history in context.chainHistory.items():
            chain_result = chain_history[-1].messages[-1]
            if chain_result.is_kind_skill and chain_result.is_ok:
                result.append(
                    ScoredChatMessage(
                        ChatMessage.agent(
                            message=chain_result.message, data=None, source=chain_name, is_error=chain_result.is_error
                        ),
                        1.0,
                    )
                )
        return result
