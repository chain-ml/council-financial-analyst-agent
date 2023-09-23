from string import Template
from typing import List

from council.contexts import AgentContext, ScoredChatMessage, LLMContext, ChatMessage
from council.filters import FilterBase
from council.llm import LLMBase, LLMMessage

import constants


class LLMFilter(FilterBase):
    def __init__(self, llm: LLMBase):
        super().__init__()
        self._llm = self.new_monitor("llm", llm)

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """Selects responses from the agent's context."""
        messages = self._build_llm_messages(context)
        llm_response = self._llm.inner.post_chat_request(
            LLMContext.from_context(context, self._llm), messages=messages
        )

        return [ScoredChatMessage(ChatMessage.agent(llm_response.first_choice), 1.0)]

    def _build_llm_messages(self, context: AgentContext) -> List[LLMMessage]:
        agent_messages = list(context.evaluation)
        query = context.chat_history.last_user_message.message
        context = ""
        for message in agent_messages:
            context += f"Response: {message.message.message}\n\n"

        select_response_prompt = Template(
            """
        # Instructions
        - The provided context is a list of research data answering the user query from different sources.
        - Combine the following data from multiple sources into a single research report to answer the query.
        - Make sure to highlight any agreements or disagreements between different responses in the final response.
        - Explicitly state from which source different parts of the final response are from.
        
        # Context:
        $context
        
        # Query:
        $query
        
        Answer:
        """
        )
        prompt = select_response_prompt.substitute(context=context, query=query)
        return [
            self._build_system_prompt(company=constants.COMPANY_NAME),
            LLMMessage.user_message(prompt),
        ]

    def _build_system_prompt(self, company: str) -> LLMMessage:
        system_prompt = Template(
            "You are a financial analyst whose job is to write a research report answering the user query based on data about $company from different sources."
        ).substitute(company=company)
        return LLMMessage.system_message(system_prompt)
