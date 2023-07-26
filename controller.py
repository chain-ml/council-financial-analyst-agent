from typing import List, Tuple
import logging
from string import Template

from council.chains import Chain
from council.runners import Budget
from council.contexts import AgentContext, ScoredChatMessage, ChatMessage, ChatMessageKind
from council.controllers import LLMController, ExecutionUnit
from council.llm import LLMMessage

import constants

logger = logging.getLogger(__name__)


class Controller(LLMController):
    """
    A controller that uses an LLM to decide the execution plan and
    reformulates the user query based on the conversational history.

    Based on LLMController: https://github.com/chain-ml/council/blob/main/council/controllers/llm_controller.py
    """

    def __init__(self, llm, response_threshold):
        """
        Initialize a new instance

        Parameters:
            llm (LLMBase): the instance of LLM to use
            response_threshold (float): a minimum threshold to select a response from its score
        """
        super().__init__(llm, response_threshold)

    def get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
        """Generates an execution plan for the agent based on the provided context, chains, and budget."""
        self.chains = chains
        answer_choices = "\n ".join([f"name: {c.name}, description: {c.description}" for c in chains])
        # Load prompts for LLM and substitute parameters
        system_prompt = "You are an assistant responsible to identify the intent of the user."

        controller_get_plan_prompt = Template("""
        Use the latest user query and the conversational history to identify the intent of the user. 
        Break this task down into 2 subtasks. First perform subtask 1 and then subtask 2.
        
        Context for subtask 1:
        Conversational history:
        $conversational_history
        
        User query: $user_query
        
        Instructions for subtask 1:
        # Use the historical conversation to update the user query to better answer the user question
        # If the query does not need to be updated, do not update the query
        # If there is no conversational history, do not update the query
        # If the conversational history is not relevant to the query, do not update the query
        # See the below examples for how to update the user query
        ************
        Example 1:
        Conversational History:
        User: Who is the CEO of OpenAI?
        Assistant: Sam Altman
        
        User Query: How old is he?
        
        Updated Query: How old is Sam Altman?
        ************
        Example 2:
        Conversational History:
        User: Who is the CEO of OpenAI?
        Assistant: Sam Altman
        
        User Query: What is the price of Bitcoin?
        
        Updated Query: What is the price of Bitcoin?
        ************
        
        Context for subtask 2: 
        Categories are given as a name and a category (name: {name}, description: {description}):
        $answer_choices
        
        Instructions for subtask 2:
        # Use the updated query to identify the intent of the user
        # score categories out of 10 using there description
        # For each category, you will answer with {name};{score}"
        # The updated query should be identical for each category
        # Each response is provided on a new line
        # When no category is relevant, you will answer exactly with 'unknown'
                                        
        Your response should always be formatted like this:
        Subtask 1: {updated_query}
        ---
        Subtask 2:
        {subtask2_results}
        """)

        user_prompt = controller_get_plan_prompt.substitute(
            conversational_history=self.build_chat_history(context),
            user_query=context.chatHistory.last_user_message.message,
            answer_choices=answer_choices,
        )
        # Send messages and receive response from model
        messages = [
            LLMMessage.system_message(system_prompt),
            LLMMessage.user_message(user_prompt),
        ]
        response = self._llm.post_chat_request(messages)[0]
        logger.debug(f"llm response: {response}")
        # Separate reformulated query and chain selection from response
        query_reformulation_result, chain_selection_result = self.parse_response(response)
        # Create execution plan and provide reformulated query to each execution unit as its initial state
        parsed = [self.parse_line(line, chains) for line in chain_selection_result.splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some() and r.unwrap()[1] > self._response_threshold]
        if (filtered is None) or (len(filtered) == 0):
            return []

        filtered.sort(key=lambda item: item[1], reverse=True)
        # List of chain names to be executed by agent in current iteration
        self.chains_current_iteration = [chain.name for chain, _ in filtered]
        result = [
            ExecutionUnit(r[0], budget, initial_state=ChatMessage.chain(query_reformulation_result))
            for r in filtered
            if r is not None
        ]

        return result[: self._top_k]

    def select_responses(self, context: AgentContext) -> List[ScoredChatMessage]:
        """Selects responses from the agent's context."""
        agent_messages = context.evaluationHistory[-1]
        query = context.chatHistory.last_user_message.message
        chain_descriptions = {chain.name: chain.description for chain in self.chains}
        context = ""
        for message in agent_messages:
            # Only aggregate results from chains executed in current iteration
            if message.message.source in self.chains_current_iteration:
                context += f"Source: {chain_descriptions[message.message.source]}\n Response: {message.message.message}\n\n"

        system_prompt = Template(
            "You are a financial analyst whose job is to write a research report answering the user query based on data about $company from different sources."
        ).substitute(company=constants.COMPANY_NAME)

        controller_select_response_prompt = Template("""
        # Instructions
        - The provided context is a list of research data answering the user query from different sources.
        - Combine the following data from muliple sources into a single research report to answer the query.
        - Make sure to highlight any agreements or disagreements between different responses in the final response.
        - Explicitly state from which source different parts of the final response are from.
        
        # Context:
        $context
        
        # Query:
        $query
        
        Answer:
        """)
        
        prompt = controller_select_response_prompt.substitute(
            context=context, query=query
        )

        messages = [
            LLMMessage.system_message(system_prompt),
            LLMMessage.user_message(prompt),
        ]

        llm_response = self._llm.post_chat_request(messages=messages)[0]

        return [ScoredChatMessage(ChatMessage.agent(llm_response), 1.0)]

    @staticmethod
    def parse_response(response: str) -> Tuple[str, str]:
        """Function to separate reformulated query and chain selection from LLM response."""
        query_reformulation_response = (
            response.split("---")[0].replace("Subtask 1:", "").replace("Subtask 1: ", "").strip()
        )
        chain_selection_response = response.split("---")[1].replace("Subtask 2:", "").replace("Subtask 2: ", "").strip()

        return query_reformulation_response, chain_selection_response

    @staticmethod
    def build_chat_history(context: AgentContext, max_history_len: int = 4) -> str:
        """Format the chat history into a string that can be added to the prompt for the query reformulation model."""
        chat_history = ""
        # Remove the user's most recent message from the chat history
        message_history = context.chatHistory.messages[:-1]
        # Return no history if there are less than 2 messages
        if len(message_history) < 1:
            return "No conversational history"

        for msg in message_history[-max_history_len:]:
            if msg.is_of_kind(ChatMessageKind.User):
                chat_history += f"User: {msg.message}\n"
            if msg.is_of_kind(ChatMessageKind.Agent):
                chat_history += f"Assistant: {msg.message}\n"

        return chat_history
