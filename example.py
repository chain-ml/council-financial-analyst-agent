import logging

logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(threadName)s %(name)s:%(funcName)s:%(lineno)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
)
logging.getLogger("council").setLevel(logging.DEBUG)

import dotenv
dotenv.load_dotenv()

from council.agents import Agent
from council.runners import Budget
from council.contexts import AgentContext, ChatHistory

from agent_config import AgentConfig

# Loading all agent configuration into an Agent class 
agent = Agent(**AgentConfig().load_config())
# Initializing context for the Agent
chat_history = ChatHistory()
chat_history.add_user_message("What is the financial performance of Microsoft?")
run_context = AgentContext(chat_history)
# Executing Agent
result = agent.execute(run_context, Budget(600))
print(result.best_message.message)
