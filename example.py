import logging

logging.basicConfig(
    format="[%(asctime)s %(levelname)s %(threadName)s %(name)s:%(funcName)s:%(lineno)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
)
logging.getLogger("council").setLevel(logging.DEBUG)

import dotenv

dotenv.load_dotenv()

from council.agents import Agent
from council.contexts import AgentContext, Budget

from agent_config import AgentConfig

# Loading all agent configuration into an Agent class
agent = Agent(**AgentConfig().load_config())
# Initializing context for the Agent
run_context = AgentContext.from_user_message(
    "What is the financial performance of Microsoft?", budget=Budget(600)
)
# Executing Agent
result = agent.execute(run_context)
print(f"\nresult:\n{result.best_message.message}")
print(f"\nexecution log:\n{run_context._execution_context._executionLog.to_json()}")
