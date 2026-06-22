from langchain.agents import create_agent
from langchain_core.tools import tool

import os
# 0. Load the .env file to set environment variables for API keys and configuration
from dotenv import load_dotenv  
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Define custom analytical tools for the agent harness
@tool
def calculate_safety_stock(lead_time_days: int, demand_std_dev: float) -> float:
    """Calculates safety stock level required given standard deviation and lead time."""
    import math
    z_score = 1.65 # 95% service level standard
    return round(z_score * demand_std_dev * math.sqrt(lead_time_days), 2)

tools = [calculate_safety_stock]

# 2. Construct the agent with its foundational model, system behavior, and tools
agent = create_agent(
    model="gpt-5.4-nano",
    #model_provider="openai",
    #api_key=OPENAI_API_KEY,
    tools=tools,
    system_prompt="You are an operations intelligence assistant. Use your tools to evaluate metric inputs."
)

# 3. Invoke the agent execution path with a target inquiry
input_state = {
    "messages": ["Calculate our safety stock if lead time is 9 days and demand standard deviation is 15 units."]
}

result = agent.invoke(input_state)
print(result["messages"][-1].content)
