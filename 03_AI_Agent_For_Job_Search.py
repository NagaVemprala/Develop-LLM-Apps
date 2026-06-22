import os
from dotenv import load_dotenv  
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 0. Load the .env file to set environment variables for API keys and configuration
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Define the desired structured output format using Pydantic
class InterviewPrepMaterial(BaseModel):
    technical_skills: list[str] = Field(
        description="A clean list of specific technical skills, tools, or methodologies required in the job description."
    )
    key_insights: list[str] = Field(
        description="Key takeaways or deep insights about what this role prioritizes based on the text."
    )
    potential_questions: list[str] = Field(
        description="Predictive behavioral or technical interview questions the candidate should prepare for."
    )

tools = [InterviewPrepMaterial]

# 2. Construct the agent with its foundational model, system behavior, and tools
# Note: Ensure you use a valid, active model string like "openai:gpt-4o" or similar available endpoints
agent = create_agent(
    model="gpt-5.4-nano",
    tools=[],
    response_format=InterviewPrepMaterial,
    system_prompt=(
        "You are a helpful assistant who prepares for job interviews. "
        "Your task is to review the provided job descriptions, extract relevant technical skills for this job"
        "and help me prepare for the interview by providing insights and potential questions based on the job descriptions."
    )
)

# 3. Define sample job descriptions containing hidden operations parameters
job_description = """
Position: Senior Analytics Engineer
Responsibilities:
- Build and maintain robust data pipelines using dbt (Data Build Tool) and SQL over Snowflake data warehouses.
- Orchestrate data workflows using Apache Airflow and develop custom dashboard tracking using Python (Pandas/NumPy) and Tableau.
- Implement data quality assertions, conduct regression testing on statistical models, and collaborate with data science teams.
"""

# 4. Invoke the agent execution path with a target inquiry embedding the documents
input_state = {
    "messages": [
        {
            "role": "user", 
            "content": f"Please process this job description and format the preparation material:\n\n{job_description}"
        }
    ]
}

result = agent.invoke(input_state)

# 5. Print out the final reasoned analysis and calculations from the agent
print(result["messages"][-1].content)