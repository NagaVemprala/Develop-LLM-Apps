import os
from dotenv import load_dotenv  
from langchain.agents import create_agent
from pydantic import BaseModel, Field
import streamlit as st

# ==========================================
# 0. STREAMLIT APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Interview Prep AI Assistant",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI Agent Interview Prep Assistant")
st.markdown("Provide a target job description below to extract structured evaluation materials, technical skill matrix sheets, and sample mock queries.")

# ==========================================
# 1. ENVIRONMENT CONFIGURATION & BACKEND INIT
# ==========================================
# Load the .env file to set environment variables for API keys and configuration
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

# 2. Construct the agent with its foundational model, system behavior, and tools
@st.cache_resource
def init_agent():
    # Caching ensures the agent graph architecture isn't rebuilt on every widget re-render
    return create_agent(
        model="openai:gpt-4o", # Adjusted string to match common valid core provider model strings
        tools=[],
        response_format=InterviewPrepMaterial,
        system_prompt=(
            "You are a helpful assistant who prepares for job interviews. "
            "Your task is to review the provided job descriptions, extract relevant technical skills for this job"
            "and help me prepare for the interview by providing insights and potential questions based on the job descriptions."
        )
    )

agent = init_agent()

# ==========================================
# 2. STREAMLIT USER INTERFACE (WIDGETS)
# ==========================================
# Provide default text - sample job description
default_jd = """Position: Senior Analytics Engineer
Responsibilities:
- Build and maintain robust data pipelines using dbt (Data Build Tool) and SQL over Snowflake data warehouses.
- Orchestrate data workflows using Apache Airflow and develop custom dashboard tracking using Python (Pandas/NumPy) and Tableau.
- Implement data quality assertions, conduct regression testing on statistical models, and collaborate with data science teams."""

# Input field for user
job_description = st.text_area(
    label="Paste Job Description Here:",
    value=default_jd,
    height=250
)

# Execution trigger
if st.button("Generate Interview Preparation Material", type="primary"):
    if not job_description.strip():
        st.warning("Please provide a valid job description before running analysis.")
    else:
        with st.spinner("Agent running text evaluation and compiling response structural schema..."):
            try:
                # 4. Invoke the agent execution path with the live UI input text
                input_state = {
                    "messages": [
                        {
                            "role": "user", 
                            "content": f"Please process this job description and format the preparation material:\n\n{job_description}"
                        }
                    ]
                }
                
                result = agent.invoke(input_state)
                
                # Fetch structured response metadata or model message content
                # Note: LangChain 1.x create_agent maps response_formats natively into structured_response data types or validated structures
                final_message = result["messages"][-1]
                
                # Render results in structured containers/tabs for clean viewing
                st.success("Analysis Complete!")
                st.divider()
                
                # Using columns or tabs to display the data cleanly
                tab1, tab2, tab3 = st.tabs([
                    "🛠️ Technical Skills List", 
                    "💡 Key Role Insights", 
                    "❓ Potential Interview Questions"
                ])
                
                # Parse out the Pydantic instance from the agent output
                # Depending on platform runtime defaults, if content is an instance or JSON string:
                if hasattr(final_message, "structured_response") and final_message.structured_response:
                    data = final_message.structured_response
                elif isinstance(final_message.content, InterviewPrepMaterial):
                    data = final_message.content
                else:
                    # Fallback structural parse if returned as raw string format
                    import json
                    parsed_json = json.loads(final_message.content)
                    data = InterviewPrepMaterial(**parsed_json)
                
                with tab1:
                    st.subheader("Required Technical Proficiencies")
                    for skill in data.technical_skills:
                        st.markdown(f"- `{skill}`")
                        
                with tab2:
                    st.subheader("Strategic Role Insights")
                    for insight in data.key_insights:
                        st.info(insight)
                        
                with tab3:
                    st.subheader("Targeted Preparation Questions")
                    for i, question in enumerate(data.potential_questions, 1):
                        st.markdown(f"**Q{i}:** {question}")
                        
            except Exception as e:
                st.error(f"An processing error occurred: {e}")