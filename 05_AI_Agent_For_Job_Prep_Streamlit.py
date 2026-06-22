import os
import json
import csv
from io import StringIO
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

st.title("💼 AI Agent Interview Prep & Practice Assistant")
st.markdown("Extract job metrics, simulate interactive interview prompts, and export your graded history to a CSV file.")

# ==========================================
# 1. ENVIRONMENT CONFIGURATION & BACKEND INIT
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

class ResponseEvaluation(BaseModel):
    score: int = Field(description="Score from 1 to 10 evaluating the quality and completeness of the candidate's answer.")
    feedback: str = Field(description="Constructive critique detailing what was good and what was missing.")
    better_answer: str = Field(description="An exemplar, high-quality response the user could have given.")

@st.cache_resource
def init_agents():
    prep_agent = create_agent(
        model="openai:gpt-4o", 
        tools=[],
        response_format=InterviewPrepMaterial,
        system_prompt=(
            "You are a helpful assistant who prepares for job interviews. "
            "Your task is to review the provided job descriptions, extract relevant technical skills for this job "
            "and help me prepare for the interview by providing insights and potential questions."
        )
    )
    
    eval_agent = create_agent(
        model="openai:gpt-4o",
        tools=[],
        response_format=ResponseEvaluation,
        system_prompt=(
            "You are an expert interviewer and technical recruiter. Your task is to evaluate the user's answer. "
            "Rate their response objectively on a scale of 1-10, provide constructive feedback, "
            "and craft a polished, exemplar 'better answer' tailored to the target skill."
        )
    )
    return prep_agent, eval_agent

prep_agent, eval_agent = init_agents()

# Initialize session states
if "prep_data" not in st.session_state:
    st.session_state.prep_data = None
if "selected_skill" not in st.session_state:
    st.session_state.selected_skill = None
if "generated_question" not in st.session_state:
    st.session_state.generated_question = None
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None
if "history" not in st.session_state:
    st.session_state.history = []  # List of dicts to store: skill, question, response, score, feedback
if "question_counter" not in st.session_state:
    st.session_state.question_counter = 1

# Helper function to generate questions dynamically
def get_new_question_text(skill, counter):
    # Appending a counter version forces variation across repeated clicks on the same skill asset
    if counter == 1:
        return f"Can you describe your experience working with {skill}? Provide a specific scenario detailing a challenge you faced and how you resolved it."
    elif counter == 2:
        return f"Walk me through a technical bottleneck or unique architecture choice you encountered while implementing {skill}."
    else:
        return f"How do you ensure best practices, testing standards, or scale optimization when managing workflows built on {skill}?"

# ==========================================
# 2. STREAMLIT USER INTERFACE (WIDGETS)
# ==========================================
default_jd = """Position: Senior Analytics Engineer
Responsibilities:
- Build and maintain robust data pipelines using dbt (Data Build Tool) and SQL over Snowflake data warehouses.
- Orchestrate data workflows using Apache Airflow and develop custom dashboard tracking using Python (Pandas/NumPy) and Tableau.
- Implement data quality assertions, conduct regression testing on statistical models, and collaborate with data science teams."""

job_description = st.text_area(label="Paste Job Description Here:", value=default_jd, height=180)

if st.button("Generate Interview Preparation Material", type="primary"):
    if not job_description.strip():
        st.warning("Please provide a valid job description.")
    else:
        with st.spinner("Analyzing job description..."):
            try:
                input_state = {"messages": [{"role": "user", "content": f"Process this job description:\n\n{job_description}"}]}
                result = prep_agent.invoke(input_state)
                final_message = result["messages"][-1]
                
                if hasattr(final_message, "structured_response") and final_message.structured_response:
                    st.session_state.prep_data = final_message.structured_response
                elif isinstance(final_message.content, InterviewPrepMaterial):
                    st.session_state.prep_data = final_message.content
                else:
                    st.session_state.prep_data = InterviewPrepMaterial(**json.loads(final_message.content))
                
                # Reset practice trackers on clean JD upload
                st.session_state.selected_skill = None
                st.session_state.generated_question = None
                st.session_state.evaluation_result = None
                st.session_state.history = []
                st.session_state.question_counter = 1
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 3. DISPLAY RESULTS & INTERACTIVE SESSION
# ==========================================
if st.session_state.prep_data:
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛠️ Technical Skills List", "💡 Key Role Insights", 
        "❓ Sample Interview Questions", "🏋️ Interactive Practice Room"
    ])
    
    with tab1:
        st.subheader("Required Technical Proficiencies")
        for skill in st.session_state.prep_data.technical_skills:
            st.markdown(f"- `{skill}`")
            
    with tab2:
        st.subheader("Strategic Role Insights")
        for insight in st.session_state.prep_data.key_insights:
            st.info(insight)
            
    with tab3:
        st.subheader("Targeted General Questions")
        for i, question in enumerate(st.session_state.prep_data.potential_questions, 1):
            st.markdown(f"**Q{i}:** {question}")
            
    with tab4:
        st.subheader("Simulate a Live Technical Interview")
        
        chosen_skill = st.selectbox("Choose a skill to test yourself on:", options=st.session_state.prep_data.technical_skills)
        
        # If the dropdown changes, refresh trackers
        if chosen_skill != st.session_state.selected_skill:
            st.session_state.selected_skill = chosen_skill
            st.session_state.question_counter = 1
            st.session_state.generated_question = get_new_question_text(chosen_skill, 1)
            st.session_state.evaluation_result = None
            
        st.info(f"**Practice Question:** {st.session_state.generated_question}")
        
        # Capture answer input text (cleared or updated via a unique dynamic practice key)
        user_answer = st.text_area(
            label="Type your practice response below:", 
            placeholder="In my experience...",
            height=120,
            key=f"user_ans_{st.session_state.selected_skill}_{st.session_state.question_counter}"
        )
        
        # Action buttons array layout
        col1, col2 = st.columns(2)
        
        with col1:
            submit_clicked = st.button("Submit Answer for Grading", type="secondary", use_container_width=True)
        with col2:
            next_clicked = st.button("Get New Question ✨", type="primary", use_container_width=True)
            
        # Action Loop 1: Handle Question Rotation Click
        if next_clicked:
            st.session_state.question_counter += 1
            st.session_state.generated_question = get_new_question_text(chosen_skill, st.session_state.question_counter)
            st.session_state.evaluation_result = None
            st.rerun()

        # Action Loop 2: Grade Answer & Log metrics
        if submit_clicked:
            if not user_answer.strip():
                st.warning("Please type a response before requesting evaluation grading metrics.")
            else:
                with st.spinner("Grading answer..."):
                    eval_state = {
                        "messages": [{
                            "role": "user",
                            "content": f"Skill: {st.session_state.selected_skill}\nQuestion: {st.session_state.generated_question}\nAnswer: {user_answer}"
                        }]
                    }
                    eval_result = eval_agent.invoke(eval_state)
                    eval_msg = eval_result["messages"][-1]
                    
                    if hasattr(eval_msg, "structured_response") and eval_msg.structured_response:
                        res_obj = eval_msg.structured_response
                    elif isinstance(eval_msg.content, ResponseEvaluation):
                        res_obj = eval_msg.content
                    else:
                        res_obj = ResponseEvaluation(**json.loads(eval_msg.content))
                        
                    st.session_state.evaluation_result = res_obj
                    
                    # Log data into session history list structure safely to prevent data leaks across turns
                    st.session_state.history.append({
                        "Skill": st.session_state.selected_skill,
                        "Question": st.session_state.generated_question,
                        "Your Answer": user_answer,
                        "Score": res_obj.score,
                        "Feedback": res_obj.feedback,
                        "Exemplar Answer": res_obj.better_answer
                    })

        # Render Grades Report and Display data values
        if st.session_state.evaluation_result:
            st.markdown("### 📊 Performance Assessment Report")
            st.metric(label="Score", value=f"{st.session_state.evaluation_result.score} / 10")
            st.markdown(f"**Feedback:** {st.session_state.evaluation_result.feedback}")
            with st.expander("💡 View Recommended Model Exemplar Answer", expanded=True):
                st.success(st.session_state.evaluation_result.better_answer)
                
        # ==========================================
        # 4. EXPORT HISTORY LEDGER TO CSV
        # ==========================================
        if st.session_state.history:
            st.markdown("---")
            st.subheader("📥 Export Performance Ledger")
            st.markdown(f"You have completed **{len(st.session_state.history)}** practice session logs. Click below to download your complete report card.")
            
            # Formulate tracking dataframe or text CSV memory streams using standard csv.DictWriter
            csv_buffer = StringIO()
            headers = ["Skill", "Question", "Your Answer", "Score", "Feedback", "Exemplar Answer"]
            writer = csv.DictWriter(csv_buffer, fieldnames=headers)
            writer.writeheader()
            for record in st.session_state.history:
                writer.writerow(record)
                
            st.download_button(
                label="Download Interview Progress Report (.csv)",
                data=csv_buffer.getvalue(),
                file_name="interview_practice_report.csv",
                mime="text/csv",
                type="primary"
            )