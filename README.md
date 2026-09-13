# 🚀 Develop LLM Apps

A hands-on, project-based learning journey into building Generative AI applications using the **LangChain** framework. 

This repository is designed for students, analysts, and developers (or anyone) who want to move beyond theory and start building practical AI agents. Starting from simple API calls, we progressively build up to a fully functional **AI Interview Coach** that reads job descriptions, quizzes you on skills, and evaluates your answers.

---

## 📂 Project Structure & File Breakdown

Here is exactly what each script does, ordered from beginner to advanced:

| # | File Name | Description |
| :---: | :--- | :--- |
| 1 | [`01_simple_LLM_chat.py`](./01_simple_LLM_chat.py) | **The Basics:** A minimalist single-prompt chat. Uses LangChain's `init_chat_model` to initialize an LLM and send a single query. Perfect for understanding model setup and prompt engineering fundamentals. |
| 2 | [`02_simple_agent.py`](./02_simple_agent.py) | **Introducing Agents:** Still a single-prompt chat, but now using the `create_agent` function. This sets the stage for giving the LLM "tools" to use, moving from a basic chatbot to a reasoning engine. |
| 3 | [`03_AI_Agent_For_Job_Search.py`](./03_AI_Agent_For_Job_Search.py) | **Practical Application (CLI):** A command-line job interview helper. Paste a job description, and the agent parses it to display **key required skills** and generates **possible interview questions**. Great for job hunting prep. |
| 4 | [`04_AI_Agent_For_Job_Search_Streamlit.py`](./04_AI_Agent_For_Job_Search_Streamlit.py) | **Web App (v1):** Takes the exact logic from `03` and puts it into a beautiful **Streamlit** web interface. No terminal needed—just a clean UI to upload job descriptions and view insights. |
| 5 | [`05_AI_Agent_For_Job_Prep_Streamlit.py`](./05_AI_Agent_For_Job_Prep_Streamlit.py) | **Web App (v2 - The Interview Coach):** The flagship of this repo! This enhanced Streamlit app doesn't just *show* you the skills—it **interviews you**. It asks specific questions based on a chosen skill, takes your spoken/typed response, evaluates your answer, and provides an expert-level suggested better response. |

---

## ✨ Features

- 🔗 **LangChain Integration:** Uses industry-standard LangChain patterns (`init_chat_model`, `create_agent`).
- 🧠 **Job Market Focus:** Tailored specifically for tech job preparation.
- 🌐 **Interactive UI:** Files 4 and 5 leverage Streamlit for a seamless user experience.
- 💬 **Response Evaluation:** The final app (as of now) actively scores your interview answers and gives constructive feedback.

---

## 🛠️ Prerequisites & Installation

Before running these scripts, ensure you have the following installed:

- Python 3.12 or higher
- An API key for an LLM provider (e.g., OpenAI, Groq, or Anthropic). OpenAI key is used in the code for demonstration. 

### 1. Clone the Repository
```bash
git clone https://github.com/NagaVemprala/Develop-LLM-Apps.git
cd Develop-LLM-Apps
```

### 2. Install Dependencies
- Create a virtual environment (recommended) and install the required packages:

```bash
pip install langchain langchain-community langchain-openai streamlit python-dotenv
```

### 3. Set up Environment Variables
- Create a .env file in the root directory and add your API keys:

```bash
OPENAI_API_KEY=your-api-key-here
# Or GROQ_API_KEY=your-groq-key-here
```

### 4. How to Run
- For Terminal/CLI scripts (Files 01, 02, 03):

```bash
python 01_simple_LLM_chat.py
# (Replace with the respective file number)
```

- For Streamlit Web Apps (Files 04 and 05):

```bash
streamlit run 04_AI_Agent_For_Job_Search_Streamlit.py
```

## 🗺️ Future Roadmap

- This project is under active development! Upcoming enhancements include:

- **Persistent Storage:** Migrating job postings and user data to permanent databases like PostgreSQL or Amazon RDS.
- **Knowledge Checks:** Maintaining separate tables to track user progress and quiz history over time.
- **Enhanced Scoring:** More sophisticated rubrics leveraging decision-modeling logic to evaluate soft skills and technical depth.



