# 1. Using `init_chat_model` with OpenAI 5.5 Nano
# The standard modern method to initialize a language model uses `init_chat_model`. 
# This structural wrapper handles provider-agnostic initialization, allowing you to quickly swap models at 
# runtime via standard application configuration.

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import os
# 0. Load the .env file to set environment variables for API keys and configuration
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Initialize the chat model using the standard unified interface
model = init_chat_model(
    model="gpt-5.4-nano", 
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    temperature=0.2,
    max_tokens=1000
)

# 2. Invoke the model directly using a message list
messages = [
    HumanMessage(content="Explain the operational advantage of an optimized RAG system.")
]

response = model.invoke(messages)
print(response.content)
