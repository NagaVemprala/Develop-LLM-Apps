import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

ModelProvider = Literal["openai", "anthropic", "gemini", "deepseek"]

def get_chat_model(
    provider: ModelProvider = "openai",
    model_name: str | None = None,
    temperature: float = 0.2
) -> BaseChatModel:
    """Factory to instantiate different LLM providers with a standardized interface."""
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    """
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name or "claude-3-5-sonnet-20241022",
                temperature=temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenAI
            return ChatGoogleGenAI(
                model=model_name or "gemini-1.5-pro",
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

        elif provider == "deepseek":
            # DeepSeek exposes an OpenAI-compatible API format
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name or "deepseek-chat",
                temperature=temperature,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            )
    """
    raise ValueError(f"Unsupported provider: {provider}")