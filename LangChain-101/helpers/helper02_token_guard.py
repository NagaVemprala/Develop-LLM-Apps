# token_guard.py
from dataclasses import dataclass
from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

# Standard context limits across popular models
MODEL_CONTEXT_LIMITS = {
    "gpt-4o-mini": 128_000,
    "gpt-5.4-nano": 128_000,
    "gpt-4o": 128_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "gemini-1.5-pro": 2_000_000,
    "deepseek-chat": 64_000,
}

class ContextWindowExceededError(Exception):
    """Raised when rendered messages exceed the designated token budget."""
    pass

@dataclass
class TokenBudgetReport:
    prompt_tokens: int
    reserved_output_tokens: int
    total_projected_tokens: int
    max_context_limit: int
    remaining_headroom: int

def check_context_budget(
    model: BaseChatModel,
    messages: List[BaseMessage],
    model_name: str = "gpt-4o-mini",
    max_output_tokens: int = 4_096,
    safety_margin: int = 500
) -> TokenBudgetReport:
    """
    Calculates prompt token usage locally before dispatching the request.
    Raises ContextWindowExceededError if the budget is breached.
    """
    context_limit = MODEL_CONTEXT_LIMITS.get(model_name, 128_000)
    
    # Calculate exact prompt tokens using the model's internal tokenizer
    prompt_tokens = model.get_num_tokens_from_messages(messages)
    total_projected = prompt_tokens + max_output_tokens + safety_margin

    if total_projected > context_limit:
        raise ContextWindowExceededError(
            f"Prompt exceeds budget: {total_projected} tokens projected "
            f"(Prompt: {prompt_tokens}, Reserved Output: {max_output_tokens}, Safety Margin: {safety_margin}) "
            f"against maximum context window of {context_limit} for {model_name}."
        )

    return TokenBudgetReport(
        prompt_tokens=prompt_tokens,
        reserved_output_tokens=max_output_tokens,
        total_projected_tokens=total_projected,
        max_context_limit=context_limit,
        remaining_headroom=context_limit - total_projected
    )