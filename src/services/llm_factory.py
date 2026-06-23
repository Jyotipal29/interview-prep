from langchain_core.language_models import BaseChatModel

from src.config.settings import LLMProvider, Settings, get_settings


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """
    Return a LangChain chat model for the configured provider.

    Accepts an optional Settings instance for dependency injection; falls
    back to loading from environment variables when omitted.
    """
    if settings is None:
        settings = get_settings()

    provider = settings.llm_provider

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

        return ChatAnthropic(  # type: ignore[call-arg]
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
        )

    if provider == LLMProvider.OPENROUTER:
        from langchain_openai import ChatOpenAI

        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
