import os
from typing import Any, Optional, Type

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel

load_dotenv()


def get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )


def chat_deployment() -> str:
    return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", os.getenv("MODEL", "gpt-4o-mini"))


def embedding_deployment() -> str:
    return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


def _messages(prompt: str, system_instruction: Optional[str] = None) -> list[dict[str, str]]:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    return messages


def _clean_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            key: _clean_schema(value)
            for key, value in schema.items()
            if key != "default"
        }
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema


def generate_text(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    response = get_azure_client().chat.completions.create(
        model=chat_deployment(),
        messages=_messages(prompt, system_instruction),
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def generate_json(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    schema: Optional[Type[BaseModel] | dict[str, Any]] = None,
    schema_name: str = "structured_response",
    temperature: float = 0.1,
) -> str:
    if schema is None:
        response_format: dict[str, Any] = {"type": "json_object"}
    else:
        raw_schema = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else schema
        json_schema = _clean_schema(raw_schema)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
            },
        }

    response = get_azure_client().chat.completions.create(
        model=chat_deployment(),
        messages=_messages(prompt, system_instruction),
        temperature=temperature,
        response_format=response_format,
    )
    return response.choices[0].message.content or "{}"


def generate_model(
    prompt: str,
    model_schema: Type[BaseModel],
    *,
    system_instruction: Optional[str] = None,
    temperature: float = 0.1,
) -> BaseModel:
    content = generate_json(
        prompt,
        system_instruction=system_instruction,
        schema=model_schema,
        schema_name=model_schema.__name__,
        temperature=temperature,
    )
    return model_schema.model_validate_json(content)


def generate_embedding(text: str) -> list[float]:
    response = get_azure_client().embeddings.create(
        model=embedding_deployment(),
        input=text,
    )
    return response.data[0].embedding
