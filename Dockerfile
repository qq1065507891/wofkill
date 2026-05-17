FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic langgraph langchain-core pyyaml pytest

COPY . .

EXPOSE 8000

CMD ["uvicorn", "werewolf_agent.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
