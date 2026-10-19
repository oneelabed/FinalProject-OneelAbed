import os
import csv
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from data_service import get_summary_statistics

logger = logging.getLogger(__name__)

# Constants
AI_LOGS_FILE = os.path.join(os.path.dirname(__file__), "ai_logs.csv")

# In-memory conversation state: conversation_id -> list of {"role": "user"|"assistant", "content": "..."}
CONVERSATION_HISTORY: Dict[str, List[Dict[str, str]]] = {}


# Schemas
class AskRequest(BaseModel):
    prompt: str = Field(..., description="User query prompt (max 500 chars)")
    conversation_id: str = Field(..., description="Conversation identifier for context tracking")


class AskResponse(BaseModel):
    conversation_id: str
    response: str
    is_mock: bool


class AILogsStatsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate_percentage: float
    average_prompt_length: float


def init_logs_file() -> None:
    """Ensure the ai_logs.csv file exists with proper headers."""
    if not os.path.exists(AI_LOGS_FILE):
        with open(AI_LOGS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "prompt_length", "status"])


def log_ai_interaction(prompt_length: int, status: str) -> None:
    """
    Log timestamp, prompt length, and status to ai_logs.csv.
    Strict privacy: Never logs API keys or raw prompt contents.
    """
    try:
        init_logs_file()
        with open(AI_LOGS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), prompt_length, status])
    except Exception as err:
        logger.error(f"Failed to write to ai_logs.csv: {err}")


def build_dynamic_system_prompt() -> str:
    """
    Build a dynamic system prompt using live analytical results:
    top 3 performers, team average completion rate, and strongest performing city.
    """
    stats = get_summary_statistics()
    if not stats:
        return (
            "You are an executive team performance analytics assistant. "
            "Data is currently loading or unavailable."
        )

    mean_rate = stats.get("team_mean_completion_rate", 0.0)
    top_city_info = stats.get("top_city", {})
    city_name = top_city_info.get("city", "Unknown")
    city_rate = top_city_info.get("average_completion_rate", 0.0)

    performers = stats.get("top_performers", [])
    performer_lines = []
    for rank, p in enumerate(performers, start=1):
        performer_lines.append(
            f"{rank}. {p.get('name', 'N/A')} ({p.get('completion_rate', 0.0)}% completed, {p.get('total_tasks', 0)} total tasks)"
        )
    performers_text = "\n".join(performer_lines) if performer_lines else "None recorded."

    system_prompt = (
        "You are an expert executive assistant specialized in team performance and task analytics.\n"
        "Here are the live team metrics based on current data analysis:\n"
        f"- Team Average Completion Rate: {mean_rate}%\n"
        f"- Strongest City: {city_name} (Average completion rate: {city_rate}%)\n"
        f"- Top 3 Performers:\n{performers_text}\n\n"
        "Instructions: Answer questions concisely, professionally, and accurately using this performance context."
    )
    return system_prompt


def generate_mock_response(prompt: str, system_prompt: str) -> str:
    """Generate a reliable, contextual mock response when no Anthropic API key is provided."""
    stats = get_summary_statistics() or {}
    mean_rate = stats.get("team_mean_completion_rate", 45.0)
    top_city = stats.get("top_city", {}).get("city", "Unknown")
    top_performers = stats.get("top_performers", [])
    top_performer_name = top_performers[0]["name"] if top_performers else "Top Employee"

    return (
        f"[Mock AI Response] Based on current team performance data, the team average completion rate "
        f"is {mean_rate}%, led by top performer {top_performer_name}. The strongest performing city is {top_city}. "
        f"Your query: \"{prompt}\" has been evaluated against these analytical benchmarks."
    )


def ask_assistant(prompt: str, conversation_id: str) -> AskResponse:
    """
    Handle the AI assistant query workflow with conversation history,
    try/except/finally logging to ai_logs.csv, and mock fallback.
    """
    prompt_len = len(prompt) if prompt else 0
    status = "error"
    is_mock = True
    reply_text = ""

    try:
        # Validate prompt length
        cleaned = prompt.strip() if prompt else ""
        if not cleaned or len(prompt) > 500:
            raise ValueError("Prompt must not be empty and cannot exceed 500 characters.")

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        system_prompt = build_dynamic_system_prompt()

        # Retrieve existing conversation history
        history = CONVERSATION_HISTORY.setdefault(conversation_id, [])

        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)

                # Format messages for Anthropic
                messages = []
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": cleaned})

                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=300,
                    system=system_prompt,
                    messages=messages
                )

                # Extract response text
                reply_text = response.content[0].text
                is_mock = False
            except Exception as anthropic_err:
                logger.warning(f"Anthropic API call failed: {anthropic_err}. Falling back to mock response.")
                reply_text = generate_mock_response(cleaned, system_prompt)
                is_mock = True
        else:
            reply_text = generate_mock_response(cleaned, system_prompt)
            is_mock = True

        # Update history
        history.append({"role": "user", "content": cleaned})
        history.append({"role": "assistant", "content": reply_text})

        status = "success"
        return AskResponse(
            conversation_id=conversation_id,
            response=reply_text,
            is_mock=is_mock
        )

    except Exception as exc:
        status = "error"
        raise exc
    finally:
        log_ai_interaction(prompt_length=prompt_len, status=status)


def get_logs_statistics() -> AILogsStatsResponse:
    """
    Read ai_logs.csv and compute total requests, success rate, and average prompt length.
    """
    init_logs_file()

    try:
        if not os.path.exists(AI_LOGS_FILE) or os.path.getsize(AI_LOGS_FILE) == 0:
            return AILogsStatsResponse(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate_percentage=0.0,
                average_prompt_length=0.0
            )

        df = pd.read_csv(AI_LOGS_FILE)
        if df.empty:
            return AILogsStatsResponse(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate_percentage=0.0,
                average_prompt_length=0.0
            )

        total_requests = len(df)
        successful_requests = int((df["status"] == "success").sum())
        failed_requests = total_requests - successful_requests

        success_rate = round((successful_requests / total_requests) * 100, 2) if total_requests > 0 else 0.0
        avg_prompt_len = round(float(df["prompt_length"].mean()), 2) if total_requests > 0 else 0.0

        return AILogsStatsResponse(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate_percentage=success_rate,
            average_prompt_length=avg_prompt_len
        )
    except Exception as err:
        logger.error(f"Error reading ai_logs.csv: {err}")
        return AILogsStatsResponse(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            success_rate_percentage=0.0,
            average_prompt_length=0.0
        )
