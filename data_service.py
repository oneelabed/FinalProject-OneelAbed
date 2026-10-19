import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, date

import requests
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://jsonplaceholder.typicode.com"
TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.json")

# In-memory global cache
_CACHE = {
    "users_df": None,
    "todos_df": None,
    "last_fetched": None
}


# Custom Exceptions
class TaskNotFoundError(Exception):
    """Raised when a requested task does not exist."""
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.message = f"Task with id {task_id} not found."
        super().__init__(self.message)


class ValidationError(Exception):
    """Raised for custom domain validation failures."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# Pydantic Models
class TaskIn(BaseModel):
    title: str = Field(..., description="Task title (cannot be empty)")
    user_id: int = Field(..., description="ID of assigned user")
    due_date: str = Field(..., description="Due date in YYYY-MM-DD format")
    priority: int = Field(..., description="Priority level between 1 and 3")
    done: bool = Field(default=False, description="Completion status")
    notes: Optional[str] = Field(default=None, description="Optional notes (excluded from TaskOut)")


class TaskOut(BaseModel):
    task_id: int
    title: str
    user_id: int
    due_date: str
    priority: int
    done: bool


class TaskOverdueOut(BaseModel):
    task_id: int
    title: str
    user_id: int
    due_date: str
    priority: int
    done: bool
    days_late: int


# Data Ingestion
def fetch_json(path: str) -> Optional[Any]:
    """
    Fetch JSON payload from upstream API with timeout=5 and raise_for_status.
    Returns None if a RequestException occurs.
    """
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as err:
        logger.error(f"Error fetching data from {url}: {err}")
        return None


def get_data(force_refresh: bool = False) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load users and todos into DataFrames and cache them globally.
    Returns (users_df, todos_df).
    """
    global _CACHE

    if not force_refresh and _CACHE["users_df"] is not None and _CACHE["todos_df"] is not None:
        return _CACHE["users_df"], _CACHE["todos_df"]

    raw_users = fetch_json("users")
    raw_todos = fetch_json("todos")

    if raw_users is None or raw_todos is None:
        logger.warning("Upstream data fetch failed.")
        return _CACHE["users_df"], _CACHE["todos_df"]

    # Build Users DataFrame using list comprehension with city extraction
    users_data = [
        {
            "user_id": u["id"],
            "name": u["name"],
            "email": u["email"],
            "city": u.get("address", {}).get("city", "Unknown") if isinstance(u.get("address"), dict) else "Unknown"
        }
        for u in raw_users
    ]
    df_users = pd.DataFrame(users_data)

    # Build Todos DataFrame and ensure completed is bool
    df_todos = pd.DataFrame(raw_todos)
    df_todos["completed"] = df_todos["completed"].astype(bool)

    # Cache transformed DataFrames
    _CACHE["users_df"] = df_users
    _CACHE["todos_df"] = df_todos
    _CACHE["last_fetched"] = datetime.now()

    return df_users, df_todos


def get_valid_user_ids() -> set[int]:
    """Retrieve all valid user IDs from cached/fetched users."""
    users_df, _ = get_data()
    if users_df is not None and not users_df.empty:
        return set(users_df["user_id"].tolist())
    return set()


# Data Analysis
def get_user_performance_df() -> Optional[pd.DataFrame]:
    """
    Merge todos and users, calculate aggregated performance metrics per user,
    and sort descending by completion percentage.
    """
    users_df, todos_df = get_data()
    if users_df is None or todos_df is None:
        return None

    # Rename userId column to user_id
    todos_copy = todos_df.copy()
    if "userId" in todos_copy.columns:
        todos_copy = todos_copy.rename(columns={"userId": "user_id"})

    # Merge todos with users
    merged = pd.merge(todos_copy, users_df, on="user_id", how="inner")

    # Groupby and aggregate
    grouped = merged.groupby(["user_id", "name", "email", "city"]).agg(
        total_tasks=("completed", "count"),
        completed_tasks=("completed", "sum")
    ).reset_index()

    # Calculate completion percentage rounded to 2 decimal places
    grouped["completion_rate"] = np.round((grouped["completed_tasks"] / grouped["total_tasks"]) * 100, 2)

    # Sort descending by completion_rate
    grouped = grouped.sort_values(by="completion_rate", ascending=False).reset_index(drop=True)
    return grouped


def get_summary_statistics() -> Optional[Dict[str, Any]]:
    """
    Calculate summary statistics including NumPy mean and standard deviation,
    top 3 performers, and strongest performing city.
    """
    perf_df = get_user_performance_df()
    if perf_df is None or perf_df.empty:
        return None

    rates = perf_df["completion_rate"].to_numpy()
    mean_rate = float(np.round(np.mean(rates), 2))
    std_rate = float(np.round(np.std(rates), 2))

    top_performers = perf_df.head(3).to_dict(orient="records")

    # Calculate top city by highest average completion rate
    city_group = perf_df.groupby("city")["completion_rate"].mean().reset_index()
    top_city_row = city_group.sort_values(by="completion_rate", ascending=False).iloc[0]
    top_city = {
        "city": str(top_city_row["city"]),
        "average_completion_rate": float(np.round(top_city_row["completion_rate"], 2))
    }

    return {
        "team_mean_completion_rate": mean_rate,
        "team_std_completion_rate": std_rate,
        "total_users": int(len(perf_df)),
        "total_tasks": int(perf_df["total_tasks"].sum()),
        "total_completed_tasks": int(perf_df["completed_tasks"].sum()),
        "top_performers": top_performers,
        "top_city": top_city
    }


def get_low_performers() -> Optional[List[Dict[str, Any]]]:
    """
    Identify employees whose completion rate is below the team average
    using boolean indexing.
    """
    perf_df = get_user_performance_df()
    if perf_df is None or perf_df.empty:
        return None

    rates = perf_df["completion_rate"].to_numpy()
    mean_rate = float(np.mean(rates))

    # Boolean indexing for below-average performance
    low_performers_df = perf_df[perf_df["completion_rate"] < mean_rate]
    return low_performers_df.to_dict(orient="records")


def get_cities_pivot() -> Optional[List[Dict[str, Any]]]:
    """
    Build a pivot table of cities versus task completion status (aggfunc='count').
    """
    users_df, todos_df = get_data()
    if users_df is None or todos_df is None:
        return None

    todos_copy = todos_df.copy()
    if "userId" in todos_copy.columns:
        todos_copy = todos_copy.rename(columns={"userId": "user_id"})

    merged = pd.merge(todos_copy, users_df, on="user_id", how="inner")

    # Pivot table with index=city, columns=completed, aggfunc='count'
    pivot = pd.pivot_table(
        merged,
        index="city",
        columns="completed",
        values="id" if "id" in merged.columns else "user_id",
        aggfunc="count",
        fill_value=0
    )

    # Format into list of structured records
    result = []
    for city_name, row in pivot.iterrows():
        completed_count = int(row.get(True, 0))
        incomplete_count = int(row.get(False, 0))
        total_count = completed_count + incomplete_count
        rate = round((completed_count / total_count * 100), 2) if total_count > 0 else 0.0
        result.append({
            "city": str(city_name),
            "completed": completed_count,
            "incomplete": incomplete_count,
            "total": total_count,
            "completion_rate": rate
        })

    result.sort(key=lambda x: x["completion_rate"], reverse=True)
    return result


# Task CRUD & JSON Persistence
def load_tasks() -> List[Dict[str, Any]]:
    """
    Load tasks from tasks.json file.
    Handles FileNotFoundError gracefully by creating tasks.json if missing.
    """
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("tasks.json not found on startup. Initializing default tasks.json file.")
        default_tasks = [
            {
                "task_id": 1,
                "title": "Initial server setup and smoke testing",
                "user_id": 1,
                "due_date": "2026-03-01",
                "priority": 1,
                "done": False,
                "notes": "Ensure all services start properly"
            },
            {
                "task_id": 2,
                "title": "Prepare analytical dashboard",
                "user_id": 2,
                "due_date": "2026-12-31",
                "priority": 2,
                "done": False,
                "notes": "Coordinate with frontend team"
            }
        ]
        save_tasks(default_tasks)
        return default_tasks
    except json.JSONDecodeError:
        logger.warning("tasks.json contains invalid JSON. Initializing empty list.")
        return []


def save_tasks(tasks: List[Dict[str, Any]]) -> None:
    """Save tasks list to tasks.json."""
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)


def validate_task_payload(task_in: TaskIn) -> None:
    """
    Validate title is not empty after strip(), priority is 1..3,
    and user exists in users dataset. Raises ValidationError on failure.
    """
    cleaned_title = task_in.title.strip()
    if not cleaned_title:
        raise ValidationError("Title cannot be empty.")

    if not (1 <= task_in.priority <= 3):
        raise ValidationError(f"Priority must be between 1 and 3. Received: {task_in.priority}")

    valid_users = get_valid_user_ids()
    if valid_users and task_in.user_id not in valid_users:
        raise ValidationError(f"User with user_id {task_in.user_id} does not exist.")


def create_task(task_in: TaskIn) -> TaskOut:
    """Create a new task, persist to tasks.json, and return TaskOut."""
    validate_task_payload(task_in)
    tasks = load_tasks()

    new_id = (max([t["task_id"] for t in tasks], default=0) + 1)
    new_task = {
        "task_id": new_id,
        "title": task_in.title.strip(),
        "user_id": task_in.user_id,
        "due_date": task_in.due_date,
        "priority": task_in.priority,
        "done": task_in.done,
        "notes": task_in.notes
    }

    tasks.append(new_task)
    save_tasks(tasks)

    return TaskOut(
        task_id=new_task["task_id"],
        title=new_task["title"],
        user_id=new_task["user_id"],
        due_date=new_task["due_date"],
        priority=new_task["priority"],
        done=new_task["done"]
    )


def get_task_by_id(task_id: int) -> TaskOut:
    """Retrieve a single task by ID or raise TaskNotFoundError."""
    tasks = load_tasks()
    for t in tasks:
        if t["task_id"] == task_id:
            return TaskOut(
                task_id=t["task_id"],
                title=t["title"],
                user_id=t["user_id"],
                due_date=t["due_date"],
                priority=t["priority"],
                done=t["done"]
            )
    raise TaskNotFoundError(task_id)


def update_task_by_id(task_id: int, task_in: TaskIn) -> TaskOut:
    """Update a task by ID or raise TaskNotFoundError."""
    validate_task_payload(task_in)
    tasks = load_tasks()

    for idx, t in enumerate(tasks):
        if t["task_id"] == task_id:
            updated_task = {
                "task_id": task_id,
                "title": task_in.title.strip(),
                "user_id": task_in.user_id,
                "due_date": task_in.due_date,
                "priority": task_in.priority,
                "done": task_in.done,
                "notes": task_in.notes
            }
            tasks[idx] = updated_task
            save_tasks(tasks)
            return TaskOut(
                task_id=updated_task["task_id"],
                title=updated_task["title"],
                user_id=updated_task["user_id"],
                due_date=updated_task["due_date"],
                priority=updated_task["priority"],
                done=updated_task["done"]
            )

    raise TaskNotFoundError(task_id)


def delete_task_by_id(task_id: int) -> None:
    """Delete a task by ID or raise TaskNotFoundError."""
    tasks = load_tasks()
    initial_length = len(tasks)
    filtered = [t for t in tasks if t["task_id"] != task_id]

    if len(filtered) == initial_length:
        raise TaskNotFoundError(task_id)

    save_tasks(filtered)


def get_overdue_tasks() -> List[TaskOverdueOut]:
    """
    Return incomplete tasks whose due date is in the past,
    including days_late computed using pd.to_datetime and dt.days.
    """
    tasks = load_tasks()
    if not tasks:
        return []

    df = pd.DataFrame(tasks)
    # Filter incomplete tasks
    incomplete = df[df["done"] == False].copy()
    if incomplete.empty:
        return []

    # Parse due_date using pd.to_datetime
    incomplete["due_datetime"] = pd.to_datetime(incomplete["due_date"], errors="coerce")
    today = pd.Timestamp.now().normalize()

    # Calculate days_late
    incomplete["days_late"] = (today - incomplete["due_datetime"]).dt.days

    # Filter tasks where days_late > 0
    overdue_df = incomplete[incomplete["days_late"] > 0].copy()
    overdue_df = overdue_df.sort_values(by="days_late", ascending=False)

    result = []
    for _, row in overdue_df.iterrows():
        result.append(
            TaskOverdueOut(
                task_id=int(row["task_id"]),
                title=str(row["title"]),
                user_id=int(row["user_id"]),
                due_date=str(row["due_date"]),
                priority=int(row["priority"]),
                done=bool(row["done"]),
                days_late=int(row["days_late"])
            )
        )
    return result
