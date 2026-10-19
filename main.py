import os
import time
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, Depends, HTTPException, Header, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from data_service import (
    get_data,
    get_user_performance_df,
    get_summary_statistics,
    get_low_performers,
    get_cities_pivot,
    load_tasks,
    create_task,
    get_task_by_id,
    update_task_by_id,
    delete_task_by_id,
    get_overdue_tasks,
    TaskIn,
    TaskOut,
    TaskOverdueOut,
    TaskNotFoundError,
    ValidationError
)
from ai_service import (
    ask_assistant,
    get_logs_statistics,
    AskRequest,
    AskResponse,
    AILogsStatsResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv()
APP_API_KEY = os.environ.get("APP_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle handler.
    Checks environment keys and initializes datasets.
    """
    logger.info("Starting up FastAPI Performance Analytics Service...")

    # Validate APP_API_KEY
    if not APP_API_KEY:
        logger.warning(
            "WARNING: [APP_API_KEY] is missing from the environment. "
            "Protected mutating endpoints will reject requests until configured."
        )
    else:
        logger.info("SUCCESS: [APP_API_KEY] loaded successfully.")

    # Validate ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        logger.warning(
            "NOTICE: [ANTHROPIC_API_KEY] is missing. "
            "AI assistant will run in intelligent Mock response mode."
        )
    else:
        logger.info("SUCCESS: [ANTHROPIC_API_KEY] loaded successfully.")

    # Warm up caches and verify tasks.json
    try:
        load_tasks()
        get_data()
        logger.info("Data caching and task persistence initialized.")
    except Exception as err:
        logger.warning(f"Initial data warmup encountered an issue: {err}")

    yield

    logger.info("Shutting down FastAPI Performance Analytics Service...")


# Initialize FastAPI app
app = FastAPI(
    title="Team Performance & AI Analytics API",
    description="FastAPI service for team analytics, task management, and AI assistance.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration - Restricted strictly to http://localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: Log method, path, status_code, and attach X-Process-Time header
@app.middleware("http")
async def process_time_and_log_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    process_time = time.perf_counter() - start_time

    # Add X-Process-Time header
    response.headers["X-Process-Time"] = f"{process_time:.6f}s"

    # Print/log method, path, status_code
    print(f"[{request.method}] {request.url.path} - Status: {response.status_code} - Process-Time: {process_time:.6f}s")
    logger.info(f"[{request.method}] {request.url.path} - Status: {response.status_code} - Process-Time: {process_time:.6f}s")

    return response


# Exception Handlers
@app.exception_handler(TaskNotFoundError)
async def task_not_found_handler(request: Request, exc: TaskNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.message, "task_id": exc.task_id}
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message}
    )


# Dependencies
def pagination_params(
    page: int = Query(1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page")
) -> Dict[str, int]:
    """Reusable dependency providing pagination parameters."""
    return {"page": page, "page_size": page_size}


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Security dependency: Validates X-API-Key header against APP_API_KEY.
    Returns 401 Unauthorized if missing or mismatched.
    """
    expected_key = os.environ.get("APP_API_KEY")
    if not expected_key or not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-API-Key header."
        )
    return x_api_key


# ==========================================
# Analytical Statistics Routes
# ==========================================

@app.get("/stats/users", summary="Get sorted and paginated user performance metrics")
def get_users_performance_stats(
    sort_by: str = Query("completion_rate", description="Field to sort by: completion_rate, total_tasks, completed_tasks"),
    order: str = Query("desc", description="Sort order: 'asc' or 'desc'"),
    pagination: Dict[str, int] = Depends(pagination_params)
):
    df = get_user_performance_df()
    if df is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Upstream data source unavailable.")

    ascending = True if order.lower() == "asc" else False
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    records = df.to_dict(orient="records")
    start = (pagination["page"] - 1) * pagination["page_size"]
    end = start + pagination["page_size"]

    return {
        "page": pagination["page"],
        "page_size": pagination["page_size"],
        "total_items": len(records),
        "data": records[start:end]
    }


@app.get("/stats/summary", summary="Get overall team performance metrics and top performers")
def get_summary_stats():
    summary = get_summary_statistics()
    if summary is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Upstream data source unavailable.")
    return summary


@app.get("/stats/low-performers", summary="Get employees performing below team average")
def get_low_performing_users(pagination: Dict[str, int] = Depends(pagination_params)):
    low_performers = get_low_performers()
    if low_performers is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Upstream data source unavailable.")

    start = (pagination["page"] - 1) * pagination["page_size"]
    end = start + pagination["page_size"]

    return {
        "page": pagination["page"],
        "page_size": pagination["page_size"],
        "total_items": len(low_performers),
        "data": low_performers[start:end]
    }


@app.get("/stats/cities", summary="Get city vs completion pivot table")
def get_cities_pivot_stats(pagination: Dict[str, int] = Depends(pagination_params)):
    pivot_data = get_cities_pivot()
    if pivot_data is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Upstream data source unavailable.")

    start = (pagination["page"] - 1) * pagination["page_size"]
    end = start + pagination["page_size"]

    return {
        "page": pagination["page"],
        "page_size": pagination["page_size"],
        "total_items": len(pivot_data),
        "data": pivot_data[start:end]
    }


# ==========================================
# Task Management (CRUD) Routes
# ==========================================

@app.get("/tasks", response_model=List[TaskOut], summary="List all tasks with pagination")
def list_tasks(pagination: Dict[str, int] = Depends(pagination_params)):
    tasks = load_tasks()
    start = (pagination["page"] - 1) * pagination["page_size"]
    end = start + pagination["page_size"]
    sliced = tasks[start:end]
    return [
        TaskOut(
            task_id=t["task_id"],
            title=t["title"],
            user_id=t["user_id"],
            due_date=t["due_date"],
            priority=t["priority"],
            done=t["done"]
        )
        for t in sliced
    ]


@app.post("/tasks", response_model=TaskOut, status_code=status.HTTP_201_CREATED, summary="Create a new task")
def add_new_task(
    task_in: TaskIn,
    _auth: str = Depends(verify_api_key)
):
    try:
        return create_task(task_in)
    except ValidationError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err.message)


# CRITICAL REQUIREMENT: /tasks/overdue must be defined BEFORE /tasks/{task_id}
@app.get("/tasks/overdue", response_model=List[TaskOverdueOut], summary="List overdue incomplete tasks with days_late")
def list_overdue_tasks(pagination: Dict[str, int] = Depends(pagination_params)):
    overdue_tasks = get_overdue_tasks()
    start = (pagination["page"] - 1) * pagination["page_size"]
    end = start + pagination["page_size"]
    return overdue_tasks[start:end]


@app.get("/tasks/{task_id}", response_model=TaskOut, summary="Retrieve a single task by ID")
def get_task(task_id: int):
    return get_task_by_id(task_id)


@app.put("/tasks/{task_id}", response_model=TaskOut, summary="Update an existing task")
def update_task(
    task_id: int,
    task_in: TaskIn,
    _auth: str = Depends(verify_api_key)
):
    try:
        return update_task_by_id(task_id, task_in)
    except ValidationError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err.message)


@app.delete("/tasks/{task_id}", summary="Delete a task by ID")
def delete_task(
    task_id: int,
    _auth: str = Depends(verify_api_key)
):
    delete_task_by_id(task_id)
    return {"detail": f"Task {task_id} deleted successfully."}


# ==========================================
# AI Assistant & Audit Logging Routes
# ==========================================

@app.post("/ai/ask", response_model=AskResponse, summary="Query AI assistant with live team context")
def ask_ai(payload: AskRequest):
    # Validation: reject empty prompts or prompts exceeding 500 characters with 400 Bad Request
    if not payload.prompt or not payload.prompt.strip() or len(payload.prompt) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt must not be empty and cannot exceed 500 characters."
        )

    try:
        return ask_assistant(prompt=payload.prompt, conversation_id=payload.conversation_id)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@app.get("/ai/logs/stats", response_model=AILogsStatsResponse, summary="Get AI audit log metrics")
def get_ai_stats():
    return get_logs_statistics()
