# FastAPI Team Performance & AI Analytics System - Project Plan

## 1. Overview
This project implements a high-performance RESTful API service built with FastAPI, Pandas, NumPy, and Pydantic. It ingests and processes organizational performance data from external sources (`jsonplaceholder.typicode.com`), provides advanced team analytics, manages task workflows with JSON persistence, secures administrative endpoints, and features an intelligent AI assistant powered by Anthropic Claude (with mock fallback).

---

## 2. Architecture & File Structure

```
FinalProject-OneelAbed/
├── PLAN.md               # Detailed architectural and implementation plan
├── .env                  # Environment variables (ANTHROPIC_API_KEY, APP_API_KEY)
├── .env.example          # Environment template for setup
├── .gitignore            # Git ignore file (excluding secrets, cache, and virtualenvs)
├── requirements.txt      # Python package dependencies
├── tasks.json            # JSON file storage for tasks CRUD persistence
├── ai_logs.csv           # CSV file storing AI request audit metrics
├── data_service.py       # Data fetching, caching, DataFrame transformations, analytics, and task CRUD
├── ai_service.py         # LLM client, conversation state management, prompt generation, and logging
└── main.py               # FastAPI application, route handlers, dependencies, middleware, and CORS
```

---

## 3. Implementation Steps & Requirements

### Step 1: Project Initialization & Configuration
- **Dependencies (`requirements.txt`)**:
  - `fastapi`
  - `uvicorn[standard]`
  - `pydantic`
  - `pandas`
  - `numpy`
  - `requests`
  - `python-dotenv`
  - `anthropic`
- **Configuration Management**:
  - Load environment variables using `load_dotenv()` and `os.environ.get()` / `os.getenv()`.
  - Keys to inspect: `APP_API_KEY` and `ANTHROPIC_API_KEY`.
  - Non-blocking startup: If any key is missing, print a descriptive warning message to the console and allow the server to continue running.

---

### Step 2: Data Ingestion & Caching (`data_service.py`)
- **External Data Fetching (`fetch_json`)**:
  - Target URL base: `https://jsonplaceholder.typicode.com/`.
  - Endpoints: `/users` and `/todos`.
  - Request configuration: `timeout=5`, `response.raise_for_status()`.
  - Exception handling: Catch `requests.exceptions.RequestException` and return `None` upon failure.
- **Data Transformation with Pandas**:
  - **Users DataFrame**: Built via list comprehension extracting `user_id`, `name`, `email`, and `city` (extracted from the nested `address.city` field).
  - **Todos DataFrame**: Created from todos records, verifying that the `completed` column is strictly of boolean type (`bool`).
- **In-Memory Cache**:
  - Cache transformed DataFrames in module-level global variables.
  - Return HTTP `503 Service Unavailable` if the remote service is unreachable and no cached data is available.

---

### Step 3: Statistical Analytics & Business Insights (`data_service.py`)
- **Data Merging**:
  - Rename column `userId` to `user_id` in the todos dataset.
  - Merge todos and users DataFrames on `user_id`.
- **Per-Employee Aggregations**:
  - Group by `user_id` and compute with `.agg()`:
    - Total task count (`total_tasks`)
    - Completed task count (`completed_tasks`)
    - Completion percentage (`completion_rate`), rounded to 2 decimal places.
  - Join employee metadata (`name`, `email`, `city`) and sort descending by `completion_rate`.
- **NumPy & Outlier Detection**:
  - Compute team-wide `mean` and `std` of completion percentages using NumPy.
  - Identify underperforming employees whose completion rate is strictly below the mean using Boolean indexing.
- **City Cross-Tabulation**:
  - Construct a pivot table of cities versus completion status (`pd.pivot_table(..., index="city", columns="completed", aggfunc="count")`).

---

### Step 4: Task Management CRUD & Persistence (`data_service.py`)
- **Pydantic Schemas**:
  - `TaskIn`:
    - `title`: string (must be non-empty after `.strip()`)
    - `user_id`: integer (must match an existing user)
    - `due_date`: string (ISO date format `YYYY-MM-DD`)
    - `priority`: integer (between 1 and 3 inclusive)
    - `done`: boolean (default `False`)
    - `notes`: optional string
  - `TaskOut`:
    - `task_id`: integer
    - `title`: string
    - `user_id`: integer
    - `due_date`: string
    - `priority`: integer
    - `done`: boolean
    - *(Note: `notes` field is explicitly excluded from `TaskOut`)*
- **Validation Rules**:
  - Return HTTP `400 Bad Request` if:
    - `title.strip()` is empty.
    - `priority` is not in the range `[1, 3]`.
    - `user_id` does not exist in the users dataset.
- **Custom Exceptions**:
  - Define `TaskNotFoundError` with an exception handler returning HTTP `404 Not Found` and a descriptive message.
- **File Persistence**:
  - Persist all mutations (create, update, delete) to `tasks.json`.
  - Handle `FileNotFoundError` gracefully on server startup by initializing a valid task repository.
  - Creating a task returns HTTP `201 Created`.

---

### Step 5: Security & Infrastructure (`main.py`)
- **Pagination Dependency**:
  - Provide reusable dependency with `page: int = 1` and `page_size: int = 10` for all list-returning endpoints.
- **Security Dependency**:
  - Extract header `X-API-Key` and compare against `APP_API_KEY`.
  - Return HTTP `401 Unauthorized` if invalid or missing.
  - Apply security dependency to all mutating endpoints (`POST`, `PUT`, `DELETE`).
- **Custom Middleware**:
  - Compute total request execution duration and append header `X-Process-Time`.
  - Log `method`, `path`, and `status_code` to the server log.
- **CORS Configuration**:
  - Restrict allowed origins strictly to `http://localhost:5173`.

---

### Step 6: AI Smart Assistant (`ai_service.py`)
- **Endpoint `POST /ai/ask`**:
  - Accept payload containing `prompt` and `conversation_id`.
  - Validate prompt: return HTTP `400 Bad Request` if empty or if length exceeds 500 characters.
  - Maintain conversational context per `conversation_id` in memory.
  - Dynamically construct a system prompt utilizing live statistical insights:
    - Top 3 performing employees
    - Team average completion rate
    - Strongest performing city
- **LLM Integration & Fallback**:
  - Query Anthropic Claude API if `ANTHROPIC_API_KEY` is present.
  - Return a contextual Mock response if the API key is not configured.
- **Audit Logging (`ai_logs.csv`)**:
  - Use `try / except / finally` block to guarantee logging on every invocation, including error scenarios.
  - Record: `timestamp`, `prompt_length`, `status` (`success` or `error`).
  - Privacy guarantee: Never log API keys or raw prompt contents.
- **Endpoint `GET /ai/logs/stats`**:
  - Parse `ai_logs.csv` and return the overall success rate and average prompt length.

---

## 4. API Endpoints Specification

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET` | `/stats/users` | No | Team performance list with sorting and pagination |
| `GET` | `/stats/summary` | No | Team-wide metrics (mean, std, top performers, top city) |
| `GET` | `/stats/low-performers` | No | Employees performing below the team average |
| `GET` | `/stats/cities` | No | City vs completion status pivot table |
| `GET` | `/tasks` | No | Paginated list of tasks |
| `POST` | `/tasks` | Yes (`X-API-Key`) | Create a new task (returns 201) |
| `GET` | `/tasks/overdue` | No | Incomplete tasks past due date with `days_late` *(declared before `/tasks/{task_id}`)* |
| `GET` | `/tasks/{task_id}` | No | Retrieve task by ID |
| `PUT` | `/tasks/{task_id}` | Yes (`X-API-Key`) | Update task details |
| `DELETE` | `/tasks/{task_id}` | Yes (`X-API-Key`) | Delete task |
| `POST` | `/ai/ask` | No | AI assistant with conversation memory and dynamic context |
| `GET` | `/ai/logs/stats` | No | AI logs analytics (success rate, avg prompt length) |

---

## 5. Verification & Testing Strategy
1. **Unit & Functional Tests**:
   - Verify network error handling and 503 response when upstream service is unavailable.
   - Validate DataFrame transformations, groupby aggregations, and NumPy calculations.
   - Test CRUD operations on `tasks.json` with all validation edge cases (empty title, invalid priority, missing user, duplicate IDs).
   - Verify that `/tasks/overdue` calculates `days_late` correctly using `pd.to_datetime` and `dt.days`.
2. **Security & Middleware Validation**:
   - Verify `401 Unauthorized` on missing or incorrect `X-API-Key` headers for mutating endpoints.
   - Inspect response headers for `X-Process-Time`.
   - Verify CORS origin restrictions (`http://localhost:5173`).
3. **AI Assistant & Log Verification**:
   - Verify prompt length validation (>500 chars returns 400).
   - Test mock response generation when `ANTHROPIC_API_KEY` is not provided.
   - Verify CSV log entries in `ai_logs.csv` across both successful and failed requests.
   - Verify metrics returned by `GET /ai/logs/stats`.
