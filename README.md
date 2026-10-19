# Team Performance Analytics & AI Assistant API

A comprehensive FastAPI backend service that aggregates organizational performance data from JSONPlaceholder, computes statistical benchmarks using Pandas and NumPy, manages operational tasks with JSON file persistence, and provides an intelligent LLM assistant with conversation tracking and audit metrics.

---

## Features

1. **Data Ingestion & Caching**:
   - Fetches users and todos from `jsonplaceholder.typicode.com` with a 5-second timeout and error handling.
   - Global in-memory cache with fallback returning HTTP 503 when the external service is unavailable.
2. **Statistical Analytics (Pandas & NumPy)**:
   - Aggregated metrics per employee: total tasks, completed tasks, and completion rate percentage.
   - NumPy calculations: team average completion rate and standard deviation.
   - Boolean indexing to identify below-average employees.
   - Cross-tabulation pivot table of cities versus completion status.
3. **Task Management (CRUD & Validation)**:
   - Pydantic models: `TaskIn` and `TaskOut` (strictly hides internal `notes`).
   - Validations: non-empty title after trimming, priority in `[1, 3]`, user existence verification.
   - File persistence in `tasks.json` with `FileNotFoundError` recovery on startup.
   - Custom `TaskNotFoundError` with HTTP 404 response handler.
   - Returns HTTP 201 on task creation.
   - Specialized `/tasks/overdue` endpoint computing `days_late` via `pd.to_datetime` and `dt.days` (declared before `/tasks/{task_id}`).
4. **Security & Infrastructure**:
   - Header authentication: `X-API-Key` compared to `APP_API_KEY` (returns HTTP 401 on mismatch) protecting all mutating routes (`POST`, `PUT`, `DELETE`).
   - Custom middleware measuring request execution duration, adding `X-Process-Time` to response headers, and logging method, path, and status code.
   - CORS origin locked strictly to `http://localhost:5173`.
   - Reusable pagination dependency (`page`, `page_size`).
5. **AI Assistant & Audit Logging**:
   - `POST /ai/ask`: Accepts prompt (validated up to 500 characters) and `conversation_id`.
   - Injects a dynamic system prompt containing live analytics (top 3 employees, team mean, top city).
   - Manages conversational history per conversation ID.
   - Powered by Anthropic Claude (with intelligent Mock response fallback when no key is set).
   - Audit logging via `try / except / finally` writing to `ai_logs.csv` (never records API keys or raw prompts).
   - `GET /ai/logs/stats`: Calculates total requests, success rate, and average prompt length.

---

## Project Structure

```
FinalProject-OneelAbed/
├── PLAN.md               # Original project plan
├── README.md             # Project documentation and guide
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── .env.example          # Environment template
├── .gitignore            # Git ignore file
├── tasks.json            # Task repository storage
├── ai_logs.csv           # AI audit metrics log
├── data_service.py       # Data fetching, caching, analytics, and task CRUD
├── ai_service.py         # LLM integration, conversation state, dynamic prompts, and logs
├── main.py               # FastAPI application, middleware, routes, and security
└── test_app.py           # Automated test suite
```

---

## Setup & Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` or edit `.env`:
```env
APP_API_KEY=my_secret_app_key_123
ANTHROPIC_API_KEY=
```

### 3. Start the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```
- Interactive API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Alternative Documentation (ReDoc): `http://127.0.0.1:8000/redoc`

### 4. Run Automated Tests
```bash
python test_app.py
```

---

## API Endpoints Reference

### Analytics Endpoints
- `GET /stats/users?page=1&page_size=10&sort_by=completion_rate&order=desc` - Paginated user performance.
- `GET /stats/summary` - Overall team benchmarks, mean, std, top 3 performers, and top city.
- `GET /stats/low-performers` - Employees performing below the team average.
- `GET /stats/cities` - Cities vs completion status pivot table.

### Task Management Endpoints
- `GET /tasks` - List tasks with pagination.
- `POST /tasks` - Create task *(requires `X-API-Key` header, returns 201)*.
- `GET /tasks/overdue` - Overdue incomplete tasks with `days_late`.
- `GET /tasks/{task_id}` - Retrieve task by ID.
- `PUT /tasks/{task_id}` - Update task *(requires `X-API-Key` header)*.
- `DELETE /tasks/{task_id}` - Delete task *(requires `X-API-Key` header)*.

### AI Assistant Endpoints
- `POST /ai/ask` - Query AI assistant with context memory and prompt length validation.
- `GET /ai/logs/stats` - Audit metrics on AI usage (success rate, average prompt length).
