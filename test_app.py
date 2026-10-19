import os
import json
from fastapi.testclient import TestClient

from main import app, APP_API_KEY
import data_service
import ai_service

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": os.environ.get("APP_API_KEY", "my_secret_app_key_123")}


def test_middleware_process_time_and_cors():
    print("Testing middleware X-Process-Time and CORS...")
    response = client.get("/stats/summary", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "X-Process-Time" in response.headers, "X-Process-Time header missing"
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173", "CORS header mismatch"
    print("[PASS] Middleware test passed.")


def test_stats_users_sorting_and_pagination():
    print("Testing /stats/users pagination and sorting...")
    response = client.get("/stats/users?page=1&page_size=3")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 3
    assert len(data["data"]) == 3
    assert "completion_rate" in data["data"][0]
    rates = [u["completion_rate"] for u in data["data"]]
    assert rates == sorted(rates, reverse=True), "Users are not sorted descending by completion_rate"
    print("[PASS] /stats/users test passed.")


def test_stats_summary():
    print("Testing /stats/summary...")
    response = client.get("/stats/summary")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    summary = response.json()
    assert "team_mean_completion_rate" in summary
    assert "team_std_completion_rate" in summary
    assert "top_performers" in summary
    assert len(summary["top_performers"]) <= 3
    assert "top_city" in summary
    print("[PASS] /stats/summary test passed.")


def test_stats_low_performers():
    print("Testing /stats/low-performers boolean indexing...")
    response = client.get("/stats/low-performers")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    summary_resp = client.get("/stats/summary")
    mean_rate = summary_resp.json()["team_mean_completion_rate"]
    for user in data["data"]:
        assert user["completion_rate"] < mean_rate, f"User {user['name']} is not below mean {mean_rate}"
    print("[PASS] /stats/low-performers test passed.")


def test_stats_cities_pivot():
    print("Testing /stats/cities pivot table...")
    response = client.get("/stats/cities")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0
    first_city = data["data"][0]
    assert "city" in first_city
    assert "completed" in first_city
    assert "incomplete" in first_city
    print("[PASS] /stats/cities test passed.")


def test_tasks_crud_lifecycle_and_security():
    print("Testing tasks CRUD, validations, and security...")
    # 1. Unauthenticated creation should return 401
    bad_auth = client.post("/tasks", json={
        "title": "Unauthorized Task",
        "user_id": 1,
        "due_date": "2026-05-01",
        "priority": 2
    })
    assert bad_auth.status_code == 401, f"Expected 401 for missing auth, got {bad_auth.status_code}"

    # 2. Validation: Empty title after strip -> 400
    empty_title = client.post("/tasks", headers=AUTH_HEADERS, json={
        "title": "   ",
        "user_id": 1,
        "due_date": "2026-05-01",
        "priority": 2
    })
    assert empty_title.status_code == 400, f"Expected 400 for empty title, got {empty_title.status_code}"

    # 3. Validation: Priority out of range [1, 3] -> 400
    bad_prio = client.post("/tasks", headers=AUTH_HEADERS, json={
        "title": "Invalid Priority Task",
        "user_id": 1,
        "due_date": "2026-05-01",
        "priority": 5
    })
    assert bad_prio.status_code == 400, f"Expected 400 for bad priority, got {bad_prio.status_code}"

    # 4. Validation: Non-existent user -> 400
    bad_user = client.post("/tasks", headers=AUTH_HEADERS, json={
        "title": "Invalid User Task",
        "user_id": 999999,
        "due_date": "2026-05-01",
        "priority": 2
    })
    assert bad_user.status_code == 400, f"Expected 400 for invalid user, got {bad_user.status_code}"

    # 5. Successful creation -> 201 Created and TaskOut excludes notes
    create_resp = client.post("/tasks", headers=AUTH_HEADERS, json={
        "title": "Automated Test Task",
        "user_id": 1,
        "due_date": "2026-01-01",
        "priority": 3,
        "done": False,
        "notes": "Secret notes that must not leak in TaskOut"
    })
    assert create_resp.status_code == 201, f"Expected 201 Created, got {create_resp.status_code}"
    created_task = create_resp.json()
    assert "task_id" in created_task
    assert "notes" not in created_task, "notes field should be excluded from TaskOut response"
    task_id = created_task["task_id"]

    # 6. Retrieve created task
    get_resp = client.get(f"/tasks/{task_id}")
    assert get_resp.status_code == 200, f"Expected 200, got {get_resp.status_code}"
    assert get_resp.json()["title"] == "Automated Test Task"
    assert "notes" not in get_resp.json()

    # 7. Update task (PUT)
    update_resp = client.put(f"/tasks/{task_id}", headers=AUTH_HEADERS, json={
        "title": "Automated Test Task Updated",
        "user_id": 1,
        "due_date": "2026-01-01",
        "priority": 1,
        "done": True
    })
    assert update_resp.status_code == 200, f"Expected 200, got {update_resp.status_code}"
    assert update_resp.json()["title"] == "Automated Test Task Updated"
    assert update_resp.json()["priority"] == 1

    # 8. Delete task (DELETE)
    del_resp = client.delete(f"/tasks/{task_id}", headers=AUTH_HEADERS)
    assert del_resp.status_code == 200

    # 9. TaskNotFoundError (404) after deletion
    not_found_resp = client.get(f"/tasks/{task_id}")
    assert not_found_resp.status_code == 404, f"Expected 404, got {not_found_resp.status_code}"
    assert "not found" in not_found_resp.json()["detail"].lower()
    print("[PASS] Tasks CRUD test passed.")


def test_overdue_tasks_route_and_days_late():
    print("Testing /tasks/overdue and days_late calculation...")
    # Add an overdue task
    client.post("/tasks", headers=AUTH_HEADERS, json={
        "title": "Overdue Verification Task",
        "user_id": 1,
        "due_date": "2025-01-01",
        "priority": 1,
        "done": False
    })

    response = client.get("/tasks/overdue")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    overdue_list = response.json()
    assert len(overdue_list) > 0, "Expected at least 1 overdue task"
    first = overdue_list[0]
    assert "days_late" in first, "days_late missing from overdue task"
    assert first["days_late"] > 0, "days_late should be > 0"
    assert first["done"] is False, "Overdue task should have done == False"
    print("[PASS] /tasks/overdue test passed.")


def test_ai_assistant_validation_and_mock_response():
    print("Testing /ai/ask validation and mock response...")
    # Empty prompt -> 400
    empty_resp = client.post("/ai/ask", json={"prompt": "   ", "conversation_id": "test_conv"})
    assert empty_resp.status_code == 400, f"Expected 400, got {empty_resp.status_code}"

    # Prompt > 500 chars -> 400
    long_prompt = "a" * 501
    long_resp = client.post("/ai/ask", json={"prompt": long_prompt, "conversation_id": "test_conv"})
    assert long_resp.status_code == 400, f"Expected 400, got {long_resp.status_code}"

    # Valid prompt -> 200
    valid_resp = client.post("/ai/ask", json={"prompt": "What is the team performance?", "conversation_id": "test_conv"})
    assert valid_resp.status_code == 200, f"Expected 200, got {valid_resp.status_code}"
    res_data = valid_resp.json()
    assert res_data["conversation_id"] == "test_conv"
    assert "response" in res_data
    assert len(res_data["response"]) > 0
    print("[PASS] /ai/ask test passed.")


def test_ai_logs_stats_and_csv_privacy():
    print("Testing /ai/logs/stats and CSV privacy...")
    stats_resp = client.get("/ai/logs/stats")
    assert stats_resp.status_code == 200, f"Expected 200, got {stats_resp.status_code}"
    stats = stats_resp.json()
    assert "total_requests" in stats
    assert "success_rate_percentage" in stats
    assert "average_prompt_length" in stats
    assert stats["total_requests"] > 0

    # Verify CSV content privacy (ensure no prompt content or keys)
    with open("ai_logs.csv", "r", encoding="utf-8") as f:
        content = f.read()
        assert "What is the team performance?" not in content, "Prompt content found in ai_logs.csv!"
        assert "my_secret_app_key_123" not in content, "API Key found in ai_logs.csv!"
    print("[PASS] /ai/logs/stats and CSV privacy test passed.")


def test_upstream_down_handling():
    print("Testing 503 handling when upstream is unavailable...")
    original_cache = data_service._CACHE.copy()
    original_fetch = data_service.fetch_json
    try:
        data_service._CACHE = {"users_df": None, "todos_df": None, "last_fetched": None}
        data_service.fetch_json = lambda path: None

        resp = client.get("/stats/summary")
        assert resp.status_code == 503, f"Expected 503, got {resp.status_code}"
        assert "unavailable" in resp.json()["detail"].lower()
    finally:
        data_service._CACHE = original_cache
        data_service.fetch_json = original_fetch
    print("[PASS] Upstream 503 error handling test passed.")


if __name__ == "__main__":
    print("\n--- RUNNING AUTOMATED TEST SUITE ---\n")
    test_middleware_process_time_and_cors()
    test_stats_users_sorting_and_pagination()
    test_stats_summary()
    test_stats_low_performers()
    test_stats_cities_pivot()
    test_tasks_crud_lifecycle_and_security()
    test_overdue_tasks_route_and_days_late()
    test_ai_assistant_validation_and_mock_response()
    test_ai_logs_stats_and_csv_privacy()
    test_upstream_down_handling()
    print("\n=== ALL 10 TESTS PASSED SUCCESSFULLY! ===\n")
