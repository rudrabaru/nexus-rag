from src.maintenance import RETENTION_DAYS


def test_retention_windows_are_recorded_for_every_pruned_table():
    assert set(RETENTION_DAYS) == {"pipeline_events", "query_logs", "fetch_log"}
    assert all(days > 0 for _, _, days in RETENTION_DAYS.values())
