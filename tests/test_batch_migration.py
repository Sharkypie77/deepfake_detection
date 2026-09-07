import sqlite3
from pathlib import Path


def test_legacy_video_ids_are_backfilled():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE videos (id VARCHAR(36) PRIMARY KEY);
        CREATE TABLE batch_jobs (id VARCHAR(36) PRIMARY KEY, video_ids JSON);
        CREATE TABLE batch_video_membership (
            batch_id VARCHAR(36), video_id VARCHAR(36), created_at TIMESTAMP
        );
        INSERT INTO videos VALUES ('video-1'), ('video-2');
        INSERT INTO batch_jobs VALUES ('batch-1', '[\"video-1\", \"video-2\"]');
        """
    )
    migration = Path("migrations/002_backfill_batch_video_membership.sql").read_text()
    connection.executescript(migration)
    rows = connection.execute(
        "SELECT batch_id, video_id, status FROM batch_video_membership"
    ).fetchall()
    assert rows == [("batch-1", "video-1", "pending"), ("batch-1", "video-2", "pending")]
