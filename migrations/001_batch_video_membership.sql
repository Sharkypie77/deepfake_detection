-- Replace legacy BatchJob.video_ids JSON storage with a normalized relation.
CREATE TABLE IF NOT EXISTS batch_video_membership (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id VARCHAR(36) NOT NULL REFERENCES batch_jobs(id) ON DELETE CASCADE,
    video_id VARCHAR(36) NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS ix_batch_video_membership_batch_id ON batch_video_membership(batch_id);
CREATE INDEX IF NOT EXISTS ix_batch_video_membership_video_id ON batch_video_membership(video_id);
CREATE INDEX IF NOT EXISTS ix_batch_video_membership_batch_video ON batch_video_membership(batch_id, video_id);

ALTER TABLE final_results ADD COLUMN disclaimer TEXT;
