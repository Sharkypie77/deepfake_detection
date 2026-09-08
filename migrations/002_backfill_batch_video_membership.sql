-- Run once on databases created with the legacy composite-key migration.
-- Preserve the old table while rebuilding it, so no video_ids data is lost.
ALTER TABLE batch_video_membership RENAME TO batch_video_membership_legacy;
CREATE TABLE batch_video_membership (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id VARCHAR(36) NOT NULL REFERENCES batch_jobs(id) ON DELETE CASCADE,
    video_id VARCHAR(36) NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'pending'
);
INSERT INTO batch_video_membership (batch_id, video_id, added_at)
SELECT b.id, json_each.value, CURRENT_TIMESTAMP
FROM batch_jobs b, json_each(b.video_ids)
WHERE b.video_ids IS NOT NULL;
CREATE INDEX ix_batch_video_membership_batch_id ON batch_video_membership(batch_id);
CREATE INDEX ix_batch_video_membership_video_id ON batch_video_membership(video_id);
CREATE INDEX ix_batch_video_membership_batch_video ON batch_video_membership(batch_id, video_id);
