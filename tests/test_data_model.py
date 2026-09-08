from datetime import datetime

from models import BatchJob, BatchVideoMembership, Video, init_db


def test_batch_video_membership_is_timestamped_many_to_many(tmp_path):
    _, session_factory = init_db(f"sqlite:///{tmp_path / 'test.db'}")
    session = session_factory()
    try:
        first = Video(
            id="video-1", filename="one.mp4", file_path="/tmp/one.mp4",
            file_size_mb=1, submission_platform="test",
        )
        second = Video(
            id="video-2", filename="two.mp4", file_path="/tmp/two.mp4",
            file_size_mb=1, submission_platform="test",
        )
        batch = BatchJob(id="batch-1", job_name="test", total_videos=2)
        batch.memberships = [
            BatchVideoMembership(video=first),
            BatchVideoMembership(video=second),
        ]
        session.add(batch)
        session.commit()

        assert {membership.video.id for membership in batch.memberships} == {"video-1", "video-2"}
        rows = session.query(BatchVideoMembership).all()
        assert len(rows) == 2
        assert all(isinstance(row.added_at, datetime) for row in rows)
    finally:
        session.close()


def test_video_can_have_independent_membership_statuses(tmp_path):
    _, session_factory = init_db(f"sqlite:///{tmp_path / 'test.db'}")
    session = session_factory()
    try:
        video = Video(
            id="video-1", filename="one.mp4", file_path="/tmp/one.mp4",
            file_size_mb=1, submission_platform="test",
        )
        first = BatchJob(id="batch-1", job_name="one", total_videos=1)
        second = BatchJob(id="batch-2", job_name="two", total_videos=1)
        session.add_all([
            first, second,
            BatchVideoMembership(batch=first, video=video, status="completed"),
            BatchVideoMembership(batch=second, video=video, status="failed"),
        ])
        session.commit()

        memberships = session.query(BatchVideoMembership).filter_by(video_id="video-1").all()
        assert {membership.batch_id for membership in memberships} == {"batch-1", "batch-2"}
        assert {membership.status for membership in memberships} == {"completed", "failed"}
    finally:
        session.close()
