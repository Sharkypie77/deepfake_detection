# Deepfake Detection Suite

This document explains the project as it exists in this repository: its purpose, technology stack, runtime flow, data model, configuration, deployment, and known limitations.

## 1. Purpose

The project is a multimodal video-analysis service intended to estimate whether a video has been manipulated. It combines:

1. **Vision analysis** for faces, spatial artifacts, frequency artifacts, and compression clues.
2. **Audio analysis** for synthetic or cloned voice indicators and acoustic anomalies.
3. **Lip-sync analysis** for disagreement between mouth movement and the audio stream.
4. **Fusion** to combine those component scores into one risk score and verdict.

The system is designed to be language-agnostic. Audio features do not require a language-specific classifier, while Whisper is used for multilingual transcription/language detection.

This is an analysis aid, not proof of authenticity. Production accuracy requires validated trained model weights and a representative evaluation set; the application reports unvalidated readiness explicitly when that provenance is absent.

## 2. Repository layout

| Path | Responsibility |
| --- | --- |
| `config.py` | Environment variables, directories, model settings, thresholds, and security limits |
| `models.py` | SQLAlchemy ORM models and database initialization |
| `pipeline.py` | End-to-end orchestrator for all analysis modules |
| `main.py` | FastAPI application, upload endpoints, result endpoints, and batch endpoints |
| `modules/vision_ai.py` | Face detection, visual features, FFT/DCT analysis, and vision score |
| `modules/audio_analysis.py` | Audio extraction, mel/MFCC/CQCC features, Whisper, and audio score |
| `modules/sync_analysis.py` | MediaPipe face mesh, mouth motion, audio/visual correlation, and sync score |
| `modules/fusion.py` | Weighted score fusion, verdict, confidence, summary, and findings |
| `celery_tasks.py` | Redis/Celery batch workers and batch aggregation |
| `telegram_bot.py` | Telegram upload and result-delivery integration |
| `fine_tuning.py` | PyTorch Lightning dataset and transfer-learning infrastructure |
| `init.py` | Directory/database/model setup helper |
| `test.py` | Manual end-to-end test script that creates synthetic media |
| `frontend/` | React dashboard for uploading videos and viewing results |
| `Dockerfile` | CUDA-based backend image |
| `docker-compose.yml` | Redis, PostgreSQL, API, workers, and Flower services |
| `.env.example` | Configuration template |

## 3. Technology stack

### Backend and API

- **Python 3.10+** application code.
- **FastAPI** exposes asynchronous HTTP endpoints and OpenAPI/Swagger documentation.
- **Uvicorn** runs the ASGI server.
- **Pydantic** validates request models.
- **python-multipart** handles uploaded files.
- **SQLAlchemy** provides ORM access to SQLite or PostgreSQL.
- **SQLite** is the default local database.
- **PostgreSQL** is intended for the Docker production-style deployment.

### Machine learning and media

- **PyTorch** supplies tensor operations and model execution.
- **torchvision** supplies image transforms and model utilities.
- **timm** supplies the EfficientNetV2 vision backbone.
- **facenet-pytorch MTCNN** detects faces without the TensorFlow-only `mtcnn` package.
- **OpenCV** reads videos, samples frames, and computes image-domain features.
- **MediaPipe Face Mesh** extracts facial landmarks for mouth movement.
- **librosa** calculates mel spectrograms, MFCCs, chroma/CQCC-like features, and fallback audio decoding.
- **torchaudio** extracts/resamples audio when the installed binary matches PyTorch.
- **Whisper** detects spoken language and creates transcription.
- **SciPy** provides signal processing and correlation helpers.
- **NumPy** is used throughout for numerical processing.

### Distributed processing and integrations

- **Celery** queues analysis jobs.
- **Redis** is the Celery broker and result backend.
- **Flower** monitors Celery workers.
- **python-telegram-bot** provides Telegram ingestion and notifications.
- Telegram is the supported messaging integration. WhatsApp is intentionally disabled.

### Frontend

- **React 18** renders the dashboard.
- **react-scripts 5** provides the Create React App build and development server.
- **Axios** calls the FastAPI API.
- The dashboard uses a small CSS file plus inline styles and is responsive on narrow screens.

## 4. End-to-end request flow

```text
Browser / Telegram / API client
              |
              v
        FastAPI upload
              |
              v
  Validate extension and size
              |
              v
       Save file and Video row
              |
      +-------+--------+
      |                |
 async/background   synchronous
      |                |
      +-------+--------+
              v
    DeepfakeDetectionPipeline
              |
      +-------+--------+--------+
      |                |        |
    Vision           Audio     Sync
      |                |        |
      +-------+--------+--------+
              v
       Multimodal fusion
              |
              v
  Save component and final rows
              |
              v
 status/results response
```

## 5. Pipeline internals

### 5.1 Pipeline initialization

`DeepfakeDetectionPipeline` creates one instance of each module. It normalizes the requested device:

- If `DEVICE=cuda` and CUDA is available, it uses CUDA.
- Otherwise it logs a warning and falls back to CPU.

The pipeline then initializes the vision, audio, sync, and fusion modules. Initialization failures are logged critically and raised so deployment fails fast; analysis is not exposed until the dependency/model problem is corrected.

### 5.2 Vision module

`modules/vision_ai.py`:

1. Opens the video with OpenCV.
2. Samples frames.
3. Converts BGR frames to RGB.
4. Uses PyTorch MTCNN to find a face and landmarks.
5. Measures temporal face-location stability and facial/skin boundary indicators.
6. Computes frequency-domain signals such as FFT/DCT artifact measurements.
7. Runs the EfficientNetV2-style classifier path.
8. Returns a dictionary containing component metrics and `vision_deepfake_score` in the range `0..1`.

The configured model name is `efficientnetv2_s`, which is the valid timm identifier. The current runtime uses `pretrained=False` because the compatible timm package does not have local weights available. Download and cache an approved checkpoint before treating the classifier score as production-grade.

### 5.3 Audio module

`modules/audio_analysis.py`:

1. Attempts to load the video's audio with torchaudio.
2. Converts stereo to mono and resamples to 16 kHz.
3. Falls back to librosa if torchaudio cannot decode the file.
4. Calculates mel spectrogram, MFCC, and chroma/CQCC-like features.
5. Uses a Whisper model to detect language and produce transcription.
6. Calculates acoustic indicators such as spectral cutoff, phase discontinuities, pitch/formant consistency, and pause/breath behavior.
7. Produces `audio_deepfake_score`.

The class is named `AAISSTModule` for backward compatibility, while it loads the published Clova AASIST topology and checkpoint layout. The checkpoint provenance safeguard prevents an unvalidated or test checkpoint from being reported as production-ready.

### 5.4 Lip-sync module

`modules/sync_analysis.py`:

1. Samples video frames.
2. Runs MediaPipe Face Mesh.
3. Extracts fixed mouth landmark indices.
4. Measures mouth-opening distance over time.
5. Normalizes mouth motion and computes mouth-region stability.
6. Compares visual motion against an audio envelope using temporal correlation.
7. Reports mismatch regions, match percentage, dubbing likelihood, and `sync_deepfake_score`.

The method uses language-independent motion features, so it does not need a phoneme inventory for Hindi, Tamil, English, or another language.

### 5.5 Fusion and verdicts

`modules/fusion.py` uses the default weights:

```text
vision = 0.35
audio  = 0.35
sync   = 0.30
```

Each component is clipped to `0..1`. The weighted average is increased slightly when component scores disagree strongly. The final normalized risk is converted to a percentage:

```text
raw = 0.35 * vision + 0.35 * audio + 0.30 * sync
final = clip(raw + inconsistency_bonus * 0.15, 0, 1)
risk_score = final * 100
```

Verdict thresholds:

| Normalized risk | Verdict | Meaning |
| --- | --- | --- |
| `< 0.30` | `AUTHENTIC` | Low estimated manipulation risk |
| `0.30 .. < 0.65` | `INCONCLUSIVE` | Evidence is ambiguous |
| `>= 0.65` | `DEEPFAKE` | High estimated manipulation risk |

Confidence is derived from the risk level and cross-modal agreement. It is not a calibrated probability.

## 6. API endpoints

### `GET /`

Returns API metadata and endpoint links.

### `GET /health`

Returns service status, API version, timestamp, and whether the pipeline initialized:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "pipeline_ready": true
}
```

### `POST /api/v1/analyze`

Accepts a multipart `file`. Supported extensions are `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, and `.webm`; the default maximum is 500 MB.

With `async_mode=true`, the API saves a `Video` row and queues FastAPI background processing:

```json
{
  "status": "queued",
  "video_id": "uuid",
  "check_status_url": "/api/v1/status/uuid"
}
```

With `async_mode=false`, the request waits for analysis and returns the pipeline result.

### `GET /api/v1/status/{video_id}`

Returns `pending`, `processing`, `completed`, or `failed`, with progress and error information.

### `GET /api/v1/results/{video_id}`

Returns the final verdict, percentage risk, confidence, summary, key findings, component scores, and generation timestamp after completion.

### Batch endpoints

- `POST /api/v1/batch/submit` accepts a job name, one or more existing video IDs, and `low`, `normal`, or `high` priority.
- `GET /api/v1/batch/{batch_id}/status` reports completed, failed, processing, and percentage progress.
- `GET /api/v1/batch/{batch_id}/results` aggregates verdict counts, average risk, deepfake percentage, and flagged videos.

Batch membership is stored in the normalized `batch_video_membership` table; it is not inferred by matching a batch UUID to a video UUID. Migration `migrations/001_batch_video_membership.sql` creates the current table, while `migrations/002_backfill_batch_video_membership.sql` rebuilds legacy installations and backfills rows from `BatchJob.video_ids` before the legacy data is retired. The API endpoint `/api/v1/videos/{video_id}/batches` exposes membership history.

Operational endpoints include `/metrics` (Prometheus text format) and `/api/v1/results/{video_id}?format=pdf`. Analysis requests are limited to ten per client per minute. Celery Beat runs `cleanup_retention` daily and removes uploads older than `RETENTION_DAYS`.

Telegram downloads and progress updates use exponential retries; `/status <video_id>` reports pipeline state and failed message edits are replaced with a new message. WhatsApp webhooks are intentionally disabled; use Telegram or the REST API.

## 7. Database schema

`models.py` creates these tables:

- **`videos`**: filename, path, size, media metadata, status, progress, submitter, and timestamps.
- **`vision_analysis`**: spatial/frequency metrics and vision score.
- **`audio_analysis`**: spoofing/acoustic metrics, languages, duration, sample rate, and audio score.
- **`sync_analysis`**: transcription, language, mouth metrics, mismatches, dubbing likelihood, and sync score.
- **`final_results`**: one final row per video, verdict, risk, confidence, component scores, weights, summary, findings, and timeline.
- **`audit_logs`**: upload and processing actions for traceability.
- **`batch_jobs`**: job name, status, counts, progress, and timestamps.
- **`batch_video_membership`**: first-class many-to-many rows with their own ID, `added_at`, and per-batch `status`; a video can appear in multiple independent batch runs.

Model checkpoint paths are configured with `AASIST_CHECKPOINT_PATH` and
`VISION_CHECKPOINT_PATH`. AASIST is mandatory: startup fails if its validated
checkpoint is missing or incompatible. Vision uses pretrained ImageNet weights
when no fine-tuned checkpoint override is configured and logs that limitation.

Each checkpoint may have a sidecar manifest beside it (`model.pth` plus
`model.json`). The manifest must contain `model_name`, `training_dataset`,
`training_date`, `validation_metric`, a SHA-256 `checksum` of the checkpoint,
and `validated: true`. Readiness is split: `pipeline_ready` means the model
objects initialized, while `models_validated` additionally requires a complete
manifest, matching checksum, and policy metric (AASIST EER <= 0.20; vision
AUC >= 0.70). `/health` reports `mode: production` only when all models pass;
otherwise it reports `unvalidated`. Set `validated` only after evaluation on a
documented held-out dataset. Unvalidated analysis results carry a disclaimer
and must not be treated as production-grade.

SQLite is initialized automatically when `main.py` imports `init_db`. For PostgreSQL, the SQLAlchemy URL must include the driver, for example:

```text
postgresql://deepfake:password@postgres:5432/deepfake_detection
```

## 8. Frontend behavior

The React dashboard:

1. Lets a user select a video through a hidden file input.
2. Rejects files over 500 MB client-side.
3. Uploads using Axios.
4. Polls `/status/{video_id}` every two seconds for asynchronous jobs.
5. Stops polling on completion, failure, or a ten-minute timeout.
6. Displays verdict, risk, confidence, component meters, key findings, metadata, and a JSON export button.

The component meters accept normalized scores (`0..1`) and clamp invalid values. This prevents the API's percentage risk score from accidentally being treated as a normalized component score.

This repository contains a browser-based Create React App only. There is no
`src-tauri/` directory, `tauri.conf.json`, Tauri Rust project, or
`@tauri-apps/*` dependency. The supported launch command is `npm start` from
`frontend/`; this checkout does not own a Tauri ACL configuration and cannot
produce `plugin:dialog|message` commands. That error comes from an external or
partially-added Tauri shell, not this frontend. Upload validation and failures
use an in-app alert banner rather than native dialogs, so the browser receives
visible feedback without a dialog plugin permission. A future Tauri wrapper
would need its own capabilities file with `dialog:allow-message` plus only the
file-system/HTTP permissions it actually uses.

Frontend commands:

```bash
cd frontend
npm install
npm start       # development server
npm run build   # production build
```

## 9. Telegram flow

`telegram_bot.py`:

1. Handles `/start` and presents inline actions.
2. Accepts Telegram videos/documents.
3. Enforces the 500 MB limit.
4. Downloads the file into `data/uploads`.
5. Creates a Telegram `Video` and audit record.
6. Runs the pipeline asynchronously.
7. Stores component/final results.
8. Edits the progress message into a verdict message.

Set `TELEGRAM_BOT_TOKEN` to a real BotFather token before starting it. Telegram
network calls use retries, `/status <video_id>` reports processing state, and
failed progress-message edits fall back to a replacement message.

## 10. Celery and Redis flow

Start a Redis broker, then run:

```bash
celery -A celery_tasks worker --loglevel=info --concurrency=2
celery -A celery_tasks flower --port=5555
```

`batch_analyze_videos` creates a `BatchJob` and dispatches one `analyze_video_task` per existing video. Individual tasks:

- Mark the video as processing.
- Create a pipeline.
- Analyze the file.
- Save component/final rows.
- Mark the video completed.
- Retry failures up to three times with exponential delay.

The API batch status/results handlers call the aggregation functions directly so an HTTP request does not enqueue a second Celery task and block waiting for a worker.

## 11. Fine-tuning infrastructure

`fine_tuning.py` expects:

```text
data/training/
├── authentic/
│   ├── hindi/*.mp4
│   ├── tamil/*.mp4
│   └── ...
└── deepfake/
    ├── wav2lip/*.mp4
    ├── sadtalker/*.mp4
    └── rvc/*.mp4
```

`IndianDeepfakeDataset` discovers files, labels authentic/deepfake samples,
splits them into train/validation sets, extracts a representative video frame,
and uses the same MTCNN face-crop strategy as inference. `DeepfakeDetectorFineTune`
wraps the pretrained EfficientNetV2-RW-S backbone in PyTorch Lightning and
replaces its classifier with two output classes.

Before training, verify that the dataset has both classes and enough samples
for stratified splitting. `train_aasist.py` trains the published AASIST
topology on ASVspoof LA audio. `evaluate.py` reports measured accuracy, AUC,
and EER from a held-out predictions CSV; it never fabricates or automatically
marks a checkpoint validated.

## 12. Model performance

No training run has been performed in this workspace: ASVspoof and
FaceForensics++ require dataset terms acceptance. FaceForensics++ access is
now available for this project, but the dataset has not been downloaded and
the available environment has no training GPU. Consequently no
local accuracy, AUC, EER, false-positive rate, or fusion score is claimed.
The published Clova AASIST benchmark (0.83% EER on ASVspoof2019-LA) is a
reference result, not a result measured by this project. The downloaded
official checkpoint is therefore intentionally unvalidated until an authorized
held-out evaluation produces a real metric and manifest checksum.

## 13. Configuration

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | HTTP port |
| `DATABASE_URL` | local SQLite URL | SQLAlchemy database |
| `DEVICE` | `cuda` | Requested `cuda` or `cpu`; runtime falls back to CPU |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker/backend |
| `TELEGRAM_BOT_TOKEN` | placeholder | Telegram credentials |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

The application creates `data/uploads`, `data/models`, `data/results`, and `logs` at import time.

## 13. Local startup

Recommended local sequence:

```bash
python -m pip install -r requirements-fixed.txt
python init.py

# Terminal 1
uvicorn main:app --reload

# Terminal 2, optional for batches
redis-server
celery -A celery_tasks worker --loglevel=info

# Terminal 3
cd frontend
npm install
npm start
```

Check the API:

```bash
curl http://localhost:8000/health
```

The first Whisper/model initialization can be slow and may download model assets. CPU inference is supported but substantially slower than a suitable CUDA installation.

## 14. Docker deployment

The intended Compose services are:

- Redis with persistent volume.
- PostgreSQL with persistent volume.
- FastAPI backend.
- Two Celery workers.
- Flower monitoring UI.

The Dockerfile is based on an NVIDIA CUDA runtime image and installs FFmpeg/OpenCV system libraries. Use a CPU-specific image or change worker device settings when deploying without NVIDIA support. Before deployment, review database credentials, mount paths, CORS origins, health-check behavior, and secrets handling in Compose.

## 15. Testing and verification

The repository's manual test script can be run with:

```bash
python test.py
```

It creates short synthetic video/audio files, exercises vision/audio/sync/fusion, runs the complete pipeline, and removes generated test files. It is a smoke test, not an accuracy benchmark.

Useful checks:

```bash
python -m compileall -q .
cd frontend && npm run build
curl http://127.0.0.1:8000/health
```

## 16. Current limitations and next engineering priorities

1. Replace the simplified audio anti-spoofing network with a validated AASIST checkpoint.
2. Download and load a validated EfficientNetV2 checkpoint; calibrate scores on held-out data.
3. Add automated API tests for upload validation, status transitions, result persistence, and batch membership.
4. Add a proper job-to-video association table if one video can belong to multiple batches with historical tracking.
5. Harden Telegram network retries, timeout handling, `/status`, and cleanup of uploaded files.
6. Supply and evaluate production AASIST and vision checkpoints, then publish their manifests with verified checksums.
7. Restrict CORS origins and move deployment secrets out of Compose source.
8. Add PDF report generation, observability, rate limiting, and retention policies.
9. Align the fine-tuning model configuration with inference and document checkpoint formats.
