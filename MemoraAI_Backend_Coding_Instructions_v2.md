# MemoraAI Backend — Coding Agent Instructions (v2)

Matches the existing repo structure: `memora_ai/Backend/services/{agent-service, auth-service, user-service}`, each with `config/`, `controllers/`, `models/`, `routes/`, `utils/`, `Dockerfile`, `main.py`, `requirements.txt`. `agent-service` additionally has `worker.py` — this is the Redis queue consumer, separate process from its FastAPI `main.py`.

---

## 0. Service responsibilities

| Service | Role |
|---|---|
| **auth-service** | Email + password signup/login, issues/validates JWT, session management |
| **user-service** | Gallery CRUD — upload, delete, folders/albums, profile — plus enqueues jobs for `agent-service` |
| **agent-service** | The AI brain: photo description generation, face detection, face matching, face-review queue, natural language search |

All three sit behind whatever gateway/reverse proxy you're using; each also independently validates the JWT issued by `auth-service` (don't trust the gateway alone).

---

## 1. auth-service

**Scope**: signup, login, logout, token refresh. Email + password only — no OAuth for this prototype.

```
auth-service/
├── config/         # db connection, jwt secret, env loading
├── controllers/     # signup_controller, login_controller, refresh_controller
├── models/          # User model (or thin wrapper if using Supabase Auth directly)
├── routes/          # /auth/signup, /auth/login, /auth/logout, /auth/refresh
├── utils/            # password hashing (if not delegating to Supabase Auth), jwt helpers
├── Dockerfile
├── main.py
└── requirements.txt
```

**Decide one**: either (a) proxy straight to Supabase Auth (simplest — Supabase already handles hashing, tokens, email verification), or (b) roll your own with `bcrypt` + custom JWT issuance. For a final-year prototype, **use Supabase Auth directly** — don't reimplement password hashing yourself. `auth-service` becomes a thin wrapper that calls Supabase Auth's API and returns normalized responses to the frontend.

**Endpoints**:
- `POST /auth/signup {email, password}` → creates Supabase Auth user, returns session
- `POST /auth/login {email, password}` → returns access + refresh token
- `POST /auth/refresh {refresh_token}` → new access token
- `POST /auth/logout` → invalidates session

---

## 2. user-service

**Scope**: everything the user directly manages — their gallery, folders, and triggering AI processing.

```
user-service/
├── config/
├── controllers/     # upload_controller, delete_controller, folder_controller, profile_controller
├── models/          # Image, Folder, UserProfile
├── routes/          # /images, /folders, /profile
├── utils/            # storage client (Supabase Storage), queue publisher
├── Dockerfile
├── main.py
└── requirements.txt
```

**Endpoints**:
- `POST /images/upload` → multipart upload → Supabase Storage → insert `images` row (`status='queued'`) → push job to Redis **`image_processing` queue** → return `image_id` immediately (don't block on AI processing)
- `DELETE /images/{id}`
- `POST /folders`, `GET /folders`, `PATCH /folders/{id}` (rename/move), `DELETE /folders/{id}`
- `POST /images/{id}/move` → assign image to a folder
- `GET /profile`, `PATCH /profile`

**Concurrency requirement**: multiple users uploading simultaneously must not block each other. This is why upload just writes to storage + Postgres + enqueues — the actual AI work happens asynchronously in `agent-service`. `user-service` should return in well under a second regardless of queue depth.

---

## 3. agent-service

**Scope**: all AI work. Two entry points — `main.py` (FastAPI, for the search endpoint and face-review endpoints the frontend calls directly) and `worker.py` (long-running Redis queue consumer, no HTTP).

```
agent-service/
├── config/          # Claude API key, ChromaDB host, InsightFace model path, Redis url
├── controllers/     # describe_controller, face_controller, search_controller
├── models/          # FaceCluster, FaceInstance, ImageTags
├── routes/          # /search, /faces/pending, /faces/{cluster_id}/confirm, /faces/{cluster_id}/reject
├── utils/            # claude_client, face_embedder (InsightFace), clustering (HDBSCAN), chroma_client
├── Dockerfile
├── main.py           # FastAPI app — serves /search and /faces/* to the frontend
├── worker.py          # Redis queue consumer — does the actual heavy lifting per image
└── requirements.txt
```

### 3.1 `worker.py` — image processing pipeline

Consumes the `image_processing` queue populated by `user-service`. For each `image_id`:

1. **Describe**: call Claude's vision API with a fixed prompt → structured JSON:
   ```json
   {
     "caption": "one sentence description",
     "scene": "e.g. beach, living room",
     "people_description": ["person in blue shirt smiling", "person in red dress"],
     "activity": "eating ice cream",
     "surroundings": "sunny park with trees",
     "mood": "cheerful"
   }
   ```
   Write this into `images.tags` (jsonb) and `images.caption`.

2. **Detect faces**: run InsightFace (RetinaFace + ArcFace) on the image. Handle **up to ~50 faces in one photo** — don't assume single-face images. For each detected face, get `(bbox, embedding, det_score)`.

3. **Match against known faces**: for each detected embedding, compare (cosine similarity) against that user's existing labeled `face_clusters.centroid_embedding`. Threshold ~0.6:
   - **Match found** → auto-assign that `cluster_id` to the new `face_instances` row. No user action needed. This is how "connects with older users" works — new photos of an already-known person are recognized automatically.
   - **No match** → this is a new/unknown face. Insert `face_instances` row with `cluster_id=null`, and run/update HDBSCAN over the user's currently-unlabeled embeddings to see if it groups with other unlabeled faces (e.g. multiple photos of the same unnamed person). Push `{cluster_id or temp_group_id}` onto the **`face_review` queue** (a Redis list, not a job queue — just a set the frontend polls via `/faces/pending`).

4. Update `images.status = 'ready'` once both description and face detection finish for that image.

### 3.2 Face-review flow (the "card" UI)

- `GET /faces/pending?user_id=` → returns unlabeled clusters, each with a representative face-crop thumbnail URL and count of photos it appears in.
- `POST /faces/{cluster_id}/confirm {name}` → labels the cluster, sets it as a known face going forward (its centroid becomes the match target for future uploads), propagates the name to every image containing that cluster.
- `POST /faces/{cluster_id}/reject` → if the auto-suggested match was wrong, unlink and either merge into a different existing cluster (`{merge_into_cluster_id}`) or leave it unlabeled for the user to name manually.
- Frontend also needs `POST /faces/new {name}` for the case where the user wants to manually add a person not yet detected in any pending cluster (e.g. pre-registering a name before any photo of them is uploaded) — store as a labeled cluster with no centroid yet, first matching face upload fills it in.

### 3.3 `/search` (RAG)

`POST /search {query, user_id}`:
1. Optionally call Claude to resolve relational terms ("my mother" → look up which labeled face the user has tagged with a "mother"/relationship field, if you add one to `face_clusters`).
2. Embed the (possibly rewritten) query.
3. Query ChromaDB, `where={"user_id": user_id}`.
4. Re-rank combining vector similarity with exact face-name matches from the query.
5. Return ranked `image_id`s with signed Supabase Storage URLs.

---

## 4. Redis queue design

Two distinct Redis structures — don't conflate them:

| Queue | Purpose | Producer | Consumer |
|---|---|---|---|
| `image_processing` | Heavy async job per uploaded image (describe + detect faces) | `user-service` on upload | `agent-service/worker.py` |
| `face_review` | Set/list of cluster IDs awaiting a human decision | `agent-service/worker.py` when a face has no confident match | Frontend polls via `agent-service` `/faces/pending`; not a job queue, just a state marker (a queued face doesn't need to "run" — it needs a human) |

Use **RQ (Redis Queue)** for `image_processing` since it's Python and gives you retries, failure tracking, and multiple worker processes for free — run several `worker.py` instances to handle concurrent uploads without one image blocking another. `face_review` is simpler: just add cluster IDs to a Redis set keyed by `user_id`, no job semantics needed.

---

## 5. Postgres schema additions

```sql
create table folders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) not null,
  name text not null,
  parent_folder_id uuid references folders(id),
  created_at timestamptz default now()
);

alter table images add column folder_id uuid references folders(id);
alter table images add column status text not null default 'queued'; -- queued | processing | ready | failed

alter table face_clusters add column relationship text; -- optional: "mother", "friend", etc, used by search
```

RLS on `folders` same pattern as `images` — user can only touch their own rows.

---

## 6. Docker Compose additions

```yaml
services:
  redis:
    image: redis:7-alpine
    volumes: ["redis_data:/data"]

  chromadb:
    image: chromadb/chroma:latest
    volumes: ["chroma_data:/chroma/chroma"]

  auth-service:
    build: ./Backend/services/auth-service
    depends_on: [redis]

  user-service:
    build: ./Backend/services/user-service
    depends_on: [redis]

  agent-service:
    build: ./Backend/services/agent-service
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    depends_on: [redis, chromadb]

  agent-worker:
    build: ./Backend/services/agent-service
    command: python worker.py
    depends_on: [redis, chromadb]
    deploy:
      replicas: 2   # multiple workers = concurrent uploads don't queue up behind each other

volumes:
  redis_data:
  chroma_data:
```

Note `agent-service` is built once but run as two different containers (`main.py` API vs `worker.py` consumer) — same image, different `command`.

---

## 7. Build order

1. `auth-service` wired to Supabase Auth, confirm signup/login works end to end.
2. `user-service` upload → Supabase Storage → Postgres row → confirm `image_processing` job actually lands in Redis (inspect with `redis-cli`).
3. `agent-service/worker.py`: describe step only first (Claude vision → tags in Postgres). Verify before adding face detection.
4. Add face detection + matching to `worker.py`. Test with a folder of ~20-30 photos with the same 3-4 people repeated, confirm clustering behaves sensibly.
5. `face_review` queue + `/faces/pending` + `/faces/{id}/confirm` endpoints, test the full "new face gets queued → user labels it → next photo of that person auto-matches" loop manually via curl/Postman before touching frontend.
6. ChromaDB upsert in the describe step, confirm `/search` returns sensible results.
7. Folders CRUD in `user-service`.
8. Frontend: auth screens → gallery/upload → swipe-card face review → search.
9. Full Docker Compose run: `docker compose up`, multiple simultaneous uploads from different test users, confirm workers process them concurrently without one blocking another.

---

## 8. Notes / things to confirm before building

- **Claude API**: used for both photo description generation and (optionally) query understanding in search. Confirm your Anthropic API key is set in `agent-service/config` — not in `user-service`, which never talks to Claude directly.
- **Face matching scope**: this spec assumes matching is scoped per-user (privacy — one user's face database never touches another's). Confirm that's the intent before building; if you actually want cross-user recognition (e.g. shared family galleries), that's a bigger design change and should be scoped separately.
- **50-face photos**: InsightFace handles multi-face detection natively, but clustering/matching 50 embeddings per image at upload time is real compute — budget for `worker.py` taking a few seconds per dense group photo, and make sure `image_processing` retries don't silently duplicate `face_instances` rows on failure (use idempotency: check if instances already exist for that `image_id` before inserting).
