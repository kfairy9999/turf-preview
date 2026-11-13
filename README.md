# Turf Preview

Phone-first web app: upload a daylight house-and-yard photo, pick a turf style, and get a photorealistic preview of professionally installed artificial grass. The house, landscaping, hardscape, and daylight stay the same; only the lawn is replaced.

Photos are not stored. Preview only — actual install may vary. Neutral branding only.

## Run locally

Set two environment variables (never commit them):

- `ACCESS_PIN` — the access PIN the app asks for
- `HF_TOKEN` — a Hugging Face token with access to the Qwen Image Edit Spaces

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8765
```

Open `http://127.0.0.1:8765`, enter the PIN, upload a photo, pick a style, and generate.

## Hosting on Render

`render.yaml` defines a free Python web service. After connecting this repo, set `ACCESS_PIN` and `HF_TOKEN` as secret environment variables (they are not in the Blueprint). The free instance sleeps after about 15 minutes of idle time; the first request after that can take a minute to wake.

Generation uses a free Qwen Image Edit model on Hugging Face (Lightning, with a Lampyy fallback), not Cloudflare Workers AI. A generate can take 1–3 minutes.

## API

- `GET /` — the phone UI
- `GET /health` — `{ok, pin_set, token_set}`
- `POST /api/check` — PIN check (`x-access-pin`)
- `POST /api/generate` — multipart photo + style + quality; returns JPEG as base64
