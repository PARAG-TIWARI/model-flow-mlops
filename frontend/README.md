# ModelFlow dashboard

This directory contains the React + Vite frontend for the ModelFlow MLOps API.

The dashboard provides pages for:
- Overview and service status
- Model metadata and expected input features
- Single-record churn prediction
- MLflow experiment comparison
- System and serving telemetry

## Local development

From the repository root, start the API first:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open this directory and install the locked dependencies:

```bash
cd frontend
npm ci
```

Create `.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Start the Vite development server:

```bash
npm run dev
```

The dashboard normally runs at `http://localhost:5173`. API requests are sent to the base URL configured by `VITE_API_BASE_URL`.

## Quality checks

```bash
npm run lint
npm run build
```

Use the root README and the API's `/docs` route for backend setup and API details.
