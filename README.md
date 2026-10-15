# Neuro-Fuzzy-GA Hybrid for Adaptive Test Suite Prioritization in CI Pipelines

A full-stack demo/portfolio project implementing a three-stage hybrid
system that reorders a CI test suite so the tests most likely to catch
a fault run first:

1. **Neural network** (pure NumPy MLP) — learns a fault-detection risk
   score per test from historical failure rate, code churn, coverage,
   staleness, and execution time.
2. **Fuzzy logic** (Mamdani-style inference, hand-written, no `skfuzzy`
   dependency) — turns the crisp risk score plus execution time and
   coverage into an interpretable priority score (low/medium/high).
3. **Genetic algorithm** — searches for the run *order* of the full
   suite that maximizes APFD (Average Percentage of Faults Detected),
   a standard test-prioritization metric.

The backend exposes a small FastAPI service; the frontend is a
dependency-free HTML/CSS/JS dashboard (Chart.js via CDN, no build step)
that lets you load a synthetic sample suite or upload your own CSV,
tune the GA's generations/population size, and see before/after
metrics, a convergence chart, and the reordered test list.

## Project structure

```
ci-test-prioritizer/
├── backend/
│   ├── app.py                  FastAPI app (all HTTP endpoints)
│   ├── sample_data.py          Synthetic test-suite generator
│   ├── sample_test_suite.csv   Example CSV you can upload
│   ├── requirements.txt
│   └── hybrid/
│       ├── neural_net.py       Stage 1: fault-risk MLP
│       ├── fuzzy_logic.py      Stage 2: fuzzy priority inference
│       ├── genetic_algorithm.py Stage 3: GA run-order optimizer
│       └── pipeline.py         Orchestrates all three stages
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

## Running it

### 1. Backend

Requires Python 3.10+.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. Check it with:

```bash
curl http://localhost:8000/api/health
```

### 2. Frontend

No build step, no npm install — it's plain HTML/CSS/JS. Just serve the
folder with any static server (opening `index.html` directly also
works in most browsers, but a static server avoids CORS quirks on some
setups):

```bash
cd frontend
python3 -m http.server 5500
```

Then open `http://localhost:5500` in your browser. The dashboard talks
to the backend at `http://localhost:8000` by default — if you run the
backend on a different host/port, set `window.API_BASE` at the top of
`app.js` (or add `<script>window.API_BASE = "http://your-host:port";</script>`
before `app.js` loads in `index.html`).

### 3. Using it

- Click **Load sample suite** to generate a synthetic 40-test suite, or
  **Upload CSV** to bring your own (must have columns: `name`,
  `exec_time`, `historical_fail_rate`, `code_churn`, `coverage`,
  `days_since_last_run` — see `backend/sample_test_suite.csv` for an
  example).
- Adjust GA generations / population size if you want.
- Click **Run prioritization** to run the full Neural → Fuzzy → GA
  pipeline and see the reordered suite plus before/after metrics.

## Adapting this to a real repo

Right now `sample_data.py` fabricates plausible test metadata. To run
this against a real project for your literature-review / demo:

- `historical_fail_rate` — compute from CI history (e.g. parse JUnit
  XML reports across recent builds, or query your CI provider's API).
- `code_churn` — from `git log --numstat` on the files each test
  covers, over some recent window.
- `coverage` — from a coverage tool (Coverage.py, Istanbul/nyc, JaCoCo)
  mapped to which tests exercise which changed lines.
- `days_since_last_run` — from your CI's test-run timestamps.

Feed those into `/api/upload` as a CSV (or extend `/api/prioritize` to
accept your own ingestion format) and the same three-stage pipeline
runs unmodified.

## Notes for the research-paper side of this project

This implementation is written to be genuinely inspectable for a
literature review:

- The neural net is a plain, from-scratch backprop MLP (`neural_net.py`)
  — you can point to the exact forward/backward equations.
- The fuzzy system (`fuzzy_logic.py`) is a real Mamdani-style rule base
  with triangular membership functions and documented rules — easy to
  compare against papers that use `skfuzzy` or MATLAB's Fuzzy Logic
  Toolbox.
- The GA (`genetic_algorithm.py`) uses order crossover (OX), swap
  mutation, tournament selection, and elitism — standard building
  blocks worth citing individually.
- `pipeline.py` reports both **APFD** (the standard test-prioritization
  metric from the literature) and a practical "time to 80% of
  fault-detection value" number, so you can discuss the hybrid's
  effect in both academic and CI-engineering terms.
