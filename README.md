# SUCAS — Smart Urban Complaint Analysis System

Streamlit app that classifies a citizen complaint into a department
(TF-IDF + Random Forest), scores its priority, and auto-assigns the
best-matched worker. Complaints can be filed via a form, or through a
conversational AI chat assistant.

## What's in this package

```
app.py              Streamlit frontend + backend
priority_engine.py  Shared priority scoring + worker assignment rules
text_utils.py        Shared text-cleaning function (used at train + predict time)
llm_workflow.py      AI Chat Assistant — LangGraph + LLM, called live (not pickled)
train_model.py       Retraining script (optional — you already have trained pickles)
requirements.txt
.env.example         Template for the LLM's API token
.gitignore
rf_model.pkl          Trained RandomForestClassifier
tfidf_vectorizer.pkl  Fitted TfidfVectorizer
```

## Setup

1. Place `Employees.csv` in this folder (`rf_model.pkl` and
   `tfidf_vectorizer.pkl` are already included).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. **Optional — AI Chat Assistant:** copy `.env.example` to `.env` and add
   your HuggingFace API token. Without this, the app still runs fine —
   the chat page just shows a short setup message instead of the chat UI,
   and the "File Complaint" form works as normal.
4. Run the app:
   ```
   streamlit run app.py
   ```

## How the two complaint-filing paths work

- **📋 File Complaint** (form): you type the complaint directly, and it's
  classified immediately.
- **🤖 AI Chat Assistant**: you chat with SUCAS, which asks follow-up
  questions (area, landmark, name, phone) until it has enough
  information, then formats your answers into a complaint, classifies it
  with the same Random Forest model, scores priority, and assigns a
  worker — same pipeline, different front door.

Both paths end up calling the same `priority_engine.py` logic, so
priority scores and worker assignment are consistent regardless of how
the complaint was filed.

## About the model & data files

- `rf_model.pkl` / `tfidf_vectorizer.pkl` were saved with `joblib.dump()`,
  so `app.py` loads them with `joblib.load()` (not the plain `pickle`
  module — that fails on sklearn's internal numpy arrays).
- `Employees.csv` is expected with these columns: `Employee_ID, Name,
  Department, Designation, Ward, Current_Workload, Max_Workload, Status,
  Phone, Experience`. Worker assignment filters directly on `Department`
  (which matches the RF model's 31 output classes), prefers workers whose
  `Status` is `Available` and have headroom under `Max_Workload`, and
  picks the least-loaded ones first. The department → "typical role"
  mapping shown on the Dashboard and Priority Simulator is derived from
  this file at load time (`priority_engine.build_role_map`), not
  hand-typed — so it always matches whatever's actually in your roster.
- The AI Chat Assistant's LLM is called live over an API — it's not a
  local object, so there's nothing to pickle for it. If `langgraph`/
  `langchain-huggingface` aren't installed, or no API token is set, the
  rest of the app is unaffected; only the chat page is disabled.
- If you ever retrain, `train_model.py` trains and saves *only* Random
  Forest — the notebook's Logistic Regression/SVM/Naive Bayes comparisons
  were experiment-only and were never the deployed model.

## Fixes made along the way

- **Duplicate/broken priority box** in the Priority Score Simulator (a
  broken string-replace hack rendered two overlapping result boxes) —
  removed the broken one.
- **Dashboard crash when models fail to load** — it now stops safely
  instead of continuing to reference data that was never loaded.
- **Silent no-op on empty complaint submission** — now shows a warning.
- **Train/inference preprocessing mismatch** — training and the live app
  now share one `preprocess()` function (`text_utils.py`) instead of two
  slightly different copies.
- **Notebook's unfinished LangGraph chatbot** — `department_prediction`
  had a bare `return` (returned `None`, would have crashed the graph);
  `priority_prediction` and `worker_assignment` were referenced as graph
  nodes but never defined. All three are implemented in `llm_workflow.py`,
  with the priority step falling back to the deterministic scorer if the
  LLM ever returns something that isn't valid JSON.
