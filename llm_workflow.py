"""
llm_workflow.py
────────────────
Conversational complaint intake via an LLM, built on the notebook's
LangGraph sketch — with its two blocking bugs actually fixed:

  1. department_prediction() had a bare `return` (returned None, so the
     graph would crash) -> now returns {"department": ...}.
  2. priority_prediction() and worker_assignment() were referenced as
     graph nodes but never defined anywhere -> both are implemented below.

Nothing here is pickled. The LLM is called live through an API
(HuggingFace Inference Endpoint via langchain_huggingface); an
API-backed model isn't a local object you can serialize, so there is
no model artifact to save for this part of the pipeline.

All the LangChain/LangGraph imports are optional at import time: if the
packages aren't installed, or no API token is configured, `llm_available`
is False and app.py shows a setup message instead of the chat page.
"""

import json
import os
from datetime import datetime
from typing import Annotated, TypedDict

from text_utils import preprocess
from priority_engine import compute_priority, assign_workers

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False

# Which HF-hosted chat model to use. Override with an env var — the
# notebook's placeholder repo id was never a real deployable endpoint.
LLM_REPO_ID = os.environ.get("SUCAS_LLM_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

llm_available = _IMPORTS_OK and bool(HF_TOKEN)

# ──────────────────────────────────────────
# Prompts (from the notebook, unchanged)
# ──────────────────────────────────────────
INTAKE_SYSTEM_PROMPT = """
You are SUCAS.

You are an AI complaint registration assistant.

Your job is to register complaints.

You should first greet the user and then ask about their problem

Collect:

- Complaint
- Area
- Landmark (optional)
- User's Name
- Phone Number

Do NOT answer unrelated questions.

Ask one question at a time.

Once you have enough information,
tell the user:

"I have enough information to register your complaint."
"""

FORMATTER_PROMPT = """
You are a complaint formatter.

Read the conversation.

Return ONLY the complaint text.

Do not explain.

Do not greet.

Do not use markdown.

Do not return JSON.

Return only one paragraph.
"""

PRIORITY_PROMPT_TEMPLATE = """
You are a civic complaint analyzer.

Return JSON only.

{{
"priority":"",
"reason":"",
"estimated_time":""
}}

Complaint:
{complaint}

Department:
{department}
"""


class ComplaintState(TypedDict):
    messages: Annotated[list, add_messages] if _IMPORTS_OK else list
    complaint: str
    department: str
    priority: dict
    worker: dict


def get_chat_model():
    """Create the HF-backed chat model. Call once and cache (e.g. with
    st.cache_resource in app.py) — don't call this per message."""
    if not _IMPORTS_OK:
        raise RuntimeError(
            "langgraph / langchain-huggingface aren't installed. "
            "Run: pip install langgraph langchain langchain-huggingface python-dotenv"
        )
    if not HF_TOKEN:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set. Add it to a .env file "
            "(see .env.example)."
        )
    endpoint = HuggingFaceEndpoint(repo_id=LLM_REPO_ID, task="text-generation")
    return ChatHuggingFace(llm=endpoint)


def new_chat_state():
    """Fresh conversation state for a new complaint-intake session."""
    if not _IMPORTS_OK:
        return {"messages": [], "completed": False}
    return {
        "messages": [SystemMessage(content=INTAKE_SYSTEM_PROMPT)],
        "completed": False,
    }


def _build_processing_graph(chat_model, rf_model, tfidf, stop_words, lemmatizer, employees_df):
    """The notebook's processing_graph, with working node implementations."""

    def complaint_gen(state: ComplaintState):
        messages = state["messages"] + [HumanMessage(content=FORMATTER_PROMPT)]
        response = chat_model.invoke(messages)
        return {"complaint": response.content.strip()}

    def department_prediction(state: ComplaintState):
        processed = preprocess(state["complaint"], stop_words, lemmatizer)
        vectorized = tfidf.transform([processed])
        department = rf_model.predict(vectorized)[0]
        return {"department": department}

    def priority_prediction(state: ComplaintState):
        prompt = PRIORITY_PROMPT_TEMPLATE.format(
            complaint=state["complaint"], department=state["department"]
        )
        try:
            response = chat_model.invoke([HumanMessage(content=prompt)])
            parsed = json.loads(response.content.strip().strip("`"))
            if parsed.get("priority") not in ("High", "Medium", "Low"):
                raise ValueError("model returned an unrecognized priority label")
            return {"priority": parsed}
        except Exception:
            # LLM output wasn't valid/expected JSON — fall back to the
            # deterministic scorer used by the manual complaint form, so
            # the ticket still gets a sensible priority either way.
            label, total, ds, ms, hs = compute_priority(state["department"], datetime.now())
            return {
                "priority": {
                    "priority": label,
                    "reason": f"Rule-based fallback (dept={ds}, month={ms}, hour={hs}, total={total}/9).",
                    "estimated_time": "Within 24-72 hours" if label == "High" else "Within 1-2 weeks",
                }
            }

    def worker_assignment(state: ComplaintState):
        label = state["priority"]["priority"]
        workers_df, role = assign_workers(state["department"], label, employees_df)
        cols = [c for c in ["name", "role", "status", "ward", "workload", "max_workload", "phone"] if c in workers_df.columns]
        return {
            "worker": {
                "role": role,
                "workers": workers_df[cols].to_dict("records"),
            }
        }

    graph = StateGraph(ComplaintState)
    graph.add_node("complaint_gen", complaint_gen)
    graph.add_node("department_prediction", department_prediction)
    graph.add_node("priority_prediction", priority_prediction)
    graph.add_node("worker_assignment", worker_assignment)

    graph.add_edge(START, "complaint_gen")
    graph.add_edge("complaint_gen", "department_prediction")
    graph.add_edge("department_prediction", "priority_prediction")
    graph.add_edge("priority_prediction", "worker_assignment")
    graph.add_edge("worker_assignment", END)

    return graph.compile()


def run_chat_turn(state, user_text, chat_model, rf_model, tfidf, stop_words, lemmatizer, employees_df):
    """Advance the conversation by one user message.

    Returns (updated_state, assistant_reply_text, ticket_or_None).
    `ticket_or_None` is populated once the assistant has gathered enough
    information and the complaint has been classified, scored, and routed.
    """
    state["messages"].append(HumanMessage(content=user_text))
    ai_response = chat_model.invoke(state["messages"])
    state["messages"].append(ai_response)

    ticket = None
    if "enough information" in ai_response.content.lower():
        proc_graph = _build_processing_graph(
            chat_model, rf_model, tfidf, stop_words, lemmatizer, employees_df
        )
        result = proc_graph.invoke({"messages": state["messages"]})
        ticket = {
            "complaint": result["complaint"],
            "department": result["department"],
            "priority": result["priority"],
            "worker": result["worker"],
        }
        state["completed"] = True

    return state, ai_response.content, ticket
