import os
import re
import json
import faiss
import numpy as np

from dotenv import load_dotenv

from fastapi import FastAPI

from pydantic import BaseModel

from typing import TypedDict, List

from sentence_transformers import SentenceTransformer

from langgraph.graph import (
    StateGraph,
    END
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)


# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()


# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="SHL Assessment Agent"
)


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():

    return {
        "message": "SHL Assessment Agent Running"
    }


# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():

    return {"status": "ok"}


# =====================================================
# GEMINI
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# =====================================================
# LOAD CATALOG
# =====================================================

with open(
    "catalog.json",
    "r",
    encoding="utf-8"
) as f:

    catalog = json.load(f)


# =====================================================
# BUILD DOCUMENTS
# =====================================================

documents = []

for item in catalog:

    text = f"""
    Assessment Name:
    {item.get("name", "")}

    Description:
    {item.get("description", "")}

    Job Levels:
    {", ".join(item.get("job_levels", []))}

    Categories:
    {", ".join(item.get("keys", []))}

    Remote:
    {item.get("remote", "")}

    Adaptive:
    {item.get("adaptive", "")}

    Languages:
    {", ".join(item.get("languages", []))}
    """

    documents.append({
        "text": text,
        "metadata": item
    })


# =====================================================
# EMBEDDING MODEL
# =====================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

dimension = (
    embedding_model
    .get_embedding_dimension()
)


# =====================================================
# FAISS INDEX
# =====================================================

index = faiss.IndexFlatIP(dimension)

stored_docs = []


def normalize_vector(vector):

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


for doc in documents:

    embedding = embedding_model.encode(
        doc["text"],
        convert_to_numpy=True
    )

    embedding = normalize_vector(
        embedding
    )

    index.add(
        np.array([embedding]).astype(
            "float32"
        )
    )

    stored_docs.append(doc)


# =====================================================
# SEARCH
# =====================================================

def semantic_search(query, k=20):

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    )

    query_embedding = normalize_vector(
        query_embedding
    )

    distances, indices = index.search(
        np.array([query_embedding]).astype(
            "float32"
        ),
        k
    )

    results = []

    for idx in indices[0]:

        if idx < len(stored_docs):

            results.append(
                stored_docs[idx]
            )

    return results


# =====================================================
# HELPERS
# =====================================================

def clean_json_response(text):

    text = text.strip()

    text = re.sub(
        r"```json",
        "",
        text
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    return text.strip()


def get_last_user_message(messages):

    for msg in reversed(messages):

        if msg["role"] == "user":
            return msg["content"]

    return ""


# =====================================================
# SENIORITY MAP
# =====================================================

SENIORITY_MAP = {

    "entry": "Entry-Level",
    "junior": "Entry-Level",

    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",

    "senior": "Manager",

    "manager": "Manager",

    "executive": "Executive"
}


# =====================================================
# FILTERING
# =====================================================

def metadata_filter(results, constraints):

    filtered = []

    seniority = (
        constraints.get(
            "seniority",
            ""
        )
        .lower()
        .strip()
    )

    mapped_seniority = None

    for key, value in SENIORITY_MAP.items():

        if key in seniority:

            mapped_seniority = value

            break

    needs_remote = constraints.get(
        "needs_remote",
        False
    )

    needs_personality = constraints.get(
        "needs_personality",
        False
    )

    needs_cognitive = constraints.get(
        "needs_cognitive",
        False
    )

    for result in results:

        meta = result["metadata"]

        if mapped_seniority:

            if mapped_seniority not in meta.get(
                "job_levels",
                []
            ):
                continue

        if needs_remote:

            if meta.get("remote") != "yes":
                continue

        keys = meta.get("keys", [])

        if needs_personality:

            if (
                "Personality & Behavior"
                not in keys
            ):
                continue

        if needs_cognitive:

            if (
                "Ability & Aptitude"
                not in keys
            ):
                continue

        filtered.append(result)

    return filtered


# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict):

    messages: list

    constraints: dict

    retrieved_docs: list

    recommendations: list

    clarification_needed: bool

    reply: str

    end_of_conversation: bool


# =====================================================
# ROUTER
# =====================================================

BLOCKED = [
    "salary",
    "legal",
    "ignore instructions",
    "hackerrank",
    "leetcode",
    "bypass",
    "politics",
    "religion"
]


def router(state):

    text = get_last_user_message(
        state["messages"]
    ).lower()

    if any(
        word in text
        for word in BLOCKED
    ):
        return "refuse"

    if (
        "compare" in text
        or "difference" in text
    ):
        return "compare"

    return "normal"


# =====================================================
# EXTRACT CONSTRAINTS
# =====================================================

def extract_constraints(state):

    messages = state["messages"]

    conversation = "\n".join(
        [
            f"{m['role']}: {m['content']}"
            for m in messages
        ]
    )

    prompt = f"""
    You are an SHL hiring assistant.

    Extract hiring requirements.

    Return ONLY valid JSON.

    JSON format:

    {{
      "role": "",
      "seniority": "",
      "skills": [],
      "needs_personality": false,
      "needs_cognitive": false,
      "needs_remote": false
    }}

    Rules:
    - Infer personality need if communication,
      leadership, stakeholder interaction,
      collaboration, teamwork mentioned.

    - Infer cognitive need if coding,
      aptitude, reasoning, developer,
      engineer, technical role mentioned.

    Conversation:
    {conversation}
    """

    response = llm.invoke(prompt)

    try:

        cleaned = clean_json_response(
            response.content
        )

        extracted = json.loads(
            cleaned
        )

    except:

        extracted = {}

    state["constraints"] = extracted

    return state


# =====================================================
# CLARIFICATION
# =====================================================

def clarification_node(state):

    constraints = state["constraints"]

    role = constraints.get(
        "role",
        ""
    )

    seniority = constraints.get(
        "seniority",
        ""
    )

    if not role:

        state["clarification_needed"] = True

        state["reply"] = (
            "What role are you hiring for?"
        )

        return state

    if not seniority:

        state["clarification_needed"] = True

        state["reply"] = (
            "What seniority level is this role?"
        )

        return state

    state["clarification_needed"] = False

    return state


# =====================================================
# RETRIEVAL
# =====================================================

def retrieval_node(state):

    constraints = state["constraints"]

    query = f"""
    Role:
    {constraints.get("role")}

    Seniority:
    {constraints.get("seniority")}

    Skills:
    {constraints.get("skills")}

    Personality:
    {constraints.get("needs_personality")}

    Cognitive:
    {constraints.get("needs_cognitive")}
    """

    semantic_results = semantic_search(
        query=query,
        k=20
    )

    filtered = metadata_filter(
        semantic_results,
        constraints
    )

    state["retrieved_docs"] = filtered[:10]

    return state


# =====================================================
# RERANK
# =====================================================

def rerank_node(state):

    docs = state["retrieved_docs"]

    constraints = state["constraints"]

    prompt = f"""
    You are an SHL assessment expert.

    Rank the most relevant assessments.

    Hiring Requirements:
    {constraints}

    Assessments:
    {docs}

    Return ONLY valid JSON list.

    Example:
    ["Assessment A", "Assessment B"]
    """

    response = llm.invoke(prompt)

    try:

        cleaned = clean_json_response(
            response.content
        )

        ranked_names = json.loads(
            cleaned
        )

    except:

        ranked_names = []

    if not ranked_names:

        ranked_names = [
            doc["metadata"]["name"]
            for doc in docs[:5]
        ]

    final = []

    for name in ranked_names:

        for doc in docs:

            if (
                doc["metadata"]["name"]
                == name
            ):

                final.append(
                    doc["metadata"]
                )

    state["recommendations"] = final[:10]

    return state


# =====================================================
# RECOMMENDATION
# =====================================================

def recommendation_node(state):

    formatted = []

    for item in state["recommendations"]:

        formatted.append({

            "name":
                item["name"],

            "url":
                item["link"],

            "test_type":
                ", ".join(
                    item.get("keys", [])
                )
        })

    state["reply"] = (
        "Here are the recommended "
        "SHL assessments based on "
        "your requirements."
    )

    state["recommendations"] = formatted

    state["end_of_conversation"] = True

    return state


# =====================================================
# COMPARISON
# =====================================================

def comparison_node(state):

    query = get_last_user_message(
        state["messages"]
    )

    docs = semantic_search(
        query,
        k=5
    )

    prompt = f"""
    Compare these SHL assessments.

    ONLY use the supplied catalog data.

    Assessments:
    {docs}

    Compare:
    - purpose
    - personality coverage
    - cognitive coverage
    - competencies
    - use cases
    """

    response = llm.invoke(prompt)

    state["reply"] = response.content

    state["recommendations"] = []

    state["end_of_conversation"] = False

    return state


# =====================================================
# REFUSAL
# =====================================================

def refusal_node(state):

    state["reply"] = (
        "I can only help with "
        "SHL assessment recommendations "
        "and assessment comparisons."
    )

    state["recommendations"] = []

    state["end_of_conversation"] = False

    return state


# =====================================================
# LANGGRAPH
# =====================================================

workflow = StateGraph(AgentState)

workflow.add_node(
    "extract",
    extract_constraints
)

workflow.add_node(
    "clarify",
    clarification_node
)

workflow.add_node(
    "retrieve",
    retrieval_node
)

workflow.add_node(
    "rerank",
    rerank_node
)

workflow.add_node(
    "recommend",
    recommendation_node
)

workflow.add_node(
    "compare",
    comparison_node
)

workflow.add_node(
    "refuse",
    refusal_node
)

workflow.set_entry_point(
    "extract"
)


workflow.add_conditional_edges(
    "extract",
    router,
    {
        "normal": "clarify",
        "compare": "compare",
        "refuse": "refuse"
    }
)


def clarification_router(state):

    if state["clarification_needed"]:
        return END

    return "retrieve"


workflow.add_conditional_edges(
    "clarify",
    clarification_router,
    {
        END: END,
        "retrieve": "retrieve"
    }
)

workflow.add_edge(
    "retrieve",
    "rerank"
)

workflow.add_edge(
    "rerank",
    "recommend"
)

workflow.add_edge(
    "recommend",
    END
)

workflow.add_edge(
    "compare",
    END
)

workflow.add_edge(
    "refuse",
    END
)

graph = workflow.compile()


# =====================================================
# REQUEST / RESPONSE SCHEMA
# =====================================================

class Message(BaseModel):

    role: str

    content: str


class ChatRequest(BaseModel):

    messages: List[Message]


class Recommendation(BaseModel):

    name: str

    url: str

    test_type: str


class ChatResponse(BaseModel):

    reply: str

    recommendations: List[
        Recommendation
    ]

    end_of_conversation: bool


# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)

def chat(payload: ChatRequest):

    state = {

        "messages": [
            msg.dict()
            for msg in payload.messages
        ],

        "constraints": {},

        "retrieved_docs": [],

        "recommendations": [],

        "clarification_needed": False,

        "reply": "",

        "end_of_conversation": False
    }

    result = graph.invoke(state)

    return {

        "reply":
            result["reply"],

        "recommendations":
            result.get(
                "recommendations",
                []
            ),

        "end_of_conversation":
            result.get(
                "end_of_conversation",
                False
            )
    }