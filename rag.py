import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from config import BASE_DIR

# =====================================
# Load Embedding Model
# =====================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================
# Load Vector Database
# =====================================

def load_database(index_file, docs_file):

    index = faiss.read_index(str(index_file))

    with open(docs_file, "rb") as file:
        documents = pickle.load(file)

    return index, documents


# =====================================
# Load All Databases
# =====================================

project_index, project_docs = load_database(
    BASE_DIR / "project.index",
    BASE_DIR / "project_docs.pkl"
)

construction_index, construction_docs = load_database(
    BASE_DIR / "construction.index",
    BASE_DIR / "construction_docs.pkl"
)

performance_index, performance_docs = load_database(
    BASE_DIR / "performance.index",
    BASE_DIR / "performance_docs.pkl"
)

procurement_index, procurement_docs = load_database(
    BASE_DIR / "procurement.index",
    BASE_DIR / "procurement_docs.pkl"
)


# =====================================
# Search One Database
# =====================================

def search_database(index, documents, query, top_k=2):

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    context = []

    for idx in indices[0]:

        if idx != -1:
            context.append(documents[idx])

    return context


# =====================================
# Retrieve Context
# =====================================

def retrieve_context(query, top_k=5):

    query_embedding = model.encode([query]).astype("float32")

    databases = [

        ("Project", project_index, project_docs),
        ("Construction", construction_index, construction_docs),
        ("Performance", performance_index, performance_docs),
        ("Procurement", procurement_index, procurement_docs)

    ]

    retrieved = []

    for name, index, docs in databases:

        distances, indices = index.search(query_embedding, top_k)

        avg_score = float(distances[0].mean())

        retrieved.append({

            "dataset": name,
            "score": avg_score,
            "indices": indices[0],
            "docs": docs

        })

    # Lower L2 distance = more similar, so the two BEST-matching
    # datasets are the ones with the lowest average score.
    retrieved.sort(key=lambda x: x["score"])

    final_context = ""

    for item in retrieved[:2]:

        final_context += f"\n===== {item['dataset']} Dataset =====\n\n"

        for idx in item["indices"]:

            # FAISS returns -1 when a database has fewer than top_k
            # documents; skip those instead of wrapping to the last doc.
            if idx == -1 or idx >= len(item["docs"]):
                continue

            final_context += item["docs"][idx] + "\n\n"

    return final_context