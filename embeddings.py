import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import BASE_DIR
from data_loader import (
    load_project_data,
    load_construction_data,
    load_performance_data,
    load_procurement_data
)

from data_preprocessing import preprocess


# ==========================================
# Convert DataFrame Rows to Text
# ==========================================

def dataframe_to_text(df):

    documents = []

    for _, row in df.iterrows():

        text = ""

        for column in df.columns:
            text += f"{column}: {row[column]}. "

        documents.append(text)

    return documents


# ==========================================
# Create FAISS Vector Database
# ==========================================

def create_vector_database(model, df, index_file, docs_file):

    print(f"\nCreating {index_file}...")

    documents = dataframe_to_text(df)

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss.write_index(index, str(index_file))

    with open(docs_file, "wb") as file:
        pickle.dump(documents, file)

    print(f"Saved {index_file}")
    print(f"Saved {docs_file}")
    print(f"Total Documents : {len(documents)}")
    print(f"Vector Dimension : {dimension}")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    # Model and dataset loading only happens when this script is run
    # directly (e.g. `python embeddings.py`) to rebuild the indexes -
    # not every time some other module imports this file.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    project_df = preprocess(load_project_data())
    construction_df = preprocess(load_construction_data())
    performance_df = preprocess(load_performance_data())
    procurement_df = preprocess(load_procurement_data())

    create_vector_database(
        model,
        project_df,
        BASE_DIR / "project.index",
        BASE_DIR / "project_docs.pkl"
    )

    create_vector_database(
        model,
        construction_df,
        BASE_DIR / "construction.index",
        BASE_DIR / "construction_docs.pkl"
    )

    create_vector_database(
        model,
        performance_df,
        BASE_DIR / "performance.index",
        BASE_DIR / "performance_docs.pkl"
    )

    create_vector_database(
        model,
        procurement_df,
        BASE_DIR / "procurement.index",
        BASE_DIR / "procurement_docs.pkl"
    )

    print("\n===================================")
    print("All Vector Databases Created")
    print("===================================")
