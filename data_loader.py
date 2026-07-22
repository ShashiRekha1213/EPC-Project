import pandas as pd

from config import DATA_DIR

# -------------------------------
# Dataset Paths
# -------------------------------

PROJECT_DATASET = DATA_DIR / "Project Management (1).csv"
CONSTRUCTION_DATASET = DATA_DIR / "construction_project_dataset.csv"
PERFORMANCE_DATASET = DATA_DIR / "construction_project_performance_dataset.csv"
PROCUREMENT_DATASET = DATA_DIR / "procurement-notices.csv"


def load_project_data():
    return pd.read_csv(PROJECT_DATASET)


def load_construction_data():
    return pd.read_csv(CONSTRUCTION_DATASET)


def load_performance_data():
    return pd.read_csv(PERFORMANCE_DATASET)


def load_procurement_data():
    return pd.read_csv(PROCUREMENT_DATASET)


if __name__ == "__main__":

    project = load_project_data()
    construction = load_construction_data()
    performance = load_performance_data()
    procurement = load_procurement_data()

    print("\nProject Dataset")
    print(project.head())

    print("\nConstruction Dataset")
    print(construction.head())

    print("\nPerformance Dataset")
    print(performance.head())

    print("\nProcurement Dataset")
    print(procurement.head())