import pandas as pd

from data_loader import (
    load_project_data,
    load_construction_data,
    load_performance_data,
    load_procurement_data
)

from data_preprocessing import preprocess


# ======================================
# Safe Column Selection
# ======================================

def select_features(df, feature_list, dataset_name):
    """Return only the requested columns that actually exist in df.
    Warns (instead of raising KeyError) about any that are missing,
    so a schema change in the source CSV doesn't crash the pipeline."""

    available = [col for col in feature_list if col in df.columns]
    missing = [col for col in feature_list if col not in df.columns]

    if missing:
        print(f"[WARNING] {dataset_name}: missing expected columns {missing}")

    return df[available]


# ======================================
# Project Dataset Features
# ======================================

project_features = [
    "project_status",
    "priority",
    "hours_spent",
    "budget",
    "actual_cost",
    "progress"
]


# ======================================
# Construction Dataset Features
# ======================================

construction_features = [
    "temperature",
    "humidity",
    "vibration_level",
    "material_usage",
    "worker_count",
    "energy_consumption",
    "task_progress",
    "cost_deviation",
    "time_deviation",
    "risk_score",
    "equipment_utilization_rate",
    "simulation_deviation"
]


# ======================================
# Performance Dataset Features
# ======================================

performance_features = [
    "temperature_c",
    "humidity_percent",
    "vibration_level_hz",
    "material_usage_kg",
    "energy_consumption_kwh",
    "worker_count",
    "task_progress_percent",
    "resource_utilization_percent",
    "risk_score",
    "simulation_accuracy_percent"
]


# ======================================
# Procurement Dataset Features
# ======================================

procurement_features = [
    "notice_type",
    "procurement_type",
    "country_name",
    "major_sector"
]


# ======================================
# Main
# ======================================

if __name__ == "__main__":

    project_df = preprocess(load_project_data())
    construction_df = preprocess(load_construction_data())
    performance_df = preprocess(load_performance_data())
    procurement_df = preprocess(load_procurement_data())

    print("\n================ PROJECT FEATURES ================\n")
    print(select_features(project_df, project_features, "Project").head())

    print("\n================ CONSTRUCTION FEATURES ================\n")
    print(select_features(construction_df, construction_features, "Construction").head())

    print("\n================ PERFORMANCE FEATURES ================\n")
    print(select_features(performance_df, performance_features, "Performance").head())

    print("\n================ PROCUREMENT FEATURES ================\n")
    print(select_features(procurement_df, procurement_features, "Procurement").head())

    # ======================================
    # ML Target Variables
    # ======================================

    print("\n================ TARGET VARIABLES ================\n")

    print("Project Progress")
    print(project_df["progress"].head())

    print("\nRisk Score")
    print(construction_df["risk_score"].head())

    print("\nPerformance Score")
    print(performance_df["performance_score"].head())

    # ======================================
    # Agent Mapping
    # ======================================

    print("\n================ AGENT DATASETS ================\n")

    print("Knowledge Agent")
    print(project_df.columns.tolist())

    print("\nCompliance Agent")
    print(construction_df.columns.tolist())

    print("\nSchedule Agent")
    print(project_df.columns.tolist())

    print("\nSupply Chain Agent")
    print(procurement_df.columns.tolist())

    print("\nCommissioning Agent")
    print(performance_df.columns.tolist())

    print("\nFeature Engineering Completed Successfully.")