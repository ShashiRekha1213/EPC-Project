import pandas as pd
from data_loader import (
    load_project_data,
    load_construction_data,
    load_performance_data,
    load_procurement_data
)

# ============================
# Clean Column Names
# ============================

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percent", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("°", "", regex=False)
    )
    return df


# ============================
# Convert Date Columns
# ============================

def convert_dates(df):

    date_columns = [
        "start_date",
        "end_date",
        "publication_date",
        "deadline_date",
        "timestamp"
    ]

    for col in date_columns:

        if col in df.columns:

            # dd/mm/yyyy
            if col in ["start_date", "end_date"]:
                df[col] = pd.to_datetime(
                    df[col],
                    dayfirst=True,
                    errors="coerce"
                )

            # yyyy-mm-dd
            else:
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

    return df

# ============================
# Fill Missing Values
# ============================
def fill_missing(df):

    for col in df.columns:

        # Skip datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        elif df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")

        else:
            median_value = df[col].median()

            # If the entire column is NaN, median() itself returns NaN
            if pd.isna(median_value):
                median_value = 0

            df[col] = df[col].fillna(median_value)

    return df


# ============================
# Remove Duplicate Rows
# ============================

def remove_duplicates(df):

    return df.drop_duplicates()


# ============================
# Complete Preprocessing
# ============================

def preprocess(df):

    df = clean_columns(df)

    df = convert_dates(df)

    df = fill_missing(df)

    df = remove_duplicates(df)

    return df


# ============================
# Display Dataset Summary
# ============================

def dataset_summary(name, df):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst Five Rows:")
    print(df.head())


# ============================
# Main
# ============================

if __name__ == "__main__":

    project = preprocess(load_project_data())

    construction = preprocess(load_construction_data())

    performance = preprocess(load_performance_data())

    procurement = preprocess(load_procurement_data())

    dataset_summary("Project Dataset", project)

    dataset_summary("Construction Dataset", construction)

    dataset_summary("Performance Dataset", performance)

    dataset_summary("Procurement Dataset", procurement)

    print("\n")
    print("=" * 60)
    print("All datasets loaded and preprocessed successfully.")
    print("=" * 60)