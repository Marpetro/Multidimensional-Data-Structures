import pandas as pd
import ast

def to_float_comma(x):
    """
    Μετατρέπει αριθμούς τύπου '6,3707' σε float 6.3707
    """
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_movies_dataset(xlsx_path: str) -> pd.DataFrame:
    # Διαβάζουμε το Excel
    df = pd.read_excel(xlsx_path)

    # --- Numeric columns ---
    numeric_cols = [
        "budget",
        "revenue",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
    ]

    for col in numeric_cols:
        df[col] = df[col].apply(to_float_comma)

    # --- Release year ---
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    # --- List-like columns ---
    list_cols = [
        "genre_names",
        "origin_country",
        "production_company_names",
    ]

    for col in list_cols:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x)
            if isinstance(x, str) and x.startswith("[")
            else []
        )

    # --- Language ---
    df["original_language"] = (
        df["original_language"].fillna("").str.lower()
    )

    # --- Filtering (όπως στην εκφώνηση) ---
    filtered = df[
        (df["release_year"] >= 2000)
        & (df["release_year"] <= 2020)
        & (df["popularity"] >= 3)
        & (df["popularity"] <= 6)
        & (df["vote_average"] >= 3)
        & (df["vote_average"] <= 5)
        & (df["runtime"] >= 30)
        & (df["runtime"] <= 60)
        & (
            df["origin_country"].apply(
                lambda lst: "US" in lst or "GB" in lst
            )
        )
        & (df["original_language"] == "en")
    ].copy()

    return filtered


def dataframe_to_vectors(df: pd.DataFrame):
    """
    Επιστρέφει:
    - ids
    - titles
    - 5D vectors για indexing
    """
    ids = df["id"].tolist()
    titles = df["title"].tolist()

    vectors = df[
        ["release_year", "popularity", "vote_average", "runtime", "budget"]
    ].values.tolist()

    return ids, titles, vectors


if __name__ == "__main__":
    xlsx_path = "data/data_movies_clean.xlsx"

    df = load_movies_dataset(xlsx_path)
    print(df.head())
    print(df.info())

    ids, titles, vectors = dataframe_to_vectors(df)
    print("Δείγμα vectors:", vectors[:3])
