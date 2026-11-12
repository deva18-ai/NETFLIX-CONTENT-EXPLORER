"""Netflix Content Explorer analysis utilities and menu-driven application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


def find_dataset_file(dataset_dir: str | os.PathLike[str] = DATASET_DIR) -> Path:
    """Find the Netflix dataset in the dataset directory, preferring CSV then Excel files."""
    directory = Path(dataset_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    files = sorted(directory.iterdir(), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No files found in dataset directory: {directory}")

    preferred_exts = [".csv", ".xlsx", ".xls"]
    for ext in preferred_exts:
        matches = [path for path in files if path.suffix.lower() == ext]
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No supported Netflix dataset file (.csv, .xlsx, .xls) found in {directory}."
    )


def load_dataset(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the Netflix dataset selecting the correct Pandas reader based on file type."""
    dataset_path = Path(path) if path is not None else find_dataset_file()
    dataset_path = dataset_path if dataset_path.exists() else find_dataset_file(dataset_path.parent)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(dataset_path)
    elif suffix in {".xlsx", ".xls"}:
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "Excel support requires the 'openpyxl' package. Install it with 'pip install openpyxl'."
            ) from exc
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset file type: {dataset_path.suffix}")

    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series | int]]:
    """Apply the required cleaning logic and return the cleaned DataFrame and diagnostics."""
    cleaned = df.copy()

    cleaned["country"] = cleaned["country"].fillna("Unknown")
    cleaned["director"] = cleaned["director"].fillna("Not Listed")
    cleaned["cast"] = cleaned["cast"].fillna("Not Listed")
    cleaned["rating"] = cleaned["rating"].fillna("Not Rated")

    raw_date_added = cleaned["date_added"].copy()
    parsed_dates = pd.to_datetime(raw_date_added, errors="coerce")
    invalid_dates = raw_date_added[parsed_dates.isna()].copy()
    cleaned["date_added"] = parsed_dates
    cleaned = cleaned.dropna(subset=["date_added"]).copy()

    cleaned["year_added"] = cleaned["date_added"].dt.year

    cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")
    invalid_release_years = cleaned["release_year"].isna()

    cleaned["country"] = cleaned["country"].astype(str).replace({"nan": "Unknown", "None": "Unknown"})
    cleaned["director"] = cleaned["director"].astype(str).replace({"nan": "Not Listed", "None": "Not Listed"})
    cleaned["cast"] = cleaned["cast"].astype(str).replace({"nan": "Not Listed", "None": "Not Listed"})
    cleaned["rating"] = cleaned["rating"].astype(str).replace({"nan": "Not Rated", "None": "Not Rated"})

    diagnostics = {
        "invalid_date_values": invalid_dates,
        "invalid_release_years": cleaned.loc[invalid_release_years, "release_year"],
        "rows_before_cleaning": len(df),
        "rows_after_cleaning": len(cleaned),
    }
    return cleaned, diagnostics


def split_and_count(series: pd.Series, unknown_label: str | None = None) -> pd.Series:
    """Split comma-separated values, strip whitespace, and count each item."""
    exploded = series.astype(str).str.split(",")
    exploded = exploded.explode()
    exploded = exploded.str.strip()
    exploded = exploded.replace({"nan": "", "None": ""})
    if unknown_label is not None:
        exploded = exploded[exploded != unknown_label]
    return exploded[exploded != ""].value_counts()


def count_movies_vs_shows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the number of Movies vs TV Shows."""
    return df["type"].value_counts().rename_axis("type").reset_index(name="count")


def titles_added_each_year(df: pd.DataFrame) -> pd.Series:
    """Return the number of titles added per year after cleaning."""
    return df["year_added"].value_counts().sort_index()


def top_producing_countries(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Return the most common countries, excluding Unknown values from the count."""
    return split_and_count(df["country"], unknown_label="Unknown").head(top_n)


def top_genres(df: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """Return the most common genres in the listed_in column."""
    return split_and_count(df["listed_in"]).head(top_n)


def rating_distribution(df: pd.DataFrame) -> pd.Series:
    """Return the distribution of the rating column."""
    return df["rating"].value_counts()


def oldest_and_newest_release_year(df: pd.DataFrame) -> tuple[int | float, int | float]:
    """Return the minimum and maximum corrected release year values."""
    valid_release_years = pd.to_numeric(df["release_year"], errors="coerce").dropna()
    if valid_release_years.empty:
        raise ValueError("No valid release years available to summarize.")
    return int(valid_release_years.min()), int(valid_release_years.max())


def search_titles(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """Search titles using a keyword in a case-insensitive way."""
    keyword = (keyword or "").strip()
    if not keyword:
        return pd.DataFrame(columns=df.columns)
    return df[df["title"].astype(str).str.contains(keyword, case=False, na=False)].copy()


def content_added_in_year(df: pd.DataFrame, year: int | str) -> pd.DataFrame:
    """Return all titles added in the requested year."""
    try:
        year_value = int(year)
    except (TypeError, ValueError):
        raise ValueError("Year must be an integer value.")
    return df[df["year_added"] == year_value].copy()


def count_unique_countries(df: pd.DataFrame) -> tuple[int, int]:
    """Return (count excluding Unknown, count of Unknown rows)."""
    unique_countries = set(
        country.strip()
        for country in df["country"].astype(str).str.split(",")
        .explode()
        .str.strip()
        .tolist()
        if country.strip() and country.strip() != "Unknown"
    )
    unknown_count = int((df["country"].astype(str).str.strip() == "Unknown").sum())
    return len(unique_countries), unknown_count


def release_year_summary(df: pd.DataFrame) -> dict[str, float]:
    """Calculate summary statistics for the release_year column using NumPy."""
    values = pd.to_numeric(df["release_year"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("No valid release_year values available for summary statistics.")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def print_menu() -> None:
    """Display the console menu for the Netflix project."""
    print("\nNETFLIX CONTENT EXPLORER")
    print("=" * 28)
    print("1. Count Movies vs TV Shows")
    print("2. Number of titles added each year")
    print("3. Top 10 producing countries")
    print("4. Top 5 most common genres")
    print("5. Distribution of content ratings")
    print("6. Oldest and newest release years")
    print("7. Search titles by keyword")
    print("8. Display content added in a specific year")
    print("9. Total number of unique countries")
    print("10. Summary statistics for release years")
    print("0. Exit")
    print("=" * 28)


def run_menu(df: pd.DataFrame) -> None:
    """Run the interactive menu-driven application."""
    while True:
        print_menu()
        choice = input("Choose an option (0-10): ").strip()

        if choice == "0":
            print("Thank you for using Netflix Content Explorer. Goodbye!")
            break

        try:
            choice_num = int(choice)
        except ValueError:
            print("Invalid choice. Please enter a number from 0 to 10.")
            continue

        if choice_num == 1:
            result = count_movies_vs_shows(df)
            print(result)
        elif choice_num == 2:
            result = titles_added_each_year(df)
            print(result)
        elif choice_num == 3:
            print(top_producing_countries(df, 10))
        elif choice_num == 4:
            print(top_genres(df, 5))
        elif choice_num == 5:
            print(rating_distribution(df))
        elif choice_num == 6:
            oldest, newest = oldest_and_newest_release_year(df)
            print(f"Oldest release year: {oldest}")
            print(f"Newest release year: {newest}")
        elif choice_num == 7:
            keyword = input("Enter a keyword to search title names: ").strip()
            matches = search_titles(df, keyword)
            if matches.empty:
                print("No matching titles found.")
            else:
                print(matches[["title", "type", "release_year", "country"]].head(20).to_string(index=False))
        elif choice_num == 8:
            try:
                year = int(input("Enter a year to view titles added in that year: ").strip())
            except ValueError:
                print("Invalid year. Please enter a whole number.")
                continue
            matches = content_added_in_year(df, year)
            if matches.empty:
                print(f"No titles were added in {year}.")
            else:
                print(matches[["title", "type", "country", "year_added"]].head(20).to_string(index=False))
        elif choice_num == 9:
            unique_count, unknown_count = count_unique_countries(df)
            print(f"Unique countries (excluding 'Unknown'): {unique_count}")
            print(f"Unknown country values: {unknown_count}")
        elif choice_num == 10:
            summary = release_year_summary(df)
            print(pd.DataFrame([summary]).to_string(index=False))
        else:
            print("Invalid menu selection. Please choose a number between 0 and 10.")

        input("\nPress Enter to return to the menu...")


def main() -> None:
    """Load the dataset, clean it, and start the interactive menu."""
    dataset_path = find_dataset_file(DATASET_DIR)
    raw_df = load_dataset(dataset_path)
    cleaned_df, diagnostics = clean_dataset(raw_df)

    print(f"Dataset loaded from: {dataset_path.name}")
    print(f"Rows: {len(raw_df)} | Columns: {len(raw_df.columns)}")
    print(f"Rows remaining after cleaning: {len(cleaned_df)}")
    if not diagnostics["invalid_date_values"].empty:
        print(f"Invalid date entries found: {len(diagnostics['invalid_date_values'])}")
    run_menu(cleaned_df)


if __name__ == "__main__":
    main()
