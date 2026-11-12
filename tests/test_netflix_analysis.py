import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netflix_analysis import (
    clean_dataset,
    content_added_in_year,
    count_movies_vs_shows,
    count_unique_countries,
    find_dataset_file,
    load_dataset,
    oldest_and_newest_release_year,
    rating_distribution,
    release_year_summary,
    run_menu,
    search_titles,
    titles_added_each_year,
    top_genres,
    top_producing_countries,
)


class TestNetflixAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df = pd.DataFrame(
            {
                "show_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
                "type": ["Movie", "TV Show", "Movie", "TV Show", "Movie", "Movie"],
                "title": ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"],
                "director": ["Alice", None, "Bob", "Carol", None, "Dan"],
                "cast": ["X", None, "Y", "Z", None, "W"],
                "country": [
                    "United States, Canada",
                    "South Africa",
                    "United States",
                    "Unknown",
                    "India",
                    "France, Germany",
                ],
                "date_added": [
                    "January 1, 2020",
                    "February 2, 2021",
                    "invalid-date",
                    "March 3, 2022",
                    "April 4, 2020",
                    "May 5, 2023",
                ],
                "release_year": [2020, 2021, 2022, 2020, 2021, 2022],
                "rating": ["PG", "TV-14", "PG-13", None, "TV-MA", "R"],
                "duration": ["90 min", "2 Seasons", "100 min", "1 Season", "90 min", "120 min"],
                "listed_in": ["Drama, Comedy", "Action, Drama", "Comedy", "Drama", "Thriller", "Sci-Fi, Drama"],
                "description": ["d1", "d2", "d3", "d4", "d5", "d6"],
            }
        )
        cls.cleaned_df, cls.diagnostics = clean_dataset(cls.raw_df)

    def test_clean_dataset_fills_values_and_creates_year_added(self):
        self.assertEqual(self.cleaned_df.loc[self.cleaned_df["title"] == "Alpha", "country"].iloc[0], "United States, Canada")
        self.assertEqual(self.cleaned_df.loc[self.cleaned_df["title"] == "Bravo", "director"].iloc[0], "Not Listed")
        self.assertEqual(self.cleaned_df.loc[self.cleaned_df["title"] == "Delta", "rating"].iloc[0], "Not Rated")
        self.assertIn(2020, set(self.cleaned_df["year_added"]))
        self.assertTrue(self.cleaned_df["year_added"].notna().all())
        self.assertEqual(len(self.diagnostics["invalid_date_values"]), 1)

    def test_count_movies_vs_shows(self):
        result = count_movies_vs_shows(self.cleaned_df)
        self.assertEqual(result["count"].sum(), len(self.cleaned_df))
        self.assertIn("Movie", result["type"].values)
        self.assertIn("TV Show", result["type"].values)

    def test_titles_added_each_year(self):
        result = titles_added_each_year(self.cleaned_df)
        self.assertTrue(result.index.tolist())
        self.assertEqual(result.sum(), len(self.cleaned_df))

    def test_top_producing_countries_excludes_unknown(self):
        result = top_producing_countries(self.cleaned_df, top_n=3)
        self.assertNotIn("Unknown", result.index)
        self.assertEqual(result.iloc[0], 1)

    def test_top_genres(self):
        result = top_genres(self.cleaned_df, top_n=3)
        self.assertIn("Drama", result.index)

    def test_rating_distribution(self):
        result = rating_distribution(self.cleaned_df)
        self.assertIn("PG", result.index)
        self.assertIn("TV-14", result.index)

    def test_oldest_and_newest_release_year(self):
        oldest, newest = oldest_and_newest_release_year(self.cleaned_df)
        self.assertEqual(oldest, 2020)
        self.assertEqual(newest, 2022)

    def test_search_titles(self):
        result = search_titles(self.cleaned_df, "alpha")
        self.assertEqual(result["title"].tolist(), ["Alpha"])

    def test_empty_keyword_search(self):
        result = search_titles(self.cleaned_df, "   ")
        self.assertTrue(result.empty)

    def test_content_added_in_year(self):
        result = content_added_in_year(self.cleaned_df, 2020)
        self.assertTrue(len(result) >= 1)

    def test_invalid_year_input(self):
        with self.assertRaises(ValueError):
            content_added_in_year(self.cleaned_df, "not-a-year")

    def test_unique_country_count(self):
        unique_count, unknown_count = count_unique_countries(self.cleaned_df)
        self.assertGreater(unique_count, 0)
        self.assertEqual(unknown_count, 1)

    def test_release_year_summary(self):
        result = release_year_summary(self.cleaned_df)
        self.assertIn("mean", result)
        self.assertIn("median", result)
        self.assertIn("std", result)
        self.assertIn("min", result)
        self.assertIn("max", result)

    def test_find_dataset_and_missing_path(self):
        dataset_path = find_dataset_file(Path(__file__).resolve().parents[1] / "dataset")
        self.assertTrue(dataset_path.exists())
        with self.assertRaises(FileNotFoundError):
            load_dataset(Path(__file__).resolve().parents[1] / "dataset" / "missing.csv")

    def test_invalid_menu_selection(self):
        output = io.StringIO()
        with patch("sys.stdout", output), patch("builtins.input", side_effect=["99", "", "0"]):
            run_menu(self.cleaned_df)
        self.assertIn("Invalid menu selection", output.getvalue())

    def test_menu_exit(self):
        output = io.StringIO()
        with patch("sys.stdout", output), patch("builtins.input", side_effect=["0"]):
            run_menu(self.cleaned_df)
        self.assertIn("Goodbye", output.getvalue())


if __name__ == "__main__":
    unittest.main()
