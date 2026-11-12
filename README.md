# Netflix Content Explorer

This project is a beginner-friendly Python application for exploring the Netflix Titles dataset using Pandas and NumPy. It includes:

- Data verification and exploration
- Data cleaning and preprocessing
- Ten required analysis operations
- A simple menu-driven console interface
- Unit tests for key logic

## Project structure

- `dataset/` – contains the Netflix dataset file
- `notebooks/netflix_analysis.ipynb` – final notebook deliverable
- `src/netflix_analysis.py` – reusable logic and menu app
- `tests/test_netflix_analysis.py` – unit tests
- `requirements.txt` – project dependencies

## Setup

1. Open the project folder.
2. Activate the existing virtual environment from `.venv`.
3. Install dependencies from `requirements.txt` only if needed.

```bash
cd "C:\Projects\Netflix Content Explorer"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the menu application

```bash
cd "C:\Projects\Netflix Content Explorer"
.\.venv\Scripts\python.exe src\netflix_analysis.py
```

## Run tests

```bash
cd "C:\Projects\Netflix Content Explorer"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Dataset notes

- The dataset is automatically detected in the `dataset/` folder.
- The project supports CSV and Excel files.
- The dataset is kept in its original form. A cleaned copy is created in memory for analysis.

## Data cleaning rules applied

- Missing `country` values become `Unknown`
- Missing `director` values become `Not Listed`
- Missing `cast` values become `Not Listed`
- Missing `rating` values become `Not Rated`
- Rows with missing `date_added` are removed
- A new `year_added` column is created from the parsed date
- Invalid dates are recorded separately for transparency
- `release_year` values are converted to numeric form and invalid values are handled safely

## Analysis menu options

1. Movies vs TV Shows
2. Titles added each year
3. Top 10 producing countries
4. Top 5 most common genres
5. Rating distribution
6. Oldest and newest release years
7. Search titles by keyword
8. Titles added in a specific year
9. Unique countries count
10. Summary statistics for release years

## Deliverable

The final notebook deliverable is located at:

`notebooks/netflix_analysis.ipynb`
