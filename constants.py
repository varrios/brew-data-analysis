import os

ROOT_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

DATAFILE_PATH = os.path.join(ROOT_PROJECT_PATH, "data", "recipeData.csv")
EXCLUDED_COLUMNS = ["BeerID", "StyleID", "UserId"]