import pickle
from .fuzzy_set import ConvexFuzzySet


def save_to_file(fset: ConvexFuzzySet, file_name: str) -> None:
    with open(file_name, "wb") as f:
        pickle.dump(fset, f)


def load_from_file(file_name: str) -> ConvexFuzzySet:
    with open(file_name, "rb") as f:
        loaded = pickle.load(f)
    return loaded
