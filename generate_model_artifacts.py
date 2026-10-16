"""Generate all local fitted model bundles from the unchanged source CSVs."""

from __future__ import annotations

from model_utils import get_classification_bundle, get_clustering_bundle, get_regression_bundle


def main() -> None:
    for name, loader in (
        ("regression", get_regression_bundle),
        ("classification", get_classification_bundle),
        ("clustering", get_clustering_bundle),
    ):
        artifact = loader()
        print(f"Generated/loaded {name} bundle ({artifact['sklearn_version']}).")


if __name__ == "__main__":
    main()
