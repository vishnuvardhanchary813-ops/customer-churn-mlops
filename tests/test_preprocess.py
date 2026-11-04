"""Tests for Telco data cleaning and train/test output generation."""

import pandas as pd
import pytest

from src.data.preprocess import load_and_clean_data, split_and_save


def test_load_and_clean_converts_total_charges_and_drops_identifier(tmp_path):
    source = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "tenure": [0, 2, 5, 10],
            "TotalCharges": ["", "50.25", "120.00", "900.50"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    ).to_csv(source, index=False)

    cleaned = load_and_clean_data(source)

    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [0, 1, 0, 1]
    assert pd.isna(cleaned.loc[0, "TotalCharges"])
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_load_and_clean_rejects_unexpected_target_values(tmp_path):
    source = tmp_path / "dataset.csv"
    pd.DataFrame(
        {"customerID": ["A"], "TotalCharges": ["10"], "Churn": ["unknown"]}
    ).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Churn must contain"):
        load_and_clean_data(source)


def test_split_saves_stratified_train_and_test_csvs(tmp_path):
    data = pd.DataFrame(
        {
            "tenure": list(range(40)),
            "TotalCharges": [float(value * 10) for value in range(40)],
            "Churn": [0, 1] * 20,
        }
    )

    train_path, test_path = split_and_save(
        data, tmp_path / "processed", test_size=0.25, random_state=17
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    assert len(train) == 30
    assert len(test) == 10
    assert set(train["Churn"].unique()) == {0, 1}
    assert set(test["Churn"].unique()) == {0, 1}
