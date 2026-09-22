"""Integration tests for end-to-end data preparation and training workflow."""


from src.data.ingest import generate_benchmark_churn_data
from src.data.validator import DataValidator
from src.training.pipeline import create_training_pipeline


def test_end_to_end_mini_pipeline():
    """Test full cycle: data generation -> validation -> pipeline fit -> prediction."""
    # 1. Generate mini dataset
    df = generate_benchmark_churn_data(n_samples=100, seed=42)
    assert len(df) == 100

    # 2. Validate
    validator = DataValidator()
    val_res = validator.validate(df, is_training=True)
    assert val_res.is_valid is True

    # 3. Clean and prepare target
    X = df.drop(columns=["customerID", "Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    # 4. Train small pipeline
    config = {
        "type": "logistic_regression",
        "C": 1.0,
        "max_iter": 200,
        "solver": "lbfgs",
    }
    pipeline = create_training_pipeline(config)
    pipeline.fit(X, y)

    # 5. Predict
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]

    assert len(preds) == 100
    assert set(preds).issubset({0, 1})
    assert len(probs) == 100
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
