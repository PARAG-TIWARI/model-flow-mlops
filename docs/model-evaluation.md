# Model evaluation and monitoring

This guide records the metrics and monitoring behaviour implemented in the repository.

## Stored held-out test report

The committed `artifacts/reports/evaluation_metrics.json` reports a test sample count of 1,057 and the following results:

| Metric | Stored result |
| --- | ---: |
| Accuracy | 0.7474 |
| Precision (positive class: Churn = Yes) | 0.5165 |
| Recall (positive class: Churn = Yes) | 0.7794 |
| F1 | 0.6213 |
| Macro F1 | 0.7159 |
| ROC-AUC | 0.8447 |
| PR-AUC / Average Precision | 0.6587 |

Confusion matrix (rows are actual class, columns are predicted class):

| | Predicted No | Predicted Yes |
| --- | ---: | ---: |
| Actual No | 571 | 205 |
| Actual Yes | 62 | 219 |

These values are read from the committed JSON artifact, not freshly generated as part of this documentation update. The report does not carry a data-source manifest, so the exact input provenance (downloaded source CSV versus synthetic fallback) should be confirmed before making dataset-specific performance claims. See [Data provenance and reproducibility](data-and-reproducibility.md).

## How the metrics are calculated

The evaluator runs `src/evaluation/evaluator.py` on the held-out test split and calls the functions in `src/evaluation/metrics.py`.

It calculates:
- Accuracy, precision, recall and F1 using the 0/1 predicted class.
- Macro F1.
- ROC-AUC and average precision (PR-AUC) using predicted probabilities.
- Confusion-matrix counts.

The positive class is `Churn = Yes`. Because the training classifiers use `class_weight="balanced"`, the default probability-to-class threshold is still the estimator's standard 0.50 threshold unless changed elsewhere.

## Candidate selection

Candidate models are scored on the validation set. The candidate with the highest configured primary metric (currently ROC-AUC) becomes the champion. The `min_roc_auc_threshold` in `params.yaml` currently generates a warning if the value is not met; it is not a hard gate and does not stop local champion artifact creation.

The Logistic Regression candidate is not probability-calibrated in the current implementation. It is a class-weighted scikit-learn `LogisticRegression` pipeline.

## Drift detection

The current implementation uses:
- **Numerical features:** PSI plus the two-sample Kolmogorov–Smirnov statistic and p-value.
- **Categorical features:** categorical PSI.
- **Aggregate alert:** the top-level drift alert becomes true when at least 30% of the analysed features are individually marked as drifted.

No Chi-Square test is currently implemented for categorical features, despite earlier README wording.

The API's `/monitoring/drift` route uses recent inference inputs when at least ten records are available in the in-memory buffer. If the buffer is not warm, it may return the saved drift report or compare the held-out test split with the training reference. The committed `drift_report.json` is a recorded analysis run, not a live service monitor.

## Operational telemetry

The `MetricsCollector` stores up to 1,000 recent predictions in memory and reports:
- Total, successful and failed request counts.
- Error rate.
- Mean and percentile inference latency (P50, P95 and P99).
- Predicted churn distribution.
- Buffered inference count.

These values reset when the API process restarts. The current repository does not implement a persistent metrics database or Prometheus exporter.

## Re-run evaluation

After the pipeline has generated the split and champion artifact:

```bash
python -m src.evaluation.evaluator
```

The report is written to the path configured in `params.yaml`. Preserve the generated JSON and plots together with the data provenance and model/run metadata if you plan to publish the results.
