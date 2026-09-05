"""Probability values, class normalization and lambda-axis regression coverage."""

import numpy as np
import pytest
from scipy.special import expit, softmax
from wrap_glmnet import GlmnetLogitNetWrapper


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("n_samples", [1, 7])
def test_path_probabilities_match_scalar_predictions(alpha, n_classes, n_samples):
    rng = np.random.default_rng(91)
    X = rng.normal(size=(90, 5))
    # String labels also check that class-column ordering is preserved.
    y = np.array(["control", "case", "other"])[np.arange(90) % n_classes]
    clf = GlmnetLogitNetWrapper(
        alpha=alpha,
        n_splits=0,
        standardize=False,
        class_weight=None,
        lambda_path=np.array([10.0, 1.0, 0.1]),
    ).fit(X, y)
    X_test = X[:n_samples]
    lambdas = clf.lambda_path_
    actual = clf.predict_proba(X_test, lamb=lambdas)
    logits = clf.decision_function(X_test, lamb=lambdas)
    if n_classes == 2:
        positive = expit(logits[:, 0, :])
        expected = np.stack([1.0 - positive, positive], axis=1)
    else:
        expected = softmax(logits, axis=1)
    assert actual.shape == (n_samples, n_classes, len(lambdas))
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    for i, lam in enumerate(lambdas):
        for request in [float(lam), np.array([lam])]:
            scalar = clf.predict_proba(X_test, lamb=request)
            assert scalar.shape == (n_samples, n_classes)
            np.testing.assert_allclose(actual[:, :, i], scalar)
