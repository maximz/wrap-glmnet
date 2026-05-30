# wrap-glmnet

[![](https://img.shields.io/pypi/v/wrap_glmnet.svg)](https://pypi.python.org/pypi/wrap_glmnet)
[![CI](https://github.com/maximz/wrap-glmnet/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/maximz/wrap-glmnet/actions/workflows/ci.yaml)
[![](https://img.shields.io/badge/docs-here-blue.svg)](https://wrap-glmnet.maximz.com)
[![](https://img.shields.io/github/stars/maximz/wrap-glmnet?style=social)](https://github.com/maximz/wrap-glmnet)

`wrap_glmnet` is a small Python wrapper around `python-glmnet`'s
`LogitNet` classifier. It keeps the glmnet elastic-net logistic regression
solver, but exposes a more sklearn-like estimator and adds control over
glmnet's internal cross-validation behavior.

## Why it exists

`python-glmnet` is useful for regularized logistic regression, but its
classifier API has a few rough edges for sklearn workflows. This package
provides `GlmnetLogitNetWrapper`, which is intended to be used in place of
`glmnet.LogitNet` when you need:

- sklearn-compatible cloning and fitted attributes such as `n_features_in_`
  and `feature_names_in_`
- explicit selection of whether predictions use `lambda_1se` or the
  best-performing cross-validation lambda
- sklearn-style class weights
- custom internal cross-validation splitters, including group-aware splitters
- multiclass ROC-AUC and deviance/log-loss scorers for glmnet's internal CV
- held-out CV predicted probabilities and per-fold CV scores
- predictable `decision_function` and `predict_proba` output shapes for binary,
  multiclass, single-lambda, and multi-lambda predictions

## How it works

Importing `wrap_glmnet` patches selected `python-glmnet` internals so glmnet's
lambda-path scoring can use an optional sklearn-style `internal_cv` splitter.
The patched scoring path also forwards `groups` to scorers that accept it,
records `_cv_scores_`, and, by default, stores held-out predicted probabilities
in `cv_pred_probs_` with shape `(n_samples, n_classes, n_lambdas)`.

`GlmnetLogitNetWrapper` delegates the actual model fit to an inner
`glmnet.LogitNet` instance. Constructor keyword arguments not handled by the
wrapper are passed through to `LogitNet`.

By default, predictions, `coef_`, `intercept_`, `cv_mean_score_final_`, and
`cv_standard_error_final_` use `lambda_1se` (`lambda_best_` in
`python-glmnet`). Set `use_lambda_1se=False` to use the lambda with the best
mean CV score (`lambda_max_`), or call `switch_lambda(...)` after fitting to
get a copied fitted model with the other lambda choice.

## Installation

The package requires Python 3.10 or newer.

```bash
pip install wrap_glmnet
```

For local development:

```bash
pip install -r requirements_dev.txt
pip install -e .
```

## Usage

```python
from sklearn.model_selection import StratifiedGroupKFold
from wrap_glmnet import GlmnetLogitNetWrapper

clf = GlmnetLogitNetWrapper(
    alpha=1.0,
    n_lambda=100,
    internal_cv=StratifiedGroupKFold(n_splits=3),
    scoring=GlmnetLogitNetWrapper.rocauc_scorer,
    require_cv_group_labels=True,
)

clf.fit(X_train, y_train, groups=groups)

labels = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Compare the default lambda_1se model to the best-CV-score lambda.
best_cv_lambda_clf = clf.switch_lambda(use_lambda_1se=False)
best_cv_probabilities = best_cv_lambda_clf.predict_proba(X_test)
```

If `internal_cv` is not supplied, the wrapper uses glmnet's normal internal CV
setup with `n_splits` (default: `3`). If `require_cv_group_labels=True`, calling
`fit(...)` without `groups` raises an error.

`class_weight` accepts sklearn-style values such as a class-to-weight mapping or
`"balanced"`. When both `class_weight` and `sample_weight` are supplied, the
wrapper multiplies them and normalizes the resulting sample weights before
fitting.

## Important behavior and limitations

- This package is focused on `glmnet.LogitNet` classification, not every glmnet
  model type.
- It patches `python-glmnet` functions at import time, so it relies on
  `python-glmnet` internals remaining compatible.
- `store_cv_predicted_probabilities=True` is the default and can use substantial
  memory for large datasets or long lambda paths. Disable it if you do not need
  `cv_pred_probs_`.
- Multiclass probabilities are computed from the wrapper's decision scores with
  softmax normalization; binary probabilities use sigmoid-style normalization.

## Development

```bash
make test
make lint
make docs
```

Tests cover sklearn cloning, scorer behavior, group-aware CV, lambda switching,
stored CV outputs, plotting, and output shapes.
