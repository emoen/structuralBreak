# language: python
# filepath: explanation.py
# Short annotated summary of the training / inference pipeline:

# train(...)
# 1) Build feature table F = _fe_table(X_train, FEATURES_TEST).reindex(y_train.index)
#    - computes per-id features (differences, tests, distances, acf, fft, etc.)
# 2) X, y = F.values, y_train
# 3) Define base_clfs: lr, rf, hgb (diverse base learners)
# 4) Do K-fold Stratified CV:
#    - For each fold clone an unfitted estimator (sk_clone) and fit on train fold.
#    - Produce OOF probability predictions for validation fold for each base clf.
# 5) Stack OOF predictions into OOF matrix and standardize (meta_scaler)
# 6) Train meta = LogisticRegression on OOF_scaled -> learns how to combine base outputs
# 7) Retrain each base classifier on full training X,y (so final base models use all data)
# 8) Save bundle = {"base": base_clfs, "meta": meta, "meta_scaler": meta_scaler, "features": F.columns.tolist()}
#    - Persisted with joblib.dump(model.joblib)

# infer(...)
# 1) Load bundle = joblib.load(model.joblib)
# 2) yield  # contract: signal ready to the runner
# 3) For each dataset in X_test:
#    - Compute feature vector f = _fe_one(dataset) in same feature order used at train
#    - For each base model: get base probability p (predict_proba or sigmoid(decision_function))
#    - Stack base probs into array; scale with meta_scaler and call meta.predict_proba -> final prob
#    - Fallback: average base probs if meta not available
#    - Clip to [0,1], yield deterministic float

# Important helpers & notes:
# - _fe_one and _fe_table: compute features per id (must be deterministic & match train order)
# - _sigmoid: numeric fallback when predict_proba missing
# - Use sk_clone(clf) to get an unfitted estimator copy for CV (avoids serializing to disk)
# - The saved bundle must include feature order; infer must produce features in same order
# - Determinism: runner expects a leading yield before producing predictions
# - OOF ROC printed during training helps diagnose which base/feature is useful