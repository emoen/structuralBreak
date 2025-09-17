# Install the Crunch CLI
#%pip install --upgrade crunch-cli

# Setup your local environment
#!crunch setup --notebook structural-break hello --token aaaabbbbccccddddeeeeffff
#!crunch setup-notebook structural-break Y5AcL5oIf905ChiIndjKLTjV


import os
import typing

# Import your dependencies
import joblib
import pandas as pd
import scipy
import sklearn.metrics


import crunch

# Load the Crunch Toolings
#crunch = crunch.load_notebook()


# Load the data simply
#X_train, y_train, X_test = crunch.load_data()


import pandas as pd

#data_dir = "/home/endrem/project/structural_break/competitions/competitions/structural-break/data"

# Train
#X_train = pd.read_parquet(f"{data_dir}/X_train.parquet")
#y_train = pd.read_parquet(f"{data_dir}/y_train.parquet")["structural_breakpoint"]

# Test (group into a list of DataFrames by dataset id, like the platform)
#_x_test_df = pd.read_parquet(f"{data_dir}/X_test.reduced.parquet")
#X_test = []
#for id_val, df in _x_test_df.groupby(_x_test_df.index.get_level_values("id")):
#    df.name = id_val
#    X_test.append(df)

# Optional: local scoring target for the reduced test
#y_test = pd.read_parquet(f"{data_dir}/y_test.reduced.parquet")["structural_breakpoint"]


#X_train


#y_train


#print("Number of datasets:", len(X_test))


#X_test[0]


import os
import typing
import numpy as np
import pandas as pd
import scipy.stats as st
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#FEATURES_TEST = [
#    "mean_diff", "abs_mean_diff", "median_diff",
#    "std_ratio", "mad_ratio",
#    "skew_diff", "kurt_diff",
#    "acf1_diff",
#    "p_t_welch", "p_ks", "p_mw", "p_levene",
#]

def _mad(x):
    m = np.median(x) if len(x) else 0.0
    return np.median(np.abs(x - m)) if len(x) else 0.0

def _acf1(x):
    if len(x) < 2:
        return 0.0
    s = pd.Series(x)
    v = s.autocorr(lag=1)
    return 0.0 if pd.isna(v) else float(v)

def _logp(p):
    p = np.clip(p, 1e-300, 1.0)
    return -np.log10(p)

def _fe_one(u: pd.DataFrame) -> np.ndarray:
    a = u["value"][u["period"] == 0].to_numpy()
    b = u["value"][u["period"] == 1].to_numpy()
    eps = 1e-12

    # robust stats
    mean_a, mean_b = (a.mean() if a.size else 0.0), (b.mean() if b.size else 0.0)
    med_a,  med_b  = (np.median(a) if a.size else 0.0), (np.median(b) if b.size else 0.0)
    std_a,  std_b  = (a.std(ddof=1) if a.size>1 else 0.0), (b.std(ddof=1) if b.size>1 else 0.0)
    mad_a,  mad_b  = _mad(a), _mad(b)
    skew_a, skew_b = (st.skew(a) if a.size>2 else 0.0), (st.skew(b) if b.size>2 else 0.0)
    kur_a,  kur_b  = (st.kurtosis(a) if a.size>3 else 0.0), (st.kurtosis(b) if b.size>3 else 0.0)
    acf_a,  acf_b  = _acf1(a), _acf1(b)

    # tests
    p_t = st.ttest_ind(a, b, equal_var=False).pvalue if a.size and b.size else 1.0
    p_ks = st.ks_2samp(a, b, alternative="two-sided", mode="asymp").pvalue if a.size and b.size else 1.0
    p_mw = st.mannwhitneyu(a, b, alternative="two-sided").pvalue if a.size and b.size else 1.0
    p_lv = st.levene(a, b, center="median").pvalue if a.size and b.size else 1.0

    feats = [
        (mean_b - mean_a),
        abs(mean_b - mean_a),
        (med_b - med_a),
        (std_b + eps) / (std_a + eps),
        (mad_b + eps) / (mad_a + eps),
        (skew_b - skew_a),
        (kur_b - kur_a),
        (acf_b - acf_a),
        _logp(p_t), _logp(p_ks), _logp(p_mw), _logp(p_lv),
    ]
    # sanitize
    feats = [0.0 if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in feats]
    return np.asarray(feats, dtype=float)

def _fe_table(X_train: pd.DataFrame, FEATURES_TEST) -> pd.DataFrame:
    rows, ids = [], []
    for gid, gdf in X_train.groupby(level="id"):
        rows.append(_fe_one(gdf))
        ids.append(gid)
    return pd.DataFrame(rows, index=pd.Index(ids, name="id"), columns=FEATURES_TEST)


def train(X_train: pd.DataFrame, y_train: pd.Series, model_directory_path: str):
    FEATURES_TEST = [
        "mean_diff", "abs_mean_diff", "median_diff",
        "std_ratio", "mad_ratio",
        "skew_diff", "kurt_diff",
        "acf1_diff",
        "p_t_welch", "p_ks", "p_mw", "p_levene",
    ]
    
    F = _fe_table(X_train, FEATURES_TEST).reindex(y_train.index)
    y = y_train.astype(int).to_numpy()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)),
    ])
    pipe.fit(F.values, y)
    joblib.dump({"model": pipe, "features": FEATURES_TEST}, os.path.join(model_directory_path, "model.joblib"))


def train2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    # For our baseline t-test approach, we don't need to train a model
    # This is essentially an unsupervised approach calculated at inference time
    model = None

    # You could enhance this by training an actual model, for example:
    # 1. Extract features from before/after segments of each time series
    # 2. Train a classifier using these features and y_train labels
    # 3. Save the trained model

    joblib.dump(model, os.path.join(model_directory_path, 'model.joblib'))


def infer(X_test: typing.Iterable[pd.DataFrame], model_directory_path: str):
    bundle = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    model: Pipeline = bundle["model"]
    yield  # ready
    for dataset in X_test:
        f = _fe_one(dataset).reshape(1, -1)
        p = float(model.predict_proba(f)[0, 1])  # probability of break
        yield p


def infer2(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    model = joblib.load(os.path.join(model_directory_path, 'model.joblib'))

    yield  # Mark as ready

    # X_test can only be iterated once.
    # Before getting the next dataset, you must predict the current one.
    for dataset in X_test:
        # Baseline approach: Compute t-test between values before and after boundary point
        # The negative p-value is used as our score - smaller p-values (larger negative numbers)
        # indicate more evidence against the null hypothesis that distributions are the same,
        # suggesting a structural break
        def t_test(u: pd.DataFrame):
            return -scipy.stats.ttest_ind(
                u["value"][u["period"] == 0],  # Values before boundary point
                u["value"][u["period"] == 1],  # Values after boundary point
            ).pvalue

        prediction = t_test(dataset)
        yield prediction  # Send the prediction for the current dataset

        # Note: This baseline approach uses a t-test to compare the distributions
        # before and after the boundary point. A smaller p-value (larger negative number)
        # suggests stronger evidence that the distributions are different,
        # indicating a potential structural break.


#crunch.test(
#    # Uncomment to disable the train
#    # force_first_train=False,
#
#    # Uncomment to disable the determinism check
#    # no_determinism_check=True,
#)


#prediction = pd.read_parquet("data/prediction.parquet")
#prediction


# Load the targets
#target = pd.read_parquet("data/y_test.reduced.parquet")["structural_breakpoint"]

# Call the scoring function
#sklearn.metrics.roc_auc_score(
#    target,
#    prediction,
#)
