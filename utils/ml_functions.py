import warnings
import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, log_loss
)
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------
# Custom ING Hubs metrics
# -------------------
def recall_at_k(y_true, y_prob, k=0.1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    P = y_true.sum()

    return float(tp_at_k / P) if P > 0 else 0.0

def lift_at_k(y_true, y_prob, k=0.1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    m = max(1, int(np.round(k * n)))
    order = np.argsort(-y_prob, kind="mergesort")
    top = order[:m]
    tp_at_k = y_true[top].sum()
    precision_at_k = tp_at_k / m
    prevalence = y_true.mean()

    return float(precision_at_k / prevalence) if prevalence > 0 else 0.0

def convert_auc_to_gini(auc):
    return 2 * auc - 1

def ing_hubs_datathon_metric(y_true, y_prob):
    score_weights = {"gini": 0.4, "recall_at_10perc": 0.3, "lift_at_10perc": 0.3}
    baseline_scores = {
        "roc_auc": 0.6925726757936908,
        "recall_at_10perc": 0.18469015795868773,
        "lift_at_10perc": 1.847159286784029,
    }

    roc_auc = roc_auc_score(y_true, y_prob)
    recall_at_10perc = recall_at_k(y_true, y_prob, k=0.1)
    lift_at_10perc = lift_at_k(y_true, y_prob, k=0.1)

    baseline_scores["gini"] = convert_auc_to_gini(baseline_scores["roc_auc"])
    new_gini = convert_auc_to_gini(roc_auc)

    # Guard against division by zero (just in case)
    eps = 1e-12
    final_gini_score   = new_gini / max(baseline_scores["gini"], eps)
    final_recall_score = recall_at_10perc / max(baseline_scores["recall_at_10perc"], eps)
    final_lift_score   = lift_at_10perc / max(baseline_scores["lift_at_10perc"], eps)

    final_score = (
        final_gini_score * score_weights["gini"] +
        final_recall_score * score_weights["recall_at_10perc"] + 
        final_lift_score * score_weights["lift_at_10perc"]
    )
    return float(final_score)

# --------------------------------------------------
# Main Optuna runner with sampling + custom metric
# --------------------------------------------------
def create_optuna_study_with_sampling(
    X,
    y,
    sampler: str = None,                          # None | "SMOTE" | "ADASYN" | "RandomOversampling" | "TomekLinks"
    model_name: str = "XGBoost",                  # "XGBoost" | "LightGBM" | "CatBoost" | "RandomForest"
    loss: str = None,                             # model-specific objective; sensible default if None
    eval_metric: str = "auc",                     # "auc" | "aucpr" | "f1" | "accuracy" | "logloss" | "ing_hubs"
    class_weights = [0.5, 0.5],                   # warning if changed and sampler chosen
    n_trials: int = 50,
    random_state: int = 42,
    use_gpu: bool = False,
    n_splits: int = 5
):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    assert set(np.unique(y)) <= {0, 1}, "Binary target {0,1} expected."

    valid_samplers = {None, "SMOTE", "ADASYN", "RandomOversampling", "TomekLinks"}
    valid_models = {"XGBoost", "LightGBM", "CatBoost", "RandomForest"}
    valid_metrics = {"auc", "aucpr", "f1", "accuracy", "logloss", "ing_hubs"}

    if sampler not in valid_samplers:
        raise ValueError(f"sampler must be one of {valid_samplers}")
    if model_name not in valid_models:
        raise ValueError(f"model_name must be one of {valid_models}")
    if eval_metric not in valid_metrics:
        raise ValueError(f"eval_metric must be one of {valid_metrics}")

    if sampler is not None and (class_weights != [0.5, 0.5]):
        warnings.warn(
            "You set both a sampling method and non-default class_weights. "
            "This can double-count imbalance handling."
        )

    def make_sampler():
        if sampler is None: return None
        if sampler == "SMOTE": return SMOTE(random_state=random_state)
        if sampler == "ADASYN": return ADASYN(random_state=random_state)
        if sampler == "RandomOversampling": return RandomOverSampler(random_state=random_state)
        if sampler == "TomekLinks": return TomekLinks()
        return None

    # Metric dispatcher
    def score_fn(y_true, y_pred_proba, y_pred_label):
        if eval_metric == "ing_hubs":
            return ing_hubs_datathon_metric(y_true, y_pred_proba)
        if eval_metric == "auc":
            return roc_auc_score(y_true, y_pred_proba)
        if eval_metric == "aucpr":
            return average_precision_score(y_true, y_pred_proba)
        if eval_metric == "f1":
            return f1_score(y_true, y_pred_label)
        if eval_metric == "accuracy":
            return accuracy_score(y_true, y_pred_label)
        if eval_metric == "logloss":
            return -log_loss(y_true, y_pred_proba, labels=[0,1])  # higher is better
        raise ValueError(f"Unsupported eval_metric: {eval_metric}")

    def suggest_params(trial):
        if model_name == "XGBoost":
            params = dict(
                objective = loss or "binary:logistic",
                # keep a stable built-in metric for early stopping/monitoring
                eval_metric = "auc",
                n_estimators = trial.suggest_int("n_estimators", 200, 1200),
                learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                max_depth = trial.suggest_int("max_depth", 3, 10),
                min_child_weight = trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
                subsample = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma = trial.suggest_float("gamma", 0.0, 10.0),
                reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                max_bin = trial.suggest_int("max_bin", 128, 512),
                tree_method = "gpu_hist" if use_gpu else "hist",
                random_state = random_state,
                n_jobs = -1,
            )
            return "xgb", params

        if model_name == "LightGBM":
            params = dict(
                objective = loss or "binary",
                n_estimators = trial.suggest_int("n_estimators", 300, 1500),
                learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                num_leaves = trial.suggest_int("num_leaves", 16, 256),
                max_depth = trial.suggest_int("max_depth", -1, 12),
                min_child_samples = trial.suggest_int("min_child_samples", 5, 200),
                subsample = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                random_state = random_state,
                n_jobs = -1,
                device = "gpu" if use_gpu else "cpu",
            )
            return "lgbm", params

        if model_name == "CatBoost":
            params = dict(
                loss_function = loss or "Logloss",
                eval_metric = "AUC",   # stable built-in metric
                iterations = trial.suggest_int("iterations", 300, 1500),
                learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                depth = trial.suggest_int("depth", 4, 10),
                l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
                random_seed = random_state,
                verbose = False,
                task_type = "GPU" if use_gpu else "CPU",
            )
            return "cat", params

        # RandomForest
        params = dict(
            n_estimators = trial.suggest_int("n_estimators", 200, 1000),
            max_depth = trial.suggest_int("max_depth", 3, 30),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            criterion = loss or "gini",   # "gini", "entropy", "log_loss"
            n_jobs = -1,
            random_state = random_state,
        )
        return "rf", params

    def objective(trial):
        model_key, params = suggest_params(trial)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Optional resampling on TRAIN ONLY
            sampler_obj = make_sampler()
            if sampler_obj is not None:
                try:
                    X_tr, y_tr = sampler_obj.fit_resample(X_tr, y_tr)
                except Exception as e:
                    warnings.warn(f"Sampler {sampler} failed on this fold ({e}). Skipping sampling.")

            # Class weights as sample weights (optional)
            sample_weight = None
            if class_weights is not None:
                w0, w1 = class_weights
                if (w0, w1) != (0.5, 0.5):
                    sw = np.zeros_like(y_tr, dtype=float)
                    sw[y_tr == 0] = w0
                    sw[y_tr == 1] = w1
                    sample_weight = sw

            # scale_pos_weight for boosting (only if not double-counting)
            neg = (y_tr == 0).sum()
            pos = (y_tr == 1).sum()
            scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

            if model_key == "xgb":
                xgb_params = params.copy()
                if sampler is None and (class_weights == [0.5, 0.5]):
                    xgb_params["scale_pos_weight"] = scale_pos_weight

                clf = XGBClassifier(**xgb_params)
                clf.fit(
                    X_tr, y_tr,
                    sample_weight=sample_weight,
                    eval_set=[(X_va, y_va)],   # built-in AUC logged; our metric computed after
                    verbose=False,
                    early_stopping_rounds=50
                )
                proba = clf.predict_proba(X_va)[:, 1]
                yhat = (proba >= 0.5).astype(int)

            elif model_key == "lgbm":
                from lightgbm import early_stopping, log_evaluation

                lgbm_params = params.copy()
                if (class_weights != [0.5, 0.5]):
                    lgbm_params["class_weight"] = {0: class_weights[0], 1: class_weights[1]}
                elif sampler is None:
                    lgbm_params["scale_pos_weight"] = scale_pos_weight

                clf = LGBMClassifier(**lgbm_params)

                # If you're optimizing ing_hubs and want per-iteration tracking in LightGBM:
                def lgb_ing_hubs_feval(y_pred, dataset):
                    y_true = dataset.get_label()
                    return ("ing_hubs", ing_hubs_datathon_metric(y_true, y_pred), True)

                eval_metric_lgb = ["auc"]
                if eval_metric == "ing_hubs":
                    eval_metric_lgb = [lgb_ing_hubs_feval, "auc"]

                clf.fit(
                    X_tr, y_tr,
                    sample_weight=sample_weight,
                    eval_set=[(X_va, y_va)],
                    eval_metric=eval_metric_lgb,
                    callbacks=[early_stopping(50), log_evaluation(0)]
                )
                proba = clf.predict_proba(X_va)[:, 1]
                yhat = (proba >= 0.5).astype(int)

            elif model_key == "cat":
                clf = CatBoostClassifier(**params)
                if (class_weights != [0.5, 0.5]):
                    clf.set_params(class_weights=[class_weights[0], class_weights[1]])

                clf.fit(
                    X_tr, y_tr,
                    sample_weight=sample_weight,
                    eval_set=[(X_va, y_va)],
                    use_best_model=True,
                    verbose=False
                )
                proba = clf.predict_proba(X_va)[:, 1]
                yhat = (proba >= 0.5).astype(int)

            else:
                rf_params = params.copy()
                if (class_weights != [0.5, 0.5]):
                    rf_params["class_weight"] = {0: class_weights[0], 1: class_weights[1]}
                clf = RandomForestClassifier(**rf_params)
                clf.fit(X_tr, y_tr, sample_weight=sample_weight)
                proba = clf.predict_proba(X_va)[:, 1]
                yhat = (proba >= 0.5).astype(int)

            score = score_fn(y_va, proba, yhat)
            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study