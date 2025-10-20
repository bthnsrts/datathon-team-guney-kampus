import warnings
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, log_loss
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks

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
class OptunaTuner:
    """
    Optuna-based hyperparameter tuner for binary classification with sampling support.
    
    Features:
    - Multiple models: XGBoost, LightGBM, CatBoost, RandomForest, Ensemble
    - Sampling methods: SMOTE, ADASYN, RandomOversampling, TomekLinks
    - Flexible evaluation metrics (sklearn functions or custom callables)
    - Cross-validation with stratified k-fold
    """
    
    VALID_SAMPLERS = {None, "SMOTE", "ADASYN", "RandomOverSampler", "TomekLinks"}
    VALID_MODELS = {"XGBoost", "LightGBM", "CatBoost", "RandomForest", "Ensemble"}
    
    def __init__(
        self,
        model_name: str = "XGBoost",
        sampler: Optional[str] = None,
        loss: Optional[str] = None,
        eval_metric: str = "roc_auc_score",
        class_weights: List[float] = [0.5, 0.5],
        n_trials: int = 50,
        random_state: int = 42,
        use_gpu: bool = False,
        n_splits: int = 5,
        study_name: Optional[str] = None,
        show_progress_bar: bool = True
    ):
        """
        Initialize the OptunaTuner.
        
        Parameters:
        -----------
        model_name : str
            Model to use: "XGBoost", "LightGBM", "CatBoost", "RandomForest", "Ensemble"
        sampler : str, optional
            Sampling method: None, "SMOTE", "ADASYN", "RandomOverSampler", "TomekLinks"
        loss : str, optional
            Model-specific objective/loss function (uses sensible defaults if None)
        eval_metric : callable
            Evaluation function for CV scoring. Can be:
            - Sklearn metric function: roc_auc_score, f1_score, accuracy_score, etc.
            - Custom function with signature: f(y_true, y_pred) -> float
              OR f(y_true, y_pred_proba, y_pred_label) -> float
            Default: roc_auc_score
        class_weights : list
            Class weights [weight_0, weight_1] for handling imbalance
        n_trials : int
            Number of Optuna trials
        random_state : int
            Random seed for reproducibility
        use_gpu : bool
            Whether to use GPU acceleration (if available)
        n_splits : int
            Number of cross-validation folds
        study_name : str, optional
            Name for the Optuna study (default: auto-generated)
        show_progress_bar : bool
            Whether to show progress bar during optimization (default: True)
        """
        if model_name not in self.VALID_MODELS:
            raise ValueError(f"Unsupported model_name: {model_name}")        
        
        self.eval_metric = eval_metric
        self.model_name = model_name
        self.sampler = sampler
        self.loss = loss        
        self.class_weights = class_weights
        self.n_trials = n_trials
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.n_splits = n_splits
        self.study_name = study_name
        self.show_progress_bar = show_progress_bar
        
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        
        if sampler is not None and class_weights != [0.5, 0.5]:
            warnings.warn(
                "You set both a sampling method and non-default class_weights. "
                "This can double-count imbalance handling."
            )     
    
    def _create_sampler(self):
        """Create a sampling object based on the sampler name."""
        if self.sampler is None:
            return None
        
        samplers = {
            "SMOTE": SMOTE(random_state=self.random_state),
            "ADASYN": ADASYN(random_state=self.random_state),
            "RandomOverSampler": RandomOverSampler(random_state=self.random_state),
            "TomekLinks": TomekLinks()
        }
        if self.sampler not in samplers:
            raise ValueError(f"Unsupported sampler: {self.sampler}")
        else: 
            return samplers.get(self.sampler)
    
    def _score(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray, 
        y_pred_label: np.ndarray
    ) -> float:
        """
        Metric routing via dictionary (exact metric names, no strip/lower, no helpers).

        Supported:
          "roc_auc_score"            -> uses y_pred_proba
          "average_precision_score"  -> uses y_pred_proba
          "f1_score"                 -> uses y_pred_label
          "accuracy_score"           -> uses y_pred_label
          "log_loss"                 -> uses y_pred_proba
          "ing_hubs_datathon_metric" -> ing_hubs_datathon_metric(y_true, y_pred_proba)

        Unknown names or any exception -> returns NaN.
        """

        if not isinstance(self.eval_metric, str):
            raise ValueError("eval_metric must be a string")

        metric_dispatch = {
            "roc_auc_score":            lambda yt, yp_proba, yp_lbl: roc_auc_score(yt, yp_proba),
            "average_precision_score":  lambda yt, yp_proba, yp_lbl: average_precision_score(yt, yp_proba),
            "f1_score":                 lambda yt, yp_proba, yp_lbl: f1_score(yt, yp_lbl),
            "accuracy_score":           lambda yt, yp_proba, yp_lbl: accuracy_score(yt, yp_lbl),
            "log_loss":                 lambda yt, yp_proba, yp_lbl: -1*log_loss(yt, yp_proba),
            "ing_hubs_datathon_metric": lambda yt, yp_proba, yp_lbl: ing_hubs_datathon_metric(yt, yp_proba),
        }

        fn = metric_dispatch.get(self.eval_metric)
        if fn is None:
            raise ValueError(f"Unsupported eval_metric: {self.eval_metric}")

        try:
            return float(fn(y_true, y_pred_proba, y_pred_label))
        except Exception:
            raise ValueError(f"Error computing metric: {self.eval_metric}")

    
    def _suggest_params(self, trial: optuna.Trial) -> Tuple[str, Dict[str, Any]]:
        """
        Suggest hyperparameters for the current trial.
        
        Returns:
        --------
        tuple : (model_key, parameters_dict)
        """
        if self.model_name == "Ensemble":
            # Suggest ensemble weights and individual model parameters
            params = {
                "weight_xgb": trial.suggest_float("weight_xgb", 0.0, 1.0),
                "weight_rf": trial.suggest_float("weight_rf", 0.0, 1.0),
                "weight_cat": trial.suggest_float("weight_cat", 0.0, 1.0),
                # XGBoost params
                "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 200, 1200),
                "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True),
                "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "xgb_subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "xgb_colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                # RandomForest params
                "rf_n_estimators": trial.suggest_int("rf_n_estimators", 200, 1000),
                "rf_max_depth": trial.suggest_int("rf_max_depth", 3, 30),
                "rf_min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "rf_max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
                # CatBoost params
                "cat_iterations": trial.suggest_int("cat_iterations", 300, 1500),
                "cat_learning_rate": trial.suggest_float("cat_learning_rate", 1e-3, 0.2, log=True),
                "cat_depth": trial.suggest_int("cat_depth", 4, 10),
                "cat_l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 10.0),
            }
            return "ensemble", params
        
        elif self.model_name == "XGBoost":
            params = {
                "objective": self.loss or "binary:logistic",
                # eval_metric will use XGBoost's default for the objective
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "max_bin": trial.suggest_int("max_bin", 128, 512),
                "tree_method": "gpu_hist" if self.use_gpu else "hist",
                "early_stopping_rounds": 50,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            return "xgb", params
        
        elif self.model_name == "LightGBM":
            params = {
                "objective": self.loss or "binary",
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "max_depth": trial.suggest_int("max_depth", -1, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self.random_state,
                "n_jobs": -1,                
                "device": "gpu" if self.use_gpu else "cpu",
            }
            return "lgbm", params
        
        elif self.model_name == "CatBoost":
            params = {
                "loss_function": self.loss or "Logloss",
                # eval_metric will use CatBoost's default for the loss function
                "iterations": trial.suggest_int("iterations", 300, 1500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_seed": self.random_state,
                "verbose": False,
                "early_stopping_rounds": 50,
                "task_type": "GPU" if self.use_gpu else "CPU",
            }
            return "cat", params
        
        else:  # RandomForest
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "criterion": self.loss or "gini",
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            return "rf", params
    
    def _train_fold(
        self,
        model_key: str,
        params: Dict[str, Any],
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train model on a single fold and return predictions.
        
        Returns:
        --------
        tuple : (y_pred_proba, y_pred_label)
        """
        # Calculate sample weights if needed
        sample_weight = None
        if self.class_weights != [0.5, 0.5]:
            w0, w1 = self.class_weights
            sample_weight = np.zeros_like(y_tr, dtype=float)
            sample_weight[y_tr == 0] = w0
            sample_weight[y_tr == 1] = w1
        
        # Calculate scale_pos_weight for imbalance handling
        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
        
        if model_key == "ensemble":
            # Extract weights and normalize
            w_xgb = params["weight_xgb"]
            w_rf = params["weight_rf"]
            w_cat = params["weight_cat"]
            total_weight = w_xgb + w_rf + w_cat
            
            if total_weight == 0:
                # If all weights are 0, use equal weights
                w_xgb = w_rf = w_cat = 1.0 / 3.0
            else:
                w_xgb /= total_weight
                w_rf /= total_weight
                w_cat /= total_weight
            
            # Train XGBoost
            xgb_params = {
                "objective": self.loss or "binary:logistic",
                "n_estimators": params["xgb_n_estimators"],
                "learning_rate": params["xgb_learning_rate"],
                "max_depth": params["xgb_max_depth"],
                "subsample": params["xgb_subsample"],
                "colsample_bytree": params["xgb_colsample_bytree"],
                "tree_method": "gpu_hist" if self.use_gpu else "hist",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            if self.sampler is None and self.class_weights == [0.5, 0.5]:
                xgb_params["scale_pos_weight"] = scale_pos_weight
            
            clf_xgb = XGBClassifier(**xgb_params)
            clf_xgb.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], verbose=False)
            proba_xgb = clf_xgb.predict_proba(X_va)[:, 1]
            
            # Train RandomForest
            rf_params = {
                "n_estimators": params["rf_n_estimators"],
                "max_depth": params["rf_max_depth"],
                "min_samples_split": params["rf_min_samples_split"],
                "max_features": params["rf_max_features"],
                "criterion": self.loss or "gini",
                "n_jobs": -1,
                "random_state": self.random_state,
            }
            if self.class_weights != [0.5, 0.5]:
                rf_params["class_weight"] = {0: self.class_weights[0], 1: self.class_weights[1]}
            
            clf_rf = RandomForestClassifier(**rf_params)
            clf_rf.fit(X_tr, y_tr, sample_weight=sample_weight)
            proba_rf = clf_rf.predict_proba(X_va)[:, 1]
            
            # Train CatBoost
            cat_params = {
                "loss_function": self.loss or "Logloss",
                "iterations": params["cat_iterations"],
                "learning_rate": params["cat_learning_rate"],
                "depth": params["cat_depth"],
                "l2_leaf_reg": params["cat_l2_leaf_reg"],
                "random_seed": self.random_state,
                "verbose": False,
                "task_type": "GPU" if self.use_gpu else "CPU",
            }
            clf_cat = CatBoostClassifier(**cat_params)
            if self.class_weights != [0.5, 0.5]:
                clf_cat.set_params(class_weights=self.class_weights)
            
            clf_cat.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], use_best_model=True, verbose=False)
            proba_cat = clf_cat.predict_proba(X_va)[:, 1]
            
            # Weighted average ensemble
            proba = w_xgb * proba_xgb + w_rf * proba_rf + w_cat * proba_cat
            yhat = (proba >= 0.5).astype(int)
            
            return proba, yhat
        
        elif model_key == "xgb":
            xgb_params = params.copy()
            if self.sampler is None and self.class_weights == [0.5, 0.5]:
                xgb_params["scale_pos_weight"] = scale_pos_weight
            
            clf = XGBClassifier(**xgb_params)
            clf.fit(
                X_tr, y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            proba = clf.predict_proba(X_va)[:, 1]
            yhat = (proba >= 0.5).astype(int)
        
        elif model_key == "lgbm":
            lgbm_params = params.copy()
            if self.class_weights != [0.5, 0.5]:
                lgbm_params["class_weight"] = {0: self.class_weights[0], 1: self.class_weights[1]}
            elif self.sampler is None:
                lgbm_params["scale_pos_weight"] = scale_pos_weight
            
            clf = LGBMClassifier(**lgbm_params)
            clf.fit(
                X_tr, y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(50), log_evaluation(0)]
            )
            proba = clf.predict_proba(X_va)[:, 1]
            yhat = (proba >= 0.5).astype(int)
        
        elif model_key == "cat":
            clf = CatBoostClassifier(**params)
            if self.class_weights != [0.5, 0.5]:
                clf.set_params(class_weights=self.class_weights)
            
            # Note: CatBoost custom metrics require different setup
            # For simplicity, using built-in metrics during training
            clf.fit(
                X_tr, y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                use_best_model=True,
                verbose=False
            )
            proba = clf.predict_proba(X_va)[:, 1]
            yhat = (proba >= 0.5).astype(int)
        
        else:  # RandomForest
            rf_params = params.copy()
            if self.class_weights != [0.5, 0.5]:
                rf_params["class_weight"] = {0: self.class_weights[0], 1: self.class_weights[1]}
            
            clf = RandomForestClassifier(**rf_params)
            clf.fit(X_tr, y_tr, sample_weight=sample_weight)
            proba = clf.predict_proba(X_va)[:, 1]
            yhat = (proba >= 0.5).astype(int)
        
        return proba, yhat
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function for Optuna optimization.
        
        Returns:
        --------
        float : Mean cross-validation score
        """
        model_key, params = self._suggest_params(trial)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            # Apply sampling if specified (only on training set)
            sampler_obj = self._create_sampler()
            if sampler_obj is not None:
                try:
                    X_tr, y_tr = sampler_obj.fit_resample(X_tr, y_tr)
                except Exception as e:
                    warnings.warn(
                        f"Sampler {self.sampler} failed on this fold ({e}). "
                        "Skipping sampling."
                    )
            
            # Train and predict
            proba, yhat = self._train_fold(model_key, params, X_tr, y_tr, X_va, y_va)
            
            # Compute score
            score = self._score(y_va, proba, yhat)
            scores.append(score)
            
            # Debug logging for first trial
            if trial.number == 0 and fold_idx == 0:
                print(f"\nStudy Name: {self.study_name}")                
                print(f"  Train size: {len(y_tr)}, Val size: {len(y_va)}")
                print(f"  Train class dist: {np.bincount(y_tr)}")
                print(f"  Val class dist: {np.bincount(y_va)}")
                print(f"  Pred proba range: [{proba.min():.4f}, {proba.max():.4f}]")
                print(f"  Pred labels dist: {np.bincount(yhat)}")                
        
        return float(np.mean(scores))
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Binary target labels (0 and 1)
            
        Returns:
        --------
        optuna.Study : Completed Optuna study object
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("Binary target {0, 1} expected.")
        
        # Set up custom logging format to show 2 decimal places
        optuna.logging.disable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Create study with optional name
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        
        # Custom callback to format output with 2 decimal places
        def logging_callback(study, trial):
            # Format value to 2 decimal places
            value_str = f"{trial.value:.2f}"
            best_value_str = f"{study.best_value:.2f}"
            
            # Format parameters to 2 decimal places
            params_formatted = {}
            for key, val in trial.params.items():
                if isinstance(val, float):
                    params_formatted[key] = f"{val:.2f}"
                else:
                    params_formatted[key] = val
            
            print(f"[I {trial.datetime_complete.strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Trial {trial.number} finished with value: {value_str} and "
                  f"parameters: {params_formatted}. Best is trial {study.best_trial.number} "
                  f"with value: {best_value_str}.")
        
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            callbacks=[logging_callback],
            show_progress_bar=self.show_progress_bar
        )
        
        self.best_params = self.study.best_params
        return self.study
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found."""
        if self.best_params is None:
            raise ValueError("No optimization run yet. Call optimize() first.")
        return self.best_params