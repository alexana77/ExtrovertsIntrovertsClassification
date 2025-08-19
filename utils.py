# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:48:58.658268Z","iopub.execute_input":"2025-08-11T09:48:58.658600Z","iopub.status.idle":"2025-08-11T09:48:58.665302Z","shell.execute_reply.started":"2025-08-11T09:48:58.658572Z","shell.execute_reply":"2025-08-11T09:48:58.664205Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np
import pandas as pd
import json
import joblib
from typing import List, Tuple
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, jaccard_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy import stats, sparse

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

LABEL_MAP = {'Introverts': 1, 'Extroverts': 0}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
POS_CLASS = 1  # always Introverts

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:48:58.666746Z","iopub.execute_input":"2025-08-11T09:48:58.667043Z","iopub.status.idle":"2025-08-11T09:48:58.695740Z","shell.execute_reply.started":"2025-08-11T09:48:58.667019Z","shell.execute_reply":"2025-08-11T09:48:58.694765Z"}}

# Association measure
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Cramér's V statistic for measuring the strength of association between
    two categorical variables. Bias‑correction (and zero‑division guard) is done.
    
    In small contingency tables, the raw phi^2 statistic tends to overestimate association strength. 
    That’s because:
            - chi-squared test is sensitive to sample size and table dimensions.
            - small tables can produce large phi^2 values even when the actual association is weak.    

    Parameters
    ----------
    x: first categorical variable
    y: second categorical variable

    Returns
    -------
    0 is no association,  1 is a perfect association
    """
    cont_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(cont_table, correction=False)[0]
    n = cont_table.values.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = cont_table.shape
    # Bias correction
    phi2_corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
    k_corr = k - (k-1)**2/(n-1)
    r_corr = r - (r-1)**2/(n-1)
    denom = min((k_corr-1), (r_corr-1)) #if min(k-1, r-1)==0, we’ll get division by zero
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))
    

# Class balance check
def check_class_distribution(y_train: pd.Series, y_test: pd.Series, tolerance=0.02)-> None:
    """
    Validate class distribution consistency between training and test sets.

    This function compares the relative frequency of each class in the training
    and test datasets. If the difference in proportions for any class exceeds
    the specified `tolerance`, it raises a ValueError. Otherwise, it confirms
    that distributions are similar.

    Parameters
    ----------
    y_train : target variable for the training dataset
    y_test  : target variable for the test dataset
    tolerance : maximum allowable absolute difference in class proportions between 
        training and test sets
    """
    
    # Calculate class distributions
    classes = sorted(set(y_train.unique()) | set(y_test.unique()))
    train_dist = y_train.value_counts(normalize=True).reindex(classes, fill_value=0.0)
    test_dist = y_test.value_counts(normalize=True).reindex(classes, fill_value=0.0)
    
    # Check that each class proportion in test set is within tolerance of train set
    for cls in train_dist.index:
        diff = abs(train_dist[cls] - test_dist[cls])
        if diff > tolerance:
            raise ValueError(
                f"Class distribution mismatch exceeds tolerance of {tolerance}:\n"
                f"Train dist: {train_dist.to_dict()}\n"
                f"Test dist:  {test_dist.to_dict()}\n"
                f"Difference: {diff}"
        )
    else:
        print("Class distributions are similar within tolerance.")


# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:48:58.748437Z","iopub.execute_input":"2025-08-11T09:48:58.748779Z","iopub.status.idle":"2025-08-11T09:48:58.772547Z","shell.execute_reply.started":"2025-08-11T09:48:58.748747Z","shell.execute_reply":"2025-08-11T09:48:58.771518Z"}}

# Training & prediction
def train_and_test(pipeline: Pipeline,X_train: pd.DataFrame,X_test: pd.DataFrame,y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a pipeline on training data and generate predictions and probabilities on test data.

    Parameters
    ----------
    pipeline : a scikit-learn compatible pipeline with a fitted classifier that supports `predict()` and `predict_proba()` methods
    X_train : training feature set
    X_test : test feature set
    y_train : target values for training

    Returns
    -------
    y_pred : predicted class labels for X_test
    y_prob : predicted class probabilities for the positive class (class=1) for X_test
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    classes_ = list(pipeline.named_steps['classifier'].classes_)
    pos_idx = classes_.index(POS_CLASS)
    y_prob = pipeline.predict_proba(X_test)[:, pos_idx]
    return y_pred, y_prob
    

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:48:58.773708Z","iopub.execute_input":"2025-08-11T09:48:58.774006Z","iopub.status.idle":"2025-08-11T09:48:58.795721Z","shell.execute_reply.started":"2025-08-11T09:48:58.773983Z","shell.execute_reply":"2025-08-11T09:48:58.794762Z"}}

# Evaluation
def evaluate(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Evaluate classification model performance using multiple metrics.

    Parameters
    ----------
    y_test : true class labels for the test set.
    y_pred : predicted class labels from the model.
    y_prob :predicted probabilities for the positive class (class=1).

    Prints
    ------
    - Classification report (precision, recall, f1-score, support)
    - Confusion matrix
    - ROC AUC score
    - Jaccard score (binary)
    """
    
    labels = np.sort(np.unique(np.r_[y_test, y_pred]))
    target_names = [INV_LABEL_MAP.get(l, str(l)) for l in labels]
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm, 
        index=[f"Actual {INV_LABEL_MAP[l]}" for l in labels], 
        columns=[f"Pred {INV_LABEL_MAP[l]}" for l in labels]
    )
       
    print(f"\033[1mClassification Report:\033[0m")
    print(classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))
    
    print(f"\033[1mConfusion Matrix:\033[0m\n{cm_df}\n")
    
    print(f"\033[1mROC AUC Score:\033[0m\n (pos={INV_LABEL_MAP[POS_CLASS]}):{roc_auc_score(y_test, y_prob):.3f}\n")
    print(f"\033[1mJaccard Score:\033[0m\n (pos={INV_LABEL_MAP[POS_CLASS]}):{jaccard_score(y_test, y_pred, pos_label=POS_CLASS):.3f}\n")


# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T09:48:58.696658Z","iopub.execute_input":"2025-08-11T09:48:58.697012Z","iopub.status.idle":"2025-08-11T09:48:58.720822Z","shell.execute_reply.started":"2025-08-11T09:48:58.696976Z","shell.execute_reply":"2025-08-11T09:48:58.719772Z"}}

# Feature importance
def calc_and_print_xgb_feat_importance(pipeline: Pipeline, top_num=15) -> pd.DataFrame:
    """
    Calculate and display the top XGBoost feature importances from a fitted pipeline.

    Parameters
    ----------
    pipeline : a trained scikit-learn pipeline containing:
        - 'preprocessor': a transformer with get_feature_names_out() method
        - 'classifier': an XGBoost model with feature_importances_ attribute

    Returns
    -------
    feature_importance_df : DataFrame sorted by importance (descending) with columns: 'Feature' and 'Importance'   
    """
        
    xgb_model = pipeline.named_steps['classifier']
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = xgb_model.feature_importances_  
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances.round(4)
    }).sort_values(by='Importance', ascending=False)
    
    print("Top Feature Importances:")
    print(feature_importance_df.head(top_num))

    return feature_importance_df


# %% [code]

# SHAP calculations
def calc_shap(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[shap.Explainer, np.ndarray, shap.Explanation, List[str]]:
    """
    Compute SHAP values for a fitted pipeline containing a preprocessor and an XGBoost classifier.

    Parameters
    ----------
    pipeline : a fitted Pipeline with steps
        - 'preprocessor': transformer supporting transform() and preferably get_feature_names_out()
        - 'classifier'  : fitted XGBoost model (scikit-learn API)
    X : raw input features (unprocessed).

    Returns
    -------
    explainer : SHAP explainer built for the fitted classifier.
    X_transformed : feature matrix after preprocessing (numeric, possibly sparse).
    shap_values : SHAP values for X_transformed.
    feature_names : transformed feature names (matches columns of X_transformed).
    """
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("pipeline must be a fitted scikit-learn Pipeline with named steps.")

    if "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise ValueError("pipeline must contain 'preprocessor' and 'classifier' steps.")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(X)

    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        # Fallback: try to build names from individual transformers
        feature_names = []
        if hasattr(preprocessor, "transformers_"):
            for _, transformer, cols in preprocessor.transformers_:
                if hasattr(transformer, "get_feature_names_out"):
                    names = transformer.get_feature_names_out(cols)
                    feature_names.extend(names)
                else:
                    # If  can't get names, append the raw column names
                    feature_names.extend(list(cols) if hasattr(cols, "__iter__") else [cols])

    # Build SHAP explainer and compute values
    explainer = shap.TreeExplainer(classifier.get_booster())
    shap_values = explainer(X_transformed)

    return explainer, X_transformed, shap_values, feature_names
    
# %% [code]
def plot_summary_shap(X_processed:np.ndarray, shap_values_obj:shap.Explanation, transformed_feature_names:list) -> None: 
    """
    Create a labeled bar plot of mean(|SHAP|) per feature (descending)

    Parameters:
    -----------
    X_processed : transformed features after preprocessing pipeline
    shap_values_obj : SHAP values object from explainer
    transformed_feature_names : list of feature names after transformation

    Returns
    -------
    summary_shap_df : DataFrame sorted by shap value (descending) with columns: 'Feature' and 'Importance'   
    
    """
    shap_arr = shap_values_obj[-1] if isinstance(shap_values_obj, list) else shap_values_obj
    plt.clf()
    shap.summary_plot(shap_arr, features=X_processed, feature_names=transformed_feature_names, plot_type="bar", show=False)
    
    mean_abs_shap = np.abs(shap_arr.values).mean(axis=0)
    order  = np.argsort(mean_abs_shap)
    
    ax = plt.gca()
    for i, v in enumerate(mean_abs_shap[order]):
        ax.text(v + 1e-3, i, f"{v:.3f}", va='center')
    plt.tight_layout()
    plt.show()
    
    return (pd.DataFrame({"Feature": np.array(transformed_feature_names)[order], "|Mean SHAP|": mean_abs_shap[order]})
              .sort_values("|Mean SHAP|", ascending=False))
   

# %% [code]
def plot_dependency_shap(X_processed:np.ndarray, shap_values_obj:shap.Explanation, transformed_feature_names:list, plot_feature='') -> None: 
    """
    Plot SHAP dependence for a single feature.

    Parameters:
    -----------
    X_processed : transformed features after preprocessing pipeline
    shap_values_obj : SHAP values object from explainer
    transformed_feature_names : list of feature names after transformation
    plot_feature : name of the feature to plot
    """
    X_df = pd.DataFrame(X_processed, columns=transformed_feature_names)
    
    shap.dependence_plot(
        plot_feature,  
        shap_values_obj.values,
        X_df,     
        feature_names=transformed_feature_names, 
        show=False
    )
    
    plt.title(f"SHAP Dependence Plot: {plot_feature}")
    plt.tight_layout()
    plt.show()
    

# Save artifacts
def save_artifacts(suffix: str, pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    '''
    Save pipeline, XGBoost model, datasets, and metadata for later reuse.
    '''

    # Extract pipeline components
    preproc = pipeline.named_steps.get('preprocessor', None)
    clf = pipeline.named_steps.get('classifier', pipeline)

    # Feature names after preprocessing
    try:
        feature_names = list(preproc.get_feature_names_out())
    except Exception:
        feature_names = []

    # Save pipeline and classifier
    joblib.dump(pipeline, f'{suffix}_xgboost_pipeline.joblib')

    if hasattr(clf, 'feature_importances_'):
        joblib.dump(clf, f'{suffix}_xgb_classifier.joblib')
        try:
            clf.get_booster().save_model(f'{suffix}_xgb_model.json')
        except Exception:
            clf.save_model(f'{suffix}_xgb_model.json')

    # Save datasets
    X_train.to_parquet(f'{suffix}_X_train.parquet', index=False)
    X_test.to_parquet(f'{suffix}_X_test.parquet', index=False)
    y_train.to_frame('target').to_parquet(f'{suffix}_y_train.parquet', index=False)
    y_test.to_frame('target').to_parquet(f'{suffix}_y_test.parquet', index=False)

    # Extract classifier parameters safely
    def sanitize_params(params):
        sanitized = {}
        for k, v in params.items():
            try:
                json.dumps(v)  # test if serializable
                sanitized[k] = v
            except TypeError:
                sanitized[k] = str(v)  # fallback: convert to string
        return sanitized

    # Save metadata
    meta = {
        'name': suffix,
        'shapes': {
            'X_train': list(X_train.shape),
            'X_test': list(X_test.shape),
            'y_train': int(y_train.shape[0]),
            'y_test': int(y_test.shape[0]),
        },
        'feature_names': feature_names,
        'classifier_params': sanitize_params(clf.get_params()) if hasattr(clf, 'get_params') else {},
        'pipeline_path': f'{suffix}_xgboost_pipeline.joblib',
        'classifier_path': f'{suffix}_xgb_classifier.joblib',
        'model_path': f'{suffix}_xgb_model.json'
    }

    with open(f'{suffix}_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    
if __name__ == '__main__':
    print('Demo run of all utils with Introverts=1 mapping')
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_num1": np.random.randn(100),
        "feature_num2": np.random.randn(100) * 5,
        "feature_cat": np.random.choice(["A", "B", "C"], size=100),
        "target": np.random.choice([0, 1], size=100, p=[0.3, 0.7])  # 1=Introverts, 0=Extroverts
    })

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    numeric_features = ["feature_num1", "feature_num2"]
    categorical_features = ["feature_cat"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], verbose_feature_names_out=False)

    pos_ratio = (y_train == POS_CLASS).mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            eval_metric="logloss", random_state=42,
            scale_pos_weight=scale_pos_weight
        ))
    ])

    # --- Tests of each function ---
    # 1) Check class balance
    check_class_distribution(y_train, y_test, 0.1)

    # 2) Train & predict
    y_pred, y_prob = train_and_test(pipeline, X_train, X_test, y_train)

    # 3) Evaluate metrics
    evaluate(y_test, y_pred, y_prob)

    # 4) Feature importance
    calc_and_print_xgb_feat_importance(pipeline)

    # 5) SHAP calculations + plots
    explainer, Xtr, shap_values, f_names = calc_shap(pipeline, X_train)
    plot_summary_shap(Xtr, shap_values, f_names)
    plot_dependency_shap(Xtr, shap_values, f_names, plot_feature=f_names[0])

    # 6) Cramér’s V demo (association between categorical col and target)
    cv = cramers_v(df["feature_cat"], df["target"])
    print(f"\nCramér’s V(feature_cat, target) = {cv:.3f}")

    # 7) Save artifacts (pipeline, data, metadata)
    save_artifacts("demo", pipeline, X_train, X_test, y_train, y_test)
    print("\nArtifacts saved with prefix 'demo_'")
     
