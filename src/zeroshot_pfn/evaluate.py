import argparse
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from zeroshot_pfn.datasets import load_all_datasets
from zeroshot_pfn.predictor import DEFAULT_CHECKPOINT_PATH, ZeroShotClassifier
from zeroshot_pfn.text_extension import TextEmbeddingPreprocessor


class OptionalSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=20):
        self.k = k
        self.selector = None

    def fit(self, X, y=None):
        if X.shape[1] > self.k:
            self.selector = SelectKBest(f_classif, k=self.k)
            self.selector.fit(X, y)
        return self

    def transform(self, X):
        if self.selector is not None:
            return self.selector.transform(X)
        return X


def build_pipeline(model, is_pfn=False, cat_cols=None, num_cols=None, text_cols=None):
    if cat_cols is None: cat_cols = []
    if num_cols is None: num_cols = []
    if text_cols is None: text_cols = []
    
    # Preprocessing asymmetry as requested by user
    # Baselines get OneHotEncoding
    # PFN gets OrdinalEncoding
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    if is_pfn:
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ('scaler', StandardScaler())
        ])
    else:
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
    transformers = [
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
    
    if text_cols:
        text_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('encoder', TextEmbeddingPreprocessor(n_components=5)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('text', text_transformer, text_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('selector', OptionalSelectKBest(k=20)),
        ('classifier', model)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy-pfn', action='store_true', help='Use untrained PFN for dry run testing.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help='Path to PFN checkpoint.',
    )
    parser.add_argument('--out', type=str, default="reports/evaluation_results.md")
    args = parser.parse_args()
    
    datasets = load_all_datasets()
    
    results = []
    
    for name, (X, y) in datasets.items():
        print(f"Evaluating {name}...")
            
        raw_cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        # Simple heuristic for text cols: unique values > 15 and avg string length > 20
        cat_cols = []
        text_cols = []
        for col in raw_cat_cols:
            if X[col].astype(str).str.len().mean() > 20:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y))
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # PFN
        pfn_model = ZeroShotClassifier(checkpoint_path=args.checkpoint, dummy=args.dummy_pfn)
        pfn_pipe = build_pipeline(pfn_model, is_pfn=True, cat_cols=cat_cols, num_cols=num_cols, text_cols=text_cols)
        
        pfn_pipe.fit(X_train, y_train)
        pfn_acc = accuracy_score(y_test, pfn_pipe.predict(X_test))
        
        # Baselines
        baselines = {
            "Dummy": DummyClassifier(strategy="prior"),
            "LogReg": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "HistGB": HistGradientBoostingClassifier(random_state=42)
        }
        
        row = {"Dataset": name, "PFN": pfn_acc}
        for b_name, b_model in baselines.items():
            pipe = build_pipeline(b_model, is_pfn=False, cat_cols=cat_cols, num_cols=num_cols, text_cols=text_cols)
            pipe.fit(X_train, y_train)
            row[b_name] = accuracy_score(y_test, pipe.predict(X_test))
            
        results.append(row)
        
    df_res = pd.DataFrame(results)
    print("\nResults:\n", df_res)
    
    Path("reports").mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Real-Data Evaluation Results\n\n")
        
        f.write("> [!WARNING]\n")
        f.write("> **Preprocessing Asymmetry Limitation:**\n")
        f.write("> To support the continuous-embedding architecture of the PFN, categorical features (like those in Titanic) are transformed using `OrdinalEncoder`. \n")
        f.write("> Baseline models (Logistic Regression, Random Forest, HistGB) are provided standard `OneHotEncoder` representations. \n")
        f.write("> This structurally handicaps the PFN on unordered categoricals (e.g. mapping {C, Q, S} to {0, 1, 2}), creating a spurious numeric ordering. \n")
        f.write("> Any performance loss for the PFN on these datasets must be weighed against this encoding limitation.\n\n")
        
        f.write("## Test Accuracy\n")
        f.write(df_res.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")

if __name__ == "__main__":
    main()
