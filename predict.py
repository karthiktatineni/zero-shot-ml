import argparse

import pandas as pd
from sklearn.metrics import accuracy_score

from zeroshot_pfn.evaluate import build_pipeline
from zeroshot_pfn.predictor import DEFAULT_CHECKPOINT_PATH, ZeroShotClassifier


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Tabular PFN - CLI Predictor")
    parser.add_argument("--support", type=str, required=True, help="Path to support/training CSV dataset")
    parser.add_argument("--query", type=str, required=True, help="Path to query/testing CSV dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to trained PFN checkpoint",
    )
    parser.add_argument("--dummy-pfn", action="store_true", help="Use untrained PFN for dry run testing")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Path to save predictions")
    args = parser.parse_args()

    print(f"Loading data from {args.support} and {args.query}...")
    df_train = pd.read_csv(args.support)
    df_test = pd.read_csv(args.query)
    
    if args.target not in df_train.columns:
        raise ValueError(f"Target column '{args.target}' not found in train CSV.")
    if args.target not in df_test.columns:
        # Assuming test set has no labels, we will just predict and save
        y_train = df_train[args.target]
        X_train = df_train.drop(columns=[args.target])
        X_test = df_test
        y_test = None
    else:
        y_train = df_train[args.target]
        X_train = df_train.drop(columns=[args.target])
        y_test = df_test[args.target]
        X_test = df_test.drop(columns=[args.target])
        
    print(f"Train context: {X_train.shape[0]} rows, {X_train.shape[1]} features.")
    print(f"Query instances: {X_test.shape[0]} rows.")
    

    # Automatically identify categorical vs continuous for our preprocessor
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    
    # Load model and pipeline
    print("Initializing PFN...")
    pfn_model = ZeroShotClassifier(checkpoint_path=args.checkpoint, dummy=args.dummy_pfn)
    pipe = build_pipeline(pfn_model, is_pfn=True, cat_cols=cat_cols, num_cols=num_cols)
    
    print("Running Zero-Shot Inference...")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    # Save predictions
    df_out = pd.DataFrame({
        "Index": range(len(preds)),
        "Prediction": preds
    })
    
    if y_test is not None:
        df_out["Actual"] = y_test.values
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy on Test Set: {acc:.4f}")
        
    df_out.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")

if __name__ == "__main__":
    main()
