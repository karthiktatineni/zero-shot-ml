import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from zeroshot_pfn.evaluate import build_pipeline
from zeroshot_pfn.predictor import DEFAULT_CHECKPOINT_PATH, ZeroShotClassifier


def infer_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    raw_categorical = [
        column
        for column in X.columns
        if (
            pd.api.types.is_object_dtype(X[column])
            or pd.api.types.is_string_dtype(X[column])
            or pd.api.types.is_bool_dtype(X[column])
            or isinstance(X[column].dtype, pd.CategoricalDtype)
        )
    ]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical, text = [], []

    for column in raw_categorical:
        if X[column].astype(str).str.len().mean() > 20:
            text.append(column)
        else:
            categorical.append(column)

    return categorical, numeric, text


def detect_target_proxy_columns(X: pd.DataFrame, y: pd.Series) -> list[dict[str, int | float | str]]:
    """Find low-cardinality features that deterministically encode the support labels."""
    proxies = []
    for column in X.columns:
        is_discrete = X[column].nunique(dropna=False) <= 20
        is_categorical = (
            pd.api.types.is_object_dtype(X[column])
            or pd.api.types.is_string_dtype(X[column])
            or pd.api.types.is_bool_dtype(X[column])
            or isinstance(X[column].dtype, pd.CategoricalDtype)
        )
        if not (is_categorical or is_discrete):
            continue

        summary = (
            pd.DataFrame({"feature": X[column], "target": y})
            .groupby("feature", dropna=False, observed=True)["target"]
            .agg(rows="size", target_classes="nunique")
        )
        well_observed = summary[summary["rows"] >= 5]
        coverage = well_observed["rows"].sum() / len(X)
        is_deterministic = (
            len(well_observed) >= 2
            and coverage >= 0.9
            and (well_observed["target_classes"] == 1).all()
        )
        if is_deterministic:
            proxies.append(
                {
                    "Feature": column,
                    "Covered support rows": int(well_observed["rows"].sum()),
                    "Observed values": len(well_observed),
                    "Coverage": coverage,
                }
            )

    return proxies


def make_pfn_pipeline(
    checkpoint_path: str, dummy_mode: bool, cat_cols: list[str], num_cols: list[str], text_cols: list[str], model_kwargs: dict = None
):
    model = ZeroShotClassifier(checkpoint_path=checkpoint_path, dummy=dummy_mode, model_kwargs=model_kwargs)
    return build_pipeline(model, is_pfn=True, cat_cols=cat_cols, num_cols=num_cols, text_cols=text_cols)


def make_baseline_pipeline(cat_cols: list[str], num_cols: list[str], text_cols: list[str]):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return build_pipeline(model, is_pfn=False, cat_cols=cat_cols, num_cols=num_cols, text_cols=text_cols)


def support_validation_score(make_pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float] | None:
    class_counts = y.value_counts()
    n_classes = y.nunique()
    if n_classes < 2 or class_counts.min() < 2:
        return None

    validation_rows = max(n_classes, round(len(y) * 0.2))
    if len(y) - validation_rows < n_classes:
        return None

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_rows, random_state=2026)
    train_idx, validation_idx = next(splitter.split(X, y))
    pipeline = make_pipeline()
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = pipeline.predict(X.iloc[validation_idx])
    actual = y.iloc[validation_idx]
    return {
        "accuracy": accuracy_score(actual, predictions),
        "balanced_accuracy": balanced_accuracy_score(actual, predictions),
    }

st.set_page_config(page_title="Tabular Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Tabular meta-learner")
st.markdown("Upload a labelled CSV to compare prediction quality on a held-out query set.")

with st.sidebar:
    st.header("Model configuration")
    prediction_mode = st.selectbox(
        "Prediction mode",
        options=["Auto reliable", "PFN only", "Classical baseline", "Untrained PFN dry run"],
        help="Auto reliable selects the stronger method using a validation split from the support rows.",
    )
    dummy_mode = prediction_mode == "Untrained PFN dry run"
    checkpoint_path = st.text_input(
        "Checkpoint path",
        str(DEFAULT_CHECKPOINT_PATH),
        disabled=dummy_mode,
    )
    use_10m_model = st.toggle(
        "Use 10M Parameter Model Architecture",
        value="10m" in str(DEFAULT_CHECKPOINT_PATH),
        help="Check this if the checkpoint is the 10M parameter model (d_model=256, 12 layers).",
        disabled=dummy_mode,
    )
    test_size = st.slider("Query set size (%)", min_value=10, max_value=90, value=30, step=5)
    exclude_target_proxies = st.toggle(
        "Exclude likely target proxies",
        value=True,
        help="Removes discrete features that perfectly determine the target within the support data.",
    )

if dummy_mode:
    st.warning("Dry run uses random PFN weights and is expected to perform poorly.")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
results_slot = st.container()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw data preview")
    st.dataframe(df.head(), hide_index=True)
    
    # Target Selection
    target_col = st.selectbox(
        "Select target column to predict",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
    )
    
    if st.button("Run inference", type="primary"):
        with results_slot, st.spinner("Preparing support and query sets..."):
            try:
                df = df.dropna(subset=[target_col])
                y = df[target_col]

                if y.nunique() > 10 and pd.api.types.is_numeric_dtype(y):
                    st.warning(
                        f"Target '{target_col}' is continuous. It has been binned into five classification buckets."
                    )
                    y = pd.qcut(
                        y,
                        q=5,
                        labels=["Very Low", "Low", "Medium", "High", "Very High"],
                        duplicates="drop",
                    )
                elif y.nunique() > 10:
                    st.error(
                        f"Target '{target_col}' has {y.nunique()} classes. This application supports up to 10."
                    )
                    st.stop()

                X = df.drop(columns=[target_col])
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size / 100.0, random_state=42)
                support_idx, query_idx = next(splitter.split(X, y))
                X_support, y_support = X.iloc[support_idx], y.iloc[support_idx]
                X_query, y_query = X.iloc[query_idx], y.iloc[query_idx]
                detected_proxies = detect_target_proxy_columns(X_support, y_support)
                proxy_columns = [proxy["Feature"] for proxy in detected_proxies]
                if exclude_target_proxies and proxy_columns:
                    X_support_model = X_support.drop(columns=proxy_columns)
                    X_query_model = X_query.drop(columns=proxy_columns)
                else:
                    X_support_model, X_query_model = X_support, X_query

                if X_support_model.shape[1] == 0:
                    st.error("All features were removed by leakage screening. Disable the screening to inspect this data.")
                    st.stop()

                cat_cols, num_cols, text_cols = infer_feature_columns(X_support_model)

                pfn_predictions = None
                baseline_predictions = None
                pfn_accuracy = None
                baseline_accuracy = None
                pfn_validation = None
                baseline_validation = None

                run_pfn = prediction_mode in {"Auto reliable", "PFN only", "Untrained PFN dry run"}
                run_baseline = prediction_mode in {"Auto reliable", "Classical baseline"}
                started_at = time.perf_counter()

                if run_pfn:
                    model_kwargs = {"d_model": 256, "n_layers": 12, "n_heads": 8, "d_ff": 1107} if use_10m_model else None
                    pfn_pipeline = make_pfn_pipeline(
                        checkpoint_path, dummy_mode, cat_cols, num_cols, text_cols, model_kwargs
                    )
                    pfn_pipeline.fit(X_support_model, y_support)
                    pfn_predictions = pfn_pipeline.predict(X_query_model)
                    pfn_accuracy = accuracy_score(y_query, pfn_predictions)

                if run_baseline:
                    baseline_pipeline = make_baseline_pipeline(cat_cols, num_cols, text_cols)
                    baseline_pipeline.fit(X_support_model, y_support)
                    baseline_predictions = baseline_pipeline.predict(X_query_model)
                    baseline_accuracy = accuracy_score(y_query, baseline_predictions)

                if prediction_mode == "Auto reliable":
                    pfn_validation = support_validation_score(
                        lambda: make_pfn_pipeline(
                            checkpoint_path, False, cat_cols, num_cols, text_cols, {"d_model": 256, "n_layers": 12, "n_heads": 8, "d_ff": 1107} if use_10m_model else None
                        ),
                        X_support_model,
                        y_support,
                    )
                    baseline_validation = support_validation_score(
                        lambda: make_baseline_pipeline(cat_cols, num_cols, text_cols),
                        X_support_model,
                        y_support,
                    )

                    if (
                        baseline_validation is not None
                        and (
                            pfn_validation is None
                            or baseline_validation["balanced_accuracy"]
                            > pfn_validation["balanced_accuracy"]
                        )
                    ):
                        selected_name, predictions = "Classical baseline", baseline_predictions
                    else:
                        selected_name, predictions = "PFN", pfn_predictions
                elif prediction_mode == "Classical baseline":
                    selected_name, predictions = "Classical baseline", baseline_predictions
                elif dummy_mode:
                    selected_name, predictions = "Untrained PFN", pfn_predictions
                else:
                    selected_name, predictions = "PFN", pfn_predictions

                elapsed = time.perf_counter() - started_at
                selected_accuracy = accuracy_score(y_query, predictions)

                st.success(f"Inference complete in {elapsed:.2f} seconds. Selected: {selected_name}.")
                if detected_proxies:
                    proxy_names = ", ".join(proxy_columns)
                    if exclude_target_proxies:
                        st.warning(
                            f"Leakage screening excluded likely target proxy columns: {proxy_names}. "
                            "Metrics below use the remaining features only."
                        )
                    else:
                        st.warning(
                            f"Potential target proxy columns detected: {proxy_names}. "
                            "Metrics may be overly optimistic while proxy screening is disabled."
                        )
                    st.dataframe(
                        pd.DataFrame(detected_proxies),
                        column_config={"Coverage": st.column_config.NumberColumn(format="percent")},
                        hide_index=True,
                    )

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Support rows", len(X_support))
                col2.metric("Query rows", len(X_query))
                col3.metric("Held-out accuracy", f"{selected_accuracy * 100:.2f}%")
                col4.metric("Prediction classes", pd.Series(predictions).nunique())
                col5.metric("Features used", X_support_model.shape[1])

                diagnostics = []
                if pfn_accuracy is not None:
                    diagnostics.append(
                        {
                            "Model": "PFN",
                            "Query accuracy": f"{pfn_accuracy * 100:.2f}%",
                            "Support validation (balanced)": (
                                f"{pfn_validation['balanced_accuracy'] * 100:.2f}%"
                                if pfn_validation is not None
                                else "Not available"
                            ),
                        }
                    )
                if baseline_accuracy is not None:
                    diagnostics.append(
                        {
                            "Model": "Classical baseline",
                            "Query accuracy": f"{baseline_accuracy * 100:.2f}%",
                            "Support validation (balanced)": (
                                f"{baseline_validation['balanced_accuracy'] * 100:.2f}%"
                                if baseline_validation is not None
                                else "Not available"
                            ),
                        }
                    )
                if diagnostics:
                    st.subheader("Model diagnostics")
                    st.dataframe(pd.DataFrame(diagnostics), hide_index=True)

                if pfn_predictions is not None and pd.Series(pfn_predictions).nunique() == 1:
                    st.warning(
                        "PFN-only predictions collapsed to one class on this dataset. "
                        "Auto reliable avoids using that output when support validation favors the baseline."
                    )

                result = pd.DataFrame(
                    {
                        "Actual": y_query.values,
                        "Predicted": predictions,
                        "Correct": y_query.values == predictions,
                    }
                )
                if pfn_predictions is not None:
                    result["PFN prediction"] = pfn_predictions
                if baseline_predictions is not None:
                    result["Baseline prediction"] = baseline_predictions

                st.subheader("Detailed predictions")
                st.dataframe(result, hide_index=True)

            except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
                st.error(f"Error during inference: {error}")
