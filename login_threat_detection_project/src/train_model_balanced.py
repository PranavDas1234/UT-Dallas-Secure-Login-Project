"""
Balanced training script for the RBA login threat detection project.

Recommended first run:
    python src/train_model_balanced.py --target "Is Attack IP" --max-rows 1000000 --negatives-per-positive 5

More direct but slower account-takeover run:
    python src/train_model_balanced.py --target "Is Account Takeover" --max-rows -1 --negatives-per-positive 20
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from rba_utils import BASE_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, MODEL_DIR, REPORT_DIR, basic_clean, find_dataset_csv, reduce_rare_categories, apply_category_maps

def build_preprocessor(X):
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X.columns] + (["is_missing_timestamp"] if "is_missing_timestamp" in X.columns else [])
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipeline, numeric_cols), ("cat", categorical_pipeline, categorical_cols)])

def build_classifier(model_name, random_state):
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    return RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)

def choose_best_threshold(y_true, y_probability, min_recall=None):
    """If min_recall is given, pick the lowest threshold that still achieves
    at least that recall (appropriate for security use cases where missing
    an attack is worse than a false alarm). Otherwise pick the F1-optimal
    threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probability)
    if min_recall is not None:
        candidates = [(t, p, r) for p, r, t in zip(precision[:-1], recall[:-1], thresholds) if r >= min_recall]
        if candidates:
            t, p, r = min(candidates, key=lambda c: c[0])  # lowest threshold meeting recall target
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return float(t), f1
        print(f"[WARN] No threshold achieves recall >= {min_recall}. Falling back to F1-optimal.")
    best_threshold, best_f1 = 0.50, -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)
    return best_threshold, best_f1

def make_features_and_target(df, target_col, use_attack_ip_as_feature=False, exclude_login_successful=False):
    feature_cols = list(BASE_FEATURES) + (["is_missing_timestamp"] if "is_missing_timestamp" in df.columns else [])
    if use_attack_ip_as_feature and target_col != "Is Attack IP":
        feature_cols.append("Is Attack IP")
    if exclude_login_successful and "Login Successful" in feature_cols:
        feature_cols.remove("Login Successful")
    feature_cols = [c for c in feature_cols if c != target_col and c in df.columns]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y, feature_cols

def load_balanced_training_data(csv_path, target_col, max_rows, chunk_size, negatives_per_positive, min_negatives_per_chunk, random_state):
    positive_chunks, negative_chunks = [], []
    total_rows_seen = total_positives_seen = 0
    reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)
    for chunk_index, chunk in enumerate(reader):
        if max_rows != -1:
            remaining = max_rows - total_rows_seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        total_rows_seen += len(chunk)
        chunk = basic_clean(chunk).dropna(subset=[target_col]).copy()
        positives = chunk[chunk[target_col] == 1]
        negatives = chunk[chunk[target_col] == 0]
        total_positives_seen += len(positives)
        if len(positives) > 0:
            positive_chunks.append(positives)
        desired_negatives = max(min_negatives_per_chunk, len(positives) * negatives_per_positive)
        desired_negatives = min(desired_negatives, len(negatives))
        if desired_negatives > 0:
            negative_chunks.append(negatives.sample(n=desired_negatives, random_state=random_state + chunk_index))
        print(f"[INFO] Chunk {chunk_index+1}: rows seen={total_rows_seen:,}, positives seen={total_positives_seen:,}, negatives sampled={sum(len(x) for x in negative_chunks):,}")
        if max_rows != -1 and total_rows_seen >= max_rows:
            break
    if not positive_chunks:
        raise ValueError(f"No positive examples found for target {target_col!r}. Increase --max-rows.")
    positives_df = pd.concat(positive_chunks, ignore_index=True)
    negatives_df = pd.concat(negative_chunks, ignore_index=True)
    if len(positives_df) < 30:
        print("\n[WARN] Fewer than 30 positive examples found. Results may still be unstable. For account takeover, use --max-rows -1.\n")
    desired_final_negatives = min(len(negatives_df), max(100, len(positives_df) * negatives_per_positive))
    negatives_df = negatives_df.sample(n=desired_final_negatives, random_state=random_state)
    balanced_df = pd.concat([positives_df, negatives_df], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("\n[INFO] Final balanced dataset")
    print(f"  Positives: {len(positives_df):,}")
    print(f"  Negatives: {len(negatives_df):,}")
    print(f"  Total:     {len(balanced_df):,}")
    print(f"  Positive rate in training table: {len(positives_df)/len(balanced_df):.4f}")
    return balanced_df

def load_natural_sample(csv_path, target_col, sample_rows, random_state):
    """Pull a fresh, UNBALANCED sample straight from the dataset (skipping
    the rows already used for training/validation as best-effort by
    sampling further into the file). Used to report realistic precision,
    since the balanced test set makes precision look artificially high."""
    reader = pd.read_csv(csv_path, chunksize=sample_rows, skiprows=range(1, sample_rows + 1))
    chunk = next(reader)
    chunk = basic_clean(chunk).dropna(subset=[target_col]).copy()
    return chunk

def evaluate_with_threshold(model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "f1": f1_score(y, predictions, zero_division=0),
        "average_precision_pr_auc": average_precision_score(y, probabilities),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y, probabilities)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics, confusion_matrix(y, predictions), classification_report(y, predictions, digits=4, zero_division=0)

def get_feature_importance_report(model, feature_columns, top_n=20):
    """Extract feature importances from the fitted pipeline, mapping
    one-hot-encoded column names back to their original feature."""
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return "Feature importances not available for this model type.\n"
    preprocessor = model.named_steps["preprocess"]
    try:
        encoded_names = preprocessor.get_feature_names_out()
    except Exception:
        return "Could not extract encoded feature names.\n"
    importances = classifier.feature_importances_
    pairs = sorted(zip(encoded_names, importances), key=lambda p: p[1], reverse=True)[:top_n]
    lines = [f"{name}: {importance:.6f}" for name, importance in pairs]
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Is Attack IP", choices=["Is Account Takeover", "Is Attack IP"])
    parser.add_argument("--max-rows", type=int, default=1_000_000, help="Use -1 to scan full dataset")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--negatives-per-positive", type=int, default=5)
    parser.add_argument("--min-negatives-per-chunk", type=int, default=200)
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic"])
    parser.add_argument("--top-categories", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-attack-ip-as-feature", action="store_true", help="Only for Is Account Takeover; behavior-only is better for reports")
    parser.add_argument("--exclude-login-successful", action="store_true", help="Use this if simulating a LIVE/pre-decision scoring system, since 'Login Successful' is not known yet at that point")
    parser.add_argument("--min-recall", type=float, default=None, help="If set, pick lowest threshold achieving at least this recall instead of the F1-optimal threshold")
    parser.add_argument("--natural-eval-rows", type=int, default=200_000, help="Rows to sample for an unbalanced, realistic-precision evaluation")
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = find_dataset_csv()
    print(f"[INFO] Using dataset: {csv_path}")
    df = load_balanced_training_data(csv_path, args.target, args.max_rows, args.chunk_size, args.negatives_per_positive, args.min_negatives_per_chunk, args.random_state)
    X, y, feature_columns = make_features_and_target(df, args.target, args.use_attack_ip_as_feature, args.exclude_login_successful)
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=args.random_state, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=args.random_state, stratify=y_temp)
    # Fit category maps on TRAIN ONLY, then apply to valid/test (fixes leakage).
    X_train, category_maps = reduce_rare_categories(X_train, categorical_cols, top_n=args.top_categories)
    X_valid = apply_category_maps(X_valid, category_maps)
    X_test = apply_category_maps(X_test, category_maps)
    print(f"[INFO] Training model: {args.model}")
    model = Pipeline([("preprocess", build_preprocessor(X_train)), ("classifier", build_classifier(args.model, args.random_state))])
    model.fit(X_train, y_train)
    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    threshold, valid_best_f1 = choose_best_threshold(y_valid, valid_probabilities, min_recall=args.min_recall)
    print(f"[INFO] Selected threshold: {threshold:.4f}")
    print(f"[INFO] F1 at that threshold (validation): {valid_best_f1:.4f}")
    test_metrics, test_matrix, test_report = evaluate_with_threshold(model, X_test, y_test, threshold)
    print("\n[TEST RESULTS - BALANCED TEST SET]")
    print(f"Threshold: {threshold:.4f}")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nConfusion Matrix:")
    print(test_matrix)
    print("\nClassification Report:")
    print(test_report)
    feature_importance_text = get_feature_importance_report(model, feature_columns)
    print("\n[FEATURE IMPORTANCES - top 20]")
    print(feature_importance_text)
    # Natural / unbalanced evaluation for realistic precision reporting.
    natural_metrics_text = ""
    try:
        natural_df = load_natural_sample(csv_path, args.target, args.natural_eval_rows, args.random_state)
        X_nat, y_nat, _ = make_features_and_target(natural_df, args.target, args.use_attack_ip_as_feature, args.exclude_login_successful)
        X_nat = apply_category_maps(X_nat[feature_columns], category_maps)
        nat_metrics, nat_matrix, nat_report = evaluate_with_threshold(model, X_nat, y_nat, threshold)
        print("\n[TEST RESULTS - NATURAL/UNBALANCED SAMPLE] (more realistic precision)")
        for key, value in nat_metrics.items():
            print(f"{key}: {value:.4f}")
        print(nat_matrix)
        natural_metrics_text = "\nNatural (Unbalanced) Sample Metrics\n" + "-"*56 + "\n" + "\n".join(f"{k}: {v:.6f}" for k, v in nat_metrics.items()) + "\n\n" + str(nat_matrix) + "\n\n" + nat_report
    except Exception as e:
        print(f"[WARN] Could not compute natural-sample evaluation: {e}")
    bundle = {"pipeline": model, "target": args.target, "threshold": threshold, "category_maps": category_maps, "feature_columns": feature_columns, "model_name": args.model}
    model_path = MODEL_DIR / "login_threat_model_balanced.joblib"
    joblib.dump(bundle, model_path)
    report_path = REPORT_DIR / "balanced_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Balanced Login Threat Detection Evaluation Report\n" + "="*56 + "\n\n")
        f.write(f"Target: {args.target}\nModel: {args.model}\nMax rows scanned: {args.max_rows}\nNegatives per positive: {args.negatives_per_positive}\nThreshold: {threshold:.6f}\nFeatures: {feature_columns}\n\n")
        f.write("Test Metrics (Balanced Test Set)\n" + "-"*56 + "\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\nConfusion Matrix\n" + "-"*56 + "\n" + str(test_matrix) + "\n\nClassification Report\n" + "-"*56 + "\n" + test_report)
        f.write("\nTop Feature Importances\n" + "-"*56 + "\n" + feature_importance_text)
        f.write(natural_metrics_text)
    print(f"[OK] Saved model to: {model_path}")
    print(f"[OK] Saved report to: {report_path}")

if __name__ == "__main__":
    main()
