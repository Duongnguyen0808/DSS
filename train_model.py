# -*- coding: utf-8 -*-
"""
Train Credit Risk Model - Theo các bước chuẩn Machine Learning
Author: DSS Credit Risk System
Date: 2026-01-03

Pipeline:
1. Load & EDA
2. Data Preprocessing & Feature Engineering
3. Handle Imbalanced Data (SMOTE)
4. Train/Test Split
5. Model Training (Random Forest, XGBoost, LightGBM)
6. Cross-Validation
7. Hyperparameter Tuning
8. Model Evaluation (Confusion Matrix, ROC-AUC, Precision-Recall)
9. Feature Importance
10. Save Model + Scaler
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, auc
)

# Handle Imbalanced Data
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced Models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not installed. Run: pip install lightgbm")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# STEP 1: LOAD DATA & EDA
# ============================================================================

def load_and_explore_data():
    """Load dữ liệu và phân tích sơ bộ"""
    print("=" * 80)
    print("STEP 1: LOADING & EXPLORING DATA")
    print("=" * 80)
    
    # Load dataset
    df = pd.read_csv('credit_risk_dataset.csv')
    
    print(f"\n✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\n📊 Dataset Info:")
    print(df.info())
    
    print(f"\n📈 Statistical Summary:")
    print(df.describe())
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values!")
    
    print(f"\n🎯 Target Distribution:")
    print(df['loan_status'].value_counts())
    print(f"Class Balance: {df['loan_status'].value_counts(normalize=True)}")
    
    return df


# ============================================================================
# STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================

def preprocess_data(df):
    """Xử lý dữ liệu và tạo features mới"""
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 80)
    
    # Copy để không ảnh hưởng dữ liệu gốc
    df = df.copy()
    
    # 1. Handle Missing Values
    print("\n1. Handling missing values...")
    if df.isnull().sum().sum() > 0:
        # Fill missing numeric với median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"   - Filled {col} with median")
        
        # Fill missing categorical với mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"   - Filled {col} with mode")
    else:
        print("   ✓ No missing values to handle")
    
    # 2. Remove Outliers (IQR method cho numerical columns)
    print("\n2. Handling outliers...")
    numeric_features = ['person_age', 'person_income', 'person_emp_length', 
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                        'cb_person_cred_hist_length']
    
    original_size = len(df)
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR  # 3*IQR để giữ nhiều data hơn
        upper = Q3 + 3 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    print(f"   ✓ Removed {original_size - len(df)} outlier rows ({((original_size - len(df))/original_size*100):.1f}%)")
    
    # 3. Feature Engineering - Transformations (GIỐNG R NOTEBOOK)
    print("\n3. Feature engineering - transformations (theo R notebook)...")
    
    # ⭐ CRITICAL: Transformations phải GIỐNG CHÍNH XÁC với lúc predict
    # Lưu original values trước khi transform
    df['loan_amnt_original'] = df['loan_amnt'].copy()
    df['person_income_original'] = df['person_income'].copy()
    
    # Transform 1: Square root cho loan_amnt (giống R: sqrt(loan_amnt))
    df['loan_amnt'] = np.sqrt(df['loan_amnt'])
    print("   ✓ Transformed: loan_amnt → sqrt(loan_amnt)")
    
    # Transform 2: 1/log transform cho person_income (giống R: 1/log(income))
    df['person_income'] = 1 / np.log(df['person_income'] + 1)
    print("   ✓ Transformed: person_income → 1/log(income+1)")
    
    # Transform 3: Cap loan_percent_income tại 1.0
    df['loan_percent_income'] = df['loan_percent_income'].clip(upper=1.0)
    print("   ✓ Capped: loan_percent_income at 1.0")
    
    # 4. Create New Features
    print("\n4. Creating new features...")
    
    # Age group
    df['age_group'] = pd.cut(df['person_age'], bins=[0, 25, 35, 50, 150], 
                              labels=['Young', 'Adult', 'Middle', 'Senior'])
    print("   ✓ Created: age_group")
    
    # Income level (dựa trên original income TRƯỚC KHI transform)
    df['income_level'] = pd.cut(df['person_income_original'], 
                                 bins=[0, 30000, 60000, 100000, float('inf')],
                                 labels=['Low', 'Medium', 'High', 'VeryHigh'])
    print("   ✓ Created: income_level")
    
    # Employment stability
    df['employment_stability'] = pd.cut(df['person_emp_length'],
                                        bins=[-1, 2, 5, 10, float('inf')],
                                        labels=['New', 'Stable', 'Experienced', 'Veteran'])
    print("   ✓ Created: employment_stability")
    
    # Debt burden ratio groups
    df['debt_burden'] = pd.cut(df['loan_percent_income'],
                                bins=[0, 0.2, 0.4, 0.6, 1.1],
                                labels=['Low', 'Medium', 'High', 'VeryHigh'])
    print("   ✓ Created: debt_burden")
    
    # Interaction features
    df['income_to_loan'] = df['person_income'] / (df['loan_amnt'] + 1)
    print("   ✓ Created: income_to_loan")
    
    df['credit_per_age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)
    print("   ✓ Created: credit_per_age")
    
    print(f"\n✓ Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


# ============================================================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ============================================================================

def encode_features(df):
    """Encode categorical variables"""
    print("\n" + "=" * 80)
    print("STEP 3: ENCODING CATEGORICAL VARIABLES")
    print("=" * 80)
    
    df = df.copy()
    label_encoders = {}
    
    categorical_columns = [
        'person_home_ownership', 'loan_intent', 'loan_grade',
        'cb_person_default_on_file', 'age_group', 'income_level',
        'employment_stability', 'debt_burden'
    ]
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"✓ Encoded: {col} -> {len(le.classes_)} classes")
    
    return df, label_encoders


# ============================================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================================

def split_data(df, test_size=0.2, random_state=42):
    """Chia train/test set với stratification"""
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN/TEST SPLIT + SCALING")
    print("=" * 80)
    
    # Separate features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Check class imbalance
    class_dist = y.value_counts(normalize=True)
    print(f"\n📊 Class Distribution:")
    print(f"   Class 0 (No Default): {class_dist[0]:.2%}")
    print(f"   Class 1 (Default):    {class_dist[1]:.2%}")
    
    imbalance_ratio = class_dist[0] / class_dist[1] if class_dist[1] > 0 else 0
    print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 1.5:
        print(f"   ⚠️  Dataset is imbalanced! Will use SMOTE later.")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    total_samples = len(X)
    train_pct = (len(X_train) / total_samples) * 100
    test_pct = (len(X_test) / total_samples) * 100
    
    print(f"\n" + "="*50)
    print(f"📊 TRAIN/TEST SPLIT SUMMARY")
    print(f"="*50)
    print(f"Total Samples:     {total_samples:,}")
    print(f"Training Set:      {X_train.shape[0]:,} samples ({train_pct:.1f}%)")
    print(f"Test Set:          {X_test.shape[0]:,} samples ({test_pct:.1f}%)")
    print(f"="*50)
    
    print(f"\n📊 Training set class distribution:")
    train_class_counts = y_train.value_counts()
    train_class_pcts = y_train.value_counts(normalize=True)
    print(f"   Class 0: {train_class_counts[0]:,} ({train_class_pcts[0]:.2%})")
    print(f"   Class 1: {train_class_counts[1]:,} ({train_class_pcts[1]:.2%})")
    
    # ⭐ SCALING: StandardScaler cho numerical features
    print(f"\n🔄 Applying StandardScaler to numerical features...")
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"✓ StandardScaler fitted and applied")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler


# ============================================================================
# STEP 5: TRAIN MULTIPLE MODELS
# ============================================================================

def train_random_forest(X_train, y_train, X_test, y_test, use_smote=False):
    """Train Random Forest"""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST")
    print("=" * 80)
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_use, y_train_use = apply_smote(X_train, y_train)
    else:
        X_train_use, y_train_use = X_train, y_train
    
    # Base model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_use, y_train_use)
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    y_train_proba = rf_model.predict_proba(X_train)[:, 1]
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    return rf_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def train_xgboost(X_train, y_train, X_test, y_test, use_smote=False):
    """Train XGBoost Classifier"""
    if not XGBOOST_AVAILABLE:
        print("\nXGBoost not available, skipping...")
        return None, None, None, None, None
    
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_use, y_train_use = apply_smote(X_train, y_train)
    else:
        X_train_use, y_train_use = X_train, y_train
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_use, y_train_use)
    
    # Predictions
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    y_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    return xgb_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def train_lightgbm_UNUSED(X_train, y_train, X_test, y_test, use_smote=True):
    """Train LightGBM Classifier"""
    if not LIGHTGBM_AVAILABLE:
        print("\n⚠️  LightGBM not available, skipping...")
        return None, None, None, None, None
    
    print("\n" + "=" * 80)
    print("STEP 5C: TRAINING LIGHTGBM")
    print("=" * 80)
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_use, y_train_use = apply_smote(X_train, y_train)
    else:
        X_train_use, y_train_use = X_train, y_train
    
    print(f"\n1. Training LightGBM...")
    lgbm_model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgbm_model.fit(X_train_use, y_train_use)
    
    # Cross-validation
    print("\n2. Cross-validation (5-fold)...")
    cv_scores = cross_val_score(lgbm_model, X_train_use, y_train_use, cv=5, scoring='roc_auc')
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predictions
    y_train_pred = lgbm_model.predict(X_train)
    y_test_pred = lgbm_model.predict(X_test)
    y_train_proba = lgbm_model.predict_proba(X_train)[:, 1]
    y_test_proba = lgbm_model.predict_proba(X_test)[:, 1]
    
    return lgbm_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def train_logistic_regression(X_train, y_train, X_test, y_test, use_smote=False):
    """Train Logistic Regression"""
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 80)
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_use, y_train_use = apply_smote(X_train, y_train)
    else:
        X_train_use, y_train_use = X_train, y_train
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_use, y_train_use)
    
    # Predictions
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    y_train_proba = lr_model.predict_proba(X_train)[:, 1]
    y_test_proba = lr_model.predict_proba(X_test)[:, 1]
    
    return lr_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def train_decision_tree_UNUSED(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Train Decision Tree
    Giải thích: Cây quyết định - dễ hiểu, visual rules
    Phù hợp: Giải thích business rules, feature interactions
    """
    print("\n" + "=" * 80)
    print("STEP 5E: TRAINING DECISION TREE")
    print("=" * 80)
    print("\n💡 Tại sao dùng Decision Tree?")
    print("   - Dễ hiểu và visualize được")
    print("   - Capture được non-linear relationships")
    print("   - Không cần feature scaling")
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_use, y_train_use = apply_smote(X_train, y_train)
    else:
        X_train_use, y_train_use = X_train, y_train
    
    print(f"\n1. Training Decision Tree với pruning...")
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42
    )
    
    dt_model.fit(X_train_use, y_train_use)
    
    # Cross-validation
    print("\n2. Cross-validation (5-fold)...")
    cv_scores = cross_val_score(dt_model, X_train_use, y_train_use, cv=5, scoring='roc_auc')
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predictions
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)
    y_train_proba = dt_model.predict_proba(X_train)[:, 1]
    y_test_proba = dt_model.predict_proba(X_test)[:, 1]
    
    print("✓ Decision Tree training completed!")
    return dt_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


def train_gradient_boosting_UNUSED(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Train Gradient Boosting (sklearn)
    Giải thích: Boosting model mạnh mẽ từ sklearn
    Phù hợp: Performance cao, ổn định
    """
    print("\n" + "=" * 80)
    print("STEP 5F: TRAINING GRADIENT BOOSTING")
    print("=" * 80)
    print("\n💡 Tại sao dùng Gradient Boosting?")
    print("   - Boosting ensemble method - học từ mistakes")
    print("   - Performance tốt, ít overfitting hơn Decision Tree")
    print("   - Sklearn built-in, không cần thư viện thêm")
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_use, y_train_use = apply_smote(X_train, y_train)
    else:
        X_train_use, y_train_use = X_train, y_train
    
    print(f"\n1. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    gb_model.fit(X_train_use, y_train_use)
    
    # Cross-validation
    print("\n2. Cross-validation (5-fold)...")
    cv_scores = cross_val_score(gb_model, X_train_use, y_train_use, cv=5, scoring='roc_auc')
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predictions
    y_train_pred = gb_model.predict(X_train)
    y_test_pred = gb_model.predict(X_test)
    y_train_proba = gb_model.predict_proba(X_train)[:, 1]
    y_test_proba = gb_model.predict_proba(X_test)[:, 1]
    
    print("✓ Gradient Boosting training completed!")
    return gb_model, y_train_pred, y_test_pred, y_train_proba, y_test_proba


# ============================================================================
# STEP 6: COMPREHENSIVE MODEL EVALUATION
# ============================================================================

def evaluate_model_simple(y_test, y_test_pred, y_test_proba, model_name="Model"):
    """
    Đánh giá model đơn giản - chỉ in Classification Report và ROC AUC
    Format giống notebook Python
    """
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_test_pred))
    
    roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\n" + "-" * 80)
    
    # Return metrics for comparison
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'specificity': specificity,
        'sensitivity': sensitivity
    }


def evaluate_model_UNUSED(y_train, y_train_pred, y_test, y_test_pred, y_train_proba, y_test_proba, model_name="Model"):
    """Đánh giá model với nhiều metrics + visualizations"""
    print("\n" + "=" * 80)
    print(f"STEP 6: EVALUATING {model_name.upper()}")
    print("=" * 80)
    
    # Training metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n📊 Training Metrics:")
    print(f"   Accuracy: {train_acc:.4f}")
    
    # Test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {test_acc:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1-Score:  {test_f1:.4f}")
    print(f"   ROC-AUC:   {test_auc:.4f}")
    print(f"   PR-AUC:    {pr_auc:.4f}")
    
    # ⭐ Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Print formatted confusion matrix like in the image
    print(f"\n📈 Confusion Matrix:")
    print(f"   array([[{cm[0][0]:>5}, {cm[0][1]:>5}],")
    print(f"          [{cm[1][0]:>5}, {cm[1][1]:>5}]])")
    
    # Calculate specificity, sensitivity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n   True Negatives:  {tn:>5}")
    print(f"   False Positives: {fp:>5}")
    print(f"   False Negatives: {fn:>5}")
    print(f"   True Positives:  {tp:>5}")
    print(f"   Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   Specificity:          {specificity:.4f}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['No Default (0)', 'Default (1)']))
    
    # Check overfitting
    if train_acc - test_acc > 0.05:
        print(f"\n⚠️  Warning: Possible overfitting (train-test gap: {train_acc - test_acc:.4f})")
    else:
        print(f"\n✓ Good generalization (train-test gap: {train_acc - test_acc:.4f})")
    
    # Return metrics dictionary
    metrics = {
        'train_accuracy': train_acc,
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'roc_auc': test_auc,
        'pr_auc': pr_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }
    
    # ⭐ Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved: confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    # ⭐ Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved: roc_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    # ⭐ Plot Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'precision_recall_curve_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Precision-Recall curve saved: precision_recall_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    return {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'roc_auc': test_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'specificity': specificity,
        'sensitivity': sensitivity
    }


# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance_UNUSED(model, feature_names, top_n=20):
    """Vẽ biểu đồ feature importance"""
    print("\n" + "=" * 80)
    print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n🏆 Top {top_n} Most Important Features:")
    for i in range(min(top_n, len(feature_names))):
        print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.barh(range(top_n), importances[indices[:top_n]], color='steelblue', alpha=0.8)
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature importance plot saved: feature_importance.png")


# ============================================================================
# STEP 8: SAVE MODEL + SCALER + METADATA
# ============================================================================

def save_model(model, label_encoders, feature_names, metrics, transformations, scaler, model_name="Random Forest"):
    """Lưu model, scaler và metadata"""
    print("\n" + "=" * 80)
    print("STEP 8: SAVING MODEL + SCALER + METADATA")
    print("=" * 80)
    
    # Save model
    with open('credit_risk_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Saved: credit_risk_model.pkl")
    
    # Save encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("✓ Saved: label_encoders.pkl")
    
    # ⭐ Save scaler (QUAN TRỌNG cho predict)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Saved: scaler.pkl")
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print("✓ Saved: feature_names.pkl")
    
    # Save metadata
    metadata = {
        'model_type': model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics.get('pr_auc', 0),
        'confusion_matrix': metrics['confusion_matrix'],
        'specificity': metrics.get('specificity', 0),
        'sensitivity': metrics.get('sensitivity', 0),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'transformations': transformations,
        'scaling': 'StandardScaler'
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("✓ Saved: model_metadata.pkl")
    
    print(f"\n{'='*80}")
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\n📊 Final Model Performance ({model_name}):")
    print(f"   Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   F1-Score:    {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:      {metrics.get('pr_auc', 0):.4f}")
    print(f"   Specificity: {metrics.get('specificity', 0):.4f}")
    print(f"   Sensitivity: {metrics.get('sensitivity', 0):.4f}")


# ============================================================================
# STEP 9: COMPARE MULTIPLE MODELS
# ============================================================================

def compare_models(models_results):
    """So sánh kết quả của nhiều models"""
    print("\n" + "=" * 100)
    print(" " * 35 + "MODEL COMPARISON TABLE")
    print("=" * 100)
    
    # Create comparison table
    comparison_df = pd.DataFrame(models_results).T
    
    # Select key metrics for display
    metrics_display = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    df_display = comparison_df[metrics_display].round(4)
    
    # Print formatted table header
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"
    ))
    print("-" * 100)
    
    # Print each row
    for idx, row in df_display.iterrows():
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            idx,
            row['accuracy'],
            row['precision'],
            row['recall'],
            row['f1_score'],
            row['roc_auc']
        ))
    
    print("=" * 100)
    
    # Save comparison
    comparison_df.to_csv('model_comparison.csv', index=True)
    print("\n✓ Saved: model_comparison.csv")
    
    # Plot comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in comparison_df.columns:
            ax = axes[idx]
            comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
            ax.set_title(metric.upper().replace('_', ' '), fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()
    
    # Find best model
    best_model_name = comparison_df['roc_auc'].idxmax()
    best_roc_auc = comparison_df.loc[best_model_name, 'roc_auc']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   ROC-AUC: {best_roc_auc:.4f}")
    
    return best_model_name


# ============================================================================
# COMPARE MULTIPLE MODELS
# ============================================================================

def compare_models_UNUSED(models_results):
    """So sánh kết quả của nhiều models"""
    print("\n" + "=" * 100)
    print(" " * 35 + "MODEL COMPARISON TABLE")
    print("=" * 100)
    
    # Create comparison table
    comparison_df = pd.DataFrame(models_results).T
    
    # Select key metrics
    metrics_display = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    if all(m in comparison_df.columns for m in metrics_display):
        comparison_df = comparison_df[metrics_display]
    
    comparison_df = comparison_df.round(4)
    
    # Print formatted table
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"
    ))
    print("-" * 100)
    
    for idx, row in comparison_df.iterrows():
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            idx,
            row.get('accuracy', 0),
            row.get('precision', 0),
            row.get('recall', 0),
            row.get('f1_score', 0),
            row.get('roc_auc', 0)
        ))
    
    print("=" * 100)
    
    # Save comparison
    comparison_df.to_csv('model_comparison.csv', index=True)
    print("\n✓ Saved: model_comparison.csv")
    
    # Plot comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in comparison_df.columns:
            ax = axes[idx]
            comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
            ax.set_title(metric.upper().replace('_', ' '), fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()
    
    # Find best model
    best_model_name = comparison_df['roc_auc'].idxmax()
    best_roc_auc = comparison_df.loc[best_model_name, 'roc_auc']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   ROC-AUC: {best_roc_auc:.4f}")
    
    return best_model_name


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    # Set UTF-8 encoding for console output
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("\n" + "="*80)
    print(" " * 20 + "CREDIT RISK MODEL TRAINING")
    print(" " * 25 + "Standard ML Pipeline")
    print("="*80 + "\n")
    
    # Step 1: Load & EDA
    df = load_and_explore_data()
    
    # Step 2: Preprocessing & Feature Engineering
    df_processed = preprocess_data(df)
    
    # Step 3: Encode categorical
    df_encoded, label_encoders = encode_features(df_processed)
    
    # Step 4: Split data & Apply StandardScaler
    X_train, X_test, y_train, y_test, feature_names, scaler = split_data(df_encoded)
    
    # Execute steps 5-9 (Train, Evaluate, Compare, Feature Importance, Save)
    train_and_compare_models(X_train, X_test, y_train, y_test, feature_names, 
                             scaler, label_encoders)


def train_and_compare_models(X_train, X_test, y_train, y_test, feature_names, 
                              scaler, label_encoders):
    """Train 3 models: Logistic Regression, Random Forest, XGBoost"""
    
    results = {}
    models = {}
    
    # 1. Logistic Regression
    print("\n" + "=" * 80)
    lr_model, lr_y_train_pred, lr_y_test_pred, lr_y_train_proba, lr_y_test_proba = train_logistic_regression(
        X_train, y_train, X_test, y_test, use_smote=False
    )
    results['Logistic Regression'] = evaluate_model_simple(y_test, lr_y_test_pred, lr_y_test_proba, "Logistic Regression")
    models['Logistic Regression'] = lr_model
    
    # 2. Random Forest
    print("\n" + "=" * 80)
    rf_model, rf_y_train_pred, rf_y_test_pred, rf_y_train_proba, rf_y_test_proba = train_random_forest(
        X_train, y_train, X_test, y_test, use_smote=False
    )
    results['Random Forest'] = evaluate_model_simple(y_test, rf_y_test_pred, rf_y_test_proba, "Random Forest")
    models['Random Forest'] = rf_model
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 80)
        xgb_model, xgb_y_train_pred, xgb_y_test_pred, xgb_y_train_proba, xgb_y_test_proba = train_xgboost(
            X_train, y_train, X_test, y_test, use_smote=False
        )
        results['XGBoost'] = evaluate_model_simple(y_test, xgb_y_test_pred, xgb_y_test_proba, "XGBoost")
        models['XGBoost'] = xgb_model
    
    # Compare models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    comparison_df = pd.DataFrame(results).T
    print("\n", comparison_df.round(4))
    
    # Find best model by ROC AUC
    best_model_name = comparison_df['roc_auc'].idxmax()
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name} (ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f})")
    
    # Feature importance for Random Forest
    if 'Random Forest' in models:
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE - RANDOM FOREST")
        print("=" * 80)
        feat_imp = pd.Series(models['Random Forest'].feature_importances_, index=feature_names).sort_values(ascending=False)
        print("\nTop 10 Features:")
        print(feat_imp.head(10))
    
    # Feature importance for XGBoost
    if 'XGBoost' in models:
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE - XGBOOST")
        print("=" * 80)
        feat_imp = pd.Series(models['XGBoost'].feature_importances_, index=feature_names).sort_values(ascending=False)
        print("\nTop 10 Features:")
        print(feat_imp.head(10))
    
    # Save best model
    transformations = {
        'person_income': 'log_transform: 1 / np.log(x + 1)',
        'loan_amnt': 'sqrt_transform: np.sqrt(x)',
        'loan_percent_income': 'clip at 1.0'
    }
    
    save_model(best_model, label_encoders, feature_names, 
               results[best_model_name], transformations, scaler, model_name=best_model_name)


if __name__ == '__main__':
    main()
