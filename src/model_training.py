import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# import lightgbm as lgb  # Imported in original notebook but not used in model training

from src import config

def split_and_scale(df: pd.DataFrame):
    """
    Splits the dataset into training and testing sets, and standardizes features.
    """
    target = df[config.TARGET_COLUMN]
    features = df.drop(config.TARGET_COLUMN, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE,
        stratify=target,
        random_state=config.RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_test, scaler

def get_models_and_grids(y_train):
    """
    Initializes ML models and defines hyperparameter tuning grids.
    """
    # Calculate scale_pos_weight for XGBoost to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            random_state=config.RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',
            random_state=config.RANDOM_STATE
        ),
        'XGBoost': xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE
        )
    }

    param_grids = {
        'Decision Tree': {
            'max_depth': [1, 4, None],
            'min_samples_leaf': [2, 5, 10],
            'min_samples_split': [2, 10]
        },
        'Random Forest': {
            'max_depth': [1, 4, None],
            'min_samples_leaf': [2, 5, 10],
            'min_samples_split': [2, 10],
            'max_features': [0.5, 1.0],
            'max_samples': [0.7, 1.0],
            'n_estimators': [50, 200]
        },
        'XGBoost': {
            'max_depth': [1, 4, 6],
            'subsample': [0.5, 1.0],
            'min_child_weight': [2, 5],
            'learning_rate': [0.1, 0.2],
            'n_estimators': [50, 200]
        }
    }

    return models, param_grids

def train_and_optimize_models(X_train_scaled, y_train):
    """
    Orchestrates the model training and hyperparameter optimization via GridSearchCV.
    """
    print("\nStarting model training and hyperparameter optimization...")
    models, param_grids = get_models_and_grids(y_train)
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    
    best_models = {}

    for name, model in models.items():
        if name in param_grids:
            print(f"Tuning and training {name}...")
            # Perform grid search with cross-validation
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train_scaled, y_train)
            best_models[name] = grid.best_estimator_
            print(f"{name} best parameters: {grid.best_params_}\n")
        else:
            print(f"Training {name} (without tuning)...")
            # Train model directly without tuning
            model.fit(X_train_scaled, y_train)
            best_models[name] = model

    print("Model training complete.\n")
    return best_models

def save_artifacts(best_models: dict, scaler: StandardScaler, save_dir: str = None) -> None:
    """
    Saves trained models and scaler as pickle files.
    """
    if save_dir is None:
        save_dir = config.MODELS_DIR
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Save best models
    for name, model in best_models.items():
        model_name_clean = name.lower().replace(" ", "_")
        model_path = os.path.join(save_dir, f'model_{model_name_clean}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model '{name}' to {model_path}")
