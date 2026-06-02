import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, make_scorer
)
from sklearn.model_selection import learning_curve, StratifiedKFold
from src import config

def evaluate_models(best_models: dict, X_train_scaled, X_test_scaled, y_train, y_test) -> tuple:
    """
    Evaluates all trained models on both train and test data, printing their AUC scores,
    and returns the name and object of the best performing model.
    """
    # print(">> Model Evaluation Results <<")
    best_score = 0
    best_model_name = None

    for name, model in best_models.items():
        # Evaluate on training data
        y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
        auc_train = roc_auc_score(y_train, y_train_prob)
        print(f">> {name} AUC (train): {auc_train * 100:.4f}%")

        # Evaluate on test data
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        auc_test = roc_auc_score(y_test, y_prob)
        print(f">> {name} AUC (test):  {auc_test * 100:.4f}%")
        print(f"[!] {name} AUC difference: {(auc_test - auc_train) * 100:.4f}%\n")

        # Track best model based on test AUC
        if auc_test > best_score:
            best_score = auc_test
            best_model_name = name

    print(f">> Best model: {best_model_name} with AUC = {best_score * 100:.4f}%\n")
    return best_model_name, best_models[best_model_name]

def plot_evaluation_charts(best_models: dict, best_model_name: str, X_test_scaled, y_test, save_dir: str = None) -> None:
    """
    Plots the ROC Curve comparison for all models and the Confusion Matrix for the best model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot ROC curves for each model
    for name, model in best_models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        auc = roc_auc_score(y_test, y_prob)

        print(f">> {name} Classification Report (AUC = {auc * 100:.4f}%) <<")
        print(classification_report(y_test, y_pred, digits=4))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

    # Plot baseline diagonal
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Graph 8 - ROC Curve Comparison')
    axes[0].legend()

    # Display confusion matrix for best model
    best_model = best_models[best_model_name]
    preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds, labels=best_model.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=best_model.classes_
    )
    disp.plot(ax=axes[1], values_format='')
    axes[1].set_title(f'Graph 9 - {best_model_name}: Confusion Matrix')

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_08_09_roc_confusion_matrix.png'), bbox_inches='tight', dpi=300)
        print(f">> Saved: graph_08_09_roc_confusion_matrix.png")
    # plt.show()
    # plt.close()

def plot_learning_curves(best_model, best_model_name: str, X_train_scaled, y_train, save_dir: str = None) -> None:
    """
    Plots learning curves for the best model to analyze over/underfitting.
    """
    # print(f"Generating learning curve for {best_model_name} [...]")
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Plot learning curves to analyze overfitting or underfitting
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=best_model,
        X=X_train_scaled,
        y=y_train,
        cv=cv,
        scoring=make_scorer(roc_auc_score),
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    # Compute mean scores
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # Plot learning curves
    plt.figure(figsize=(16, 6))
    plt.plot(train_sizes, train_mean, label="Training AUC")
    plt.plot(train_sizes, val_mean, label="Validation AUC")
    plt.xlabel("Training Set Size")
    plt.ylabel("AUC Score")
    plt.title(f"Graph 10 - Learning Curve - {best_model_name}")
    plt.legend()
    plt.grid(True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_10_learning_curve.png'), bbox_inches='tight', dpi=300)
        print(f">> Saved: graph_10_learning_curve.png")
    # plt.show()
    # plt.close()
