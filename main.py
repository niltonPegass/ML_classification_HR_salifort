import os
from src import config
from src.data_loader import load_data, display_dataset_overview
from src.eda import run_eda
from src.model_training import split_and_scale, train_and_optimize_models, save_artifacts
from src.model_evaluation import evaluate_models, plot_evaluation_charts, plot_learning_curves
from src.insights import (
    generate_df_result, plot_feature_importance_and_box,
    plot_risk_segment_distribution, print_department_top_list_details,
    plot_department_analysis
)

def main():
    print("=" * 80)
    print("STARTING HR SALIFORT MOTORS CHURN PREDICTION PIPELINE")
    print("=" * 80)

    # 1. Ingest Data
    df = load_data()

    # 2. Dataset Overview
    print("\n" + "=" * 50)
    print("STEP 1: Dataset Overview")
    print("=" * 50)
    display_dataset_overview(df)

    # 3. Exploratory Data Analysis (EDA)
    print("\n" + "=" * 50)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 50)
    run_eda(df)

    # 4. Data Splitting & Feature Scaling
    print("\n" + "=" * 50)
    print("STEP 3: Splitting and Scaling Features")
    print("=" * 50)
    X_train_scaled, X_test_scaled, y_train, y_test, X_test, scaler = split_and_scale(df)
    print(f"Features scaled successfully. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    # 5. Model Training & Optimization
    print("\n" + "=" * 50)
    print("STEP 4: Training & Hyperparameter Optimization")
    print("=" * 50)
    best_models = train_and_optimize_models(X_train_scaled, y_train)

    # 6. Save Scaler and Models
    print("\n" + "=" * 50)
    print("STEP 5: Saving Trained Model Artifacts")
    print("=" * 50)
    save_artifacts(best_models, scaler)

    # 7. Model Evaluation
    print("\n" + "=" * 50)
    print("STEP 6: Evaluating Models")
    print("=" * 50)
    best_model_name, best_model = evaluate_models(best_models, X_train_scaled, X_test_scaled, y_train, y_test)

    # 8. Evaluation Plots
    print("\n" + "=" * 50)
    print("STEP 7: Generating Evaluation Charts")
    print("=" * 50)
    plot_evaluation_charts(best_models, best_model_name, X_test_scaled, y_test, save_dir=config.OUTPUT_DIR)
    plot_learning_curves(best_model, best_model_name, X_train_scaled, y_train, save_dir=config.OUTPUT_DIR)

    # 9. Knowledge Acquisition & Risk Segmentation
    print("\n" + "=" * 50)
    print("STEP 8: Risk Segmentation & Insights Generation")
    print("=" * 50)
    df_result = generate_df_result(best_model, X_test_scaled, X_test, y_test)

    # 10. Insights Plots
    print("\n" + "=" * 50)
    print("STEP 9: Plotting Feature Importances and Risk Distribution")
    print("=" * 50)
    plot_feature_importance_and_box(best_model, best_model_name, X_test, df_result, y_test, save_dir=config.OUTPUT_DIR)
    plot_risk_segment_distribution(df_result, save_dir=config.OUTPUT_DIR)

    # 11. Department Analysis
    print("\n" + "=" * 50)
    print("STEP 10: Department Specific Averages and Visualizations")
    print("=" * 50)
    print_department_top_list_details(df_result)
    
    # Bottom departments deep dive (Graph 15 - 17)
    plot_department_analysis(
        df_result, 
        departments=['hr', 'technical'], 
        start_graph_num=14, 
        save_dir=config.OUTPUT_DIR
    )
    
    # Top departments deep dive (Graph 19 - 22)
    plot_department_analysis(
        df_result, 
        departments=['management', 'RandD'], 
        start_graph_num=18, 
        save_dir=config.OUTPUT_DIR
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All figures have been saved to: {config.OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
