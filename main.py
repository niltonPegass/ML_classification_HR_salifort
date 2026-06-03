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

    print("\n" + "=" * 50)
    print("STEP 1: DATA INGESTION AND DATASET OVERVIEW")
    print("=" * 50)
    df = load_data()
    display_dataset_overview(df)

    print("\n" + "=" * 50)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50)
    run_eda(df)

    print("\n" + "=" * 50)
    print("STEP 3: SPLITTING & SCALING FEATURES")
    print("=" * 50)
    X_train_scaled, X_test_scaled, y_train, y_test, X_test, scaler = split_and_scale(df)
    print(f"Features scaled successfully\n")
    print(f">> Train shape: {X_train_scaled.shape}")
    print(f">> Test shape: {X_test_scaled.shape}")

    print("\n" + "=" * 50)
    print("STEP 4: TRAINING & HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)
    best_models = train_and_optimize_models(X_train_scaled, y_train)

    print("\n" + "=" * 50)
    print("STEP 5: SAVING TRAINED MODEL ARTIFACTS")
    print("=" * 50)
    save_artifacts(best_models, scaler)

    print("\n" + "=" * 50)
    print("STEP 6: EVALUATING MODELS")
    print("=" * 50)
    best_model_name, best_model = evaluate_models(best_models, X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n" + "=" * 50)
    print("STEP 7: GENERATING EVALUATION CHARTS")
    print("=" * 50)
    plot_evaluation_charts(best_models, best_model_name, X_test_scaled, y_test, save_dir=config.OUTPUT_DIR)
    plot_learning_curves(best_model, best_model_name, X_train_scaled, y_train, save_dir=config.OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("STEP 8: RISK SEGMENTATION & INSIGHTS GENERATION")
    print("=" * 50)
    df_result = generate_df_result(best_model, X_test_scaled, X_test, y_test)

    print("\n" + "=" * 50)
    print("STEP 9: PLOTTING FEATURE IMPORTANCES AND RISK DISTRIBUTION")
    print("=" * 50)
    plot_feature_importance_and_box(best_model, best_model_name, X_test, df_result, y_test, save_dir=config.OUTPUT_DIR)
    plot_risk_segment_distribution(df_result, save_dir=config.OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("STEP 10: DEPARTMENT-SPECIFIC AVERAGES AND VISUALIZATIONS")
    print("=" * 50)
    print_department_top_list_details(df_result)
    
    plot_department_analysis(
        df_result, 
        departments=['hr', 'technical'], 
        start_graph_num=14, 
        save_dir=config.OUTPUT_DIR
    )
    
    plot_department_analysis(
        df_result, 
        departments=['management', 'RandD'], 
        start_graph_num=18, 
        save_dir=config.OUTPUT_DIR
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f'>> Model artifacts have been saved:\n{config.MODELS_DIR}')
    print(f">> All figures have been saved:\n{config.OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
