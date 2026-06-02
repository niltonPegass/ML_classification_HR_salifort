import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def segment_risk(probability: float) -> str:
    """
    Segments the risk into categories based on the probability value.
    - 'high' for probability >= 0.90
    - 'medium' for probability between 0.25 and 0.90
    - 'low' for probability < 0.25
    """
    return (
        'high' if probability >= 0.90 else
        'medium' if probability >= 0.25 else
        'low'
    )

def generate_df_result(best_model, X_test_scaled, X_test, y_test) -> pd.DataFrame:
    """
    Predicts churn probabilities on the test set, adds risk segments,
    reconstructs department columns, and prints the summary by risk segment.
    """
    print("Reassembling test dataset with predicted churn probability and risk segments [...]")
    
    # Get the churn probabilities from the best model's prediction
    churn_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

    # Reset index of the original test DataFrame
    X_test_reset = X_test.reset_index(drop=True)

    # Create a copy of the DataFrame and add the churn probability
    df_result = X_test_reset.copy()
    df_result['churn_probability'] = churn_probabilities

    # Apply the risk segmentation function to the churn probabilities
    df_result['risk_segment'] = df_result['churn_probability'].apply(segment_risk)

    # Retrieve the department column from the dummy variables
    department_columns = [col for col in df_result.columns if col.startswith('department_')]
    no_department_mask = df_result[department_columns].sum(axis=1) == 0

    df_result['department'] = df_result[department_columns].idxmax(axis=1)
    df_result['department'] = df_result['department'].str.replace('department_', '', regex=False)
    df_result.loc[no_department_mask, 'department'] = 'IT'

    # Create a summary by risk segment
    segment_summary = df_result.groupby('risk_segment').mean(numeric_only=True)
    segment_summary['employee_count'] = df_result['risk_segment'].value_counts()

    print('>> Dataset reassembled with the new data obtained (churn_probability, risk_segment)')
    
    try:
        # Check if running in Jupyter/IPython environment
        get_ipython = globals().get('get_ipython')
        if get_ipython is not None:
            from IPython.display import display
            display(df_result.head())
            print("\nSegment Summary:")
            display(segment_summary)
            return df_result
    except Exception:
        pass
        
    # print("\nFirst 5 rows of result dataset:")
    # print(df_result.head().to_string())
    # print("\nSegment Summary:")
    # print(segment_summary.to_string())
    
    return df_result

def plot_feature_importance_and_box(best_model, best_model_name: str, X_test, df_result, y_test, save_dir: str = None) -> None:
    """
    Plots feature importances for the best model and a boxplot of the most important
    feature grouped by employee status.
    """
    # print("\nGenerating feature importance and box plots [...]")
    
    # Create a DataFrame for feature importances from the best model
    feature_importances = pd.DataFrame(
        best_model.feature_importances_,
        columns=['gini_importance'],
        index=X_test.columns
    )

    # Sort feature importances and select the top 5
    top_features = feature_importances.sort_values(by='gini_importance', ascending=False).head(5)

    # Select the most important feature
    most_important_feature = top_features.index[0]

    # Create a DataFrame with the selected feature and the target variable 'left'
    df_plot = df_result[[most_important_feature]].copy()
    df_plot['left'] = y_test.reset_index(drop=True)

    # Create a 1x2 subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: Feature importance barplot
    sns.barplot(
        data=top_features,
        x="gini_importance",
        y=top_features.index,
        hue=top_features.index,
        orient='h',
        palette=sns.color_palette("Blues", n_colors=len(top_features))[::-1],
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title(f'Graph 11 - {best_model_name}: Feature Importances')
    axes[0].set_xlabel('Gini Importance')
    axes[0].set_ylabel('')

    # Plot 2: Boxplot of the most important feature by employee status
    sns.boxplot(
        data=df_plot,
        x='left',
        y=most_important_feature,
        hue='left',
        palette='Set2',
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title(f'Graph 12 - Boxplot: {most_important_feature} by Employee Status (0 = Stayed, 1 = Left)')
    axes[1].set_xlabel('Employee Status')
    axes[1].set_ylabel(most_important_feature)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_11_12_feature_importance.png'), bbox_inches='tight', dpi=300)
        print(">> Saved: graph_11_12_feature_importance.png")
    # plt.show()
    # plt.close()

def plot_risk_segment_distribution(df_result: pd.DataFrame, save_dir: str = None) -> None:
    """
    Plots employee distribution by risk segment and high-risk percentage by department.
    """
    # print("Generating risk distribution charts [...]")
    
    # === Heatmap: percentage distribution of risk level by department ===
    heatmap_table = pd.crosstab(
        df_result['department'],
        df_result['risk_segment'],
        normalize='index'
    ) * 100

    # Reorder columns and sort by 'high' risk
    heatmap_table = heatmap_table[['low', 'medium', 'high']]
    heatmap_table = heatmap_table.sort_values(by='high', ascending=False)

    # Create a table focused only on high risk
    high_risk_table = heatmap_table[['high']].copy()

    # === Plotting ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [4, 1]})

    # Barplot: Number of employees by risk segment
    sns.countplot(
        data=df_result,
        x='risk_segment',
        hue='risk_segment',
        order=['low', 'medium', 'high'],
        palette='viridis',
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title('Graph 13 - Employees per Risk Segment')
    axes[0].set_xlabel('Risk Segment')
    axes[0].set_ylabel('Number of Employees')

    # Heatmap: High-risk percentage by department
    sns.heatmap(
        high_risk_table,
        annot=True,
        fmt=".1f",
        cmap='Reds',
        ax=axes[1]
    )
    axes[1].set_title('Graph 14 - High-Risk Percentage by Department')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Department')

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_13_14_risk_by_department.png'), bbox_inches='tight', dpi=300)
        print(">> Saved: graph_13_14_risk_by_department.png")
    # plt.show()
    # plt.close()

def print_department_top_list_details(df_result: pd.DataFrame) -> None:
    """
    Prints aggregated metrics grouped by project count and risk segment for key departments.
    """
    # print("\n>> Key Department Detailed Averages <<")
    department_top_list = ['hr', 'technical', 'management', 'RandD']

    # Iterate through the selected departments
    for dept in department_top_list:
        # print(f'\n>> Department: {dept}')
        
        # Filter data for the selected department and valid working hours
        filtered_df = df_result[
            (df_result['department'] == dept) &
            (df_result['average_montly_hours'] >= 0)
        ].copy()

        # Set 'risk_segment' as an ordered categorical variable
        filtered_df['risk_segment'] = pd.Categorical(
            filtered_df['risk_segment'],
            categories=['high', 'medium', 'low'],
            ordered=True
        )

        # Group data by number of projects and risk segment, and calculate averages
        grouped = filtered_df.groupby(['number_project', 'risk_segment'], observed=True)[
            ['average_montly_hours', 'satisfaction_level', 'last_evaluation', 'tenure', 'churn_probability']
        ].mean().dropna().sort_index(level='risk_segment')

        try:
            get_ipython = globals().get('get_ipython')
            if get_ipython is not None:
                from IPython.display import display
                display(grouped)
                continue
        except Exception:
            pass
            
        # print(grouped.to_string())

def plot_department_analysis(df_result: pd.DataFrame, departments: list, start_graph_num: int, save_dir: str = None) -> None:
    """
    Plots satisfaction level, evaluation, and average working hours for specified departments
    across projects and risk segments.
    """
    n = start_graph_num
    for department in departments:
        # print(f'\n>> Graph Department: {department}')
        
        # Filter the DataFrame for the selected department and valid working hours
        filtered_df = df_result[
            (df_result['department'] == department) & 
            (df_result['average_montly_hours'] >= 0)
        ].copy()

        # Define the order of risk categories for sorting
        filtered_df['risk_segment'] = pd.Categorical(
            filtered_df['risk_segment'],
            categories=['high', 'medium', 'low'],
            ordered=True
        )

        # Group the data by number of projects and risk segment, then calculate the averages
        grouped_df = (
            filtered_df
            .groupby(['number_project', 'risk_segment'], observed=True)[
                ['average_montly_hours', 'satisfaction_level', 'last_evaluation', 'tenure', 'churn_probability']
            ].mean().dropna().sort_index(level='risk_segment').reset_index())

        # Create a combined column for X-axis representation
        grouped_df['project_risk'] = (
            grouped_df['number_project'].astype(str) + ' | ' + grouped_df['risk_segment'].astype(str)
        )

        # Create subplots for visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Melt the dataframe for plotting satisfaction and evaluation metrics
        melted_df = grouped_df.melt(
            id_vars=['project_risk', 'churn_probability'],
            value_vars=['satisfaction_level', 'last_evaluation'],
            var_name='metric',
            value_name='value'
        )

        # Plot 1: Satisfaction Level and Last Evaluation
        sns.barplot(
            ax=axes[0],
            data=melted_df,
            x='project_risk',
            y='value',
            hue='metric',
            palette='Set2'
        )
        n += 1
        axes[0].set_title(f'Graph {n} - Satisfaction and Last Evaluation')
        axes[0].set_xlabel('Number of Projects | Risk Segment')
        axes[0].set_ylabel('Average Value')
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Average Monthly Hours
        sns.barplot(
            ax=axes[1],
            data=grouped_df,
            x='project_risk',
            y='average_montly_hours',
            hue='project_risk',
            palette='Blues_d'
        )
        n += 1
        axes[1].set_title(f'Graph {n} - Average Monthly Hours by Project and Risk')
        axes[1].set_xlabel('Number of Projects | Risk Segment')
        axes[1].set_ylabel('Average Hours')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f'graph_{n-1}_{n}_dept_{department}.png'
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
            print(f">> Saved: {filename}")
        # plt.show()
        # plt.close()
