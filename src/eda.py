import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = None) -> None:
    """Plots correlation heatmap for numerical variables."""
    numeric_columns_df = df.select_dtypes(include=['number'])
    
    plt.figure(figsize=(15, 6))
    correlation_heatmap = sns.heatmap(
        numeric_columns_df.corr(),
        vmin=-1, vmax=1,
        annot=True,
        cmap=sns.color_palette("vlag", as_cmap=True)
    )
    correlation_heatmap.set_title('Graph 1 - Correlation Heatmap', fontdict={'fontsize': 10}, pad=12)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_01_correlation_heatmap.png'), bbox_inches='tight', dpi=300)
        print(f">> Saved: graph_01_correlation_heatmap.png")
        # print(f"Successfully saved: {save_dir}")
    # plt.show()
    # plt.close()

def plot_satisfaction_vs_evaluation(df: pd.DataFrame, save_dir: str = None) -> None:
    """Plots satisfaction level distribution and scatter plot vs evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram: satisfaction level by employee status
    sns.histplot(
        ax=axes[0],
        data=df,
        x='satisfaction_level',
        hue='left',
        multiple='stack',
        bins=40,
        palette=config.STATUS_PALETTE
    )
    axes[0].set_title('Graph 2 - Satisfaction Level Distribution by Exit Status')
    axes[0].legend(title='Employee Status', labels=['Left', 'Stayed'])

    # Scatterplot: satisfaction vs evaluation, colored by status
    sns.scatterplot(
        ax=axes[1],
        data=df,
        x='satisfaction_level',
        y='last_evaluation',
        hue='left',
        palette=config.STATUS_PALETTE,
        alpha=0.4
    )
    # Add mean and quantile lines
    axes[1].axvline(
        x=df['satisfaction_level'].mean(),
        color='#ff6361',
        label='Mean Satisfaction',
        linestyle='--'
    )
    axes[1].axvline(
        x=df['satisfaction_level'].quantile(0.25),
        color='#5BE776',
        label='1st Quantile',
        linestyle='--'
    )
    # Fix legend
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=handles, labels=labels, title='Legend')
    axes[1].set_title('Graph 3 - Satisfaction vs Evaluation')
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_02_03_satisfaction_evaluation.png'), bbox_inches='tight', dpi=300)
        print(f">> Saved: graph_02_03_satisfaction_evaluation.png")
        # print(f"Successfully saved: {save_dir}")
    # plt.show()
    # plt.close()

def plot_scatter_additional_vars(df: pd.DataFrame, save_dir: str = None) -> None:
    """Plots satisfaction vs evaluation colored by monthly hours and number of projects."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Satisfaction vs Evaluation by Monthly Hours
    sns.scatterplot(
        ax=axes[0],
        data=df,
        x='satisfaction_level',
        y='last_evaluation',
        hue='average_montly_hours',
        palette='plasma',
        alpha=0.3
    )
    axes[0].set_title("Graph 4 - Satisfaction vs Evaluation by Monthly Hours")

    # Satisfaction vs Evaluation by Number of Projects
    sns.scatterplot(
        ax=axes[1],
        data=df,
        x='satisfaction_level',
        y='last_evaluation',
        hue='number_project',
        palette='viridis',
        alpha=0.3
    )
    axes[1].set_title("Graph 5 - Satisfaction vs Evaluation by Number of Projects")
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_04_05_satisfaction_evaluation_hours_projects.png'), bbox_inches='tight', dpi=300)
        print(f">> Saved: graph_04_05_satisfaction_evaluation_hours_projects.png")
        # print(f"Successfully saved: {save_dir}")
    # plt.show()
    # plt.close()

def plot_scatter_salary_tenure(df: pd.DataFrame, save_dir: str = None) -> None:
    """Plots satisfaction vs evaluation colored by salary and tenure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Satisfaction vs Evaluation by Salary Level
    sns.scatterplot(
        ax=axes[0],
        data=df,
        x='satisfaction_level',
        y='last_evaluation',
        hue='salary',
        palette='crest',
        alpha=0.3
    )
    axes[0].legend(title='Salary', labels=['Low', 'Medium', 'High'])
    axes[0].set_title("Graph 6 - Satisfaction vs Evaluation by Salary")

    # Satisfaction vs Evaluation by Tenure
    sns.scatterplot(
        ax=axes[1],
        data=df,
        x='satisfaction_level',
        y='last_evaluation',
        hue='tenure',
        palette='cubehelix',
        alpha=0.3
    )
    axes[1].set_title("Graph 7 - Satisfaction vs Evaluation by Tenure")
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'graph_06_07_satisfaction_evaluation_salary_tenure.png'), bbox_inches='tight', dpi=300)
        print(f">> Saved: graph_06_07_satisfaction_evaluation_salary_tenure.png")
        # print(f"Successfully saved: {save_dir}")
    # plt.show()
    # plt.close()

def run_eda(df: pd.DataFrame, save_dir: str = None) -> None:
    """Runs all EDA visualizations."""
    if save_dir is None:
        save_dir = config.OUTPUT_DIR
        
    # print("Running Exploratory Data Analysis (EDA)\n")
    plot_correlation_heatmap(df, save_dir)
    plot_satisfaction_vs_evaluation(df, save_dir)
    plot_scatter_additional_vars(df, save_dir)
    plot_scatter_salary_tenure(df, save_dir)
    print("\nEDA Visualizations complete")
