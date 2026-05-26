import pandas as pd
import os
from src import config

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Loads the processed dataset from the specified path.
    Defaults to the path configured in config.DATA_PATH.
    """
    if file_path is None:
        file_path = config.DATA_PATH
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
        
    print(f"Loading dataset from: {file_path}")
    return pd.read_csv(file_path)

def display_dataset_overview(df: pd.DataFrame) -> None:
    """
    Display general information about the dataset,
    including shape, variable descriptions, sample data, and summary statistics.
    """
    print(
        f">> Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns "
        f"({df.shape[0] * df.shape[1]} data points)\n"
    )

    # Description of key dataset variables
    variable_descriptions = {
        'satisfaction_level':     "Employee-reported job satisfaction level",
        'last_evaluation':        "Score of employee's last performance review",
        'number_project':         "Number of projects employee contributes to",
        'average_montly_hours':  "Average number of hours employee worked per month",
        'tenure':                 "Years the employee has been with the company",
        'work_accident':          "Whether the employee experienced a work accident",
        'left':                   "Whether the employee left the company",
        'promotion_last_5years':  "Whether the employee was promoted in the last 5 years",
        'department':             "Employee's department",
        'salary':                 "Employee's salary (in USD)"
    }

    variable_df = (
        pd.DataFrame.from_dict(variable_descriptions, orient='index', columns=['Description'])
        .reset_index()
        .rename(columns={'index': 'Variable'})
    )

    try:
        # Check if running in Jupyter/IPython environment
        get_ipython = globals().get('get_ipython')
        if get_ipython is not None:
            from IPython.display import display
            display(variable_df)
            print(">> General dataset information:")
            df.info()
            print(">> First 5 rows:")
            display(df.head(5))
            print(">> Descriptive statistics:")
            display(df.describe())
            return
    except Exception:
        pass

    # Fallback to plain print in terminal
    print(">> Variable Descriptions:")
    print(variable_df.to_string(index=False))
    print("\n>> General dataset information:")
    df.info()
    print("\n>> First 5 rows:")
    print(df.head(5).to_string())
    print("\n>> Descriptive statistics:")
    print(df.describe().to_string())
