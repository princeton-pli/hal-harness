"""
Analyze data from the results folder.
"""

# Standard library imports
import os
import sys
import glob
import json
import logging
import argparse
import pandas as pd
import seaborn as sns
from enum import Enum
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

# Setup logging
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# global constants
RESULTS_DIR: str = "results"
METRICS_DIR: str = os.path.join(RESULTS_DIR, "metrics")

# Set default plotting styles
#plt.style.use('seaborn')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# result type
# Make sure this is a copy of ResultType in hal/benchmarks/USACO/USACOBench/evaluation/result_type.py
# not importing ResultType directly from that file as it pulls in other dependencies
class ResultType(int, Enum):
    ACCEPTED              = 1
    WRONG_ANSWER          = 2
    TIME_LIMIT_EXCEEDED   = 3
    MEMORY_LIMIT_EXCEEDED = 4
    COMPILATION_ERROR     = 5
    RUNTIME_ERROR         = 6
    UNKNOWN               = 7

def value_counts_with_percentage(series):
    counts = series.value_counts()
    percentages = series.value_counts(normalize=True) * 100
    return pd.DataFrame({
        'count': counts,
        'percentage': percentages.round(2)
    })

def calculate_category_stats(df, grouping_cols, main_groupiong_col):
    # Get total counts and percentages
    total_stats = (df.groupby(grouping_cols)
                    .size()
                    .reset_index(name='count'))
    
    # Calculate overall percentage
    total_records = len(df)
    total_stats['percentage'] = (total_stats['count'] / total_records * 100).round(2)
    
    # Calculate percentage within each category
    total_stats['percentage_within_category'] = (total_stats.groupby(main_groupiong_col)['count']
                                                          .transform(lambda x: (x / x.sum() * 100).round(2)))
    
    return total_stats.sort_values([main_groupiong_col, 'count'], ascending=[True, False])


def get_most_recent_file_with_timestamp(file_pattern: str) -> tuple[str, datetime]:
    """
    Find the most recent file and its timestamp for files matching the pattern.
    
    Args:
        file_pattern (str): Glob pattern to match files
        
    Returns:
        tuple: (file_path, timestamp) for the most recent file
        
    Raises:
        FileNotFoundError: If no files match the pattern
    """
    # Get all matching files with their timestamps
    files_with_timestamps = [
        (f, datetime.fromtimestamp(os.path.getmtime(f)))
        for f in glob.glob(file_pattern)
    ]

    if not files_with_timestamps:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    # Get the most recent file
    most_recent_file, timestamp = max(files_with_timestamps, key=lambda x: x[1])

    return most_recent_file, timestamp

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Data analysis script with configurable data directory')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing the data files'
    )

    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        help='Model-id for the model that generated the results being analyzed'
    )
    
    return parser.parse_args()

def load_json_files(data_dir: str) -> list:
    """
    Load all JSON files matching the pattern '*rdict*.json' from the specified directory.
    Although we say "All" there is only ever a single rdict.json file in the directory, the
    reason we are using a pattern is that the file path contains a timestamp which make it 
    different for each run so you have to really know the filename. In case we do find multiple files
    we take the most recent one for analysis, thus there is always a single file that we are reading.
    
    Args:
        data_dir (str): Directory containing JSON files
        
    Returns:
        list: List of dictionaries containing the data from JSON files
    """
    try:
        # Create the file pattern
        file_pattern = os.path.join(data_dir, '*rdict*.json')
        
        # Find all matching files
        file_path, _ = get_most_recent_file_with_timestamp(file_pattern)
        
        if not file_path:
            logger.warning(f"No files matching pattern '*rdict*.json' found in {data_dir}")
            return {}
        
        # Load data from the most recent matching file
        logger.info(f"Loading data from {file_path}")            
        with open(file_path, 'r') as f:
            data = json.load(f)                
        
        logger.info(f"Successfully loaded {file_path} file")
        return data
    
    except Exception as e:
        logger.error(f"Error loading JSON files: {str(e)}")
        raise

def convert_to_dataframe(data_list: list) -> pd.DataFrame:
    """
    Convert list of dictionaries to a pandas DataFrame
    
    Args:
        data_list (list): List of dictionaries containing the data
        
    Returns:
        pd.DataFrame: Converted DataFrame
    """
    try:
        result_list = []
        for key, item in data_list.items():
            #print(item)
            #print(len(item))
            problem_category = key.split("_")[1]
            results = item[0]['result_list']
            if results is None:
                print(f"problem_id={key}, results={results}")
                continue
            #print(results)
            for i, result in enumerate(results):
                result_type = result['result_type']
                status = result['status'].split("\n")[0]
                info = dict(problem_id=key, problem_category=problem_category, task=i+1, result_type=result_type, status=status)
                result_list.append(info)
            #print(result_list)
            #print(len(result_list))
        logger.info(f"extracted relevant information into a list of length {len(result_list)}, first few elements are {result_list[:5]}")
        df = pd.DataFrame(result_list)
        logger.info(f"Successfully converted data to DataFrame with shape {df.shape}")

        # convert result_type to string
        df.result_type = df.result_type.map(lambda x: ResultType(x).name)
        logger.info(df.head())
        return df
    except Exception as e:
        logger.error(f"Error converting data to DataFrame: {str(e)}")
        raise

def perform_eda(df: pd.DataFrame, args) -> None:
    """
    Perform initial exploratory data analysis
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    # Basic information about the dataset
    print("\nDataset Info:")
    print("-" * 50)
    print(df.info())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(df.describe())
    
    # Missing values
    print("\nMissing Values:")
    print("-" * 50)
    print(df.isnull().sum())

    # list out only the tests that passed
    df_passed = df.groupby(["problem_id", "problem_category"])["result_type"].apply(lambda x: (x == 'ACCEPTED').all()).reset_index()
    df_passed = df_passed[df_passed.result_type == True]
    logger.info(f"out of a total of {len(df.problem_id.unique())} problems we had {len(df_passed)} problems for which the generated code passed all unit tests")
    logger.info(df_passed)
    fpath: str = args.passed_tests_file_path
    logger.info(f"going to save passed problems list to {fpath}")
    df_passed.to_csv(fpath, index=False)

    # passed by category
    df_problems_by_category = value_counts_with_percentage(df[['problem_id', 'problem_category']].drop_duplicates()['problem_category'])
    logger.info(f"problem counts by category")
    logger.info(df_problems_by_category)
    fpath: str = args.problem_counts_by_category_file_path
    logger.info(f"going to save problems by category to {fpath}")
    df_problems_by_category.to_csv(fpath, index=True)

    # passed by category
    df_passed_by_category = value_counts_with_percentage(df_passed['problem_category'])
    logger.info(f"passed problem counts by category")
    logger.info(df_passed_by_category)
    fpath: str = args.passed_tests_by_category_file_path
    logger.info(f"going to save passed problems by category to {fpath}")
    df_passed_by_category.to_csv(fpath, index=True)


    # result_type counts to get a sense of how many tasks overall (problems contain tasks)
    # generated results which were accepted, timedout etc
    df_result_type_metrics = value_counts_with_percentage(df['result_type'])
    logger.info(f"result_type counts")
    logger.info(df_result_type_metrics)
    fpath: str = args.result_type_counts_file_path
    logger.info(f"going to save result_type counts to {fpath}")
    df_result_type_metrics.to_csv(fpath, index=True)

    # result type counts broken down by problem category
    # so that we can understand which types of problem i.e. bronze, silver, gold, platinum
    # had which types of failures (runtime errors, timeouts etc) and in what proportion, is there
    # a pattern to it (foe example: do platinum category problems have the most timeouts?)
    logger.info(df.columns)
    df_result_type_by_problem_category_metrics = calculate_category_stats(df, ['problem_category', 'result_type'], 'problem_category') # value_counts_with_percentage(df[])
    #df_result_type_by_problem_category_metrics = df_result_type_by_problem_category_metrics.reset_index().sort_values(by="problem_category")
    logger.info(f"result_type by problem_category counts")
    logger.info(df_result_type_by_problem_category_metrics)
    fpath: str = args.result_type_by_probpem_category_counts_file_path
    logger.info(f"going to save result_type by problem_category counts to {fpath}")
    df_result_type_by_problem_category_metrics.to_csv(fpath, index=False)

    # plot the result type by problem category counts in a chart for better understanding
    # Order of problem_category on x-axis
    category_order = ["bronze", "silver", "gold", "platinum"]

    # Plotting grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_result_type_by_problem_category_metrics,
        x="problem_category",
        y="count",
        hue="result_type",
        order=category_order,
        palette="Set2"
    )

    # Customizing the chart
    plt.title("Breakdown of result type counts by problem category", fontsize=14)
    plt.suptitle(f"Model: {args.model_id}", fontsize=12, color="gray")
    plt.xlabel("Problem Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Result Type", title_fontsize=10, fontsize=9)
    plt.tight_layout()

    # save
    fpath: str = args.result_type_by_problem_category_plot_file_path
    plt.savefig(fpath)
    plt.clf()

    

    # plot result_type by problem_category but this time for percentage within category
    # this helps answer the question what percentage of tasks within each category timeout or are accepted etc.
    # Pivot to wide format
    wide_df = df_result_type_by_problem_category_metrics.pivot(
        index="problem_category", 
        columns="result_type", 
        values="percentage_within_category"
    ).reset_index()

    # Define the desired category order
    category_order = ["bronze", "silver", "gold", "platinum"]

    # Set the category order for problem_category
    wide_df["problem_category"] = wide_df["problem_category"].astype(
        CategoricalDtype(categories=category_order, ordered=True)
    )

    # Sort the DataFrame by problem_category to ensure the order in the plot
    wide_df = wide_df.sort_values("problem_category")

    # Increase the canvas size
    plt.figure(figsize=(12, 8))  # Width = 12, Height = 8

    # Create the horizontal stacked bar chart
    ax = wide_df.plot(
        kind="barh", 
        stacked=True, 
        x="problem_category", 
        figsize=(12, 8)  # Optional: Include size here as well
    )

    # Move the legend outside the plot
    plt.legend(
        loc="upper left", 
        bbox_to_anchor=(1.05, 1),  # Place legend outside plot area
        title="Result Type"
    )

    # Labels for x & y axis
    plt.xlabel("Percentage")
    plt.ylabel("Problem Category")

    # Title of plot
    plt.title("Breakdown of result_type for tasks in each problem_category")
    plt.suptitle(f"Model: {args.model_id}", fontsize=12, color="gray")


    # Adjust layout
    plt.tight_layout()

    # Save the plot
    fpath = args.result_type_by_problem_category_within_category_pct_plot_file_path
    plt.savefig(fpath)
    plt.clf()

def main():
    """Main function to run the analysis"""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # create model specific metrics dir
        metrics_dir = os.path.join(METRICS_DIR, args.model_id.replace(":", "-"))
        os.makedirs(metrics_dir, exist_ok=True)
        args.problem_counts_by_category_file_path = os.path.join(metrics_dir, "problem_counts_by_category.csv")
        args.passed_tests_file_path = os.path.join(metrics_dir, "passed.csv")
        args.passed_tests_by_category_file_path = os.path.join(metrics_dir, "passed_by_category.csv")
        args.result_type_counts_file_path = os.path.join(metrics_dir, "result_type_counts.csv")
        args.result_type_by_probpem_category_counts_file_path = os.path.join(metrics_dir, "result_type_by_problem_category_counts.csv")
        args.result_type_by_problem_category_plot_file_path = os.path.join(metrics_dir, "result_type_by_problem_category_plot.png")
        args.result_type_by_problem_category_within_category_pct_plot_file_path = os.path.join(metrics_dir, "result_type_by_problem_category_within_category_pct_plot.png")
        
        # Load data from JSON files
        data_list = load_json_files(args.data_dir)
        
        if not data_list:
            logger.error("No data loaded. Exiting.")
            sys.exit(1)
        
        # Convert to DataFrame
        df = convert_to_dataframe(data_list)
        
        # Perform EDA
        perform_eda(df, args)

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()