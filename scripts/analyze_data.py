# -*- coding: utf-8 -*-

"""
analyze_data.py

This script loads the CSV data from `data/raw/unified_results.csv`,
performs a more in-depth analysis, computes descriptive statistics 
(including percentiles), visualizes the data, and saves a summary CSV in
`data/processed/metrics_summary.csv`.

Usage:
    python analyze_data.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
#                            HELPER FUNCTIONS                                 #
###############################################################################

def summarize_grades(df, grade_columns):
    """
    Given a DataFrame and a list of grade columns, compute key descriptive stats
    (count, mean, std, min, 25%, 50%, 75%, max) for each column.
    Returns a dictionary of these statistics for each column.
    """
    stats_dict = {}
    for col in grade_columns:
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            # No valid data
            stats_dict[col] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": None,
                "75%": None,
                "max": None
            }
        else:
            stats_dict[col] = {
                "count": len(valid_data),
                "mean": valid_data.mean(),
                "std": valid_data.std(),
                "min": valid_data.min(),
                "25%": valid_data.quantile(0.25),
                "50%": valid_data.quantile(0.50),  # median
                "75%": valid_data.quantile(0.75),
                "max": valid_data.max()
            }
    return stats_dict


def find_best_worst(df, grade_columns, n=5):
    """
    Find the best and worst 'n' entries for each grade column.
    Returns a nested dictionary with 'best' and 'worst' data subsets.
    """
    best_worst_dict = {}
    for col in grade_columns:
        # Sort ascending for worst
        sorted_df = df.sort_values(by=col, ascending=True)
        worst_n = sorted_df.dropna(subset=[col]).head(n)

        # Sort descending for best
        best_n = df.dropna(subset=[col]).nlargest(n, col)

        best_worst_dict[col] = {
            "best": best_n,
            "worst": worst_n
        }
    return best_worst_dict


def compute_top_bottom_averages(df, grade_columns, n=5):
    """
    For each grade column, compute the average of the top n and bottom n values.
    Returns a dictionary of top/bottom means for each column.
    """
    top_bottom_dict = {}
    for col in grade_columns:
        valid_data = df.dropna(subset=[col]).copy()
        # Sort ascending
        valid_data_sorted = valid_data.sort_values(by=col, ascending=True)
        bottom_n = valid_data_sorted.head(n)[col].mean() if len(valid_data_sorted) >= n else np.nan
        top_n = valid_data_sorted.tail(n)[col].mean() if len(valid_data_sorted) >= n else np.nan

        top_bottom_dict[col] = {
            "bottom_n_avg": bottom_n,
            "top_n_avg": top_n
        }
    return top_bottom_dict


def correlation_analysis(df, grade_columns):
    """
    Compute the correlation matrix for the grade columns
    and return it as a DataFrame.
    """
    return df[grade_columns].corr()


###############################################################################
#                                MAIN SCRIPT                                  #
###############################################################################

def main():
    """
    Main function to orchestrate data analysis.
    """
    # 1. Set paths
    data_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    input_file = os.path.join(data_dir, "unified_results.csv")
    output_summary_file = os.path.join(processed_dir, "metrics_summary.csv")
    output_correlation_file = os.path.join(processed_dir, "correlation_matrix.csv")

    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # 2. Load the data
    df = pd.read_csv(input_file)  # Adjust encoding or separator if needed

    # 3. Select relevant columns
    grade_columns = [
        "grade_without_examples",
        "grade_base",
        "grade_fine_tuned"
    ]

    # Convert these columns to numeric; coerce invalid parsing to NaN
    for col in grade_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Compute descriptive statistics (including percentiles)
    stats_dict = summarize_grades(df, grade_columns)

    # 5. Find the best/worst n entries for each grade column
    best_worst_dict = find_best_worst(df, grade_columns, n=5)

    # 6. Compute average of top 5 and bottom 5 for each grade column
    top_bottom_dict = compute_top_bottom_averages(df, grade_columns, n=5)

    # 7. Correlation analysis across the grade columns
    corr_df = correlation_analysis(df, grade_columns)

    ###############################################################################
    #                            PRINT ANALYSIS TO CONSOLE                        #
    ###############################################################################

    print("===== DESCRIPTIVE STATISTICS =====")
    for col in grade_columns:
        col_stats = stats_dict[col]
        print(f"\nColumn: {col}")
        print(f"  Count: {col_stats['count']}")
        print(f"  Mean:  {col_stats['mean']}")
        print(f"  Std:   {col_stats['std']}")
        print(f"  Min:   {col_stats['min']}")
        print(f"  25%:   {col_stats['25%']}")
        print(f"  50%:   {col_stats['50%']}")  # median
        print(f"  75%:   {col_stats['75%']}")
        print(f"  Max:   {col_stats['max']}")

        print("\n  --> Worst 5 Entries:")
        if len(best_worst_dict[col]["worst"]) > 0:
            print(best_worst_dict[col]["worst"][["question", "human_answer", col]])
        else:
            print("No data")

        print("\n  --> Best 5 Entries:")
        if len(best_worst_dict[col]["best"]) > 0:
            print(best_worst_dict[col]["best"][["question", "human_answer", col]])
        else:
            print("No data")

        # Print top/bottom 5 average
        print(f"\n  --> Average of Bottom 5 for {col}: {top_bottom_dict[col]['bottom_n_avg']}")
        print(f"  --> Average of Top 5 for {col}: {top_bottom_dict[col]['top_n_avg']}")

    # Print correlation matrix
    print("\n===== CORRELATION MATRIX =====")
    print(corr_df)

    ###############################################################################
    #                         SAVE RESULTS TO CSV/TABLES                          #
    ###############################################################################
    # We'll build a DataFrame that holds all our stats, including percentiles

    # Define the order of metrics for the summary table
    metrics_order = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    summary_data = {"metric": metrics_order}
    for col in grade_columns:
        col_stats = stats_dict[col]
        summary_data[col] = [
            col_stats["count"],
            col_stats["mean"],
            col_stats["std"],
            col_stats["min"],
            col_stats["25%"],
            col_stats["50%"],
            col_stats["75%"],
            col_stats["max"]
        ]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_summary_file, index=False)
    print(f"\nSummary statistics (including percentiles) saved to: {output_summary_file}")

    # 2. Save the correlation matrix to CSV
    corr_df.to_csv(output_correlation_file)
    print(f"Correlation matrix saved to: {output_correlation_file}")

    ###############################################################################
    #                          DATA VISUALIZATIONS                                #
    ###############################################################################
    
    # -- Distribution plots (histograms + KDE) for each grade column
    for col in grade_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=10, color='blue')
        plt.title(f"Histogram of {col}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.tight_layout()
        hist_path = os.path.join(processed_dir, f"{col}_hist.png")
        plt.savefig(hist_path, dpi=100)
        plt.close()
        print(f"Histogram for {col} saved to {hist_path}")

    # -- Boxplots for each grade column
    #    (Check if at least one column has valid numeric data to avoid errors)
    valid_data = df[grade_columns].dropna(how='all')  # Drop rows that are all NaN
    if not valid_data.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=valid_data[grade_columns], orient="v")
        plt.title("Box Plot of Grades for Each Model")
        plt.ylabel("Score")
        plt.xticks(range(len(grade_columns)), grade_columns, rotation=10)
        plt.tight_layout()
        boxplot_path = os.path.join(processed_dir, "grades_boxplot.png")
        plt.savefig(boxplot_path, dpi=100)
        plt.close()
        print(f"Boxplot of grades saved to {boxplot_path}")
    else:
        print("No valid numeric data found for boxplots. Skipping boxplot creation.")

    # -- Correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, cmap="Blues", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap of Grade Columns")
    plt.tight_layout()
    heatmap_path = os.path.join(processed_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=100)
    plt.close()
    print(f"Correlation heatmap saved to {heatmap_path}")

    print("\nAnalysis complete. All visualizations and summary files are saved in 'data/processed/'.")

if __name__ == "__main__":
    main()
