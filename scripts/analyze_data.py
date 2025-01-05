# -*- coding: utf-8 -*-

"""
analyze_data.py

This script loads the CSV data from `data/raw/House of Spencer_unified_results.csv`,
performs a more in-depth analysis, computes descriptive statistics (including percentiles),
visualizes the data, and saves a summary CSV in `data/processed/metrics_summary.csv`.
It also saves best/worst cases in `data/processed/best_worst_cases.csv`.

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
    Find the best (top 'n') and worst (bottom 'n') entries for each grade column.
    Returns a nested dictionary with 'best' and 'worst' DataFrames for each column.
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
    input_file = os.path.join(data_dir, "House of Spencer_unified_results.csv")
    output_summary_file = os.path.join(processed_dir, "metrics_summary.csv")
    output_correlation_file = os.path.join(processed_dir, "correlation_matrix.csv")
    output_best_worst_file = os.path.join(processed_dir, "best_worst_cases.csv")

    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # 2. Load the data
    df = pd.read_csv(input_file)  # Adjust encoding or separator if needed

    # 3. Define the relevant grade columns and map them to the machine response columns
    grade_columns = [
        "grade_without_examples",
        "grade_base",
        "grade_fine-tuned_whithout_examples",
        "grade_fine-tuned_with_examples"
    ]

    # This dictionary maps each grade column to its corresponding model-response column in CSV
    # Make sure the keys match your grade_columns exactly.
    response_map = {
        "grade_without_examples": "House of Spencer_without examples",
        "grade_base": "House of Spencer_base",
        "grade_fine-tuned_whithout_examples": "House of Spencer_fine-tuned_without_examples",
        "grade_fine-tuned_with_examples": "House of Spencer_fine-tuned_with_examples"
    }

    # Convert these columns (the numeric ones) to numeric; coerce invalid parsing to NaN
    for col in grade_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Compute descriptive statistics (including percentiles)
    stats_dict = summarize_grades(df, grade_columns)

    # 5. Find the best/worst n entries for each grade column
    #    We'll only take 3 here as per your request, but you can adjust
    best_worst_dict = find_best_worst(df, grade_columns, n=3)

    # 6. Compute average of top 3 and bottom 3 for each grade column
    top_bottom_dict = compute_top_bottom_averages(df, grade_columns, n=3)

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

        # Print top/bottom 3 average
        print(f"\n  --> Average of Bottom 3 for {col}: {top_bottom_dict[col]['bottom_n_avg']}")
        print(f"  --> Average of Top 3 for {col}: {top_bottom_dict[col]['top_n_avg']}")

    # Print correlation matrix
    print("\n===== CORRELATION MATRIX =====")
    print(corr_df)

    ###############################################################################
    #                         SAVE RESULTS TO CSV/TABLES                          #
    ###############################################################################
    # 1. Build a DataFrame that holds all our stats, including percentiles
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

    # 3. Save best/worst 3 in a dedicated CSV
    #    We'll create a row for each best/worst instance, storing the question,
    #    human_response, model column, grade, AND the machine response for that column.
    bw_rows = []
    for col in grade_columns:
        # For clarity, let's figure out which model response column to use
        model_col_name = response_map[col]

        best_df = best_worst_dict[col]["best"]
        worst_df = best_worst_dict[col]["worst"]

        for _, row in best_df.iterrows():
            bw_rows.append({
                "type": "best",
                "model_column": col,
                "grade": row[col],
                "question": row["question"],
                "human_response": row["human_response"],
                "machine_response": row.get(model_col_name, None)
            })

        for _, row in worst_df.iterrows():
            bw_rows.append({
                "type": "worst",
                "model_column": col,
                "grade": row[col],
                "question": row["question"],
                "human_response": row["human_response"],
                "machine_response": row.get(model_col_name, None)
            })

    bw_df = pd.DataFrame(bw_rows)
    bw_df.to_csv(output_best_worst_file, index=False)
    print(f"Best/worst 3 cases saved to: {output_best_worst_file}")

    ###############################################################################
    #                          DATA VISUALIZATIONS                                #
    ###############################################################################
    
    # -- Distribution plots (histograms + KDE) for each grade column
    #    We'll force the same x-axis range for all. For example, [0,5].
    x_min, x_max = 0, 5  # Adjust if your data can go beyond 5 or below 0

    for col in grade_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=10, color='blue')
        
        # Force the same x-axis on all histograms
        plt.xlim(x_min, x_max)
        
        # Plot the average as a vertical line
        mean_val = df[col].mean()
        if not np.isnan(mean_val):
            plt.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean={mean_val:.2f}')
        
        plt.title(f"Histogram of {col}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        
        hist_path = os.path.join(processed_dir, f"{col}_hist.png")
        plt.savefig(hist_path, dpi=100)
        plt.close()
        print(f"Histogram for {col} saved to {hist_path}")

    # -- Boxplots for each grade column
    valid_data = df[grade_columns].dropna(how='all')  # Drop rows that are all NaN
    if not valid_data.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=valid_data[grade_columns], orient="v")
        plt.title("Box Plot of Grades for Each Model")
        plt.ylabel("Score")
        plt.ylim(x_min, x_max)  # Force same scale on boxplot
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
