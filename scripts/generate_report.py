# -*- coding: utf-8 -*-

"""
generate_report.py

Generate an HTML report using the summary statistics and correlation matrix
produced by analyze_data.py. Renders data into templates/report_template.html
and saves the result to reports/final_report.html.

Usage:
    python generate_report.py
"""

import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader

def main():
    # 1. Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    templates_dir = os.path.join(base_dir, "templates")
    reports_dir = os.path.join(base_dir, "reports")

    # Input CSV files
    summary_csv = os.path.join(processed_dir, "metrics_summary.csv")
    corr_csv = os.path.join(processed_dir, "correlation_matrix.csv")

    # Output HTML file
    output_file = os.path.join(reports_dir, "final_report.html")

    # 2. Load CSVs into DataFrames
    summary_df = pd.read_csv(summary_csv)
    corr_df = pd.read_csv(corr_csv, index_col=0)

    # 3. Convert summary_df to a dictionary of lists to be easily iterated in Jinja2
    summary_data = summary_df.to_dict(orient="list")

    # 4. Prepare the Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("report_template.html")

    # 5. Determine the grade columns from the summary_data (excluding 'metric')
    grade_columns = [col for col in summary_data.keys() if col != "metric"]

    # 6. Render the template
    html_content = template.render(
        summary_df=summary_data,
        corr_df=corr_df,
        grade_columns=grade_columns,
        processed_dir="../data/processed"  # Correct relative path from final_report.html to images
    )

    # 7. Save the rendered HTML
    os.makedirs(reports_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    main()
