import argparse
import json
import os
import sys

import pandas as pd


def validate_taxonomic_names(input_file, json_file, output_file):
    # Load the JSON file with valid names
    with open(json_file, "r") as f:
        valid_names = json.load(f)
    
    # Submissions have used a mix of tsv and csv, so accounting for this here
    file_extension = os.path.splitext(input_file)[1].lower()
    sep = "\t" if file_extension == ".tsv" else ","
    df = pd.read_csv(input_file, sep=sep)
    
    # Rename columns: remove rank information (e.g., (-viria))
    df.columns = [col.split(" (")[0] for col in df.columns]
    
    # Identify relevant columns (excluding '_score' columns and 'sequenceid')
    columns_to_validate = [col for col in df.columns if not col.endswith("_score") and col != "sequenceid"]
    
    # Store validation failures
    failed_rows = []
    
    for index, row in df.iterrows():
        for col in columns_to_validate:
            if pd.notna(row[col]) and col in valid_names:  # Ignore empty cells
                if row[col] not in valid_names[col]:
                    failed_rows.append({"Row": index + 1, "Column": col, "Invalid Value": row[col]})
    
    # Save failing entries into failure report
    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(output_file, index=False)
        print(f"Validation completed - errors saved to {output_file}")
    else:
        print(f"Validation completed - no errors found for {input_file}")

    
def process_folder(input_folder, json_file, output_folder):
    # Does output_folder exist? (overwrites)
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each CSV or TSV file in the input_folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv") or filename.endswith(".tsv"):
            print(filename)
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_error_report.txt")
            validate_taxonomic_names(input_file, json_file, output_file)

def main():
    parser = argparse.ArgumentParser(description="Validate taxonomic names against a reference JSON file.")
    parser.add_argument("--input_folder", help="Path to the folder containing results to validate")
    parser.add_argument("--json_file", help="Path to the json file containing valid names")
    parser.add_argument("--output_folder", help="Path to save the validation failure reports")
    
    args = parser.parse_args()
    process_folder(args.input_folder, args.json_file, args.output_folder)

if __name__ == "__main__":
    main()
