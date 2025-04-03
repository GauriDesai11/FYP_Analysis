import pandas as pd

def flip_ratio(input_file, output_file):
    # Define mapping for flipping the ratio
    ratio_mapping = {
        0.2: 2.0, 0.4: 1.8, 0.6: 1.6, 0.8: 1.4,
        1.2: 1.2, 1.4: 0.8, 1.6: 0.6, 1.8: 0.4, 2.0: 0.2
    }
    
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Replace Ratio values using the mapping
    df['Ratio'] = df['Ratio'].map(ratio_mapping)
    
    # Save the updated CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Updated CSV saved to {output_file}")

# Example usage
input_csv = "NEW_FYP_User_study.csv"  # Replace with actual input file path
output_csv = "output.csv"  # Replace with desired output file path
flip_ratio(input_csv, output_csv)
