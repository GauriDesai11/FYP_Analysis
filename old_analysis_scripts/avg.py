import pandas as pd

# Load the cleaned data
df = pd.read_csv("experiment_data_cleaned.csv")

# Convert categorical columns to categories
df["Size"] = df["Size"].astype("category")
df["VR_exp"] = df["VR_exp"].astype("category")
# Ensure 'Similar' is treated as a binary variable (1 for 'similar', 0 otherwise)
df["Similar"] = df["Similar"].apply(lambda x: 1 if x == "similar" else 0)

# Function to calculate the average percentage of Correct_cd
def calculate_average_correct_cd(group_var):
    avg_correct = df.groupby(group_var)["Correct_cd"].mean() * 100
    print(f"Average % of Correct_cd for {group_var}:")
    print(avg_correct)
    print("\n")
    return avg_correct

# Calculate averages for VR_exp and Size
# average_correct_vr = calculate_average_correct_cd("VR_exp")
# average_correct_size = calculate_average_correct_cd("Size")

# avg_correct = df["Correct_cd"].mean() * 100
# print(avg_correct)


def calculate_percentage_similar():
    total_small = df[df["Size"] == "Small"].shape[0]
    total_large = df[df["Size"] == "Large"].shape[0]
    
    similar_small = df[df["Size"] == "Small"]["Similar"].sum()
    similar_large = df[df["Size"] == "Large"]["Similar"].sum()
    
    percentage_small = (similar_small / total_small) * 100 if total_small > 0 else 0
    percentage_large = (similar_large / total_large) * 100 if total_large > 0 else 0
    
    print("Percentage of 'Similar' answers out of all responses for Small and Large cubes:")
    print(f"Small: {percentage_small:.2f}%")
    print(f"Large: {percentage_large:.2f}%")
    print("\n")
    
    return percentage_small, percentage_large

# Run the function
percentage_small, percentage_large = calculate_percentage_similar()
