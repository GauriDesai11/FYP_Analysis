import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the cleaned data
df = pd.read_csv("experiment_data_cleaned.csv")

# Convert categorical columns to categories
df["Size"] = df["Size"].astype("category")

# Function to plot number of correct answers vs. ratio with variance
def plot_correct_ratio_variance():
    plt.figure(figsize=(8,6))
    
    for size in ["Small", "Large"]:
        subset = df[df["Size"] == size]
        grouped = subset.groupby("Ratio")["Correct_cd"]
        mean_correct = grouped.sum()
        variance = grouped.var()
        
        plt.errorbar(mean_correct.index, mean_correct.values, yerr=np.sqrt(variance.values), fmt='-o', label=size, capsize=5)
    
    plt.xlabel("Ratio")
    plt.ylabel("Number of Correct Answers")
    plt.title("Number of Correct Answers vs. Ratio (with Variance)")
    plt.legend()
    plt.grid()
    plt.savefig("correct_ratio_variance.png")
    plt.show()

# Function to plot combined correct answers for Small and Large with variance
def plot_combined_correct_ratio_variance():
    plt.figure(figsize=(8,6))
    
    grouped = df.groupby("Ratio")["Correct_cd"]
    mean_correct = grouped.sum()
    variance = grouped.var()
    
    plt.errorbar(mean_correct.index, mean_correct.values, yerr=np.sqrt(variance.values), fmt='-o', label="All Data", capsize=5)
    
    plt.xlabel("Ratio")
    plt.ylabel("Number of Correct Answers")
    plt.title("Number of Correct Answers vs. Ratio (with Variance)")
    plt.legend()
    plt.grid()
    plt.savefig("combined_correct_ratio_variance.png")
    plt.show()

# Run the functions
# plot_correct_ratio_variance()
plot_combined_correct_ratio_variance()
