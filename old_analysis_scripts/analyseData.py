import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# Load the cleaned data
df = pd.read_csv("experiment_data_cleaned.csv")

# Convert categorical columns to categories
df["Size"] = df["Size"].astype("category")
df["Color"] = df["Color"].astype("category")
df["VR_exp"] = df["VR_exp"].astype("category")

# Function to plot % correct vs. a given x-axis variable
def plot_percentage_correct(x_var, title, filename):
    plt.figure(figsize=(8,6))
    
    for size in ["Small", "Large"]:
        subset = df[df["Size"] == size]
        group = subset.groupby(x_var)["Correct_cd"].mean() * 100
        plt.plot(group.index, group.values, marker='o', label=size)
        
        # Fit a linear regression line
        z = np.polyfit(group.index, group.values, 1)
        p = np.poly1d(z)
        plt.plot(group.index, p(group.index), linestyle='dashed', label=f"Best fit {size}")
        print(f"Gradient of best fit line for {size} ({x_var} vs Correct_cd): {z[0]:.4f}")
    
    plt.xlabel(x_var)
    plt.ylabel("% Correct CD")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Generate plots
plot_percentage_correct("Ratio", "% of CD-correct vs. Ratio", "cd_correct_vs_ratio.png")
plot_percentage_correct("Order", "% of CD-correct vs. Order", "cd_correct_vs_order.png")

# Function to plot % responses for a given answer type
def plot_response_percentage(y_var, title, filename):
    plt.figure(figsize=(8,6))
    
    for size in ["Small", "Large"]:
        subset = df[df["Size"] == size]
        group = subset.groupby("Ratio")[y_var].mean() * 100
        plt.plot(group.index, group.values, marker='o', label=size)
        
        # Fit a linear regression line
        z = np.polyfit(group.index, group.values, 1)
        p = np.poly1d(z)
        plt.plot(group.index, p(group.index), linestyle='dashed', label=f"Best fit {size}")
        print(f"Gradient of best fit line for {size} ({y_var} vs Ratio): {z[0]:.4f}")
    
    plt.xlabel("Ratio")
    plt.ylabel(f"% {y_var} Responses")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Generate additional plots
plot_response_percentage("Odd", "% Odd Responses vs. Ratio", "odd_vs_ratio.png")
plot_response_percentage("Similar", "% Same Responses vs. Ratio", "same_vs_ratio.png")

# Function to plot VR vs Non-VR
def plot_vr_comparison(y_var, x_var, title, filename):
    plt.figure(figsize=(8,6))
    
    for vr in [0, 1]:
        subset = df[df["VR_exp"] == vr]
        group = subset.groupby(x_var)[y_var].mean() * 100
        plt.plot(group.index, group.values, marker='o', label=f"VR = {vr}")
        
        # Fit a linear regression line
        z = np.polyfit(group.index, group.values, 1)
        p = np.poly1d(z)
        plt.plot(group.index, p(group.index), linestyle='dashed', label=f"Best fit VR = {vr}")
        print(f"Gradient of best fit line for VR={vr} ({x_var} vs {y_var}): {z[0]:.4f}")
    
    plt.xlabel(x_var)
    plt.ylabel(f"% {y_var} Responses")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Generate VR vs Non-VR plots
plot_vr_comparison("Correct_cd", "Ratio", "% of CD-correct vs. Ratio (VR vs Non-VR)", "cd_correct_vs_ratio_vr.png")
plot_vr_comparison("Correct_cd", "Order", "% of CD-correct vs. Order (VR vs Non-VR)", "cd_correct_vs_order_vr.png")
plot_vr_comparison("Odd", "Ratio", "% Odd Responses vs. Ratio (VR vs Non-VR)", "odd_vs_ratio_vr.png")
plot_vr_comparison("Similar", "Ratio", "% Same Responses vs. Ratio (VR vs Non-VR)", "same_vs_ratio_vr.png")

# ANOVA Analysis
def perform_anova(dependent_var):
    model = stats.f_oneway(
        df[df["Size"] == "Small"][dependent_var],
        df[df["Size"] == "Large"][dependent_var]
    )
    print(f"ANOVA for {dependent_var} - F-value: {model.statistic:.4f}, p-value: {model.pvalue:.4f}")

# Perform ANOVA on key dependent variables
perform_anova("Correct_cd")
perform_anova("Odd")
perform_anova("Similar")
