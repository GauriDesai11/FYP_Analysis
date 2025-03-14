import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import curve_fit

# Load the cleaned data
df = pd.read_csv("experiment_data_cleaned.csv")

# Convert categorical columns to categories
df["Size"] = df["Size"].astype("category")
df["Color"] = df["Color"].astype("category")
df["VR_exp"] = df["VR_exp"].astype("category")

# Define a quadratic function for curve fitting
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

# Function to plot % correct vs. a given x-axis variable with a curved best fit line
def plot_curved_best_fit(x_var, y_var, title, filename):
    plt.figure(figsize=(8,6))
    
    for size in ["Small", "Large"]:
        subset = df[df["Size"] == size]
        group = subset.groupby(x_var)[y_var].mean() * 100
        
        plt.scatter(group.index, group.values, label=size)
        
        # Fit a quadratic curve
        popt, _ = curve_fit(quadratic, group.index, group.values)
        x_fit = np.linspace(min(group.index), max(group.index), 100)
        y_fit = quadratic(x_fit, *popt)
        plt.plot(x_fit, y_fit, linestyle='dashed', label=f"Best Fit {size}")
        
        print(f"Quadratic coefficients for {size} ({x_var} vs {y_var}): {popt}")
    
    plt.xlabel(x_var)
    plt.ylabel(f"% {y_var} Responses")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Generate curved best fit plots
plot_curved_best_fit("Ratio", "Correct_cd", "% of CD-correct vs. Ratio (Curved Fit)", "cd_correct_vs_ratio_curved.png")
plot_curved_best_fit("Order", "Correct_cd", "% of CD-correct vs. Order (Curved Fit)", "cd_correct_vs_order_curved.png")
plot_curved_best_fit("Ratio", "Odd", "% Odd Responses vs. Ratio (Curved Fit)", "odd_vs_ratio_curved.png")
plot_curved_best_fit("Ratio", "Similar", "% Same Responses vs. Ratio (Curved Fit)", "same_vs_ratio_curved.png")

# Function to plot VR vs Non-VR comparison with a curved best fit line
def plot_vr_curved_best_fit(y_var, x_var, title, filename):
    plt.figure(figsize=(8,6))
    
    for vr in [0, 1]:
        subset = df[df["VR_exp"] == vr]
        group = subset.groupby(x_var)[y_var].mean() * 100
        
        plt.scatter(group.index, group.values, label=f"VR = {vr}")
        
        # Fit a quadratic curve
        popt, _ = curve_fit(quadratic, group.index, group.values)
        x_fit = np.linspace(min(group.index), max(group.index), 100)
        y_fit = quadratic(x_fit, *popt)
        plt.plot(x_fit, y_fit, linestyle='dashed', label=f"Best Fit VR = {vr}")
        
        print(f"Quadratic coefficients for VR={vr} ({x_var} vs {y_var}): {popt}")
    
    plt.xlabel(x_var)
    plt.ylabel(f"% {y_var} Responses")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Generate VR vs Non-VR curved best fit plots
plot_vr_curved_best_fit("Correct_cd", "Ratio", "% of CD-correct vs. Ratio (VR vs Non-VR, Curved Fit)", "cd_correct_vs_ratio_vr_curved.png")
plot_vr_curved_best_fit("Correct_cd", "Order", "% of CD-correct vs. Order (VR vs Non-VR, Curved Fit)", "cd_correct_vs_order_vr_curved.png")
plot_vr_curved_best_fit("Odd", "Ratio", "% Odd Responses vs. Ratio (VR vs Non-VR, Curved Fit)", "odd_vs_ratio_vr_curved.png")
plot_vr_curved_best_fit("Similar", "Ratio", "% Same Responses vs. Ratio (VR vs Non-VR, Curved Fit)", "same_vs_ratio_vr_curved.png")
