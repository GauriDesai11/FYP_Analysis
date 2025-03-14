import pandas as pd
import scipy.stats as stats

# Load the cleaned data
df = pd.read_csv("experiment_data_cleaned.csv")

# Convert categorical columns to categories
df["Size"] = df["Size"].astype("category")
df["VR_exp"] = df["VR_exp"].astype("category")

# Function to perform ANOVA
def perform_anova(dependent_var, independent_var, group1, group2, label1, label2):
    data1 = df[df[independent_var] == group1][dependent_var]
    data2 = df[df[independent_var] == group2][dependent_var]
    
    f_value, p_value = stats.f_oneway(data1, data2)
    print(f"ANOVA for {dependent_var} vs {independent_var} ({label1} vs {label2}) - F-value: {f_value:.4f}, p-value: {p_value:.4f}")
    
    return f_value, p_value

# List of comparisons to analyze
comparisons = [
    # VR vs Non-VR
    ("Correct_cd", "VR_exp", 0, 1, "Non-VR", "VR"),
    ("Correct_cd", "Order", 0, 1, "Non-VR", "VR"),
    ("Correct_cd", "Ratio", 0, 1, "Non-VR", "VR"),
    ("Odd", "Ratio", 0, 1, "Non-VR", "VR"),
    ("Similar", "Ratio", 0, 1, "Non-VR", "VR"),
    
    # Large vs Small
    ("Correct_cd", "Size", "Small", "Large", "Small", "Large"),
    ("Correct_cd", "Order", "Small", "Large", "Small", "Large"),
    ("Correct_cd", "Ratio", "Small", "Large", "Small", "Large"),
    ("Odd", "Ratio", "Small", "Large", "Small", "Large"),
    ("Similar", "Ratio", "Small", "Large", "Small", "Large"),
    
    # Additional ANOVA for VR vs Non-VR in each graph type
    ("Correct_cd", "Ratio", 0, 1, "Non-VR", "VR"),
    ("Correct_cd", "Order", 0, 1, "Non-VR", "VR"),
    ("Odd", "Ratio", 0, 1, "Non-VR", "VR"),
    ("Similar", "Ratio", 0, 1, "Non-VR", "VR"),
    
    # Additional ANOVA for Large vs Small in each graph type
    ("Correct_cd", "Ratio", "Small", "Large", "Small", "Large"),
    ("Correct_cd", "Order", "Small", "Large", "Small", "Large"),
    ("Odd", "Ratio", "Small", "Large", "Small", "Large"),
    ("Similar", "Ratio", "Small", "Large", "Small", "Large"),
]

# Run ANOVA for all comparisons
for dep_var, indep_var, g1, g2, l1, l2 in comparisons:
    perform_anova(dep_var, indep_var, g1, g2, l1, l2)

# Explanation of the F-value
f_value_explanation = """
The F-value in ANOVA represents the ratio of the variance between the groups to the variance within the groups. 
A higher F-value indicates that the group means are more different from each other compared to the variation within each group. 
A low F-value suggests that the differences between group means are relatively small compared to the within-group variation.

- If the F-value is large and the p-value is small (typically < 0.05), we conclude that there is a statistically significant difference between the groups.
- If the F-value is small and the p-value is large (>= 0.05), we conclude that the differences between groups are likely due to random variation.
"""
print(f_value_explanation)
