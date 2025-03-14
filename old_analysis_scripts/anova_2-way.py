import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the cleaned data
df = pd.read_csv("experiment_data_cleaned.csv")

# Convert categorical columns to categories
df["Size"] = df["Size"].astype("category")
df["VR_exp"] = df["VR_exp"].astype("category")

# Function to perform two-way ANOVA
def perform_two_way_anova(dependent_var, x_var, group_var, label_x, label_group):
    formula = f"{dependent_var} ~ {x_var} * {group_var}"
    model = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"ANOVA for {dependent_var} vs {x_var} with grouping by {group_var} ({label_group})")
    print(anova_table)
    print("\n")
    return anova_table

# List of two-way ANOVA tests
anova_tests = [
    ("Correct_cd", "Ratio", "VR_exp", "Ratio", "VR vs Non-VR"),
    ("Correct_cd", "Ratio", "Size", "Ratio", "Large vs Small"),
    ("Correct_cd", "Order", "VR_exp", "Order", "VR vs Non-VR"),
    ("Correct_cd", "Order", "Size", "Order", "Large vs Small"),
    ("Odd", "Ratio", "VR_exp", "Ratio", "VR vs Non-VR"),
    ("Odd", "Ratio", "Size", "Ratio", "Large vs Small"),
    ("Similar", "Ratio", "VR_exp", "Ratio", "VR vs Non-VR"),
    ("Similar", "Ratio", "Size", "Ratio", "Large vs Small"),
]

# Run ANOVA for all comparisons
for dep_var, x_var, group_var, label_x, label_group in anova_tests:
    perform_two_way_anova(dep_var, x_var, group_var, label_x, label_group)

# Explanation of Two-Way ANOVA
anova_explanation = """
Two-way ANOVA tests whether two independent variables (X and a grouping factor) influence the dependent variable (Y), and whether their interaction is significant.

- **Main Effect of X:** Checks if X (e.g., Ratio, Order) significantly affects Y (e.g., Correct_cd, Odd, Similar).
- **Main Effect of Grouping Factor:** Checks if the grouping factor (VR_exp or Size) significantly affects Y.
- **Interaction Effect (X * Grouping Factor):** Checks if the effect of X on Y differs between the two groups (VR vs Non-VR, Large vs Small).

If the p-value for the interaction is significant (p < 0.05), it means that the effect of X on Y depends on the grouping factor.
"""
print(anova_explanation)
