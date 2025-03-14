import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM

# Load the cleaned data
df = pd.read_csv("experiment_data_cleaned.csv")

# Convert categorical columns to categories
df["Size"] = df["Size"].astype("category")
df["VR_exp"] = df["VR_exp"].astype("category")
df["UserID"] = df["UserID"].astype("category")  # For repeated measures

# Aggregate data to ensure one observation per subject per condition
def aggregate_for_repeated_measures(dependent_var, within_var):
    return df.groupby(["UserID", within_var])[dependent_var].mean().reset_index()

# Repeated-Measures ANOVA (Checking within-subject effects across trials)
def repeated_measures_anova(dependent_var, within_var, subject_var):
    df_agg = aggregate_for_repeated_measures(dependent_var, within_var)
    print(f"Repeated-Measures ANOVA for {dependent_var} with {within_var} as within-subject factor")
    model = AnovaRM(df_agg, depvar=dependent_var, subject=subject_var, within=[within_var]).fit()
    print(model.summary())
    print("\n")
    return model

# Run Repeated-Measures ANOVA for key variables
repeated_measures_anova("Correct_cd", "Ratio", "UserID")
repeated_measures_anova("Correct_cd", "Order", "UserID")
repeated_measures_anova("Odd", "Ratio", "UserID")
repeated_measures_anova("Similar", "Ratio", "UserID")

# Mixed-Effects Model (Parsing out individual differences)
def mixed_effects_model(dependent_var, fixed_var, random_var):
    print(f"Mixed-Effects Model for {dependent_var} with {fixed_var} as fixed effect and {random_var} as random effect")
    model = smf.mixedlm(f"{dependent_var} ~ {fixed_var}", df, groups=df[random_var]).fit()
    print(model.summary())
    print("\n")
    return model

# Run Mixed-Effects Models for key variables
mixed_effects_model("Correct_cd", "Ratio", "UserID")
mixed_effects_model("Correct_cd", "Order", "UserID")
mixed_effects_model("Odd", "Ratio", "UserID")
mixed_effects_model("Similar", "Ratio", "UserID")

# Logistic Regression (Binary outcome: Correct_cd)
def logistic_regression(dependent_var, independent_var):
    print(f"Logistic Regression for {dependent_var} ~ {independent_var}")
    model = smf.logit(f"{dependent_var} ~ {independent_var}", data=df).fit()
    print(model.summary())
    print("\n")
    return model

# Run Logistic Regression for Correct_cd
logistic_regression("Correct_cd", "Ratio")
logistic_regression("Correct_cd", "Order")
logistic_regression("Correct_cd", "VR_exp")

# Multinomial Regression (Categorical outcome: Odd or Similar responses)
def multinomial_regression(dependent_var, independent_var):
    print(f"Multinomial Regression for {dependent_var} ~ {independent_var}")
    model = smf.mnlogit(f"{dependent_var} ~ {independent_var}", data=df).fit()
    print(model.summary())
    print("\n")
    return model

# Run Multinomial Regression for Odd and Similar responses
multinomial_regression("Odd", "Ratio")
multinomial_regression("Odd", "VR_exp")
multinomial_regression("Similar", "Ratio")
multinomial_regression("Similar", "VR_exp")

# Suggesting Survival Analysis (if timing data is available)
def survival_analysis(time_var, event_var):
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[time_var], event_observed=df[event_var])
    kmf.plot_survival_function()
    print(f"Survival Analysis for {time_var} (time-to-response) and {event_var}")
    return kmf

# If we had response timing data, we could run:
# survival_analysis("ResponseTime", "Correct_cd")
