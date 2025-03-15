import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

# Suppose df columns:
#   UserID (identifier), Correct_cd (0/1), Size (Small/Large), Ratio (numeric)
# We'll treat Size as a categorical factor and Ratio as numeric or factor.

df = pd.read_csv("NEW_FYP_User_study.csv")

# GEE approach (approx):
model = smf.gee(
    formula="Correct_cd ~ C(Size) + Ratio",
    groups="UserID",
    data=df,
    family=sm.families.Binomial()
)
result = model.fit()
print(result.summary())


model = smf.gee(
    formula="Correct_cd ~ C(Size)*Ratio",
    groups="UserID",
    data=df,
    family=sm.families.Binomial()
)
result = model.fit()
print(result.summary())