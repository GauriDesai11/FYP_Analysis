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

# Convert 'Order' and 'Size' to categorical if they are not already
df['Order'] = df['Order'].astype('category')
df['Size'] = df['Size'].astype('category')

# OLS model to test main effects of Size, Order, and their interaction on Correct_cd
# WARNING: This treats Correct_cd as continuous, ignoring that it's actually binary data
#          and also does NOT account for repeated measures.
model_anova = smf.ols("Correct_cd ~ C(Size)*C(Order)", data=df).fit()

# Perform Type II ANOVA
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print(anova_table)