"""
Data Analysis Script: VR Experiment
===================================
Analyzes data from a CSV file with columns:
    UserID, Order, Size, Color, Ratio, Heavier_blue, Heavier_other, Similar, Correct_cd, Odd, VR_exp

Example row:
    User4,3,Small,Green,0.2,FALSE,TRUE,0,1,1,0

Steps to be performed:
1)  Plot %Correct_cd=1 vs. Order (single line + curved best fit).
2)  Plot %Correct_cd=1 vs. Order with two lines (Size='Small' vs Size='Large'), each with curved best fit.
3)  ANOVA to test significance between the two lines from Step 2.
4)  Determine the stationary point(s) for the curve(s) in Steps 1 & 2.
5)  Plot %Odd=1 vs. Ratio (single line + curved best fit).
6)  Repeat Step 5 but with two lines (Size='Small' vs Size='Large').
7)  Calculate %Similar=1 for Small vs Large; do significance test.
8)  Calculate %Correct_cd=1 for Small vs Large; do significance test.
9)  Plot %Similar=1 vs. Ratio (single line + curved best fit).
10) Plot %Correct_cd=1 (or “Correct” if that’s a different variable) vs. Ratio (single line + curved best fit).
11) Calculate %Correct_cd=1 for VR_exp=1 vs VR_exp=0.
12) Statistical difference test for Step 11 results.

All results and figures are saved in the folder "Result_analysis".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

###############################################################################
# 1. Utility functions
###############################################################################

def ensure_result_dir_exists(dir_name="Result_analysis"):
    """
    Checks if the results directory exists; creates it if it does not.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def read_data(csv_path):
    """
    Reads the CSV file into a pandas DataFrame.
    Returns the DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df

def compute_proportion(df, group_cols, filter_col):
    """
    Given:
      df          - DataFrame
      group_cols  - list of columns to group by (e.g., ['Order'] or ['Order', 'Size'])
      filter_col  - column name for which we want the proportion of 1's (e.g., 'Correct_cd')
    Returns a grouped DataFrame with columns [group_cols] + ['Count', 'Total', 'Proportion'].
    The 'Proportion' is the fraction of rows within each group where filter_col == 1.
    """
    grouped = (
        df.groupby(group_cols)
          .agg(
              Count=(filter_col, lambda x: (x == 1).sum()),
              Total=(filter_col, 'size')
          )
          .reset_index()
    )
    grouped['Proportion'] = grouped['Count'] / grouped['Total']
    return grouped

def polynomial_fit_and_predict(x, y, deg=2):
    """
    Fit a polynomial of specified degree (default=2) to (x, y).
    Returns coefficients of the polynomial fit (numpy.polyfit).
    """
    # Filter out any NaN or infinite values (in case data is incomplete)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    coeffs = np.polyfit(x_clean, y_clean, deg)
    return coeffs

def find_polynomial_stationary_point(coeffs):
    """
    For a polynomial of degree 2: y = a x^2 + b x + c,
    the stationary point occurs where dy/dx = 0 => x = -b/(2a), if a != 0.

    If the polynomial degree or structure changes, adjust accordingly.
    Returns the x-value of stationary point (or None if deg != 2 or a==0).
    """
    # coeffs = [a, b, c] for a 2nd-degree polynomial
    # if higher degree, you can implement a more general derivative approach.
    if len(coeffs) == 3:
        a, b, c = coeffs
        if abs(a) < 1e-14:  # avoid division by zero
            return None
        x_stat = -b / (2*a)
        return x_stat
    else:
        return None

def save_text(text, filename):
    """
    Save text content to a file in the results directory.
    """
    ensure_result_dir_exists()
    with open(os.path.join("Result_analysis", filename), "w", encoding="utf-8") as f:
        f.write(text)

def save_plot(fig, filename):
    """
    Save a matplotlib figure to a PNG file in the results directory.
    """
    ensure_result_dir_exists()
    fig.savefig(os.path.join("Result_analysis", filename), dpi=300, bbox_inches='tight')
    plt.close(fig)  # close figure to free up memory

###############################################################################
# 2. Plotting functions
###############################################################################

def plot_correct_cd_vs_order(df, filter_col='Correct_cd', poly_degree=2):
    """
    Step 1: Plot overall %Correct_cd=1 vs. Order with a curved (polynomial) line of best fit.
    1) Aggregate data to get proportion by Order
    2) Plot scatter + polynomial curve
    3) Save figure + stationary point
    """
    # Compute proportions
    group_cols = ['Order']
    grouped = compute_proportion(df, group_cols, filter_col)

    # Sort by Order to ensure a proper x sequence
    grouped.sort_values(by='Order', inplace=True)
    x = grouped['Order'].values
    y = grouped['Proportion'].values

    # Polynomial fit
    coeffs = polynomial_fit_and_predict(x, y, deg=poly_degree)
    # Evaluate curve on a dense grid of x for smoothness
    x_dense = np.linspace(x.min(), x.max(), 200)
    poly = np.poly1d(coeffs)
    y_dense = poly(x_dense)

    # Stationary point
    x_stat = find_polynomial_stationary_point(coeffs)
    if x_stat is not None and x.min() <= x_stat <= x.max():
        y_stat = poly(x_stat)
        stat_msg = f"Stationary point at x={x_stat:.3f}, y={y_stat:.3f}"
    else:
        stat_msg = "No valid stationary point within data range."

    # Plot
    fig = plt.figure()
    plt.scatter(x, y, label='% Correct_cd=1 (Data)', alpha=0.7)
    plt.plot(x_dense, y_dense, label='Polynomial fit', linewidth=2)
    if x_stat is not None and x.min() <= x_stat <= x.max():
        plt.scatter([x_stat], [y_stat], marker='X', s=100, label='Stationary point')

    plt.title('%Correct_cd=1 vs. Order')
    plt.xlabel('Order')
    plt.ylabel('Proportion Correct_cd=1')
    plt.legend()
    save_plot(fig, 'Correct_cd_vs_Order.png')

    # Save text with details
    summary_text = (
        "Plot: %Correct_cd=1 vs. Order\n"
        f"Polynomial degree: {poly_degree}\n"
        f"Coefficients (highest degree first): {coeffs}\n"
        f"{stat_msg}\n"
    )
    save_text(summary_text, 'Correct_cd_vs_Order_summary.txt')


def plot_correct_cd_vs_order_by_size(df, filter_col='Correct_cd', poly_degree=2):
    """
    Step 2: Plot %Correct_cd=1 vs. Order for Size='Small' and Size='Large' on the same figure,
    with separate polynomial fits. Also find stationary points for each.
    """
    group_cols = ['Order', 'Size']
    grouped = compute_proportion(df, group_cols, filter_col)
    grouped.sort_values(by='Order', inplace=True)

    sizes = grouped['Size'].unique()

    fig = plt.figure()
    summary_lines = []
    for sz in sizes:
        temp = grouped[grouped['Size'] == sz]
        x = temp['Order'].values
        y = temp['Proportion'].values

        # Polynomial fit
        coeffs = polynomial_fit_and_predict(x, y, deg=poly_degree)
        poly = np.poly1d(coeffs)
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = poly(x_dense)

        # Stationary point
        x_stat = find_polynomial_stationary_point(coeffs)
        if x_stat is not None and x.min() <= x_stat <= x.max():
            y_stat = poly(x_stat)
            stat_msg = f"Size={sz}: stationary point at x={x_stat:.3f}, y={y_stat:.3f}"
        else:
            stat_msg = f"Size={sz}: no valid stationary point within data range."

        summary_lines.append(f"Size={sz}\nCoeffs={coeffs}\n{stat_msg}\n")

        # Plot
        plt.scatter(x, y, label=f'{sz} (data)', alpha=0.7)
        plt.plot(x_dense, y_dense, label=f'{sz} fit')
        if x_stat is not None and x.min() <= x_stat <= x.max():
            plt.scatter([x_stat], [y_stat], marker='X', s=100)

    plt.title('%Correct_cd=1 vs. Order by Size')
    plt.xlabel('Order')
    plt.ylabel('Proportion Correct_cd=1')
    plt.legend()
    save_plot(fig, 'Correct_cd_vs_Order_by_Size.png')

    # Save text with details
    summary_text = "Plot: %Correct_cd=1 vs. Order (Small vs Large)\n"
    summary_text += "\n".join(summary_lines)
    save_text(summary_text, 'Correct_cd_vs_Order_by_Size_summary.txt')


###############################################################################
# 3. Significance tests
###############################################################################

def anova_correct_cd_by_size(df, filter_col='Correct_cd'):
    """
    Step 3: Perform an ANOVA to see if there's a significant difference in %Correct_cd=1
    between Size='Small' and Size='Large' across all Orders. 

    Implementation notes:
    - This is a simple one-way ANOVA using statsmodels. 
    - If you have repeated measures (the same User in multiple conditions), 
      you might consider a repeated-measures approach or a mixed-effects model.
    """
    # We can collapse by (User, Size) to get each user's overall average for each size,
    # or we can keep the full data and do a simpler model. There are multiple approaches.
    # Here we’ll do a simpler approach: for each row (trial), we code whether correct_cd=1 or not,
    # then run an ANOVA with “Size” as the factor.
    # More advanced approach: we could incorporate 'Order' or 'UserID' as factors or random effects.

    model = ols(f"{filter_col} ~ C(Size)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Save the ANOVA results to text file
    save_text(str(anova_table), 'ANOVA_Correct_cd_by_Size.txt')

    # Also return it if needed for interactive use
    return anova_table


def compare_small_large_proportions(df, column='Similar'):
    """
    Steps 7 & 8 approach:
    - Calculate % of 'Similar'=1 for Size='Small' and Size='Large'
    - Perform significance test (two-sample t-test or z-test) comparing these proportions.

    You can similarly use this for 'Correct_cd' by changing column='Correct_cd'.
    """
    # Compute proportion for each participant or each group, then do a test.

    # Example approach: group by (UserID, Size), compute average of 'column' => which is actually 1/0
    grouped = df.groupby(['UserID', 'Size'])[column].mean().reset_index()

    # Now we have each user’s mean of `column` for each size (if a user has both sizes).
    # We'll split into small vs large.
    small_vals = grouped.loc[grouped['Size'] == 'Small', column].dropna()
    large_vals = grouped.loc[grouped['Size'] == 'Large', column].dropna()

    # t-test (assuming parametric data). 
    # If data are not normally distributed or sample sizes are small, consider non-parametric (e.g. Mann-Whitney).
    t_stat, p_val = stats.ttest_ind(small_vals, large_vals, equal_var=False)

    # Save results
    result_text = (
        f"Comparing {column} (proportion) between Small and Large:\n"
        f"Mean (Small): {small_vals.mean():.3f}, Mean (Large): {large_vals.mean():.3f}\n"
        f"T-statistic: {t_stat:.3f}, p-value: {p_val:.6f}\n"
        "Interpretation: if p < 0.05, difference is significant.\n"
    )
    filename = f"Compare_{column}_Small_vs_Large.txt"
    save_text(result_text, filename)

    return (t_stat, p_val)


def compare_vr_exp_proportions(df, column='Correct_cd'):
    """
    Steps 11 & 12:
    - Calculate %Correct_cd=1 for VR_exp=1 and VR_exp=0 separately
    - Do significance test between these two values
    """
    # Similar approach as compare_small_large_proportions
    grouped = df.groupby(['UserID', 'VR_exp'])[column].mean().reset_index()

    vr1 = grouped.loc[grouped['VR_exp'] == 1, column].dropna()
    vr0 = grouped.loc[grouped['VR_exp'] == 0, column].dropna()

    t_stat, p_val = stats.ttest_ind(vr1, vr0, equal_var=False)

    result_text = (
        f"Comparing {column} (proportion) between VR_exp=1 vs. VR_exp=0:\n"
        f"Mean (VR=1): {vr1.mean():.3f}, Mean (VR=0): {vr0.mean():.3f}\n"
        f"T-statistic: {t_stat:.3f}, p-value: {p_val:.6f}\n"
    )
    filename = f"Compare_{column}_VR_exp.txt"
    save_text(result_text, filename)

    return (t_stat, p_val)

###############################################################################
# 4. Further plots: Ratio vs. Odd, Similar, Correct_cd
###############################################################################

'''
def plot_odd_vs_ratio(df, filter_col='Odd', poly_degree=2):
    """
    Steps 5 & 6: 
    - Plot %Odd=1 vs. Ratio (with a polynomial curve)
    - Then split by Size for a two-line version.

    We'll do it in one function to illustrate how to handle them both.
    """
    # First do overall
    grouped_overall = compute_proportion(df, ['Ratio'], filter_col)
    grouped_overall.sort_values(by='Ratio', inplace=True)
    x = grouped_overall['Ratio'].values
    y = grouped_overall['Proportion'].values

    coeffs = polynomial_fit_and_predict(x, y, poly_degree)
    poly = np.poly1d(coeffs)
    x_dense = np.linspace(x.min(), x.max(), 200)
    y_dense = poly(x_dense)

    fig = plt.figure()
    plt.scatter(x, y, label='Odd=1 (all data)', alpha=0.7)
    plt.plot(x_dense, y_dense, label='Polynomial fit', linewidth=2)

    plt.title('%Odd=1 vs. Ratio (Overall)')
    plt.xlabel('Ratio')
    plt.ylabel('Proportion Odd=1')
    plt.legend()
    save_plot(fig, 'Odd_vs_Ratio_Overall.png')

    # Then by size
    grouped_size = compute_proportion(df, ['Ratio', 'Size'], filter_col)
    grouped_size.sort_values(by='Ratio', inplace=True)

    fig2 = plt.figure()
    for sz in grouped_size['Size'].unique():
        temp = grouped_size[grouped_size['Size'] == sz]
        x = temp['Ratio'].values
        y = temp['Proportion'].values
        coeffs = polynomial_fit_and_predict(x, y, poly_degree)
        poly = np.poly1d(coeffs)
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = poly(x_dense)

        plt.scatter(x, y, label=f'Odd=1, {sz}', alpha=0.7)
        plt.plot(x_dense, y_dense, label=f'{sz} fit')

    plt.title('%Odd=1 vs. Ratio (by Size)')
    plt.xlabel('Ratio')
    plt.ylabel('Proportion Odd=1')
    plt.legend()
    save_plot(fig2, 'Odd_vs_Ratio_by_Size.png')
'''

def plot_odd_vs_ratio_line_through_points(df, filter_col='Odd'):
    """
    Plots %Odd=1 vs. Ratio as a simple line connecting the points (no polynomial fit).
    Saves the figure as 'Odd_vs_Ratio_Line.png'.
    """
    # 1. Compute proportions
    grouped = compute_proportion(df, ['Ratio'], filter_col)
    # 2. Sort by ratio to ensure correct ordering for line plot
    grouped.sort_values(by='Ratio', inplace=True)

    x = grouped['Ratio'].values
    y = grouped['Proportion'].values

    # 3. Plot a line through the points
    fig = plt.figure()
    plt.plot(x, y, marker='o', label='%Odd=1 line')
    plt.title('%Odd=1 vs. Ratio (Line Plot)')
    plt.xlabel('Ratio')
    plt.ylabel('Proportion Odd=1')
    plt.legend()
    save_plot(fig, 'Odd_vs_Ratio_Line.png')
    
    # 4. (Optional) Save any summary text if needed
    summary_text = (
        "Plot: %Odd=1 vs. Ratio (simple line through points).\n"
        f"Data points:\n{grouped[['Ratio','Proportion']]}\n"
    )
    save_text(summary_text, 'Odd_vs_Ratio_Line_summary.txt')



def plot_similarity_vs_ratio(df, filter_col='Similar', poly_degree=2):
    """
    Step 9: Plot %Similar=1 vs. Ratio (single line + best fit).
    """
    grouped = compute_proportion(df, ['Ratio'], filter_col)
    grouped.sort_values(by='Ratio', inplace=True)
    x = grouped['Ratio'].values
    y = grouped['Proportion'].values

    coeffs = polynomial_fit_and_predict(x, y, poly_degree)
    poly = np.poly1d(coeffs)
    x_dense = np.linspace(x.min(), x.max(), 200)
    y_dense = poly(x_dense)

    fig = plt.figure()
    plt.scatter(x, y, label='Similar=1 (data)', alpha=0.7)
    plt.plot(x_dense, y_dense, label='Polynomial fit', linewidth=2)

    plt.title('%Similar=1 vs. Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('Proportion Similar=1')
    plt.legend()
    save_plot(fig, 'Similar_vs_Ratio.png')

def plot_similarity_vs_ratio_with_stationary(df, filter_col='Similar', poly_degree=2):
    """
    Plots %Similar=1 vs. Ratio with polynomial best fit and marks the stationary point.
    """
    grouped = compute_proportion(df, ['Ratio'], filter_col)
    grouped.sort_values(by='Ratio', inplace=True)

    x = grouped['Ratio'].values
    y = grouped['Proportion'].values

    # Fit polynomial
    coeffs = polynomial_fit_and_predict(x, y, poly_degree)
    poly = np.poly1d(coeffs)
    x_dense = np.linspace(x.min(), x.max(), 200)
    y_dense = poly(x_dense)

    # Stationary point
    x_stat = find_polynomial_stationary_point(coeffs)
    stat_text = ""
    if x_stat is not None and x.min() <= x_stat <= x.max():
        y_stat = poly(x_stat)
        stat_text = f"Stationary point at x={x_stat:.3f}, y={y_stat:.3f}"
    else:
        stat_text = "No valid stationary point within data range."

    # Plot
    fig = plt.figure()
    plt.scatter(x, y, label='%Similar=1 (data)', alpha=0.7)
    plt.plot(x_dense, y_dense, label='Polynomial fit', linewidth=2)
    if x_stat is not None and x.min() <= x_stat <= x.max():
        plt.scatter([x_stat], [y_stat], marker='X', s=100, label='Stationary point')

    plt.title('%Similar=1 vs. Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('Proportion Similar=1')
    plt.legend()
    save_plot(fig, 'Similar_vs_Ratio_with_stationary.png')

    # Save summary
    summary_text = (
        "Plot: %Similar=1 vs. Ratio (with polynomial fit)\n"
        f"Coefficients: {coeffs}\n"
        f"{stat_text}\n"
    )
    save_text(summary_text, 'Similar_vs_Ratio_with_stationary_summary.txt')


def plot_correct_vs_ratio(df, filter_col='Correct_cd', poly_degree=2):
    """
    Step 10: Plot %Correct=1 (or 'Correct_cd'=1 if that's the correct column) vs. Ratio.
    """
    grouped = compute_proportion(df, ['Ratio'], filter_col)
    grouped.sort_values(by='Ratio', inplace=True)
    x = grouped['Ratio'].values
    y = grouped['Proportion'].values

    coeffs = polynomial_fit_and_predict(x, y, poly_degree)
    poly = np.poly1d(coeffs)
    x_dense = np.linspace(x.min(), x.max(), 200)
    y_dense = poly(x_dense)
    
    # 4. Find stationary point
    x_stat = find_polynomial_stationary_point(coeffs)
    stat_msg = ""
    if x_stat is not None and x.min() <= x_stat <= x.max():
        y_stat = poly(x_stat)
        stat_msg = f"Stationary point at x={x_stat:.3f}, y={y_stat:.3f}"
    else:
        stat_msg = f"No valid stationary point within data range"


    fig = plt.figure()
    plt.scatter(x, y, label='Correct=1 (data)', alpha=0.7)
    plt.plot(x_dense, y_dense, label='Polynomial fit', linewidth=2)
    if x_stat is not None and x.min() <= x_stat <= x.max():
            plt.scatter([x_stat], [y_stat], marker='X', s=100)
            
    print(f"Poly coeffs={coeffs}, {stat_msg}")

    plt.title('%Correct_cd=1 vs. Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('Proportion Correct_cd=1')
    plt.legend()
    save_plot(fig, 'Correct_vs_Ratio.png')

def plot_correct_vs_ratio_by_size_with_fit(df, filter_col='Correct_cd', poly_degree=2):
    """
    Plots %Correct_cd=1 vs. Ratio for two groups (Size='Small' & Size='Large'),
    each with its own polynomial (curved) line of best fit.
    Also computes and marks stationary points if within the data range.
    """
    # 1. Compute proportions grouped by (Ratio, Size)
    grouped = compute_proportion(df, ['Ratio', 'Size'], filter_col)
    grouped.sort_values(by='Ratio', inplace=True)

    sizes = grouped['Size'].unique()
    fig = plt.figure()
    
    # Will store results for summary text
    summary_lines = []

    for sz in sizes:
        temp = grouped[grouped['Size'] == sz]
        x = temp['Ratio'].values
        y = temp['Proportion'].values

        # 2. Fit polynomial
        coeffs = polynomial_fit_and_predict(x, y, deg=poly_degree)
        poly = np.poly1d(coeffs)

        # 3. Evaluate polynomial on dense grid
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = poly(x_dense)

        # 4. Find stationary point
        x_stat = find_polynomial_stationary_point(coeffs)
        stat_msg = ""
        if x_stat is not None and x.min() <= x_stat <= x.max():
            y_stat = poly(x_stat)
            stat_msg = f"Stationary point for {sz} at x={x_stat:.3f}, y={y_stat:.3f}"
        else:
            stat_msg = f"No valid stationary point within data range for {sz}"

        # 5. Plot data points + polynomial fit
        plt.scatter(x, y, label=f'{sz} (data)', alpha=0.7)
        plt.plot(x_dense, y_dense, label=f'{sz} fit')
        if x_stat is not None and x.min() <= x_stat <= x.max():
            plt.scatter([x_stat], [y_stat], marker='X', s=100)

        summary_lines.append(f"Size={sz}, Poly coeffs={coeffs}, {stat_msg}")

    plt.title('%Correct_cd=1 vs. Ratio by Size')
    plt.xlabel('Ratio')
    plt.ylabel('Proportion Correct_cd=1')
    plt.legend()
    save_plot(fig, 'Correct_vs_Ratio_by_Size.png')

    # 6. Save summary text
    summary_text = (
        "Plot: %Correct_cd=1 vs. Ratio by Size (curved best fit)\n"
        + "\n".join(summary_lines) + "\n"
    )
    save_text(summary_text, 'Correct_vs_Ratio_by_Size_summary.txt')

def plot_correct_cd_mean_variance_by_ratio_and_size(df, filter_col='Correct_cd'):
    """
    Groups data by (Ratio, Size) and calculates:
      - Mean of Correct_cd (i.e., proportion or average if 1/0)
      - Variance of Correct_cd
    Then plots two lines (Size='Small' vs 'Large'), with error bars or
    some representation of variance.

    Finally, saves the variance data to a text file.
    """
    # 1. Group by (Ratio, Size) to compute mean and variance
    grouped = (
        df.groupby(['Ratio', 'Size'])[filter_col]
          .agg(['mean', 'var'])
          .reset_index()
          .rename(columns={'mean': 'Mean', 'var': 'Variance'})
    )
    # Sort for plotting
    grouped.sort_values(by='Ratio', inplace=True)

    # 2. Create a plot
    fig = plt.figure()

    sizes = grouped['Size'].unique()
    variance_output_lines = []
    for sz in sizes:
        temp = grouped[grouped['Size'] == sz]
        x = temp['Ratio'].values
        y_mean = temp['Mean'].values
        y_var = temp['Variance'].values  # could also consider std dev

        # We'll plot a line for the means; optionally add error bars for std dev
        # If you prefer standard error, do y_err = np.sqrt(y_var)/sqrt(n).
        plt.plot(x, y_mean, marker='o', label=f'{sz} Mean')
        
        # Store variance lines for text file
        # One row per ratio
        for row in temp.itertuples():
            variance_output_lines.append(
                f"Ratio={row.Ratio}, Size={row.Size}, Mean={row.Mean:.3f}, Variance={row.Variance:.3f}"
            )

    plt.title('Mean of Correct_cd=1 vs. Ratio (by Size) with Variance Info')
    plt.xlabel('Ratio')
    plt.ylabel('Mean Correct_cd=1')
    plt.legend()
    save_plot(fig, 'Correct_cd_MeanVariance_by_Ratio_and_Size.png')

    # 3. Save the variance info to text
    variance_text = "Mean/Variance of Correct_cd=1 grouped by Ratio and Size:\n"
    variance_text += "\n".join(variance_output_lines)
    save_text(variance_text, 'Correct_cd_MeanVariance_by_Ratio_and_Size.txt')


def plot_mean_correct_cd_errorbars_by_ratio_size(
    df, filter_col='Correct_cd', error_mode='std'
):
    """
    1) Group data by (Ratio, Size), compute:
         - Mean of Correct_cd
         - Variance (and possibly standard deviation)
       Plot each point at the mean, with error bars for variation.
       Saves figure as 'Correct_cd_MeanVariance_by_Ratio_and_Size.png'.

    Parameters:
      df (DataFrame): your dataset
      filter_col (str): typically 'Correct_cd'
      error_mode (str): 'std' for standard deviation or 'sem' for standard error

    Y-axis: mean Correct_cd (average number of correct=1)
    X-axis: Ratio
    Error bars: ±1 SD or ±1 SE for that group
    """
    # 1. Group by (Ratio, Size) to compute metrics
    grouped = (
        df.groupby(['Ratio', 'Size'])[filter_col]
          .agg(['mean', 'var', 'count'])
          .reset_index()
          .rename(columns={'mean': 'Mean', 'var': 'Variance', 'count': 'N'})
    )

    # Standard Deviation or Standard Error
    grouped['StdDev'] = np.sqrt(grouped['Variance'])
    if error_mode.lower() == 'std':
        grouped['Error'] = grouped['StdDev']
    else:
        # standard error of the mean = std / sqrt(N)
        grouped['Error'] = grouped['StdDev'] / np.sqrt(grouped['N'])

    # 2. Make the plot
    fig = plt.figure()
    for sz in grouped['Size'].unique():
        temp = grouped[grouped['Size'] == sz].sort_values(by='Ratio')
        x = temp['Ratio']
        y = temp['Mean']
        y_err = temp['Error']
        plt.errorbar(
            x, y, yerr=y_err, fmt='-o', capsize=5, label=f"Size={sz}"
        )

    plt.title('Mean of Correct_cd=1 vs. Ratio (by Size) with Error Bars')
    plt.xlabel('Ratio')
    plt.ylabel('Mean Correct_cd=1')
    plt.legend()
    save_plot(fig, 'Correct_cd_MeanVariance_by_Ratio_and_Size.png')

    # 3. Save summary text about means, variances, etc.
    lines = []
    for row in grouped.itertuples():
        lines.append(
            f"Ratio={row.Ratio}, Size={row.Size}, Mean={row.Mean:.3f}, "
            f"Variance={row.Variance:.3f}, N={row.N}, Error={row.Error:.3f}"
        )
    summary_text = "Mean/Variance of Correct_cd=1 by Ratio & Size\n" + "\n".join(lines)
    save_text(summary_text, "Correct_cd_MeanVariance_by_Ratio_and_Size.txt")


def plot_variance_correct_cd_vs_ratio_by_size(
    df, filter_col='Correct_cd', poly_degree=2
):
    """
    2) For each (Ratio, Size), compute the variance of Correct_cd.
       Then plot two lines (Size='Small', 'Large') of variance vs Ratio,
       each with a polynomial best-fit curve.
       Then run an ANOVA comparing the two lines (Size as a factor).
       Saves figure as 'Correct_cd_Variance_by_Ratio_and_Size.png'
       Saves ANOVA table to 'ANOVA_Variance_by_Size.txt'
    """
    # 1. Group by (Ratio, Size) to compute variance
    grouped = (
        df.groupby(['Ratio', 'Size'])[filter_col]
          .agg(['var', 'count'])
          .reset_index()
          .rename(columns={'var': 'Variance', 'count': 'N'})
    )
    # Some groups might have variance=NaN if there's only 1 data point in that group
    # or if all values are identical. Filter them out if needed or handle carefully:
    grouped = grouped.dropna(subset=['Variance'])

    # 2. Plot with polynomial fit for each Size
    fig = plt.figure()
    size_labels = grouped['Size'].unique()
    summary_lines = []

    for sz in size_labels:
        temp = grouped[grouped['Size'] == sz].copy()
        temp.sort_values(by='Ratio', inplace=True)

        x = temp['Ratio'].values
        y = temp['Variance'].values

        # Fit polynomial
        coeffs = polynomial_fit_and_predict(x, y, deg=poly_degree)
        poly = np.poly1d(coeffs)
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = poly(x_dense)

        # Stationary point (if relevant for variance)
        x_stat = find_polynomial_stationary_point(coeffs)
        stat_msg = ""
        if x_stat is not None and x.min() <= x_stat <= x.max():
            y_stat = poly(x_stat)
            stat_msg = f"Stationary point for {sz} at x={x_stat:.3f}, variance={y_stat:.3f}"
        else:
            stat_msg = f"No valid stationary point for {sz} within data range."

        # Plot
        plt.scatter(x, y, alpha=0.7, label=f'{sz} data')
        plt.plot(x_dense, y_dense, label=f'{sz} fit')
        if x_stat is not None and x.min() <= x_stat <= x.max():
            plt.scatter([x_stat], [y_stat], marker='X', s=100)

        summary_lines.append(
            f"Size={sz}, Coeffs={coeffs}, {stat_msg}"
        )

    plt.title('Variance of Correct_cd vs. Ratio (by Size)')
    plt.xlabel('Ratio')
    plt.ylabel('Variance of Correct_cd')
    plt.legend()
    save_plot(fig, "Correct_cd_Variance_by_Ratio_and_Size.png")

    # 3. Run an ANOVA to compare the variance between the two lines
    #    Approaches can vary. A straightforward approach is to treat
    #    "Variance" as the response, "Size" as the factor, ignoring Ratio or
    #    or modeling Ratio as a covariate. *But* if each Ratio is a repeated measure
    #    across multiple users, we need a more advanced approach.
    #    For demonstration, here is a simple approach ignoring repeated measures:

    # Rebuild a DataFrame that has columns: "Variance" and "Size"
    # each row = one ratio group. This is a simplistic approach:
    model_data = grouped.copy()
    # Basic ANOVA
    model = ols("Variance ~ C(Size)", data=model_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # 4. Save ANOVA results
    anova_txt = "ANOVA on variance of Correct_cd by Size:\n" + str(anova_table)
    save_text(anova_txt, "ANOVA_Variance_by_Size.txt")

    # 5. Save summary lines about polynomial fits
    summary_text = (
        "Plot: Variance of Correct_cd vs. Ratio (by Size), polynomial fits.\n"
        + "\n".join(summary_lines) + "\n\n"
        + "ANOVA Results:\n" + str(anova_table)
    )
    save_text(summary_text, "Correct_cd_Variance_by_Ratio_and_Size_Summary.txt")


def plot_mean_correct_cd_errorbars_by_ratio_all(
    df, filter_col='Correct_cd', error_mode='std'
):
    """
    3) Repeat step 1 but combine all data (ignore size).
       Plot mean Correct_cd vs Ratio, with error bars.
       Saves figure as 'Correct_cd_MeanVariance_by_Ratio_ALL.png'.
    """
    grouped = (
        df.groupby('Ratio')[filter_col]
          .agg(['mean','var','count'])
          .reset_index()
          .rename(columns={'mean':'Mean','var':'Variance','count':'N'})
    )
    grouped['StdDev'] = np.sqrt(grouped['Variance'])
    if error_mode.lower() == 'std':
        grouped['Error'] = grouped['StdDev']
    else:
        grouped['Error'] = grouped['StdDev'] / np.sqrt(grouped['N'])

    grouped.sort_values(by='Ratio', inplace=True)

    # Plot
    x = grouped['Ratio']
    y = grouped['Mean']
    y_err = grouped['Error']

    fig = plt.figure()
    plt.errorbar(x, y, yerr=y_err, fmt='-o', capsize=5, label='All Data')
    plt.title('Mean of Correct_cd=1 vs. Ratio (All Data) with Error Bars')
    plt.xlabel('Ratio')
    plt.ylabel('Mean Correct_cd=1')
    plt.legend()
    save_plot(fig, 'Correct_cd_MeanVariance_by_Ratio_ALL.png')

    # Save summary
    lines = []
    for row in grouped.itertuples():
        lines.append(
            f"Ratio={row.Ratio}, Mean={row.Mean:.3f}, "
            f"Variance={row.Variance:.3f}, N={row.N}, Error={row.Error:.3f}"
        )
    summary_text = "Mean/Variance of Correct_cd=1 by Ratio (All)\n" + "\n".join(lines)
    save_text(summary_text, "Correct_cd_MeanVariance_by_Ratio_ALL.txt")


def plot_variance_correct_cd_vs_ratio_all_fitted(
    df, filter_col='Correct_cd', poly_degree=2
):
    """
    3) Repeat step 2 but for the single line combining all data (ignore Size).
       Plot variance of Correct_cd vs. Ratio with polynomial fit.
       There's only one group, so no ANOVA between groups here.

       Saves figure as: 'Correct_cd_Variance_by_Ratio_ALL.png'
    """
    grouped = (
        df.groupby('Ratio')[filter_col]
          .agg(['var','count'])
          .reset_index()
          .rename(columns={'var':'Variance','count':'N'})
    )
    grouped = grouped.dropna(subset=['Variance'])  # drop any row with NaN variance
    grouped.sort_values(by='Ratio', inplace=True)

    x = grouped['Ratio'].values
    y = grouped['Variance'].values

    # Fit polynomial
    coeffs = polynomial_fit_and_predict(x, y, deg=poly_degree)
    poly = np.poly1d(coeffs)
    x_dense = np.linspace(x.min(), x.max(), 200)
    y_dense = poly(x_dense)

    # Stationary point
    x_stat = find_polynomial_stationary_point(coeffs)
    stat_msg = ""
    if x_stat is not None and x.min() <= x_stat <= x.max():
        y_stat = poly(x_stat)
        stat_msg = f"Stationary point at x={x_stat:.3f}, variance={y_stat:.3f}"
    else:
        stat_msg = "No valid stationary point within data range."

    # Plot
    fig = plt.figure()
    plt.scatter(x, y, alpha=0.7, label='Variance (Data)')
    plt.plot(x_dense, y_dense, label='Polynomial fit')
    if x_stat is not None and x.min() <= x_stat <= x.max():
        plt.scatter([x_stat], [y_stat], marker='X', s=100)

    plt.title('Variance of Correct_cd vs. Ratio (All Data)')
    plt.xlabel('Ratio')
    plt.ylabel('Variance of Correct_cd')
    plt.legend()
    save_plot(fig, "Correct_cd_Variance_by_Ratio_ALL.png")

    summary_text = (
        "Variance of Correct_cd vs. Ratio (All Data), with polynomial fit\n"
        f"Coefficients: {coeffs}\n"
        f"{stat_msg}\n"
    )
    # Save details
    # Also include the raw table of variance
    variance_lines = []
    for row in grouped.itertuples():
        variance_lines.append(
            f"Ratio={row.Ratio}, Variance={row.Variance:.3f}, N={row.N}"
        )
    summary_text += "\nData Points (Ratio, Variance, N):\n" + "\n".join(variance_lines)
    save_text(summary_text, "Correct_cd_Variance_by_Ratio_ALL_Summary.txt")


###############################################################################
# 5. Main or demonstration of usage
###############################################################################
def run_all_analysis(csv_file_path):
    """
    Example function that orchestrates all the steps, calling individual functions.
    Adjust as needed. 
    """
    # 1) Read data
    df = read_data(csv_file_path)

    # 2) Plot %Correct_cd vs. Order (single line + best fit)
    plot_correct_cd_vs_order(df, filter_col='Correct_cd', poly_degree=2)

    # 3) Plot %Correct_cd vs. Order (two lines, small vs large) + best fit
    plot_correct_cd_vs_order_by_size(df, filter_col='Correct_cd', poly_degree=2)

    # 4) ANOVA for difference between lines (small vs large)
    anova_correct_cd_by_size(df, filter_col='Correct_cd')

    # Steps 5 & 6) Plot %Odd=1 vs. Ratio (all data), then by Size
    # plot_odd_vs_ratio(df, filter_col='Odd', poly_degree=2)
    plot_odd_vs_ratio_line_through_points(df, filter_col='Odd')

    # Steps 7 & 8) Compare proportions of 'Similar' and 'Correct_cd' for small vs large
    compare_small_large_proportions(df, column='Similar')
    compare_small_large_proportions(df, column='Correct_cd')

    # 9) Plot %Similar=1 vs. Ratio
    # plot_similarity_vs_ratio(df, filter_col='Similar', poly_degree=2)
    plot_similarity_vs_ratio_with_stationary(df, filter_col='Similar', poly_degree=2)

    # 10) Plot %Correct=1 vs. Ratio
    #     If your actual column is 'Correct_cd', just confirm the naming. 
    plot_correct_vs_ratio(df, filter_col='Correct_cd', poly_degree=2)
    plot_correct_vs_ratio_by_size_with_fit(df, 'Correct_cd')


    # 11 & 12) Compare %Correct_cd for VR_exp=1 vs VR_exp=0
    compare_vr_exp_proportions(df, column='Correct_cd')
    
    plot_correct_cd_mean_variance_by_ratio_and_size(df, 'Correct_cd')
    
    # 1) Mean with error bars, by Ratio & Size
    plot_mean_correct_cd_errorbars_by_ratio_size(df, filter_col='Correct_cd', error_mode='std')

    # 2) Variance vs. Ratio by Size with polynomial fits + ANOVA
    plot_variance_correct_cd_vs_ratio_by_size(df, filter_col='Correct_cd', poly_degree=2)

    # 3) Same idea, but combining all data (ignore size) for a single line
    plot_mean_correct_cd_errorbars_by_ratio_all(df, filter_col='Correct_cd', error_mode='std')
    plot_variance_correct_cd_vs_ratio_all_fitted(df, filter_col='Correct_cd', poly_degree=2)



    # All results will be placed in "Result_analysis" folder as .png or .txt files.

###############################################################################
# 6. Additional Explanations
###############################################################################

analysis_explanation = r'''
Analysis and Significance Tests Explanation
===========================================
1) Proportions Computation:
   - We group the data by certain columns (e.g., Order) and calculate the fraction
     of rows where a binary variable (e.g., Correct_cd) equals 1.
   - This gives us a proportion for each group (e.g., for each Order).

2) Polynomial Fitting:
   - We often fit a second-degree polynomial (a "curved line") via np.polyfit(x, y, 2).
   - The resulting polynomial can be evaluated for plotting or for finding a stationary point 
     (where the first derivative is 0).

3) Stationary Points:
   - For a quadratic polynomial a*x^2 + b*x + c, derivative is 2*a*x + b.
   - Setting derivative=0 => x = -b/(2*a). This is the stationary point (could be a max or min).
   - We check whether that x lies within the data range.

4) ANOVA (Analysis of Variance):
   - We use a simple one-way ANOVA to check if there's a significant difference in means among groups.
   - For example, if we want to see if "Size" (Small vs Large) has an effect on Correct_cd, we can
     compare the distribution of Correct_cd=1 vs 0 for each group. The null hypothesis is 
     "both groups have the same mean proportion of Correct_cd=1." 
   - If p < 0.05, we typically conclude there's a significant difference.

5) Two-Sample T-tests:
   - When we specifically compare two groups (e.g., Small vs. Large or VR_exp=0 vs. VR_exp=1),
     a simple approach is a two-sample t-test. This checks if the means are significantly different.
   - If your data might be non-normal or you have outliers, consider non-parametric alternatives 
     like the Mann-Whitney U test.

6) Plotting:
   - We produce scatter plots of each aggregated proportion vs. an independent variable 
     (Order or Ratio), plus a smooth polynomial curve.
   - Plots and numeric summaries are saved to the "Result_analysis" folder.

7) Additional Tests (Optional Suggestions):
   - If repeated measurements exist (the same user tried multiple Orders or multiple Sizes),
     repeated-measures ANOVAs or mixed-effects models may be more appropriate.
   - For proportions, logistic regression can be considered, especially for analyzing the
     effect of multiple factors (e.g., Size, VR_exp, Ratio) simultaneously.

All these steps are provided in separate functions so that you can call them individually
and in any order that you prefer, making the analysis modular and repeatable.
'''

def explanation():
    """
    Prints (or returns) a detailed explanation of the analysis. 
    """
    return analysis_explanation


###############################################################################
# End of Script
###############################################################################

if __name__ == "__main__":
    # Example usage (uncomment and provide the correct CSV path):
    run_all_analysis("NEW_FYP_User_study.csv")
    # run_all_analysis("experiment_data_cleaned.csv")

    # Print the full explanation if you want
    # print(explanation())
