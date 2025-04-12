"""
Data Analysis Script: VR Experiment
===================================
Analyzes data from a CSV file with columns:
    UserID, Order, Size, Color, Ratio, Heavier_blue, Heavier_other, Similar, Correct_cd, Odd, VR_exp

Example row:
    User4,3,Small,Green,0.2,FALSE,TRUE,0,1,1,0

All results and figures are saved in the folder "Result".
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import statistics

results_folder = "Results"

class AnalysisData():
    def __init__(self, results_folder="Results"):
        self.results_folder = results_folder
        self.ensure_result_dir_exists()
        
    def ensure_result_dir_exists(self, dir_name="Results"):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def read_data(self, csv_path):
        df = pd.read_csv(csv_path)
        return df

    def compute_proportion(self, df, group_cols, filter_col):
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

    def polynomial_fit_and_predict(self, x, y, deg=2):
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

    def find_polynomial_stationary_point(self, coeffs):
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

    def save_text(self, text, filename):
        self.ensure_result_dir_exists()
        with open(os.path.join(results_folder, filename), "w", encoding="utf-8") as f:
            f.write(text)

    def save_plot(self, fig, filename):
        """
        Save a matplotlib figure to a PNG file in the results directory.
        """
        self.ensure_result_dir_exists()
        fig.savefig(os.path.join(results_folder, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)  # close figure to free up memory

    def plot_correct_cd_vs_order(self, df, filter_col='Correct_cd', poly_degree=2):
        """
        Step 1: Plot overall %Correct_cd=1 vs. Order with a curved (polynomial) line of best fit.
        1) Aggregate data to get proportion by Order
        2) Plot scatter + polynomial curve
        3) Save figure + stationary point
        """
        # Compute proportions
        group_cols = ['Order']
        grouped = self.compute_proportion(df, group_cols, filter_col)

        # Sort by Order to ensure a proper x sequence
        grouped.sort_values(by='Order', inplace=True)
        x = grouped['Order'].values
        y = grouped['Proportion'].values

        # Polynomial fit
        coeffs = self.polynomial_fit_and_predict(x, y, deg=poly_degree)
        # Evaluate curve on a dense grid of x for smoothness
        x_dense = np.linspace(x.min(), x.max(), 200)
        poly = np.poly1d(coeffs)
        y_dense = poly(x_dense)

        # Stationary point
        x_stat = self.find_polynomial_stationary_point(coeffs)
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
        self.save_plot(fig, 'Correct_cd_vs_Order.png')

        # Save text with details
        summary_text = (
            "Plot: %Correct_cd=1 vs. Order\n"
            f"Polynomial degree: {poly_degree}\n"
            f"Coefficients (highest degree first): {coeffs}\n"
            f"{stat_msg}\n"
        )
        self.save_text(summary_text, 'Correct_cd_vs_Order_summary.txt')


    def plot_correct_cd_vs_order_by_size(self, df, filter_col='Correct_cd', poly_degree=2):
        """
        Step 2: Plot %Correct_cd=1 vs. Order for Size='Small' and Size='Large' on the same figure,
        with separate polynomial fits. Also find stationary points for each.
        """
        group_cols = ['Order', 'Size']
        grouped = self.compute_proportion(df, group_cols, filter_col)
        grouped.sort_values(by='Order', inplace=True)

        sizes = grouped['Size'].unique()

        fig = plt.figure()
        summary_lines = []
        for sz in sizes:
            temp = grouped[grouped['Size'] == sz]
            x = temp['Order'].values
            y = temp['Proportion'].values

            # Polynomial fit
            coeffs = self.polynomial_fit_and_predict(x, y, deg=poly_degree)
            poly = np.poly1d(coeffs)
            x_dense = np.linspace(x.min(), x.max(), 200)
            y_dense = poly(x_dense)

            # Stationary point
            x_stat = self.find_polynomial_stationary_point(coeffs)
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
        self.save_plot(fig, 'Correct_cd_vs_Order_by_Size.png')

        # Save text with details
        summary_text = "Plot: %Correct_cd=1 vs. Order (Small vs Large)\n"
        summary_text += "\n".join(summary_lines)
        self.save_text(summary_text, 'Correct_cd_vs_Order_by_Size_summary.txt')

    def anova_correct_cd_by_size(self, df, filter_col='Correct_cd'):
        """
        Step 3: Perform an ANOVA to see if there's a significant difference in %Correct_cd=1
        between Size='Small' and Size='Large' across all Orders. 

        This is a simple one-way ANOVA using statsmodels. 
        """
        model = ols(f"{filter_col} ~ C(Size)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Save the ANOVA results to text file
        self.save_text(str(anova_table), 'ANOVA_Correct_cd_by_Size.txt')

        # Also return it if needed for interactive use
        return anova_table


    def compare_small_large_proportions(self, df, column='Similar'):
        """
        Steps 7 & 8 approach:
        - Calculate % of 'Similar'=1 for Size='Small' and Size='Large'
        - Perform significance test (two-sample t-test or z-test) comparing these proportions.

        You can similarly use this for 'Correct_cd' by changing column='Correct_cd'.
        """
        # Compute proportion for each participant or each group, then do a test.

        # group by (UserID, Size), compute average of 'column' => which is actually 1/0
        grouped = df.groupby(['UserID', 'Size'])[column].mean().reset_index()

        # Split into small vs large.
        small_vals = grouped.loc[grouped['Size'] == 'Small', column].dropna()
        large_vals = grouped.loc[grouped['Size'] == 'Large', column].dropna()

        # t-test (assuming parametric data). 
        t_stat, p_val = stats.ttest_ind(small_vals, large_vals, equal_var=False)

        # Save results
        result_text = (
            f"Comparing {column} (proportion) between Small and Large:\n"
            f"Mean (Small): {small_vals.mean():.3f}, Mean (Large): {large_vals.mean():.3f}\n"
            f"T-statistic: {t_stat:.3f}, p-value: {p_val:.6f}\n"
            "Interpretation: if p < 0.05, difference is significant.\n"
        )
        filename = f"Compare_{column}_Small_vs_Large.txt"
        self.save_text(result_text, filename)

        return (t_stat, p_val)


    def compare_vr_exp_proportions(self, df, column='Correct_cd'):
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
        self.save_text(result_text, filename)

        return (t_stat, p_val)

    def plot_odd_vs_ratio_line_through_points(self, df, filter_col='Odd'):
        """
        Plots %Odd=1 vs. Ratio as a simple line connecting the points (no polynomial fit).
        Saves the figure as 'Odd_vs_Ratio_Line.png'.
        """
        # 1. Compute proportions
        grouped = self.compute_proportion(df, ['Ratio'], filter_col)
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
        self.save_plot(fig, 'Odd_vs_Ratio_Line.png')
        
        # 4. (Optional) Save the summary text
        summary_text = (
            "Plot: %Odd=1 vs. Ratio (simple line through points).\n"
            f"Data points:\n{grouped[['Ratio','Proportion']]}\n"
        )
        self.save_text(summary_text, 'Odd_vs_Ratio_Line_summary.txt')

    def plot_similarity_vs_ratio_with_stationary(self, df, filter_col='Similar', poly_degree=2):
        """
        Plots %Similar=1 vs. Ratio with polynomial best fit and marks the stationary point.
        """
        grouped = self.compute_proportion(df, ['Ratio'], filter_col)
        grouped.sort_values(by='Ratio', inplace=True)

        x = grouped['Ratio'].values
        y = grouped['Proportion'].values

        # Fit polynomial
        coeffs = self.polynomial_fit_and_predict(x, y, poly_degree)
        poly = np.poly1d(coeffs)
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = poly(x_dense)

        # Stationary point
        x_stat = self.find_polynomial_stationary_point(coeffs)
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
        self.save_plot(fig, 'Similar_vs_Ratio_with_stationary.png')

        # Save summary
        summary_text = (
            "Plot: %Similar=1 vs. Ratio (with polynomial fit)\n"
            f"Coefficients: {coeffs}\n"
            f"{stat_text}\n"
        )
        self.save_text(summary_text, 'Similar_vs_Ratio_with_stationary_summary.txt')


    def plot_correct_vs_ratio(self, df, filter_col='Correct_cd', poly_degree=2):
        """
        Step 10: Plot %Correct=1 (or 'Correct_cd'=1 if that's the correct column) vs. Ratio.
        """
        grouped = self.compute_proportion(df, ['Ratio'], filter_col)
        grouped.sort_values(by='Ratio', inplace=True)
        x = grouped['Ratio'].values
        y = grouped['Proportion'].values

        coeffs = self.polynomial_fit_and_predict(x, y, poly_degree)
        poly = np.poly1d(coeffs)
        x_dense = np.linspace(x.min(), x.max(), 200)
        y_dense = poly(x_dense)
        
        # 4. Find stationary point
        x_stat = self.find_polynomial_stationary_point(coeffs)
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
        self.save_plot(fig, 'Correct_vs_Ratio.png')

    def plot_correct_vs_ratio_by_size_with_fit(self, df, filter_col='Correct_cd', poly_degree=2):
        """
        Plots %Correct_cd=1 vs. Ratio for two groups (Size='Small' & Size='Large'),
        each with its own polynomial (curved) line of best fit.
        Also computes and marks stationary points if within the data range.
        """
        # 1. Compute proportions grouped by (Ratio, Size)
        grouped = self.compute_proportion(df, ['Ratio', 'Size'], filter_col)
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
            coeffs = self.polynomial_fit_and_predict(x, y, deg=poly_degree)
            poly = np.poly1d(coeffs)

            # 3. Evaluate polynomial on dense grid
            x_dense = np.linspace(x.min(), x.max(), 200)
            y_dense = poly(x_dense)

            # 4. Find stationary point
            x_stat = self.find_polynomial_stationary_point(coeffs)
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
        self.save_plot(fig, 'Correct_vs_Ratio_by_Size.png')

        # 6. Save summary text
        summary_text = (
            "Plot: %Correct_cd=1 vs. Ratio by Size (curved best fit)\n"
            + "\n".join(summary_lines) + "\n"
        )
        self.save_text(summary_text, 'Correct_vs_Ratio_by_Size_summary.txt')

    def run_all_analysis(self, csv_file_path):
        df = self.read_data(csv_file_path)
        self.plot_correct_cd_vs_order(df, filter_col='Correct_cd', poly_degree=2)
        self.plot_correct_cd_vs_order_by_size(df, filter_col='Correct_cd', poly_degree=2)
        self.anova_correct_cd_by_size(df, filter_col='Correct_cd')
        self.plot_odd_vs_ratio_line_through_points(df, filter_col='Odd')
        self.compare_small_large_proportions(df, column='Similar')
        self.compare_small_large_proportions(df, column='Correct_cd')
        self.plot_similarity_vs_ratio_with_stationary(df, filter_col='Similar', poly_degree=2)
        self.plot_correct_vs_ratio(df, filter_col='Correct_cd', poly_degree=2)
        self.plot_correct_vs_ratio_by_size_with_fit(df, 'Correct_cd')
        self.compare_vr_exp_proportions(df, column='Correct_cd')
    
class AnalysisQuestionnaire():
    
    @staticmethod
    def plotSubjectiveScores(csv_file="subjective_FYP_User_study.csv",
                            output_file="subjective_scores_boxplot.png"):
        """
        Reads the CSV:
        - column = User ID and their scores
        - row = question (Q1..Q10)

        Creates a boxplot with Q1..Q10 on the x-axis, scores on y-axis.
        An orange 'X' marks the mean inside each box.
        """

        # 1. Read the CSV file
        df = pd.read_csv(csv_file)

        # 2. Convert from wide → long format
        #    We'll get columns: ['Question', 'User', 'Score']
        melted = df.melt(id_vars="Question", var_name="User", value_name="Score")

        # 3. Create the boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=melted,
            x="Question",
            y="Score",
            showmeans=True,
            meanprops={
                "marker": "X",
                "markerfacecolor": "orange",
                "markeredgecolor": "black",
                "markersize": 8
            }
        )
        
        plt.title("Subjective Questionnaire Scores")
        plt.xlabel("Question")
        plt.ylabel("Score (Likert scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, output_file), dpi=300)
        plt.show()

    @staticmethod
    def calculateMeanStdDev(csv_file="subjective_FYP_User_study.csv"):
        # Read the CSV
        df = pd.read_csv(csv_file)

        # For each row (question), gather all user columns into a list
        for idx, row in df.iterrows():
            question_label = row['Question']
            # The rest of columns are user scores
            scores = row.drop(labels='Question').values.astype(float)
            
            mean = np.mean(scores)
            std_dev = np.std(scores, ddof=1)
            
            print(f"{question_label}: mean={mean:.2f}, std={std_dev:.2f}")
        return    

def processAges():
    # List of the ages
    numbers = [23, 21, 21, 20, 22, 22, 23, 21, 23, 21, 22, 25]

    # Calculate average (mean)
    average = statistics.mean(numbers)

    # Calculate standard deviation
    std_dev = statistics.stdev(numbers)  # For sample standard deviation

    # Print results
    print(f"Numbers: {numbers}")
    print(f"Average: {average}")
    print(f"Standard Deviation: {std_dev}")

if __name__ == "__main__":
    # Example usage (uncomment and provide the correct CSV path):
    # run_all_analysis("NEW_FYP_User_study.csv")
    
    analysis = AnalysisData()
    analysis.run_all_analysis("output.csv")
        
    AnalysisQuestionnaire.plotSubjectiveScores()
    AnalysisQuestionnaire.calculateMeanStdDev()
    
    processAges()
    
