import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Statistical libraries
from scipy.stats import wilcoxon
from statsmodels.stats.anova import AnovaRM  # repeated-measures ANOVA

def plot_subjective_scores(csv_file="subjective_FYP_User_study.csv",
                           output_file="subjective_scores_boxplot.png"):
    """
    Reads the CSV:
        Question,User15,User14,User13,User12,User11,User10,User9,User8,User7,User6,User5,User4
    where each row is a question (Q1..Q10) and each column after 'Question' is a user's score.

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
    # Optional: overlay points to show individual data
    # sns.stripplot(data=melted, x="Question", y="Score", color='black', alpha=0.5)

    plt.title("Subjective Questionnaire Scores")
    plt.xlabel("Question")
    plt.ylabel("Score (Likert scale)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


def run_one_sample_wilcoxon_tests(csv_file="subjective_FYP_User_study.csv", 
                                  compare_value=0.0):
    """
    Runs a one-sample Wilcoxon signed rank test for each question, 
    testing if the distribution of scores is significantly different 
    from 'compare_value' (default 0).

    Prints p-values for each question. Also returns a dictionary with results.
    """

    # Read the CSV
    df = pd.read_csv(csv_file)

    # For each row (question), gather all user columns into a list, then do the test
    results = {}
    for idx, row in df.iterrows():
        question_label = row['Question']
        # The rest of columns are user scores
        scores = row.drop(labels='Question').values.astype(float)
        
        # Wilcoxon checks median vs. 'compare_value'
        # We do 'scores - compare_value' so that if median of scores is > compare_value,
        # we should get a significant p-value (2-sided).
        stat, p_val = wilcoxon(scores - compare_value, alternative='two-sided')
        
        results[question_label] = (stat, p_val)
        print(f"Question {question_label}: Wilcoxon stat={stat:.3f}, p-value={p_val:.4g}")

    return results

def run_overall_wilcoxon_test(csv_file="subjective_FYP_User_study.csv", 
                               compare_value=0.0):
    """
    Combines all scores across all questions and users into a single list,
    then runs a Wilcoxon signed-rank test against `compare_value`.
    """
    df = pd.read_csv(csv_file)
    scores = df.drop(columns=["Question"]).values.flatten().astype(float)

    stat, p_val = wilcoxon(scores - compare_value, alternative='two-sided')
    print(f"Overall Wilcoxon test vs {compare_value}: stat={stat:.3f}, p-value={p_val:.4g}")

    return stat, p_val


if __name__ == "__main__":
    # 1) Plot the boxplot
    plot_subjective_scores()

    # 2) Perform one-sample Wilcoxon vs. 0 (or vs. 4 if you want to see if median>4)
    wilcoxon_results = run_one_sample_wilcoxon_tests(compare_value=0.0)
    
    run_overall_wilcoxon_test(compare_value=0.0)
    
