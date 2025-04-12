import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        plt.savefig(output_file, dpi=300)
        plt.show()

    @staticmethod
    def calculateMeanStdDev(csv_file="subjective_FYP_User_study.csv"):
        # Read the CSV
        df = pd.read_csv(csv_file)

        # For each row (question), gather all user columns into a list
        results = {}
        for idx, row in df.iterrows():
            question_label = row['Question']
            # The rest of columns are user scores
            scores = row.drop(labels='Question').values.astype(float)
            
            mean = np.mean(scores)
            std_dev = np.std(scores, ddof=1)
            
            print(f"{question_label}: mean={mean:.2f}, std={std_dev:.2f}")
        return 


if __name__ == "__main__":
    AnalysisQuestionnaire.plotSubjectiveScores()
    AnalysisQuestionnaire.calculateMeanStdDev()
    
