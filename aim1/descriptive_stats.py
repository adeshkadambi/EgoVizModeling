'''This module contains code for computing descriptive statistics from aim 1 quenstionnaires.'''

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# dictionary of questions and their corresponding indices
questions = {
    0: "I predominantly deal with technical systems because I am forced to.",
    1: "I enjoy spending time becoming acquainted with a new technical system.",
    2: "It is enough for me that a technical system works; I do not care how or why.",
    3: "I try to understand how a technical system exactly works.",
    4: "It is enough for me to know the basic functions of a technical system.",
    5: "I try to make full use of the capabilities of a technical system.",
    6: "I would personally use this CDSS regularly to understand and assess patient performance.",
    7: "I found the CDSS easy to use overall, and the various functions in this CDSS were well integrated.",
    8: "I felt like I could trust the metrics and information provided to me by the CDSS.",
    9: "Using this CDSS regularly would substantially interrupt my workflow.",
    10: "I feel apprehensive about using this CDSS with future patients.",
    11: "Using this CDSS is useful in understanding and assessing patient performance.",
    12: "I found the CDSS very cumbersome to use, and the presented metrics were not clear.",
    13: "I would imagine that most people would learn to use this CDSS very quickly.",
    14: "The information presented by the CDSS would potentially influence patient therapy plans.",
    15: "Presenting the information in this format is NOT useful to me or my assessment of patients."
}

# list of questions to be reverse scored
reverse_scored = [0, 2, 4, 9, 10, 12, 15]

# scores for each question from each participant (rows are participants, columns are questions)
scores = np.array([
    [1, 4, 2, 4, 2, 5, 4, 4, 4, 2, 2, 5, 2, 4, 4, 1], # T-01
    [1, 5, 1, 5, 1, 5, 5, 5, 5, 2, 1, 5, 1, 4, 5, 1], # T-02
    [3, 2, 4, 2, 4, 3, 4, 4, 4, 3, 2, 4, 2, 4, 4, 2], # T-03
])

# reverse score questions
for question in reverse_scored:
    scores[:, question] = 6 - scores[:, question]

# compute mean and standard deviation for each question
means = np.mean(scores, axis=0)
stds = np.std(scores, axis=0)

# plot means and standard deviations
plt.figure(figsize=(10, 5))
bars = plt.bar(range(len(means)), means, yerr=stds, align="center", alpha=0.5, ecolor="black", capsize=10)

# Add mean and standard deviation annotations
for i, bar in enumerate(bars):
    plt.annotate(f"Mean: {means[i]:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(-5, -120),
                 textcoords="offset points", ha='center', va='center', rotation='vertical', color='black')
    plt.annotate(f"SD: {stds[i]:.2f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(5, -120),
                 textcoords="offset points", ha='center', va='center', rotation='vertical', color='black')

plt.xticks(range(len(means)), [f"Q{i+1}" for i in range(len(means))], rotation=0)
plt.ylabel("Mean Score")
plt.title("Mean Scores for Each Question")
plt.tight_layout()
plt.savefig("aim1/mean_scores.png")

# compute mean and standard deviation for each participant
means = np.mean(scores, axis=1)
stds = np.std(scores, axis=1)

# plot means and standard deviations
plt.figure(figsize=(10, 5))
plt.bar(range(len(means)), means, yerr=stds, align="center", alpha=0.5, ecolor="black", capsize=10)
plt.xticks(range(len(means)), ["T-01", "T-02", "T-03"])
plt.ylabel("Mean Score")
plt.title("Mean Scores for Each Participant")
plt.tight_layout()
plt.savefig("aim1/mean_scores_participants.png")
plt.show()