"""This module computes descriptive statistics for the system usability scale (SUS) questionnaire."""

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def calculate_sus_scores(raw_scores):
    """
    Calculates the System Usability Scale (SUS) scores based on raw Likert scale scores.

    Args:
        raw_scores (numpy array): 2D array of raw Likert scale scores.
            Rows represent participants, and columns represent statements.

    Returns:
        numpy array: 1D array of SUS scores for each participant.
    """
    # Define the conversion factors for odd- and even-numbered statements
    odd_conversion = np.array([1, 3, 5, 7, 9])
    even_conversion = np.array([2, 4, 6, 8, 10])

    # Convert the raw scores to SUS scores
    sus_scores = np.zeros(raw_scores.shape[0])  # Initialize array to store SUS scores
    for i, scores in enumerate(raw_scores):
        odd_scores = scores[odd_conversion - 1] - 1
        even_scores = 5 - scores[even_conversion - 1]
        sus_scores[i] = (np.sum(odd_scores) + np.sum(even_scores)) * 2.5

    return sus_scores


# Participant responses to SUS statements (1 to 10)
sus_responses = np.array(
    [
        [4, 2, 4, 2, 4, 2, 4, 1, 4, 3],  # T-01
        [5, 1, 5, 2, 5, 1, 5, 1, 5, 1],  # T-02
        [2, 2, 4, 2, 4, 2, 5, 2, 4, 2],  # T-03
        [4, 2, 5, 2, 4, 1, 5, 1, 4, 2],  # T-04
    ]
)

# Calculate SUS scores
sus_scores = calculate_sus_scores(sus_responses)

# Print SUS scores for each participant
for i, score in enumerate(sus_scores):
    print(f"T-0{i + 1}: {score:.2f}")

# Print mean and standard deviation of SUS scores
print(f"Mean: {np.mean(sus_scores):.2f}")
print(f"Standard deviation: {np.std(sus_scores):.2f}")

# Plot SUS scores on bar plot, add a horizontal line at 68
plt.figure(figsize=(10, 4))
plt.bar(np.arange(1, 5), sus_scores, alpha=0.5, ecolor="black", capsize=10)
plt.axhline(y=68, color="red", linestyle="--")
plt.xticks(np.arange(1, 5), labels=["T-01", "T-02", "T-03", "T-04"])
plt.xlabel("Participant")
plt.ylabel("SUS score")
plt.title("System Usability Scale (SUS) Scores for Each Participant")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("aim1/sus_scores_participants.png")
