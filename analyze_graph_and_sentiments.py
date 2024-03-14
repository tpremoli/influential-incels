import pandas as pd
import networkx as nx
import numpy as np
import pickle
from scipy import stats

EMOTIONS = ["admiration", "amusement", "approval", "caring", "anger", "annoyance",
            "disappointment", "disapproval", "confusion", "desire", "excitement",
            "gratitude", "joy", "disgust", "embarrassment", "fear", "grief",
            "curiosity", "love", "optimism", "pride", "relief", "nervousness",
            "remorse", "sadness", "realization", "surprise", "neutral"]

if __name__ == "__main__":
    print("Loading graph from file...")
    with open('forum_graph.pkl', 'rb') as f:
        incel_graph = pickle.load(f)

    # Apply the PageRank algorithm
    print("applying pagerank algorithm to graph")
    pagerank = nx.pagerank(incel_graph, weight='weight')

    # Sort users by their PageRank scores
    print("sorting users by pagerank scores")
    sorted_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    # Display the top 10 influential users
    print("Network analysis complete. Top 10 users:")
    for user_id, score in sorted_users[:10]:
        print(f"User ID: {user_id}, Score: {score}")

    print("loading sentiment files...")
    df = pd.read_csv('sentiment_analysis.csv')
    
    # Get the top 10 users based on PageRank
    top_10_user_ids = [user[0] for user in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]]

    # Calculate the average sentiment scores for the top 10 users
    top_10_avg = df[df['user_id'].isin(top_10_user_ids)][EMOTIONS].mean()

    # Calculate the average sentiment scores for all other users
    other_users_avg = df[~df['user_id'].isin(top_10_user_ids)][EMOTIONS].mean()

    # Calculate the differences
    differences = top_10_avg - other_users_avg

    # Combine all the data into one DataFrame
    comparison = pd.DataFrame({
        'Top 10 Users': top_10_avg,
        'Other Users': other_users_avg,
        'Difference': differences
    })

    # Variance analysis across all users for each emotion
    emotion_variance = df[EMOTIONS].var()

    # Display the results
    print("Average sentiment comparison between top 10 users and other users:")
    print(comparison)

    print("\nVariance in sentiment scores across all users:")
    print(emotion_variance)
        
    print("running confidence interval analysis...")
    # Number of bootstrap samples
    n_bootstraps = 1000

    # Initialize dictionaries to hold the bootstrap results
    top_10_bootstrap_means = {emotion: [] for emotion in EMOTIONS}
    other_users_bootstrap_means = {emotion: [] for emotion in EMOTIONS}

    # Bootstrap for top 10 users
    for _ in range(n_bootstraps):
        sample = df[df['user_id'].isin(top_10_user_ids)].sample(frac=1, replace=True)
        for emotion in EMOTIONS:
            top_10_bootstrap_means[emotion].append(sample[emotion].mean())

    # Bootstrap for other users
    for _ in range(n_bootstraps):
        sample = df[~df['user_id'].isin(top_10_user_ids)].sample(frac=1, replace=True)
        for emotion in EMOTIONS:
            other_users_bootstrap_means[emotion].append(sample[emotion].mean())

    print("95% confidence intervals for the average sentiment scores:")
    # Calculate the confidence intervals
    confidence_intervals = {
        'Top 10 Users': {emotion: np.percentile(top_10_bootstrap_means[emotion], [5, 95]) for emotion in EMOTIONS},
        'Other Users': {emotion: np.percentile(other_users_bootstrap_means[emotion], [5, 95]) for emotion in EMOTIONS}
    }

    # Display the confidence intervals
    for emotion in EMOTIONS:
        print(f"{emotion}:\nTop 10 Users: {confidence_intervals['Top 10 Users'][emotion]}\nOther Users: {confidence_intervals['Other Users'][emotion]}\n")
        
    print("checking statistical significance of means...")
    comparison_results = []

    for emotion in EMOTIONS:
        # Extract the emotion scores for top 5 users and other users
        top_10_scores = df[df['user_id'].isin(top_10_user_ids)][emotion]
        other_users_scores = df[~df['user_id'].isin(top_10_user_ids)][emotion]

        # Calculate the averages
        top_10_avg = top_10_scores.mean()
        other_users_avg = other_users_scores.mean()

        # Calculate the difference
        difference = top_10_avg - other_users_avg

        # Perform Levene's test for equality of variances
        levene_test = stats.levene(top_10_scores, other_users_scores)

        # Perform the t-test based on the variances
        if levene_test.pvalue < 0.05:
            t_test_result = stats.ttest_ind(top_10_scores, other_users_scores, equal_var=False)
        else:
            t_test_result = stats.ttest_ind(top_10_scores, other_users_scores, equal_var=True)

        # Determine if the result is statistically significant (using p < 0.05 as the criterion)
        is_significant = t_test_result.pvalue < 0.05

        # Append the results to the DataFrame
        comparison_results.append({
            'Emotion': emotion,
            'Top 10 Users': top_10_avg,
            'Other Users': other_users_avg,
            'Difference': difference,
            'Is Significant': is_significant
        })

    # Convert the results list to a DataFrame
    comparison_df = pd.DataFrame(comparison_results)

    # Display the results
    print(comparison_df)
    