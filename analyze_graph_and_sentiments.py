import pandas as pd
import networkx as nx
import numpy as np
import pickle
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

EMOTIONS = ["admiration", "amusement", "approval", "caring", "anger", "annoyance",
            "disappointment", "disapproval", "confusion", "desire", "excitement",
            "gratitude", "joy", "disgust", "embarrassment", "fear", "grief",
            "curiosity", "love", "optimism", "pride", "relief", "nervousness",
            "remorse", "sadness", "realization", "surprise", "neutral"]

def conf_interval_analysis(df, top_10_user_ids, n_bootstraps=1000):
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

def avgs_per_decile(sorted_users, df, emotions):
    # Determine the number of users in each decile
    num_users = len(sorted_users)
    decile_size = num_users // 10
    
    # Initialize a list to store the average sentiment scores for each decile
    decile_averages = []
    
    # Calculate the average sentiment scores for each decile
    for i in range(10):
        print("looking at decile",i)
        start_idx = i * decile_size
        end_idx = start_idx + decile_size if i < 9 else num_users  # Ensure all users are included in the last decile
        
        # Get the user IDs for the current decile
        decile_user_ids = [user[0] for user in sorted_users[start_idx:end_idx]]
        
        # Calculate the average sentiment scores for the current decile
        decile_avg = df[df['user_id'].isin(decile_user_ids)][emotions].mean()
        
        # Add the averages to the list
        decile_averages.append(decile_avg)

    # Convert the list of averages to a DataFrame for easier plotting
    decile_averages_df = pd.DataFrame(decile_averages)
    
    print("plotting")
    # Plotting
    plt.figure(figsize=(16, 8))
    for emotion in emotions:
        if emotion == "neutral": continue
        plt.plot(range(1, 11), decile_averages_df[emotion], marker='o', label=emotion)
    
    plt.title('Average Sentiment Scores by Decile')
    plt.xlabel('Decile')
    plt.ylabel('Average Score')
    plt.xticks(range(1, 11), [f'{i*10}%' for i in range(1, 11)])
    
    # Adjust the legend position
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Improve layout to prevent overlap and ensure everything fits
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.grid(True)
    plt.savefig("deciles.png")

if __name__ == "__main__":
    print("Loading graph from file...")
    with open('forum_graph.pkl', 'rb') as f:
        incel_graph = pickle.load(f)

    # Apply the PageRank algorithm
    print("applying pagerank algorithm to graph")
    pagerank = nx.pagerank(incel_graph, weight='weight')
    # betweenness = nx.betweenness_centrality(incel_graph, weight='weight')
    # eigen = nx.eigenvector_centrality(incel_graph, weight='weight')
    # closeness = nx.closeness_centrality(incel_graph, distance='weight')

    # Sort users by their PageRank scores
    print("sorting users by pagerank scores")
    sorted_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    # Display the top 10 influential users
    print("Network analysis complete. Top 10 users:")
    for user_id, score in sorted_users[:10]:
        print(f"User ID: {user_id}, Score: {score}")

    print("loading sentiment files...")
    df = pd.read_csv('sentiment_analysis_goemotionsd.csv')
    
    # Clamp emotion values
    for emotion in EMOTIONS:
        df[emotion] = df[emotion].apply(lambda x: 0 if x < 0.5 else 1)
            
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
        
    # print("running confidence interval analysis...")
    # conf_interval_analysis(df, top_10_user_ids)
    
    print("checking deciles")
    avgs_per_decile(sorted_users, df, EMOTIONS)
        
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
    