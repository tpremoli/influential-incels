import pandas as pd
import networkx as nx
import pickle

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