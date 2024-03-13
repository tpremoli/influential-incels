import json
import networkx as nx
import pickle
import os
from tqdm import tqdm

from transformers import pipeline

def calculate_graph(users, posts):
    # Initialize a directed graph
    incel_graph = nx.DiGraph()

    # Add nodes (users) to the graph
    print("adding users (nodes) to graph")
    for user in tqdm(users, desc="Processing users"):
        incel_graph.add_node(user['user_id'], username=user['username'])

    # Define weights for different types of interactions
    mention_quote_weight = 1.0
    general_comment_weight = 0.5

    # Add edges (interactions) to the graph
    print("Adding edges (interactions) to graph...")
    for post in tqdm(posts, desc="Processing posts"):
        for comment in post['comments']:
            # Direct interaction through mentions or quotes
            has_direct_interaction = False

            # If the comment mentions other users
            if comment['mentioned_users']:
                has_direct_interaction = True
                for mentioned_user in comment['mentioned_users']:
                    incel_graph.add_edge(comment['user_id'], mentioned_user, weight=mention_quote_weight)

            # If the comment quotes other posts
            if comment['quoted_posts']:
                has_direct_interaction = True
                for quoted_post in comment['quoted_posts']:
                    # Find the original poster and add an edge
                    original_poster = next((item['user_id'] for item in posts if item['post_id'] == quoted_post), None)
                    if original_poster:
                        incel_graph.add_edge(comment['user_id'], original_poster, weight=mention_quote_weight)

            # General comment interaction with the original post's author
            if not has_direct_interaction and comment['user_id'] != post['user_id']:
                incel_graph.add_edge(comment['user_id'], post['user_id'], weight=general_comment_weight)

    print("graph calculation complete. Saving to file")
    with open('forum_graph.pkl', 'wb') as f:
        pickle.dump(incel_graph, f)
    
    return incel_graph

def get_mean_emotion_score(emotion_probs):
    # Apply threshold and calculate the mean of the probabilities
    filtered_probs = [prob for prob in emotion_probs if prob >= 0.5]
    if not filtered_probs:  # If no emotions exceed the threshold, consider all for the mean
        return sum(emotion_probs) / len(emotion_probs)
    return sum(filtered_probs) / len(filtered_probs)

def compare_top_n_emotions(N, pagerank, users,posts):
    roberta = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    top_n_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:N]
    top_n_user_ids = [user[0] for user in top_n_users]

    # Initialize dictionaries to store sentiment scores
    top_user_mean_scores = []
    other_user_mean_scores = []

    for post in posts:
        # Analyze the post's text sentiment
        post_emotions = roberta(post['text_content'])
        post_mean_score = get_mean_emotion_score(post_emotions)
        
        if post['user_id'] in top_n_user_ids:
            top_user_mean_scores.append(post_mean_score)
        else:
            other_user_mean_scores.append(post_mean_score)

        # Analyze sentiments in the comments
        for comment in post['comments']:
            comment_emotions = roberta(comment['text_content'])
            comment_mean_score = get_mean_emotion_score(comment_emotions)

            if comment['user_id'] in top_n_user_ids:
                top_user_mean_scores.append(comment_mean_score)
            else:
                other_user_mean_scores.append(comment_mean_score)
                
    avg_top_user_score = sum(top_user_mean_scores) / len(top_user_mean_scores) if top_user_mean_scores else 0
    avg_other_user_score = sum(other_user_mean_scores) / len(other_user_mean_scores) if other_user_mean_scores else 0

    print("Average Emotion Score of Top Users:", avg_top_user_score)
    print("Average Emotion Score of Other Users:", avg_other_user_score)

if __name__ == "__main__":
    # Load the JSON data
    print("loading users")
    with open('unique_users.json', encoding='utf-8') as f:
        users = json.load(f)

    print("loading posts")
    with open('posts.json', encoding='utf-8') as f:
        posts = json.load(f)
    
    # check if forum_graph.pkl exists, if so load it, else calculate it
    if os.path.exists('forum_graph.pkl'):
        print("Loading graph from file...")
        with open('forum_graph.pkl', 'rb') as f:
            incel_graph = pickle.load(f)
    else:
        print("Calculating graph...")
        incel_graph = calculate_graph(users,posts)
        
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
        
    print("calculating and comparing sentiments")
    compare_top_n_emotions(10, pagerank,users,posts)