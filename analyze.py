import json
import networkx as nx
import pickle
import os
from tqdm import tqdm

import pandas as pd
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

def split_text(text, max_length):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def get_weighted_mean_emotion_score(chunks, model):
    total_length = sum(len(chunk) for chunk in chunks)
    weighted_scores = [0] * 28  # Initialize a list of zeros for 28 emotion labels

    for chunk in chunks:
        chunk_scores = model(chunk)  # This returns a list of dictionaries
        weight = len(chunk) / total_length

        for score_dict in chunk_scores:
            # Assuming the model returns a 'label' key that corresponds to the emotion index
            label_index = int(score_dict['label'].split('_')[-1])  # Extract index from label (e.g., 'LABEL_0' -> 0)
            weighted_scores[label_index] += score_dict['score'] * weight

    return weighted_scores

def compare_top_n_emotions(N, pagerank, users, posts):
    roberta = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    top_n_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:N]
    top_n_user_ids = [user[0] for user in top_n_users]

    rows = []

    for post in posts:
        # Split and analyze the post's text sentiment
        post_chunks = split_text(post['text_content'], 512)
        post_mean_score = get_weighted_mean_emotion_score(post_chunks, roberta)
        
        rows.append([post['post_id'], post['user_id']] + post_mean_score)

        # Analyze sentiments in the comments
        for comment in post['comments']:
            comment_chunks = split_text(comment['text_content'], 512)
            comment_mean_score = get_weighted_mean_emotion_score(comment_chunks, roberta)

            rows.append([comment['post_id'], comment['user_id']] + comment_mean_score)

    # Save to CSV
    columns = ['post_id', 'user_id'] + [f'sentiment{i+1}' for i in range(28)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('sentiment_analysis.csv', index=False)
    
    # TODO: get aggregate data
    
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