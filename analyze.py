import json
import networkx as nx
import pickle
import os
from tqdm import tqdm

import pandas as pd
import torch
from transformers import pipeline
from transformers import AutoTokenizer

EMOTIONS = ["admiration", "amusement", "approval", "caring", "anger", "annoyance",
            "disappointment", "disapproval", "confusion", "desire", "excitement",
            "gratitude", "joy", "disgust", "embarrassment", "fear", "grief",
            "curiosity", "love", "optimism", "pride", "relief", "nervousness",
            "remorse", "sadness", "realization", "surprise", "neutral"]


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

def split_text_to_tokens(text, tokenizer, max_length):
    # Tokenize the text and split it into chunks of max_length tokens
    tokens = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

def get_weighted_mean_emotion_score(chunks, model):
    total_length = sum(len(chunk) for chunk in chunks)
    
    # Initialize a dictionary for emotion labels with initial score of 0
    weighted_scores = {emotion: 0 for emotion in EMOTIONS}

    for chunk in chunks:
        chunk_scores = model(chunk)  # This returns a list of dictionaries for each chunk
        weight = len(chunk) / total_length

        for score_dict in chunk_scores[0]:  # chunk_scores[0] contains our list of dictionaries
            emotion = score_dict['label']
            score = score_dict['score']
            if emotion in weighted_scores:
                weighted_scores[emotion] += score * weight

    return weighted_scores

def compare_top_n_emotions(N, pagerank, users, posts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roberta = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

    top_n_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:N]
    top_n_user_ids = [user[0] for user in top_n_users]

    rows = []

    for post in tqdm(posts, desc="Processing posts"):
        # Split and analyze the post's text sentiment
        post_chunks = split_text_to_tokens(post['text_content'], tokenizer, 512)
        post_mean_score = get_weighted_mean_emotion_score(post_chunks, roberta)
        
        # Ensure the order of scores matches the order of emotions
        post_scores = [post_mean_score[emotion] for emotion in EMOTIONS]
        rows.append([post['post_id'], post['user_id']] + post_scores)

        # Analyze sentiments in the comments
        for comment in post['comments']:
            comment_chunks = split_text_to_tokens(comment['text_content'], tokenizer, 512)
            comment_mean_score = get_weighted_mean_emotion_score(comment_chunks, roberta)

            # Ensure the order of scores matches the order of emotions
            comment_scores = [comment_mean_score[emotion] for emotion in EMOTIONS]
            rows.append([comment['post_id'], comment['user_id']] + comment_scores)

    # Save to CSV
    columns = ['post_id', 'user_id'] + [emotion + "_score" for emotion in EMOTIONS]
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