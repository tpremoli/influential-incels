import json
import networkx as nx
from tqdm import tqdm

# Load the JSON data
print("loading users")
with open('unique_users.json', encoding='utf-8') as f:
    users = json.load(f)

print("loading posts")
with open('posts.json', encoding='utf-8') as f:
    posts = json.load(f)

# Initialize a directed graph
G = nx.DiGraph()

# Add nodes (users) to the graph
print("adding users (nodes) to graph")
for user in tqdm(users, desc="Processing users"):
    G.add_node(user['user_id'], username=user['username'])

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
                G.add_edge(comment['user_id'], mentioned_user, weight=mention_quote_weight)

        # If the comment quotes other posts
        if comment['quoted_posts']:
            has_direct_interaction = True
            for quoted_post in comment['quoted_posts']:
                # Find the original poster and add an edge
                original_poster = next((item['user_id'] for item in posts if item['post_id'] == quoted_post), None)
                if original_poster:
                    G.add_edge(comment['user_id'], original_poster, weight=mention_quote_weight)

        # General comment interaction with the original post's author
        if not has_direct_interaction and comment['user_id'] != post['user_id']:
            G.add_edge(comment['user_id'], post['user_id'], weight=general_comment_weight)

# Apply the PageRank algorithm
print("applying pagerank algorithm to graph")
pagerank = nx.pagerank(G, weight='weight')

# Sort users by their PageRank scores
print("sorting users by pagerank scores")
sorted_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

# Display the top 10 influential users
print("Analysis complete. Top 10 users:")
for user_id, score in sorted_users[:10]:
    print(f"User ID: {user_id}, Score: {score}")
