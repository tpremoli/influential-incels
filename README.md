

## Welcome to my program for Social Networks and Text Analysis!

Data is stored in the scraped_data folder, and the full text analysis output is stored in text_analysis.
The results ultimately analyzed in the report are stored in the results.txt file.
The report is found in "report.pdf"

The scripts for the web scraper is found in the "incels_forum_scraper" folder.
    To run it, you can simply use "sh scrape.sh" in the root directory of the project.
    To run this, you need scrapy installed.
    When running the scraper through scrape.sh, the outputs will be stored in "scraped_data/posts.json"
    and "scraped_data/unique_users.json

The scripts for the generation of the emotion scores and the user graph is found
in create_graph_and_sentiments.py. To run this, you need to have the following
packages installed:
- pandas
- networkx
- huggingface transformers
- numpy
- tqdm
- torch
    The outputs are placed in "forum_graph.pkl" for the graph, and the sentiment files go 
    in "text_analysis/sentiment_analysis_cardiff.csv" for simple sentiments, and 
    "text_analysis/sentiment_analysis_goemotions.csv" for the emotion classification.

To run the final analysis, you can run the "analyze_graph_and_sentiments.py" script. This script will run
the analysis as explained in the report. The output will be printed to the console.
The outputs used in the report are stored in "results.txt"

Some extra scripts are included in the "extra_utils" folder which are kept for the sake of completeness.
- "lightbox_cleanup.py" cleans up the posts.json file to remove lightbox links.
    These links are placeholders for images, but as we are not interested in images, we remove them.
    The posts.json file included with this submission already has the cleaned data.
- "count_items.py" counts the number of unique users, comments, threads, and overall posts.