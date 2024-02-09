# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ThreadItem(scrapy.Item):
    post_id = scrapy.Field()
    
    user_id = scrapy.Field()
    title = scrapy.Field()
    comments = scrapy.Field()
    original_post_content = scrapy.Field()
    
    url = scrapy.Field()
    post_date = scrapy.Field()
    views = scrapy.Field()  # New field for thread views
    likes = scrapy.Field()  # New field if threads can be liked or upvoted
    tags = scrapy.Field()  # New field for thread tags or labels


class CommentItem(scrapy.Item):
    post_id = scrapy.Field()
    
    user_id = scrapy.Field()
    comment_text = scrapy.Field()
    comment_timestamp = scrapy.Field()
    
    reply_to_post_id = scrapy.Field()  # New field for tracking replies
    
    page_number = scrapy.Field()


class UserItem(scrapy.Item):
    user_id = scrapy.Field()
    username = scrapy.Field()
    avatar_url = scrapy.Field()
    join_date = scrapy.Field()
    post_count = scrapy.Field()
