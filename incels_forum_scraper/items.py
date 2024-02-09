# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ThreadItem(scrapy.Item):
    post_id = scrapy.Field()
    
    title = scrapy.Field()
    author = scrapy.Field()
    comments = scrapy.Field()
    original_post_content = scrapy.Field()
    
    url = scrapy.Field()
    post_date = scrapy.Field()
    views = scrapy.Field()  # New field for thread views
    likes = scrapy.Field()  # New field if threads can be liked or upvoted
    tags = scrapy.Field()  # New field for thread tags or labels


class CommentItem(scrapy.Item):
    post_id = scrapy.Field()
    comment_text = scrapy.Field()
    commenter_username = scrapy.Field()
    commenter_user_title = scrapy.Field()
    comment_timestamp = scrapy.Field()
    
    reply_to_post_id = scrapy.Field()  # New field for tracking replies
    
    page_number = scrapy.Field()


