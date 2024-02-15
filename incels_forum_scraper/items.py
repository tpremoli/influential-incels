# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ThreadItem(scrapy.Item):
    post_id = scrapy.Field()

    user_id = scrapy.Field()
    title = scrapy.Field()
    text_content = scrapy.Field()
    comments = scrapy.Field()
    mentioned_users= scrapy.Field()

    url = scrapy.Field()
    post_date = scrapy.Field()
    


class CommentItem(scrapy.Item):
    post_id = scrapy.Field()

    user_id = scrapy.Field()
    text_content = scrapy.Field()
    post_date = scrapy.Field()

    quoted_posts = scrapy.Field()  # New field for tracking replies
    mentioned_users = scrapy.Field()  # New field for tracking mentions

    page_number = scrapy.Field()


class UserItem(scrapy.Item):
    user_id = scrapy.Field()
    username = scrapy.Field()

    join_date = scrapy.Field()
    post_count = scrapy.Field()
