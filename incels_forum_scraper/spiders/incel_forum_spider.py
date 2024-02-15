import scrapy
from incels_forum_scraper.items import ThreadItem, CommentItem, UserItem

import re

from datetime import datetime
import re
from html import unescape


def extract_and_remove_quotes(html_text):
    # Initialize an empty list to hold all reply_to_post_ids
    reply_to_post_ids = []

    # Find all blockquotes and extract reply_to_post_id
    blockquotes = re.findall(
        r'<blockquote[^>]+data-source="post: (\d+)"[^>]*>', html_text, flags=re.DOTALL)
    reply_to_post_ids.extend(blockquotes)

    # Remove all occurrences of blockquote elements
    cleaned_html = re.sub(r'<blockquote.*?/blockquote>',
                          '', html_text, flags=re.DOTALL)

    # Convert HTML entities back to characters, if necessary
    cleaned_html = unescape(cleaned_html)

    return cleaned_html, [int(reply) for reply in reply_to_post_ids]


class ForumSpider(scrapy.Spider):
    name = 'incelsis'
    allowed_domains = ['incels.is']
    start_urls = ['https://incels.is/forums/inceldom-discussion.2/']

    def parse(self, response):
        # Extract thread URLs and iterate over them
        threads = response.css(
            'div.structItem--thread a[data-tp-primary="on"]::attr(href)').getall()
        for thread_url in threads:
            yield response.follow(thread_url, self.check_thread_date)

        # Handle forum pagination if exists
        next_page = response.css('a.pageNav-jump--next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def check_thread_date(self, response):

        # Example: Extract the thread start date (you'll need to adjust the selector based on actual page structure)
        # Parse the date assuming the format is '2024-01-29T16:48:37-0500'
        thread_datetime = response.css(
            'a.u-concealed time::attr(datetime)').get()
        thread_date = datetime.strptime(
            thread_datetime, "%Y-%m-%dT%H:%M:%S%z").date()

        # Only process threads starting from 2023 onwards
        if thread_date.year >= 2023:
            yield from self.parse_thread(response)

    def parse_thread(self, response):
        is_first_page = not any(part.startswith('page-')
                                for part in response.url.split('/'))
        if is_first_page:
            page_number = 1
        else:
            page_number = int(response.url.split('/')[-1].split('-')[-1])
        
        if 'thread_item' in response.meta:
            thread_item = response.meta['thread_item']
        else:
            # We are on the first page, so initialize thread_item and its properties
            thread_item = ThreadItem()
            thread_item['title'] = response.css('h1.p-title-value::text').get().strip()
            thread_item['url'] = response.url
            thread_item['comments'] = []  # Initialize comments list here
            
            # Process the first post only if on the first page
            if is_first_page:
                first_post = response.css('article.message--post').extract_first()
                if first_post:
                    first_post = response.css('article.message--post')[0]
                    # Adjust the selector based on the actual structure
                    thread_item['user_id'] = int(first_post.css(
                        '.avatar::attr(data-user-id)').get())
                    thread_item['post_date'] = first_post.css(
                        'time.u-dt::attr(datetime)').get()
                    thread_item['text_content'] = ' '.join(
                        first_post.css('.message-content ::text').getall()).strip()

                    #         "post_id": "post-13448716",
                    thread_item['post_id'] = int(
                        first_post.xpath('@data-content').get()[5:])

                    username = first_post.css(
                        '.message-userDetails .message-name span::text').get()
                    join_date = first_post.css(
                        '.message-userExtras dt:contains("Joined") + dd::text').get()
                    post_count = first_post.css(
                        '.message-userExtras dt:contains("Posts") + dd::text').get()
                    post_count_clean = re.sub(r'\D', '', post_count) if post_count else '0'
                    post_count_int = int(post_count_clean)
                    # saving user data
                    yield UserItem(
                        user_id=thread_item['user_id'],
                        username=username,
                        join_date=join_date,
                        post_count=post_count_int
                    )
                else:
                    # Fallback or default values if the first post isn't found
                    thread_item['user_id'] = -1
                    thread_item['text_content'] = 'Content not found'

        # For the first page, skip the first post when scraping comments
        # For other pages, scrape all posts as comments
        start_index = 0 if not is_first_page else 1
        comments = response.css('article.message--post')[start_index:]
        for comment in comments:
            comment_item = CommentItem()
            # Extracting post ID from each comment
            comment_item['post_id'] = int(
                comment.xpath('@data-content').get()[5:])

            # Removing quoted content before extracting text
            comment_html = comment.get()
            cleaned_html, reply_to_post_ids = extract_and_remove_quotes(
                comment_html)
            # Convert back to a Selector for further processing
            comment_selector = scrapy.Selector(text=cleaned_html)
            # text and if it's a reply to another post
            comment_item['text_content'] = ' '.join(
                comment_selector.css('.message-content ::text').getall()).strip()
            
            # we count a quote as a reply, otherwise we use the thread's post_id
            comment_item['quoted_posts'] = reply_to_post_ids if reply_to_post_ids else []

            # post metadata
            comment_item['user_id'] = int(comment.css(
                '.avatar::attr(data-user-id)').get())
            comment_item['post_date'] = comment.css(
                'time.u-dt::attr(datetime)').get()
            comment_item['page_number'] = page_number

            username = comment.css(
                '.message-userDetails .message-name span::text').get()
            join_date = comment.css(
                '.message-userExtras dt:contains("Joined") + dd::text').get()
            post_count = comment.css(
                '.message-userExtras dt:contains("Posts") + dd::text').get()
            post_count_clean = re.sub(r'\D', '', post_count) if post_count else '0'
            post_count_int = int(post_count_clean)
            yield UserItem(
                user_id=comment_item['user_id'],
                username=username,
                join_date=join_date,
                post_count=post_count_int
            )

            thread_item['comments'].append(comment_item)

        # Handle pagination and pass along the thread_item
        next_page = response.css('a.pageNav-jump--next::attr(href)').get()
        if next_page:
            request = response.follow(next_page, callback=self.parse_thread)
            request.meta['thread_item'] = thread_item  # Pass the thread_item for continuity
            yield request
        else:
            # Finalize and yield the thread_item if there's no next page
            yield thread_item