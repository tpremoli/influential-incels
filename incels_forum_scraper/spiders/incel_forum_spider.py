import scrapy
from incels_forum_scraper.items import ThreadItem, CommentItem

import re

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

    return cleaned_html, reply_to_post_ids


class ForumSpider(scrapy.Spider):
    name = 'incelsis'
    allowed_domains = ['incels.is']
    start_urls = ['https://incels.is/forums/inceldom-discussion.2/']

    def parse(self, response):
        # Extract thread URLs and iterate over them
        threads = response.css(
            'div.structItem--thread a[data-tp-primary="on"]::attr(href)').getall()
        for thread in threads:
            yield response.follow(thread, self.parse_thread)

        # Handle forum pagination if exists
        next_page = response.css('a.pageNav-jump--next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_thread(self, response):
        thread_item = ThreadItem()
        thread_item['title'] = response.css('h1.p-title-value::text').get()
        thread_item['url'] = response.url

        # Check if the URL is for the first page of the thread
        is_first_page = not any(part.startswith('page-')
                                for part in response.url.split('/'))
        if is_first_page:
            page_number = 1
        else:
            page_number = int(response.url.split('/')[-1].split('-')[-1])

        if is_first_page:
            first_post = response.css('article.message--post').extract_first()
            if first_post:
                first_post = response.css('article.message--post')[0]
                # Adjust the selector based on the actual structure
                thread_item['user_id'] = int(first_post.css(
                    '.avatar::attr(data-user-id)').get())
                thread_item['post_date'] = first_post.css(
                    'time.u-dt::attr(datetime)').get()
                thread_item['original_post_content'] = ' '.join(
                    first_post.css('.message-content ::text').getall()).strip()

                #         "post_id": "post-13448716",
                thread_item['post_id'] = int(
                    first_post.xpath('@data-content').get()[5:])

            else:
                # Fallback or default values if the first post isn't found
                thread_item['user_id'] = -1
                thread_item['original_post_content'] = 'Content not found'
        else:
            # For subsequent pages, do not include the original post content
            thread_item['author'] = 'N/A'
            thread_item['original_post_content'] = 'N/A'

        thread_item['comments'] = []

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
            comment_item['comment_text'] = ' '.join(
                comment_selector.css('.message-content ::text').getall()).strip()
            comment_item['reply_to_post_id'] = reply_to_post_ids[0] if reply_to_post_ids else None

            # post metadata
            comment_item['user_id'] = int(comment.css(
                '.avatar::attr(data-user-id)').get())
            comment_item['comment_timestamp'] = comment.css(
                'time.u-dt::attr(datetime)').get()
            comment_item['page_number'] = page_number

            thread_item['comments'].append(comment_item)

        # Handle pagination and pass along the thread_item as before
        next_page = response.css('a.pageNav-jump--next::attr(href)').get()
        if next_page:
            request = response.follow(next_page, self.parse_thread)
            # Pass the thread_item through meta only if it's the first page
            if is_first_page:
                request.meta['thread_item'] = thread_item
            yield request
        else:
            # If no next page, yield the completed thread item
            yield thread_item
