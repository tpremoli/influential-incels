# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
from incels_forum_scraper.items import UserItem


class IncelsForumScraperPipeline:
    def process_item(self, item, spider):
        return item

class UserPipeline:
    def open_spider(self, spider):
        self.users_seen = set()
        self.file = open('unique_users.json', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        if isinstance(item, UserItem):  # Ensure this pipeline only processes UserItems
            user_identifier = item['user_id']  # Assuming 'user_id' is a unique identifier
            if user_identifier not in self.users_seen:
                self.users_seen.add(user_identifier)
                line = json.dumps(dict(item)) + "\n"
                self.file.write(line)
            return item  # Returning the item is important for pipeline chaining
        else:
            return item  # Non-UserItem items are not processed but passed through
