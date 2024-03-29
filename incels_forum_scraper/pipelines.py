# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from scrapy.exceptions import DropItem
from scrapy.exporters import JsonItemExporter
from incels_forum_scraper.items import UserItem
import json


class IncelsForumScraperPipeline:
    def process_item(self, item, spider):
        return item

class UserPipeline:
    def open_spider(self, spider):
        self.file = open('scraped_data/unique_users.json', 'wb')  # Open in binary mode for JsonItemExporter
        self.exporter = JsonItemExporter(self.file, encoding='utf-8', ensure_ascii=False)
        self.exporter.start_exporting()
        self.users_seen = set()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()
    
    def process_item(self, item, spider):
        if isinstance(item, UserItem):  # Ensure this pipeline only processes UserItems
            user_identifier = item['user_id']  # Assuming 'user_id' is a unique identifier
            if user_identifier not in self.users_seen:
                self.users_seen.add(user_identifier)
                self.exporter.export_item(item)
            raise DropItem("Duplicate user found: %s" % item['username'])
        return item  # Non-UserItem items are passed through