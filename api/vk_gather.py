""""
Gathering from VK without labeling
"""
from api.corpus_db import Database
from api.vk_api import VkApi

db = Database()

query = "новости"
vk_api = VkApi()

groups = vk_api.groups_search(query)
for group in groups:
    print(group["name"])
    if not group["is_closed"]:
        topics = vk_api.get_topics(group["id"], count=500)
        for topic in topics:
            if not topic["marked_as_ads"]:
                comments = vk_api.get_comments(group["id"], topic["id"], topic["comments"])
                print(topic["comments"])
                for comment in comments:
                    db.add_comment(comment["text"])
