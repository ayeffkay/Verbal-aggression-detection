""""
Gathering comments from OK (without labeling)
"""
from api.corpus_db import Database
from api.ok_api import OkApi

db = Database()
query = "новости"
ok_api = OkApi()

groups = ok_api.search_quick(query)
for group in groups:
    if group["access_type"] == "OPEN":
        print(group["name"])
        topics = ok_api.stream_get(group["uid"], count=150)
        for topic in topics:
            comments = ok_api.get_comments(topic["id"],
                                           topic["comments_count"])
            print(topic["comments_count"])
            for comment in comments:
                db.add_comment(comment["text"])
