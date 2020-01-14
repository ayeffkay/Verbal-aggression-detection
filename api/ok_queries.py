"""
Gathering and labeling comments from OK
"""
import api
from api import dialog as d
from api.ok_api import OkApi
from api.dataset_db import Database

query = "новости"
ok_api = OkApi()
db = Database()

groups = ok_api.search_quick(query)
for group in groups:
    if group["access_type"] == "OPEN" and d.display_group(group["name"]):
        topics = ok_api.stream_get(group["uid"])
        for topic in topics:
            if d.display_topic(topic["text"], topic["comments_count"]):
                comments = ok_api.get_comments(topic["id"], topic["comments_count"])
                ct = topic["comments_count"]
                for comment in comments:
                    answ = d.display_comment(comment["text"])
                    if answ == 0 or answ == 1:
                        db.add_group(group["uid"],
                                    group["name"],
                                    group["description"],
                                    group["members_count"])
                        db.add_topic(group["uid"],
                                    topic["id"],
                                    topic["text"],
                                    topic["likes"],
                                    topic["reposts"],
                                    topic["comments_count"],
                                    topic["date"])
                        usr = ok_api.users_get_info(comment["author_id"])
                        db.add_user(comment["author_id"],
                                    usr["name"],
                                    usr["gender"],
                                    usr["age"],
                                    usr["city"],
                                    usr["friends"])
                        is_member = ok_api.is_member(group["uid"], comment["author_id"])
                        emoji = api.emoji_count(comment["text"])
                        db.add_comment(group["uid"], topic["id"], comment["id"],
                                      comment["author_id"], is_member,
                                      comment["text"], emoji,
                                      comment["like_count"],
                                      comment["reply_to_comment_id"],
                                      comment["reply_to_id"],
                                      comment["date"], answ)
                    ct -= 1
                    if d.ask_exit(ct):
                        break
