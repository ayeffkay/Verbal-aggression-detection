import api
from api import dialog as d
from api.vk_api import VkApi
from api.dataset_db import Database

db = Database()
query = "новости"
vk_api = VkApi()

groups = vk_api.groups_search(query)
for group in groups:
    if not group["is_closed"] and d.display_group(group["name"]):
        topics = vk_api.get_topics(group["id"])
        for topic in topics:
            if not topic["marked_as_ads"] and d.display_topic(topic["text"], topic["comments"]):
                comments = vk_api.get_comments(group["id"], topic["id"], topic["comments"])
                ct = topic["comments"]
                for comment in comments:
                    answ = d.display_comment(comment["text"])
                    if answ == 0 or answ == 1:
                        try:
                            db.add_group(group["id"],
                                        group["name"],
                                        group["description"],
                                        group["members_count"])
                            db.add_topic(group["id"],
                                        topic["id"],
                                        topic["text"],
                                        topic["likes"],
                                        topic["reposts"],
                                        topic["comments"],
                                        topic["date"])
                            usr = vk_api.user_info(comment["from_id"])
                            name = usr["first_name"] + " " + usr["last_name"]
                            db.add_user(usr["id"], name,
                                       usr["sex"], usr["age"],
                                       usr["city"], usr["friends"])
                            is_member = vk_api.is_member(group["id"], comment["from_id"])
                            emoji = api.emoji_count(comment["text"])
                            db.add_comment(group["id"], topic["id"], comment["id"],
                                           usr["id"], is_member,
                                           comment["text"],
                                           emoji, comment["likes"],
                                           comment["reply_to_comment"],
                                           comment["reply_to_user"],
                                           comment["date"], answ)
                        except:
                            pass
                    ct -= 1
                    if d.ask_exit(ct):
                        break
