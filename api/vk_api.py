import vk
import re
from datetime import datetime, date
import time
import api


class VkApi:
    def __init__(self, version=5.74):
        self.access_token = api.vk_token
        self.session = vk.Session(access_token=self.access_token)
        self.api = vk.API(self.session, v=version)
        self.groups_lim = 15
        self.topics_lim = 100
        self.comments_lim = 100


    def groups_search(self, query):
        groups = self.api.groups.search(q=query, type="group, page",
                                        sort=4, count=self.groups_lim,
                                        fields="description,members_count")
        return groups['items']


    @staticmethod
    def to_datetime(time):
        return datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')

    def get_topics(self, id, count=100):
        offset = 0
        total_topics = []
        for i in range(count, 0, -self.topics_lim):
            topics = self.api.wall.get(owner_id=-id, count=self.topics_lim, offset=offset)["items"]
            total_topics += topics
            offset += self.topics_lim
        for topic in total_topics:
            topic["date"] = self.to_datetime(topic["date"])
            count = topic.pop("comments", None)
            topic["comments"] = count["count"]
            likes = topic.pop("likes", None)
            topic["likes"] = likes["count"]
            reposts = topic.pop("reposts", None)
            topic["reposts"] = reposts["count"]

        return total_topics

    def get_comments(self, owner_id, post_id, count=100):
        total_comments = []
        j = 0  # offset
        for i in range(count, 0, -self.comments_lim):
            ct = i if i <= self.comments_lim else self.comments_lim
            comments = self.api.wall.getComments(owner_id=-owner_id,
                                                 post_id=post_id,
                                                 need_likes=1,
                                                 count=ct,
                                                 offset=j)["items"]
            total_comments += comments
            j += self.comments_lim
            time.sleep(1)
        for comment in total_comments:
            comment["date"] = self.to_datetime(comment["date"])
            likes = comment.pop("likes", None)
            comment["likes"] = likes["count"]

            reply = comment.pop("reply_to_comment", None)
            comment["reply_to_comment"] = reply

            reply = comment.pop("reply_to_user", None)
            comment["reply_to_user"] = reply
        return total_comments

    def is_member(self, group_id, user_id):
        return self.api.groups.isMember(group_id=group_id, user_id=user_id)

    def user_info(self, user_id):
        fields = "bdate,city,counters,sex"
        usr = self.api.users.get(user_ids=user_id, fields=fields)[0]

        bdate = usr.pop("bdate", None)
        usr["age"] = self.calculate_age(bdate) if bdate else None

        city = usr.pop("city", None)
        usr["city"] = city["title"] if city else None

        counters = usr.pop("counters", None)
        usr["friends"] = counters["friends"] if counters else None
        return usr


    @staticmethod
    def calculate_age(bdate):
        bdate_pattern = re.compile(r'\d+\.\d+\.\d{4}', re.DOTALL)
        bdate_match = bdate_pattern.fullmatch(bdate)
        if not bdate_match:
            return None
        today = date.today()
        bdate = datetime.strptime(bdate_match.group(0), '%d.%m.%Y')
        age = today.year - bdate.year
        full_year_passed = (today.month, today.day) < (bdate.month, bdate.day)
        if not full_year_passed:
            age -= 1
        return age
