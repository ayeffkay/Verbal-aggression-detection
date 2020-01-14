from hashlib import md5
import requests
import json
import re
import api
import time


class OkApi:
    def __init__(self, fmt='json'):
        self.secret = api.ok_session_key
        self.token = api.ok_token

        self.base_params = {"application_key": api.ok_app_key,
                            "format": fmt}

        self.groups_lim = 20
        self.topics_lim = 100
        self.comments_lim = 100


    def generate_sig(self, params):
        p = ["{}={}".format(key, params[key]) for key in sorted(params)]
        p = "".join(p)
        return md5((p + self.secret).encode('utf-8')).hexdigest().lower()


    def generate_request(self, params):
        sig = self.generate_sig(params)
        request = 'https://api.ok.ru/fb.do?'
        for key in sorted(params):
            request += "{}={}&".format(key, params[key])
        request += 'sig=' + sig + '&access_token=' + self.token
        return json.loads(requests.get(request).text)


    def search_quick(self, query):
        params = dict()
        params["method"] = "search.quick"
        params["query"] = query
        params["types"] = "group"
        params["count"] = self.groups_lim
        params.update(self.base_params)
        params["fields"] = "group.*"
        groups = self.generate_request(params)
        return groups["entities"]["groups"]

    @staticmethod
    def get_id_and_text(topic):
        pattern = re.compile("\{media_topic:(?P<id>\d+)\}(?P<text>.*)\{media_topic\}", re.DOTALL)
        info = re.match(pattern, topic)
        topic_id = int(info.group("id"))
        topic_text = info.group("text")
        return topic_id, topic_text

    def stream_get(self, id, count=100):
        total_topics = []
        params = dict()
        params["method"] = "stream.get"
        params["gid"] = id
        params["count"] = self.topics_lim
        params["patterns"] = "post"
        params.update(self.base_params)
        for i in range(count, 0, -self.topics_lim):
            topics = self.generate_request(params)
            total_topics += topics["feeds"]
            if "anchor" in topics:
                params["anchor"] = topics["anchor"]
            time.sleep(1)
        for topic in total_topics:
            topic["id"], topic["text"] = self.get_id_and_text(topic["message"])
            info = self.discussion_get(topic["id"])
            topic["likes"] = info["like_count"]
            topic["reposts"] = info["reshare_summary"]["count"]
            topic["comments_count"] = info["total_comments_count"]
        return total_topics

    def discussion_get(self, id):
        params = dict()
        params["method"] = "discussions.get"
        params["discussionId"] = id
        params["discussionType"] = "group_topic"
        params["fields"] = "discussion.*"
        params.update(self.base_params)
        discussion_params = self.generate_request(params)
        return discussion_params['discussion']

    def get_comments(self, id, count=100):
        params = dict()
        params["method"] = "discussions.getComments"
        params["discussionId"] = id
        params["discussionType"] = "group_topic"
        params["fields"] = "comment.*"
        params.update(self.base_params)
        total_comments = []
        for i in range(count, 0, -self.comments_lim):
            params["count"] = i if i <= self.comments_lim\
                                            else self.comments_lim
            comments = self.generate_request(params)
            if "comments" in comments:
                total_comments += comments["comments"]
            # чтобы "подцепить" хвост комментариев
            if "anchor" in comments:
                params["anchor"] = comments["anchor"]
            time.sleep(1)
        """for comment in total_comments:
            reply = comment.pop("reply_to_comment_id", None)
            comment["reply_to_comment_id"] = reply[:-1] if reply else None

            reply = comment.pop("reply_to_id", None)
            comment["reply_to_id"] = reply"""
        return total_comments


    def users_get_info(self, id):
        params = dict()
        params["method"] = "users.getInfo"
        params["uids"] = id
        params["fields"] = "age,allows_anonym_access,location,gender,name"
        params.update(self.base_params)
        usr = self.generate_request(params)[0]

        gender = usr.pop("gender", None)
        if gender == "male":
            usr["gender"] = 2
        elif gender == "female":
            usr["gender"] = 1
        else:
            usr["gender"] = 0

        age = usr.pop("age", None)
        usr["age"] = age

        city = usr.pop("city", None)
        usr["city"] = city

        access = usr.pop("allows_anonym_access", None)
        usr["friends"] = self.friends(id) if access else None
        return usr


    def is_member(self, group_id, usr_id):
        params = dict()
        params["method"] = "group.getUserGroupsByIds"
        params["group_id"] = group_id
        params["uids"] = usr_id
        params.update(self.base_params)
        membership = self.generate_request(params)
        if membership:
            if membership[0]["status"] == "ACTIVE" or \
                    membership[0]["status"] == "ADMIN" or \
                    membership[0]["status"] == "MODERATOR":
                return 1
        return 0


    def friends(self, id):
        params = dict()
        params["method"] = "friends.get"
        params["fid"] = id
        params.update(self.base_params)
        friends = self.generate_request(params)
        return len(friends)
