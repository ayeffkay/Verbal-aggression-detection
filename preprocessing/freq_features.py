"""Comment features based on frequencies"""

from preprocessing.load_db import Dataset
from preprocessing import PATH
import pandas as pd
import numpy as np
import re


class FreqFeatures:
    def __init__(self):
        self.dataset = Dataset()
        columns = ["comments_ct", "replies_to_usr", "usr_replies", "is_reply",
                   "is_member", "gender", "age", "friends", "norm_len", "likes",
                   "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                   "excl", "quest", "comma", "quotes", "uppercase",
                   "smiles", "emoji", "aggression"
                   ]
        self.df = pd.DataFrame(columns=columns)
        self.generate_features()
        self.df.to_pickle(PATH + "dataset_db/freq_features.pkl")

    @staticmethod
    def length(text):
        return len(text.replace(" ", ""))

    @staticmethod
    def symbols(text, s='аеёиоуыэюя!?,"«»'):
        res = list(map(text.lower().count, s))
        res[-3] += res[-2] + res[-1]
        return np.array(res[:-2])

    @staticmethod
    def smiles(text):
        smiles_pattern = re.compile(r"[:;=]?[-~]?[)D(]")
        return len(re.findall(smiles_pattern, text))

    @staticmethod
    def uppercase(text):
        uppercase_pattern = re.compile(r"[А-Я]")
        return len(re.findall(uppercase_pattern, text))

    def add_row(self, row):
        self.df.loc[-1] = row
        self.df.index = self.df.index + 1
        self.df = self.df.sort_index()


    def max_age(self):
        field = "MAX(AGE)"
        age = self.dataset.get_users(field)[field][0]
        return age

    def normalized_age(self, usr_age):
        if not pd.isnull(usr_age):
            age = self.max_age()
            return usr_age / age
        return 0

    def max_friends(self):
        field = "MAX(friends_count)"
        friends = self.dataset.get_users(field)[field][0]
        return friends

    def normalized_friends(self, usr_friends):
        if not pd.isnull(usr_friends):
            friends = self.max_friends()
            return usr_friends / friends
        return 0

    def get_users(self, fields="user_id, age, gender, friends_count"):
        users = self.dataset.get_users(fields)
        return users

    def usr_comments(self, usr_id, fields="id, comment_id, comment_text, "
                                          "emoji_count, is_member, aggression"):
        condition = "user_id = \"{}\"".format(usr_id)
        comments = self.dataset.get_comments(fields, condition)
        return comments

    def total_usr_comments(self, usr_id):
        field = "COUNT(*)"
        condition = "user_id = \"{}\"".format(usr_id)
        total = self.dataset.get_comments(field, condition)[field][0]
        return total


    def normalized_comments(self, id, usr_id):
        ct1 = self.usr_total_comments(id, usr_id)
        ct2 = self.total_usr_comments(usr_id)
        if ct1 and ct2:
            return ct1 / ct2
        return 0

    def replies_to_comment(self, id, usr_id, comment_id):
        field = "COUNT(reply_to_comment_id)"
        condition = "id = \"{}\" AND comment_id = \"{}\" AND user_id=\"{}\"".\
            format(id, comment_id, usr_id)
        ct = self.dataset.get_comments(field, condition)[field][0]
        return ct


    def replies_to_usr(self, id, usr_id):
        field = "COUNT(reply_to_id)"
        condition = "id = \"{}\" AND reply_to_id = \"{}\"".format(id, usr_id)
        ct = self.dataset.get_comments(field, condition)[field][0]
        return ct


    def normalized_replies_to_usr(self, id, usr_id, comment_id):
        ct1 = self.replies_to_comment(id, usr_id, comment_id)
        ct2 = self.replies_to_usr(id, usr_id)
        if ct1 and ct2:
            return ct1 / ct2
        return 0

    def usr_repl_in_disc(self, id, usr_id):
        field = "COUNT(*)"
        condition = "id = \"{}\" AND user_id = \"{}\" " \
                    "AND reply_to_comment_id IS NOT NULL".format(id, usr_id)
        ct = self.dataset.get_comments(field, condition)[field][0]
        return ct


    def usr_total_comments(self, id, usr_id):
        field = "COUNT(*)"
        condition = "id = \"{}\" AND user_id = \"{}\"".format(id, usr_id)
        ct = self.dataset.get_comments(field, condition)[field][0]
        return ct


    def normalized_usr_replies(self, id, usr_id):
        usr_repl = self.usr_repl_in_disc(id, usr_id)
        usr_total = self.usr_total_comments(id, usr_id)
        if usr_repl and usr_total:
            return usr_repl / usr_total
        return 0

    def is_reply(self, comment_id):
        field = "reply_to_comment_id"
        condition = "comment_id = \"{}\"".format(comment_id)
        res = self.dataset.get_comments(field, condition)[field][0]
        if pd.isnull(res):
            return 0
        return 1

    def sum_len(self, id, usr_id):
        field = "comment_text"
        condition = "id = \"{}\" AND user_id=\"{}\"".format(id, usr_id)
        texts = self.dataset.get_comments(field, condition)
        l = 0
        for ind, row in texts.iterrows():
            l += self.length(row[field])
        return l


    def normalized_len(self, text, id, usr_id):
        l1 = self.length(text)
        l2 = self.sum_len(id, usr_id)
        return l1 / l2

    def likes_ct(self, id, usr_id, comment_id):
        field = "likes"
        condition = "id = \"{}\" AND user_id = \"{}\" " \
                     "AND comment_id = \"{}\"".format(id, usr_id, comment_id)
        ct = self.dataset.get_comments(field, condition)[field][0]
        return ct


    def total_likes(self, id, usr_id):
        field = "SUM(likes)"
        condition = "id = \"{}\" AND user_id = \"{}\"".format(id, usr_id)
        ct = self.dataset.get_comments(field, condition)[field][0]
        return ct

    def normalized_likes(self, id, usr_id, comment_id):
        ct1 = self.likes_ct(id, usr_id, comment_id)
        ct2 = self.total_likes(id, usr_id)
        if not pd.isnull(ct1) and ct2:
            return ct1 / ct2
        return 0

    def generate_features(self):
        users = self.get_users()
        for index, row in users.iterrows():
            usr_id = row["user_id"]
            gender = row["gender"]
            age = self.normalized_age(row["age"])
            friends = self.normalized_friends(row["friends_count"])
            comments = self.usr_comments(usr_id)
            for index1, row1 in comments.iterrows():
                id = row1["id"]
                comment_id = row1["comment_id"]
                comments_ct = self.normalized_comments(id, usr_id)
                text = row1["comment_text"]
                repl_to_usr = self.normalized_replies_to_usr(id, usr_id, comment_id)
                usr_repl = self.normalized_usr_replies(id, usr_id)
                is_reply = self.is_reply(comment_id)
                norm_len = self.normalized_len(text, id, usr_id)
                norm_likes_ct = self.normalized_likes(id, usr_id, comment_id)
                sym = self.symbols(text) / self.length(text)
                upper = self.uppercase(text) / self.length(text)
                smiles = self.smiles(text) / self.length(text)
                emoji = row1["emoji_count"] / self.length(text)
                features = [comments_ct, repl_to_usr, usr_repl, is_reply,
                            row1["is_member"], gender, age, friends,
                            norm_len, norm_likes_ct] + sym.tolist() + \
                           [upper, smiles, emoji, row1["aggression"]]
                self.add_row(features)


features = FreqFeatures()
