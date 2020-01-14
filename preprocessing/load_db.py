import pandas as pd
from sqlalchemy import create_engine
from preprocessing import params1, params2


class Dataset:
    def __init__(self):
        self.engine = create_engine(params1, echo=False)
        self.texts, self.labels = self.get_texts()

    def get_texts(self):
        q = "SELECT comment_text, aggression FROM comments"
        comments = pd.read_sql_query(q, self.engine)
        return comments["comment_text"].get_values(), comments["aggression"].get_values()

    def get_users(self, fields, condition=None):
        if condition:
            q = "SELECT {} FROM users WHERE {}".format(fields, condition)
        else:
            q = "SELECT {} FROM users".format(fields)
        users = pd.read_sql_query(q, self.engine)
        return users

    def get_comments(self, fields, condition=None):
        if condition:
            q = "SELECT {} FROM comments WHERE {}".format(fields, condition)
        else:
            q = "SELECT {} FROM comments".format(fields)
        comments = pd.read_sql_query(q, self.engine)
        return comments


class Corpus:
    def __init__(self):
        self.engine = create_engine(params2, echo=False)
        self.texts = self.get_texts()
        self.labels = None

    def get_texts(self):
        q = "SELECT comment FROM comments"
        comments = pd.read_sql_query(q, self.engine)
        return comments["comment"].get_values()
