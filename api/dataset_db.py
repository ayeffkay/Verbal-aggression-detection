"""
For labeled comments
"""

from mysql.connector import (connection)
from mysql.connector.errors import Error
import api


class Database:
    def __init__(self):
        self.cnct = connection.MySQLConnection(user=api.user,
                                               password=api.password,
                                               host=api.host,
                                               database=api.dataset)
        self.cursor = self.cnct.cursor(buffered=True)

        # number of gathered messages
        self.total_count = dict.fromkeys(['aggressive', 'nonaggressive', 'total'], 0)

        # aggressive comments
        self.cursor.execute("SELECT COUNT(aggression) FROM comments where aggression=1;")
        self.total_count['aggressive'] = self.cursor.fetchone()[0]

        # nonagressive comments
        self.cursor.execute("SELECT COUNT(aggression) FROM comments where aggression=0;")
        self.total_count['nonaggressive'] = self.cursor.fetchone()[0]

        # total comments number
        self.cursor.execute("SELECT COUNT(*) FROM comments;")
        self.total_count['total'] = self.total_count["aggressive"] + \
                                    self.total_count["nonaggressive"]
        print(self.total_count)

    def get_group_id(self, group_id):
        q = "SELECT group_name FROM groups WHERE group_id = \'{}\';".format(group_id)
        try:
            self.cursor.execute(q)
            if self.cursor.rowcount:
                return True
        except Error as e:
            print(e)
            return e
        return None

    def add_group(self, group_id, group_name, description, members_count):
        res = self.get_group_id(group_id)
        if not res:
            try:
                self.cursor.execute("INSERT INTO groups VALUES(%s, %s, %s, %s);",
                                    (group_id, api.clear(group_name),
                                     api.clear(description), members_count,))
                self.cnct.commit()
            except Error as e:
                print(e)
                return False
        return True

    def get_group_topic_id(self, group_id, topic_id):
        q = "SELECT id FROM topics WHERE topic_id = \'{}\' AND group_id = \'{}\'".\
            format(topic_id, group_id)
        self.cursor.execute(q)
        if self.cursor.rowcount:
            return self.cursor.fetchone()[0]
        return None


    def add_topic(self, group_id, topic_id, topic_text, likes, reposts, total_ct, date):
        id = self.get_group_topic_id(group_id, topic_id)
        if not id:
            try:
                self.cursor.execute("INSERT INTO topics VALUES(0, %s, %s, %s, %s, %s, %s, %s);",
                                    (group_id, topic_id, api.clear(topic_text),
                                     likes, reposts, total_ct, date))
                self.cnct.commit()
            except Error as e:
                print(e)
                return False
        return True

    def get_user_id(self, user_id):
        q = "SELECT user_name FROM users WHERE user_id = \'{}\';".format(user_id)
        self.cursor.execute(q)
        if self.cursor.rowcount:
            return True
        return False

    def add_user(self, user_id, user_name, gender, age, city, friends_count):
        res = self.get_user_id(user_id)
        if not res:
            try:
                self.cursor.execute("INSERT INTO users VALUES(%s, %s, %s, %s, %s, %s);",
                                    (user_id, api.clear(user_name), gender,
                                     age, city, friends_count))
                self.cnct.commit()
            except Error as e:
                print(e)
                return False
        return True

    def get_comment_id(self, id, user_id, comment_id):
        q = "SELECT comment_text FROM comments WHERE id = \"{}\" AND " \
            "user_id = \"{}\" AND comment_id=\"{}\";".format(id, user_id, comment_id)
        self.cursor.execute(q)
        if self.cursor.rowcount:
            return True
        return False

    def add_comment(self, group_id, topic_id, comment_id, user_id, is_member, text, emoji,
                    likes, reply_comment, reply_id, date, aggression):
        id = self.get_group_topic_id(group_id, topic_id)
        res = self.get_comment_id(id, user_id, comment_id)
        if not res:
            try:
                self.cursor.execute("INSERT INTO comments VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
                                    (id, comment_id, user_id, is_member, api.clear(text), emoji, likes,
                                     reply_comment, reply_id, date, aggression, ))
                self.cnct.commit()
                if aggression == 0:
                    self.total_count["nonaggressive"] += 1
                elif aggression == 1:
                    self.total_count["aggressive"] += 1
                self.total_count["total"] += 1
                print(self.total_count)
            except Error as e:
                print(e)
                return False
        return True
