"""
Unlabeled corpus gathering (only for w2v learning)
"""
from mysql.connector import (connection)
import api


class Database:
    def __init__(self):
        self.cnct = connection.MySQLConnection(user=api.user,
                                       password=api.password,
                                       host=api.host,
                                       database=api.corpus)
        self.cursor = self.cnct.cursor(buffered=True)

    def add_comment(self, text):
        text = api.clear(text)
        q = "INSERT INTO comments VALUES({}, \"{}\")".format(0, text)
        try:
            self.cursor.execute(q)
            self.cnct.commit()
        except:
            pass
