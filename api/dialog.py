"""
Dialog with user (for corpus labeling)
"""
import api
from sys import stdin
from textwrap import wrap



def display_group(group_name):
    print("Название сообщества:", group_name)
    print("Посмотреть топики?(enter)")
    answ = stdin.readline().rstrip()
    if len(answ):
        return False
    return True


def display_topic(topic_text, comments_count):
    print("Текст топика:")
    splitted = wrap(topic_text, api.wrap_len)
    for part in splitted:
        print(part)
    print("Количество комментариев:", comments_count)
    print("Посмотреть комментарии?(enter)")
    answ = stdin.readline().rstrip()
    if len(answ):
        return False
    return True


def display_comment(comment_text):
    print("Текст комментария:")
    splitted = wrap(comment_text, api.wrap_len)
    for part in splitted:
        print(part)
    print("Агрессивный(1), неагрессивный(0), пропустить-enter")
    answ = stdin.readline().rstrip()
    if not len(answ):
        return 3
    return int(answ)


def ask_exit(comments_count):
    print("Осталось комментариев:", comments_count)
    print("Выйти из дискуссии?(клавиша кроме enter)")
    answ = stdin.readline().rstrip()
    if not len(answ):
        return False
    return True
