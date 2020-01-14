import re

vk_token = ""

ok_app_key = ""
ok_session_key = ""
ok_token = ""

user = "root"
password = ""
host = "127.0.0.1"
dataset = "dataset"
# for word2vec learning
corpus = "corpus"
dialect = "mysql+mysqlconnector"
port = 3306
# len for wrapping message
wrap_len = 100
# smiles pattern
emoji = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]+|"
                                u"[\U0001F600-\U0001F64F"
                                u"\U0001F300-\U0001F5FF"
                                u"\U0001F680-\U0001F6FF"
                                u"\U0001F1E0-\U0001F1FF"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251]+", flags=re.UNICODE)
base_pattern = re.compile("(^\[.*\], )|<br>|\n+|#|\(с\)|\(c\)|\d+|\([a-zA-z]+\)|"
                                  "(Сообщение содержит прикрепленные файлы \(смотрите в полной версии сайта\))$")
# hypperref replacement pattern
url = re.compile("http.*")


def emoji_count(comment):
    return len(re.findall(emoji, comment))


# message clearing
def clear(text):
    # replace hyperref by url
    text = re.sub(url, "url", text)
    # removing smiles
    return re.sub(emoji, " ",
                  re.sub(base_pattern, " ", text))
