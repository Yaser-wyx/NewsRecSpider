import random

from mongoengine import connect

import redis

from sina.models.channel_names import ChannelName
from sina.models.news import News
from mongoengine.context_managers import switch_db

rd = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)
connect("news_db", alias="source")
connect("news_recommender", alias="target")


def start_data_cleaning():
    # 进行数据清洗
    """数据清洗步骤:  1.读取数据库中所有flag为False的数据
                    2.将新闻的channel添加到channel表中，并统计channel数据
                    3.设置新闻的viewCount与commentTotal为随机数
                    4.将新闻加入到另一个数据库中，并对新闻设置flag为True"""
    news_list = News.objects(flag=False)
    channels = ChannelName.objects()
    channel_dict = {channel["channel_name"]: channel for channel in channels}
    for news in news_list:
        if 0 < len(news["channel_name"]) < 5:
            news["channel_name"] = news["channel_name"].replace("新浪", "")
        else:
            news["channel_name"] = "其它"
        news["view_count"] = random.randint(10, 1000)
        news["comment_total"] = random.randint(10, 1000)
        news["flag"] = True
        channel_dict["推荐"]["count"] += 1
        if channel_dict.get(news["channel_name"], None):
            channel_dict[news["channel_name"]]["count"] += 1
        else:
            channel_tmp = ChannelName()
            channel_tmp["count"] = 1
            channel_tmp["channel_name"] = news["channel_name"]
            channel_dict[news["channel_name"]] = channel_tmp
        # news.save()
    with switch_db(News, "target"):
        for news in news_list:
            news.save()
        for channel in channel_dict:
            print(channel)


if __name__ == '__main__':
    # news_list = News.objects(flag=False)
    # print(type(news_list))
    start_data_cleaning()
