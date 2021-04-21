from mongoengine import connect, Q
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from flask import Flask, request
from response import *
from logUtils import *
import random
from multiprocessing import Process
import json
import threading
from modelHelper import *
from sina.models.userRecommenderList import UserRecommenderList

app = Flask(__name__)
rd = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)

"""spiderStatus:
                1 spider运行中
                2 spider运行结束"""


def start_spider(spider_name, crawl_num):
    # 清空日志数据
    with open(LOG_FILEPATH(spider_name), 'w') as f:
        f.write("")

    process = CrawlerProcess(get_project_settings())
    # 设置爬虫状态信息
    rd.set(LINE_INDEX(spider_name), 0)
    rd.set(SPIDER_INFO(spider_name),
           json.dumps({
               "spiderStatus": 1,
               "startTime": int(time.time()),
               "endTime": -1,
               "repeatCount": 0,
               "crawlCount": 0,
               "saveCount": 0
           }))
    # 启动子线程，定时查询爬虫信息
    spider_monitor_thread = threading.Thread(target=spider_monitor, args=(process, spider_name, crawl_num))

    spider_monitor_thread.start()
    # 获取爬虫
    process.crawl(spider_name)
    # 启动
    process.start()


@app.route("/runSpider")
def run_spider():
    spider_name = request.args.get("spiderName")
    crawl_num = int(request.args.get("crawlNum"))
    p = Process(target=start_spider, args=(spider_name, crawl_num))
    p.start()
    return success()


@app.route("/spiderStatus")
def get_spider_status():
    spider_name = request.args.get("spiderName")
    spider_info = json.loads(rd.get(SPIDER_INFO(spider_name)))
    return success(spider_info)


@app.route("/setNewsEmbedding", methods=["POST"])
def set_news_embedding():
    titles_obj = json.loads(request.get_data(), encoding="utf-8")
    titles = []
    for title_obj in titles_obj:
        titles.append(title_obj["title"])
    title_embedding = get_title_embedding(titles)
    for index in range(len(titles)):
        news_embedding = NewsEmbedding()
        news_embedding["doc_id"] = titles_obj[index]["id"]
        news_embedding["embedding"] = title_embedding[index]
        news_embedding["create_time"] = titles_obj[index]["createTime"]
        news_embedding.save()

    return success()


@app.route("/setUserRecommenderList", methods=["POST"])
def set_user_recommender_list():
    user_info = json.loads(request.get_data(), encoding="utf-8")
    user_interest_embedding = get_user_interest_embedding(user_info["labels"])
    user_history_embedding = get_user_history_embedding(user_info["history"])

    news_recommender_list = cal_news_score(user_interest_embedding=user_interest_embedding,
                                           user_history_embedding=user_history_embedding,
                                           user_history=user_info["history"])
    # 将推荐列表存入数据库中
    user_recommender_list = UserRecommenderList()
    user_recommender_list["user_id"] = int(user_info["userId"])
    user_recommender_list["recommender_list"] = news_recommender_list
    user_recommender_list.save()
    return success()


@app.route("/getSimilarNewsList")
def get_similar_news_list():
    doc_id = request.args.get("docId")
    key = doc_id + "_similar"
    if rd.exists(key) > 0:
        print("存在类似")
        similar_news_list = json.loads(rd.get(key))
    else:
        print("不存在类似")
        similar_news_list = cal_similar_news(doc_id)
        rd.set(key, json.dumps(similar_news_list), ex=60 * 24)
    res = []
    idx = random.sample(range(1, 90), 5)
    for index in idx:
        res.append(similar_news_list[index])
    return success({"similarNews": res})


if __name__ == "__main__":
    model_init()
    connect("news_recommender")
    app.run()
