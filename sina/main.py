from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from flask import Flask, request
from response import *
from logUtils import *
import time
from multiprocessing import Process
import json
import threading
from dataCleaningUtils import *

app = Flask(__name__)

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


# @app.route("/getSpiderStatus")
# def spider_status():
#     spider_name = request.args.get("spiderName")
#     if spider_name is not None:
#         spider_info = get_spider_info(spider_name)
#         if spider_info:
#             return success(spider_info)
#         else:
#             return error(1102, "{} 还没有运行！".format(spider_name))
#     else:
#         return error(1101, "缺少spider名称")


if __name__ == "__main__":
    app.run()
    # run_spider()
#     # start_spider("sina")
