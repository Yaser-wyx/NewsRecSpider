from time import sleep

import redis
import json
import time

# 获取redis数据库连接
rd = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)

LINE_INDEX = lambda name: name + "_" + "line_index"
LOG_FILEPATH = lambda name: name + "_" + "spider.log"
SPIDER_INFO = lambda name: name + "_" + "spider_info"


# 从上一次分析结束的位置开始继续分析日志
# 主要分析已经爬取了多少新闻
def get_logger(spider_name):
    line_index = int(rd.get(LINE_INDEX(spider_name)) or 0)  # 获取开始读取的位置
    print(line_index)
    with open(file=LOG_FILEPATH(spider_name), mode='r', encoding="utf-8") as file:
        lines_tmp = file.readlines()[line_index:] or []
    lines = []
    for line in lines_tmp:
        if len(line.strip()) > 0:
            lines.append(line)
    if len(lines) > 0:
        line_index += len(lines) + 1  # 设置下一次开始读取的位置
        rd.set(LINE_INDEX(spider_name), line_index)
    return lines


# 用于监视爬虫运行状态的
def spider_monitor(spider, spider_name, crawl_num):
    stop_spider = False
    while True:
        # 爬虫运行中
        # 获取日志并进行分析
        logger = get_logger(spider_name)
        spider_info = analyse_log(spider_name=spider_name, logger=logger)

        if spider_info:  # 如果有数据
            if not stop_spider and int(spider_info["saveCount"]) >= crawl_num:
                # 停止爬虫
                spider.stop()
                stop_spider = True

            rd.set(SPIDER_INFO(spider_name), json.dumps(spider_info))  # 设置spider信息

            if spider_info["spiderStatus"] == 2:
                # 退出
                break
        sleep(5)  # 休眠5秒



def analyse_log(spider_name, logger):
    if len(logger) == 0:
        return False
    spider_info = json.loads(rd.get(SPIDER_INFO(spider_name)))
    for line in logger:
        res = line.split("|")
        if res[-1].strip() == "news_save_suc":
            spider_info["saveCount"] += 1
            spider_info["crawlCount"] += 1
        elif res[-1].strip() == "news_exist":
            spider_info["repeatCount"] += 1
            spider_info["crawlCount"] += 1
        elif "Dumping Scrapy stats" in res[-1]:
            spider_info["spiderStatus"] = 2
            spider_info["endTime"] = int(time.time())

    return spider_info
