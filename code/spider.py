from selenium import webdriver
from lxml import etree
from urllib import parse
from time import sleep
import datetime
from xlutils.copy import copy
import xlrd
import time
import re
from selenium.webdriver.chrome.service import Service
keyword = '日常'  #关键词
y = 2024  # 起始年
m = 1  # 起始月
d = 1  # 起始日
days = 1000  # 爬days天
url_keyword = parse.quote(keyword)  # 将关键词转换成为网址可识别

option = webdriver.ChromeOptions()
option.add_argument("headless")

def getday(y, m, d, n):  # 封装日期
    the_date = datetime.datetime(y, m, d)
    result_date = the_date + datetime.timedelta(days=n)
    d = result_date.strftime('%Y-%m-%d')
    return d


def p(days, x):  # 爬取解析存储

    for i in range(days):
        data = getday(y, m, d, +i)

        for j in range(24):  # 获取24小时的网址
            if j == 23:
                data_add_hour = data + '-' + str(j) + ':' + getday(y, m, d, -(i - 1)) + '-' + str(0)
            else:
                data_add_hour = data + '-' + str(j) + ':' + data + '-' + str(j + 1)
            # selenium
            service = Service(r"E:\chromedriver-win64\chromedriver.exe")
            bro = webdriver.Chrome(service=service,options=option)
            url = 'https://s.weibo.com/weibo?q=' + url_keyword + '&typeall=1&suball=1&timescope=custom:' + data_add_hour
            print(url)
            bro.get(url)
            sleep(1)  # 等待完整加载
            page_text = bro.page_source  # 完整页面
            sleep(1)
            bro.quit()  # 关闭网页
            # 开始解析
            tree = etree.HTML(page_text)
            wb_list = tree.xpath("//div[@class='card-feed']")
            for li in wb_list:

                wb_time = li.xpath("./div[2]/p[3]/a[1]/text()|./div[2]/p[2]/a[1]/text()")
                time_re = '[0-9][0-9]月[0-9][0-9]日 [0-9][0-9]:[0-9][0-9]'
                rst = re.compile(time_re).findall(str(wb_time))

                wb_name = li.xpath("./div[2]/div[1]/div[2]/a[1]/text()")
                print(wb_name)
                wb_text = li.xpath("./div[2]/p[1]//text()")
                print(wb_text)
                wb_from = li.xpath("./div[2]/p[@class='from']/a[2]/text()")
                wb_href = li.xpath("./div[2]/p[@class='from']/a[1]/@href")
                print(wb_href)
                rb = xlrd.open_workbook('wb_py.xls')  # 打开文件

                wb = copy(rb)  # 利用xlutils.copy下的copy函数复制
                ws = wb.get_sheet(0)  # 获取表单0

                ws.write(x, 1, wb_name)

                ws.write(x, 2, wb_href)

                ws.write(x, 3, wb_text)

                ws.write(x, 4, wb_time)
                # print(wb_time)
                ws.write(x, 5, wb_from)

                x = x + 1
                print(x)
                wb.save('wb_py.xls')  # 保存文件

if __name__ == '__main__':
    p(days, 1)


