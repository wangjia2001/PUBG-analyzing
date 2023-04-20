import requests
import urllib.request
# 导入正则匹配包
import re
import time
import random
import numpy as np
import sys


# -*- encoding:utf-8 -*-

#获得玩家姓名，用于得到网址
def geturl():
    html = 'https://pubg.op.gg/leaderboard/?platform=steam&mode=competitive-tpp&queue_size=1'
    # 该网址的源码(以该网页的原编码方式进行编码，特殊字符编译不能编码就设置ignore)
    webSourceCode = urllib.request.urlopen(html).read().decode("utf-8", "ignore")
    # print(webSourceCode)
    # 匹配数据的正则表达式
    Url_Re = re.compile(r'<a class="leader-board__nickname" data-link="leaderboard-link" href=".*?">(.*?)</a>')
    Url = Url_Re.findall(webSourceCode)
    #print(Url)
    #存储玩家姓名
    f = open("username.txt", "w")
    str = '\n'
    f.write(str.join(Url))
    f.close()
    return(Url)

#获取数据
def getdata(url):
    #url = 'https://pubg.op.gg/user/TheLuckiest_BF'
    webSourceCode = urllib.request.urlopen(url).read().decode("utf-8", "ignore")
    # print(webSourceCode)
    # 匹配数据的正则表达式
    #获取比赛场次id：match_id
    match_id_Re = re.compile(r'data-u-match_id="(.*?)="')
    match_id = match_id_Re.findall(webSourceCode)
    #print(match_id)

    # 获取该场比赛的全部击杀信息:death
    for i in range(0, len(match_id)):
        url = 'https://pubg.op.gg/api/matches/'+match_id[i]+'%3D/deaths'
        time.sleep(np.random.rand() * 5)
        try:
            death = urllib.request.urlopen(url).read().decode("utf-8", "ignore")
        except :
            print("error")
            continue
        with open('PUBG_death.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(death)  # 写入
            file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

    #获取该场比赛所有玩家表现：MatchData
    for i in range(0, len(match_id)):
        url = 'https://pubg.op.gg/api/matches/'+match_id[i]+'%3D'
        time.sleep(np.random.rand() * 5)
        try:
            MatchData = urllib.request.urlopen(url).read().decode("utf-8", "ignore")
        except:
            print("error")
            continue
        with open('PUBG_match_data.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write(MatchData)  # 写入
            file_handle.write('\n')

if __name__=='__main__':
    lst = geturl()
    for i in range(0, 100):
        getdata('https://pubg.op.gg/user/'+lst[i])