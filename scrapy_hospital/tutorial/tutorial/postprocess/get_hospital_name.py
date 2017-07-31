# coding=utf-8
import json
import os
import sys

yiyuan = u"医院"
for line in sys.stdin.readlines():
    d = json.loads(line.strip().decode("utf-8"))
    title = d.get("title", "")
    res = title.find(yiyuan)
    #print res
    if (-1 == res):
        #print("-1")
        print(title.encode("utf-8"))
    if (res != len(title) - len(yiyuan)):
        first_yiyuan = title.find(yiyuan)
        second_yiyuan = title[title.find(yiyuan) + len(yiyuan):].find(yiyuan)
        if (-1 == second_yiyuan):
            print(title.encode("utf-8"))
            continue
        print(title[: title.find(yiyuan) + len(yiyuan)].encode("utf-8"))
        print(title[first_yiyuan + len(yiyuan): first_yiyuan + second_yiyuan + 2 * len(yiyuan)].encode("utf-8"))
    else:
        print(title.encode("utf-8"))

    #print(d.get("title", "").encode("utf-8"))
