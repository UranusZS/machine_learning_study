# -*- coding: utf-8 -*-
import scrapy
import os
import urllib
import sys
import re
import time
import json
reload(sys)
sys.setdefaultencoding("utf-8")

def parse_hospital(s):
    '''
    /w/%E5%B9%BF%E5%B7%9E%E5%A4%A7%E8%82%BF%E7%98%A4%E5%8C%BB%E9%99%A2" title="\xe5\xb9\xbf\xe5\xb7\x9e\xe5\xa4\xa7\xe8\x82\xbf\xe7\x98\xa4\xe5\x8c\xbb\xe9\x99\xa2">\xe5\xb9\xbf\xe5\xb7\x9e\xe5\xa4\xa7\xe8\x82\xbf\xe7\x98\xa4\xe5\x8c\xbb\xe9\x99\xa2</a></b>
    '''
    url = s[:s.find('"')]
    s = s[s.find('"'):]
    title = s[s.find('">')+2:s.find('</a>')]
    s = title
    if (-1 < s.find('</a>')):
        title = s[:s.find('</a>')]
    return url, title


def merge_info(hospital_list, info_list):
    merge_result = []
    if (len(hospital_list) != len(info_list)):
        return merge_result
    for i in range(len(hospital_list)):
        tmp_dict = {}
        h_i = hospital_list[i]
        i_i = info_list[i]
        tmp_dict[h_i[1]] = {}
        tmp_dict[h_i[1]]["name"] = h_i[1]
        tmp_dict[h_i[1]]["url"] = h_i[0]
        for (key, value) in i_i.items():
            tmp_dict[h_i[1]][key] = value
        merge_result.append(tmp_dict)
    return merge_result

def write_result(fp, res_list):
    for item in res_list:
        try:
            json_str = json.dumps(item)
            fp.write(json_str + "\n")
        except:
            pass

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        #'http://www.a-hospital.com/w/%E5%85%A8%E5%9B%BD%E5%8C%BB%E9%99%A2%E5%88%97%E8%A1%A8',
        #'http://www.yixue.com/%E5%85%A8%E5%9B%BD%E5%8C%BB%E9%99%A2%E5%88%97%E8%A1%A8'
        'http://www.a-hospital.com/w/%E5%85%A8%E5%9B%BD%E5%8C%BB%E9%99%A2%E5%88%97%E8%A1%A8'
    ]

    liebiao = u"列表"
    quanguo_liebiao = u"全国医院列表"
    index = 0

    ###############################################################################################################
    def parse_list_test(self, response):
        #time.sleep(1)
        print(1111111)
        current_url = response.url
        body = response.body
        unicode_body = response.body_as_unicode()
        print(current_url)
        self.log("list current_url-> " + current_url)
        print(current_url)
        hospital_str_list = re.findall(r'<li><b><a href="(.*)', response.body)
        hospital_list = []
        for item in hospital_str_list:
            try:
                url, name = parse_hospital(item)
                self.log("get_hospital_name_info href-> " + url + " title-> " + name)
                hospital_list.append([url, name])
            except:
                pass
        info_list = []
        for ul in response.css("ul"):
            tmp_dict = {}
            is_pass = 0
            try:
                for li in ul.css("li"):
                    t_i = li.css("::text").extract()
                    if (2 != len(t_i)):
                        is_pass = 1
                        break
                    tmp_dict[t_i[0]] = t_i[1]
                if (0 == is_pass):
                    info_list.append(tmp_dict)
            except:
                pass
        if (len(info_list) != len(hospital_list)):
            self.log("len_info_list_not_equal_to_len_hospital_list in url-> " + current_url)
        else:
            merge_result = merge_info(hospital_list, info_list)
        try:
            url_arr = current_url.split("/")
            hos_name = urllib.unquote(url_arr[-1])
            self.index += 1
            with open("data/" + str(self.index) + ".html", "wb") as f:
                f.write(body)
        except:
            pass
        for quote in response.css('p'):
            try:
                for link in quote.css('a'):
                    href = link.xpath("@href").extract_first()
                    title = link.xpath("@title").extract_first()
                    if (href is not None) and (-1 < href.find("w")) and (title is not None) and \
                            (-1 < title.find(self.liebiao)) \
                            and (-1 == title.find(self.quanguo_liebiao)) and (-1 == title.find(hos_name)):
                        self.log("list current: href-> " + href + " title-> " + title)
                        yield response.follow(href, callback=self.parse_list)
            except:
                pass

    ###############################################################################################################
    def parse(self, response):
        current_url = response.url
        body = response.body
        unicode_body = response.body_as_unicode()
        self.log("current_url-> " + current_url)
        #print("unicode_body-> " + unicode_body)
        #print(response.body)
        #file_name = "quotes.html"
        #with open(file_name, "wb") as f:
        #    f.write(response.body)
        #response.css('li a::attr(href)').extract_first():
        for quote in response.css('li'):
            quote_href = quote.css('a::attr(href)')
            quote_title = quote.css('a::attr(title)')
            href = quote_href.extract_first()
            title = quote_title.extract_first()
            try:
                print("in_parse href-> " + quote_href.extract_first() + " title-> " + quote_title.extract_first())
            except:
                href_none = 0
                title_none = 0
                if (quote_href.extract_first() is None):
                    href_none = 1
                if (quote_title.extract_first() is None):
                    title_none = 1
                print("in_parse href_none[%d] title_none[%d]" % (href_none, title_none))
                pass
            if (href is not None) and (-1 < href.find("w")) and (title is not None) and (-1 < title.find(self.liebiao)):
                try:
                    self.log("in_parse_next href-> " + href + " title-> " + title)
                except:
                    pass
                yield response.follow(href, callback=self.parse_list)
                
        for quote in response.css('p'):
            quote_href = quote.css('a::attr(href)')
            quote_title = quote.css('a::attr(title)')
            href = quote_href.extract_first()
            title = quote_title.extract_first()
            try:
                print("in_parse href-> " + quote_href.extract_first() + " title-> " + quote_title.extract_first())
            except:
                href_none = 0
                title_none = 0
                if (quote_href.extract_first() is None):
                    href_none = 1
                if (quote_title.extract_first() is None):
                    title_none = 1
                print("in_parse href_none[%d] title_none[%d]" % (href_none, title_none))
                pass
            if (href is not None) and (-1 < href.find("w")) and (title is not None) and (-1 < title.find(self.liebiao)):
                try:
                    self.log("in_parse_next href-> " + href + " title-> " + title)
                except:
                    pass
                yield response.follow(href, callback=self.parse_list)

    ###############################################################################################################
    def get_hospital_list(self, response):
        hospital_str_list = re.findall(r'<li><b><a href="(.*)', response.body)
        hospital_list = []
        for item in hospital_str_list:
            try:
                url, name = parse_hospital(item)
                self.log("get_hospital_name_info href-> " + url + " title-> " + name)
                hospital_list.append([url, name])
            except:
                pass
        return hospital_list

    def get_info_list(self, response):
        info_list = []
        for ul in response.css("ul"):
            tmp_dict = {}
            is_pass = 0
            try:
                for li in ul.css("li"):
                    t_i = li.css("::text").extract()
                    if (2 != len(t_i)):
                        is_pass = 1
                        break
                    tmp_dict[t_i[0]] = t_i[1]
                if (0 == is_pass):
                    info_list.append(tmp_dict)
            except:
                pass
        return info_list

    def parse_list(self, response):
        #time.sleep(1)
        current_url = response.url
        body = response.body
        unicode_body = response.body_as_unicode()
        self.log("list current_url-> " + current_url)
        hospital_list = self.get_hospital_list(response)
        info_list = self.get_info_list(response)
        merge_result = []
        try:
            if (len(info_list) != len(hospital_list)):
                self.log("in_parse_list len_info_list_not_equal_to_len_hospital_list in url[%s] hospital_len[%s] info_len[%s]" % (current_url, str(len(hospital_list)), str(len(info_list))))
            else:
                self.log("in_parse_list len_info_list_equal_to_len_hospital_list in url[%s] hospital_len[%s] info_len[%s]" % (current_url, str(len(hospital_list), str(len(info_list)))))
                merge_result = merge_info(hospital_list, info_list)
        except:
            pass
        try:
            url_arr = current_url.split("/")
            hos_name = urllib.unquote(url_arr[-1])
            self.index += 1
            with open("data/list" + str(self.index) + ".html", "wb") as f:
                f.write(body)
        except:
            pass
        for quote in response.css('p'):
            try:
                for link in quote.css('a'):
                    href = link.xpath("@href").extract_first()
                    title = link.xpath("@title").extract_first()
                    if (href is not None) and (-1 < href.find("w")) and (title is not None) and \
                            (-1 < title.find(self.liebiao)) \
                            and (-1 == title.find(self.quanguo_liebiao)) and (-1 == title.find(hos_name)):
                        self.log("in_parse_list_next list current: href-> " + href + " title-> " + title)
                        yield response.follow(href, callback=self.parse_list)
            except:
                pass
        
                
    ###############################################################################################################
    download_delay = 10
    
    user_agent_list = [\
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"\
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",\
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",\
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",\
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",\
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",\
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",\
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",\
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",\
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",\
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",\
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",\
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",\
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",\
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",\
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",\
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",\
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
       ]

