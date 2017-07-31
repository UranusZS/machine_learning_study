import sys
import os
import json



def get_item(s, item):
    if (-1 == s.find(item)):
        return ""
    s = s[s.find(item):]
    s = s[s.find("=\"") + 2:]
    s = s[:s.find("\"")]
    return s

def extract_kv(s):
    k_begin = s.find("<li><b>")
    k_end = s.find("</b>")
    k = s[k_begin + len("<li><b>"): k_end]
    s = s[k_end:]
    v_begin = s.find("</b>")
    s = s[v_begin:]
    v_end = s.find("</li>")
    v = s[v_begin + len("</b>") + 3: v_end]
    if (-1 < v.find("href")):
        v = get_item(v, "href")
    return k, v


def extract_hospital_info(filename):
    find_first = 0
    hospital_dict = {}
    with open(filename) as fp:
        line = fp.readline()
        while line:
            try:
                tmp_dict = {}
                while (0 == find_first) and line:
                    res = line.strip().find("<li><b><a")
                    if (-1 == res):
                        line = fp.readline()
                    else:
                        find_first = 1
                # get hospital name
                if not line:
                    line = fp.readline()
                    break
                href = get_item(line.strip(), "href")
                title = get_item(line.strip(), "title")
                tmp_dict["href"] = href
                tmp_dict["title"] = title
                tmp_dict["info_detail"] = line.strip()
                line = fp.readline()
                if not line:
                    line = fp.readline()
                    break
                while (-1 == line.strip().find("<li><b><a")) and line:
                    #line = fp.readline()
                    if (-1 == line.strip().find("<li><b>")):
                        line = fp.readline()
                        continue
                    key, value = extract_kv(line.strip()) 
                    tmp_dict[key] = value
                    line = fp.readline()
                hospital_dict[title] = tmp_dict
            except:
                pass
        return hospital_dict


root_dir = "./data/"
hospital_info_dict = {}
for parent, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if (-1 == filename.find("list")):
            continue
        hospital_dict = extract_hospital_info(root_dir + filename)
        for (key, value) in hospital_dict.items():
            if key not in hospital_info_dict:
                hospital_info_dict[key] = value
            else:
                for (k, v) in value.items():
                    hospital_info_dict[key][k] = v

for (k, v) in hospital_info_dict.items():
    print(json.dumps(v, ensure_ascii=False))
