#!/usr/bin/python
# -*- coding: <encoding name> -*-
from selenium import webdriver
import requests as req
import time
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen

# 브라우저를 크롬으로 만들어주고 인스턴스를 생성해준다.
browser = webdriver.Chrome()
# 브라우저를 오픈할 때 시간간격을 준다.
browser.implicitly_wait(3)

count = 0
keywords = '화보'

photo_list = []
before_src = ""

# 개요에서 설명했다시피 google이 아니라 naver에서 긁어왔으며,
# 추가적으로 나는 1027x760이상의 고화질의 해상도가 필요해서 아래와 같이 추가적인 옵션이 달려있다.
path_keyword = "https://search.naver.com/search.naver?where=image&section=image&query=" + keywords + "&res_fr=786432&res_to=100000000&sm=tab_opt&face=0&color=0&ccl=0&nso=so%3Ar%2Ca%3Aall%2Cp%3Aall&datetype=0&startdate=0&enddate=0&start=1"

# 해당 경로로 브라우져를 오픈해준다.
browser.get(path_keyword)
time.sleep(1)

# 아래 태그의 출처는 사진에서 나오는 출처를 사용한것이다.
# 여기서 주의할 점은 find_element가 아니라 elements를 사용해서 아래 span태그의 img_border클래스를
# 모두 가져왔다.
src_list = []
for i in range(100):
    photo_list = browser.find_elements_by_tag_name("span.img_border")
    for index, img in enumerate(photo_list[count:]):
        # 위의 큰 이미지를 구하기 위해 위의 태그의 리스트를 하나씩 클릭한다.
        img.click()

        # 한번에 많은 접속을 하여 image를 크롤링하게 되면 naver, google서버에서 우리의 IP를 10~20분
        # 정도 차단을 하게된다. 때문에 Crawling하는 간격을 주어 IP 차단을 피하도록 장치를 넣어주었다.
        time.sleep(1)

        # 확대된 이미지의 정보는 img태그의 _image_source라는 class안에 담겨있다.
        html_objects = browser.find_element_by_tag_name('img._image_source')
        current_src = html_objects.get_attribute('src')
        print("=============================================================")
        print("현재 src :" + current_src)
        if current_src in src_list:
            continue
        else:
            t = urlopen(current_src).read()
            if count < 4000:
                filename = "./downloads/화보/img_" + str(count) + ".jpg"
                with open(filename, "wb") as f:
                    f.write(t)
                    count += 1
                print("Img Save Success")
                src_list.append(current_src)
            else:
                # browser.close()
                break
