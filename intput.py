# -*- coding: utf-8 -*-
# 使用內建的 urllib.request 裡的 urlopen 這個功能來送出網址
import random
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle
import pandas as pd
from styleframe import StyleFrame, Styler
import operator as op
from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import requests

# 臺東
tung = '%E5%8F%B0%E6%9D%B1%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 花蓮
hua = '%E8%8A%B1%E8%93%AE%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 彰化
chua = '%E5%BD%B0%E5%8C%96%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 苗栗
mia = '%E8%8B%97%E6%A0%97%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 嘉義市
chiaa = '%E5%98%89%E7%BE%A9%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 嘉義縣
chia = '%E5%98%89%E7%BE%A9%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 新竹市
shnn = '%E6%96%B0%E7%AB%B9%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 新竹縣
shn = '%E6%96%B0%E7%AB%B9%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 桃園
tao = '%E6%A1%83%E5%9C%92%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 基隆
ki = '%E5%9F%BA%E9%9A%86%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 臺北
taip = '%E5%8F%B0%E5%8C%97%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 新北
newp = '%E6%96%B0%E5%8C%97%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 臺中
taic = '%E5%8F%B0%E4%B8%AD%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 南投
to = '%E5%8D%97%E6%8A%95%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 屏東
pin = '%E5%B1%8F%E6%9D%B1%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 雲林
yun = '%E9%9B%B2%E6%9E%97%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 高雄
kao = '%E9%AB%98%E9%9B%84%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 宜蘭
il = '%E5%AE%9C%E8%98%AD%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 臺南
nan = '%E5%8F%B0%E5%8D%97%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 連江
lian ='%E9%80%A3%E6%B1%9F%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 澎湖
pon = '%E6%BE%8E%E6%B9%96%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
# 金門
mon = '%E9%87%91%E9%96%80%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
no = tung
response = urlopen('https://vegemap.merit-times.com/restaurant_list/'+no)
#使用 beautiful soup 解析網站回傳的 html response
html = BeautifulSoup(response.read(),'lxml')
divs = html.find_all(class_='B_item_productlist typeA')
df01 = pd.DataFrame(columns=["店名","電話",'初始評分','素食種類','可配合多少素食種類加權分','休息時間','全勤加權分','可手機聯絡加權分','地址'])
for div in divs:
    relax = div.find("div",class_='B_item_info Bbox_table')
    rela = relax.find_all("div")
    rel = rela[-1].get_text()
    '''
    種類
    '''
    text = div.find("div", class_='proList_itemAbbr')
    ts = text.find_all('span')
    nam = div.find('h3').get_text()
    print('店名： '+nam)
    '''
    初始評分與計算
    '''
    star = div.find("div", itemprop="aggregateRating").attrs['class'][1]
    sart = 0
    if(star[-1]=='0'):
        sart=1
    elif(star[-1]=='1'):
        sart=1.2
    else:
        sart = int(star[-1])
    '''
    手機加權分與計算************************************************************
    '''
    pho = div.find("span", itemprop="telephone").get_text()
    pho = pho.strip('\t')
    w1 = 0
    if(pho[0:2]=='09'or'&'in pho):
        w1 = 1.2
        print('手機： '+pho)
    else:
        w1 = 1
        print('市話： '+pho)
    print('可手機聯絡加權分： '+str(w1))
    '''
    可配合多少素食種類加權分
    '''
    count = 0
    tex=[]
    for t in ts:
        count = count+1
        tex.append(t.get_text())
        if(op.contains(t.get_text(), "五辛")):
            count=count-1.5
    vetype = ' '.join([str(elem) for elem in tex])
    if(count==0):
        count=0.9
    print("可配合： "+vetype)
    print('可配合多少素食類別加權分： '+ str(count))
    '''
    全勤加權分
    '''
    hou = op.contains(rel, "5:")
    mond = op.contains(rel, "一")
    tue = op.contains(rel, "二")
    wed = op.contains(rel, "三")
    thu = op.contains(rel, "四")
    fri = op.contains(rel, "五")
    sat = op.contains(rel, "六")
    sun = op.contains(rel, "日")
    sunn = op.contains(rel, "天")
    w2 = 2
    xre = []
    if(op.contains(rel, "無"or'無休')or hou):
       w2 = 2
       #print('休息日： '+rel)
       xre.append(''+rel)
    elif(mond or tue or wed or thu or fri or sat or sun or sunn): 
       w2 = w2-0.5
       #print('休息日： '+rel)
       xre.append(''+rel)
    else:
       #print('休息日：沒特別說'+rel)
       xre.append('沒特別說 '+rel)
       w2 = 1
    relat = ' '.join([str(elem) for elem in xre])
    #print(relat)
    #print('全勤加權分： '+str(w2))
    '''
    地址
    '''
    addr = div.find("span", itemprop="address").get_text()
    print('地址： '+addr+'\n')
    '''
    增加至清單
    '''
    s01 = pd.Series([nam,pho,sart,vetype,count,relat,w2,w1,addr], index=['店名','電話','初始評分','素食種類','可配合多少素食種類加權分','休息時間','全勤加權分','可手機聯絡加權分','地址'])
    df01 = df01.append(s01, ignore_index=True)
    df01.index = df01.index+1
#print(df01.to_string())
sf = StyleFrame(df01)
sf.set_column_width_dict(col_width_dict={
    ("店名"): 30,
    ("電話"): 27,
    ("初始評分"): 12,
    ("素食種類"): 26,
    ("可配合多少素食種類加權分"): 23,
    ("休息時間"): 26,
    ("全勤加權分"): 13,
    ("地址"): 28
    })
sf.to_excel('log.xlsx').save()
