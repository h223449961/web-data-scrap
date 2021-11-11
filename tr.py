# for python 3
from urllib.request import urlopen, urlretrieve
import numpy as np
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
import operator as op
from bs4 import BeautifulSoup
from styleframe import StyleFrame, Styler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
class NeuralNetwork():
    def __init__(self):
        # 随机数生成的种子随机数生成的种子
        np.random.seed(1)
        # 将权重转换为值为-1到1且平均值为0的3乘1矩阵
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

    # 定义signoid函数的导数
    def sigmoid(self, x):
        x = x.astype(float)
        return (1 / (1 + np.exp(-x)))
    # 计算sigmoid函数的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 训练
    def train(self, train_inputs, train_outputs,i): # 输入 输出 迭代次数
        # 训练模型在不断调整权重的同时做出准确预测
        for iteration in range(i):
            # 通过神经元提取训练数据
            output = self.think(train_inputs)
            # 反向传播错误率
            error = train_outputs - output
            # 进行权重调整
            adjustments = np.dot(train_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights = self.synaptic_weights + adjustments
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
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
root = tk.Tk()
root.geometry("1600x800")
radioValue = tk.IntVar()
exfile_frame = tk.Frame(root)
exfile_frame.pack(side=tk.TOP)
exfile_label = tk.Label(exfile_frame, text='快速點擊爬取',font=(24),fg = "#30aabc")
exfile_label.pack(side=tk.LEFT)
def validate():
    value = radioValue.get()
    global mode
    if (value == 1):
        mode = tung
    if( value == 2):
        mode = lian
    if( value == 3):
        mode = hua
    if (value == 4):
        mode = mia
    if( value == 5):
        mode = pon
    if( value == 6):
        mode = pin
    if (value == 7):
        mode = kao
    if( value == 8):
        mode = shn
    if( value == 9):
        mode = shnn
def oas():
    global mode
    response = urlopen('https://vegemap.merit-times.com/restaurant_list/'+mode)
    html = BeautifulSoup(response.read(),'lxml')
    divs = html.find_all(class_='B_item_productlist typeA')
    df01 = pd.DataFrame(columns=["店名","電話",'初始分','總分','素食種類','可配合素食種類加權分','休息時間','全勤加權分','可手機聯絡加權分','地址'])
    alsum = 0
    ave = 0
    for div in divs:
        nam = div.find('h3').get_text()
        print('店名： '+nam)
        relax = div.find("div",class_='B_item_info Bbox_table')
        rela = relax.find_all("div")
        rel = rela[-1].get_text()
        '''
        種類
        '''
        text = div.find("div", class_='proList_itemAbbr')
        ts = text.find_all('span')
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
        if(pho[0:2]=='09'or('&'in pho)or(','in pho)):
            w1 = 1.2
            print('手機： '+pho)
        else:
            w1 = 1
            print('市話： '+pho)
        count = 0
        tex=[]
        for t in ts:
            count = count+1
            tex.append(t.get_text())
        vetype = ' '.join([str(elem) for elem in tex])
        if(count==0):
            count=0.9
        if(op.contains(vetype, "五辛")):
            count=count-1.5
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
            xre.append(''+rel)
        elif(mond or tue or wed or thu or fri or sat or sun or sunn): 
           w2 = w2-0.5
           xre.append(''+rel)
        else:
           xre.append('沒特別說 '+rel)
           w2 = 1
        relat = ' '.join([str(elem) for elem in xre])
        addr = div.find("span", itemprop="address").get_text()
        '''
        列印區
        '''
        iw = float(w1*w2*count+sart)
        iw = round(iw,2)
        alsum = alsum+iw
        print('初始分： '+str(sart))
        print('可手機聯絡加權分： '+str(w1))
        print('休息時間： '+relat)
        print('全勤加權分： '+str(w2))
        print('可配合素食種類： '+vetype)
        print('可配合多少素食種類加權分： '+str(count))
        print('總分： '+str(iw))
        print('地址： '+addr+'\n')
        if(div==divs[-1]):
            ave = alsum/len(divs)
            print('平均分： '+str(ave))
        s01 = pd.Series([nam,pho,sart,iw,vetype,count,relat,w2,w1,addr], index=['店名','電話','初始分','總分','素食種類','可配合素食種類加權分','休息時間','全勤加權分','可手機聯絡加權分','地址'])
        df01 = df01.append(s01, ignore_index=True)
        df01.index = df01.index+1
    result = []
    for ind in df01.index:
        if(df01['總分'][ind]>ave):
            print(df01['店名'][ind], df01['總分'][ind],'推薦')
            result.append("推薦")
        else:
            print(df01['店名'][ind], df01['總分'][ind],'普通')
            result.append("普通")
    df01["公式計算"] = result
    dft = df01[['初始分','可手機聯絡加權分','可配合素食種類加權分','全勤加權分']]
    dftt = df01[['初始分','可手機聯絡加權分','可配合素食種類加權分','全勤加權分']]
    rab = []
    for ind in df01.index:
        if(df01['公式計算'][ind]=='推薦'):
            rab.append(1)
        else:
            rab.append(0)
    dft['rabel'] = rab
    num = dftt.to_numpy()
    numt = num.reshape(1,-1).T
    ranu = dft['rabel'].to_numpy()
    ranut = ranu.reshape(1,-1).T
    def softmax(x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x
    def clo(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    print(num)
    print(ranut)
    b = NeuralNetwork()
    b.train(num, ranut,5000)
    classif = softmax(b.think(num))
    print(classif)
    cround = np.round(classif, 2)
    cro = pd.DataFrame(cround,columns = ['機率'])
    print(cro)
    print('總機率： '+str(np.sum(cround)))
    print('最接近總機率的值： '+str(clo(cround,1)))
    flg = clo(cro,1)
    nlearn = []
    for ind in cro.index:
        if(cro['機率'][ind]==flg):
            nlearn.append('推薦')
        else:
            nlearn.append('普通')
    df01["神經網路"] = nlearn
    acoun = 0
    bcoun = 0
    altrue = 0
    preci = 0
    for i in df01.index:
        truflg = []
        if(df01['公式計算'][i]==df01['神經網路'][i]):
            acoun = acoun+1
        else:
            bcoun = bcoun+1
        if(df01['神經網路'][i]=='推薦'):
            altrue = altrue +1
        if(df01['公式計算'][i]=='推薦'):
            truflg = df01['公式計算'][i]
        if(truflg==df01['神經網路'][i]):
            preci= preci+1
    print((acoun/(acoun+bcoun))*100,(preci/altrue)*100)
    acpre_label.configure(text ='準確： '+str((acoun/(acoun+bcoun))*100)+' %')
    sf = StyleFrame(df01)
    sf.set_column_width_dict(col_width_dict={("店名"): 30,("電話"): 27,("初始分"): 7.39,("總分"): 7,("素食種類"): 26,("可配合素食種類加權分"): 23,("休息時間"): 26,("全勤加權分"): 13,("地址"): 28})
    sname = 'llogg.xlsx'
    output = sf.to_excel(sname).save()
    df = pd.read_excel(sname)
    df.fillna('官方沒有收集到', inplace=True)
    cols = list(df.columns)
    def treeview_sort_column(tv, col, reverse):
        try:
            l = [float((tv.set(k, col)), k) for k in tv.get_children('')]
        except:
            l = [(tv.set(k, col), k) for k in tv.get_children('')]
        l.sort(reverse=reverse)

    # rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

    # reverse sort next time
        tv.heading(col, command=lambda:
            treeview_sort_column(tv, col, not reverse))
    tree = ttk.Treeview(root)
    tree.pack()
    tree["columns"] = cols
    tree.column("#0", width=0, stretch=False)
    for i in cols:
        # 店名
        tree.column('# 1',width =118,anchor="center")
        # 電話
        tree.column('# 2',width =82,anchor="center")
        # 初始分
        tree.column('# 3',width =0,anchor="center")
        # 總分
        tree.column('# 4',width =0,anchor="center")
        # 素食種類
        tree.column('# 5',width =82,anchor="center")
        # 可配合素食加權分
        tree.column('# 6',width =49,anchor="center")
        # 休息時間
        tree.column('# 7',width =22,anchor="center")
        # 全勤加權分
        tree.column('# 8',width =3,anchor="center")
        # 可手機聯絡加權分
        tree.column('# 9',width =39,anchor="center")
        # 地址
        tree.column('# 10',width =139,anchor="center")
        # 推薦
        tree.column('# 11',width =0,anchor="center")
        # 推薦
        tree.column('# 12',width =10,anchor="center")
        tree.heading(i,text=i,anchor='center')
        tree.heading(i, text=i,command=lambda c=i: treeview_sort_column(tree, c, False))
    for index, row in df.iterrows():
        tree.insert("",'end',text = index,values=list(row))
    tree.place(relx=0,rely=0.4,relheight=0.5,relwidth=1)
r1 = tk.Radiobutton(root,text = "臺東",font=(24),variable=radioValue, value=1,command = validate).pack()
r2 = tk.Radiobutton(root,text = "連江",font=(24),variable=radioValue, value=2,command = validate).pack()
r3 = tk.Radiobutton(root,text = "花蓮",font=(24),variable=radioValue, value=3,command = validate).pack()
r4 = tk.Radiobutton(root,text = "苗栗",font=(24),variable=radioValue, value=4,command = validate).pack()
r5 = tk.Radiobutton(root,text = "澎湖",font=(24),variable=radioValue, value=5,command = validate).pack()
r6 = tk.Radiobutton(root,text = "屏東",font=(24),variable=radioValue, value=6,command = validate).pack()
r7 = tk.Radiobutton(root,text = "高雄",font=(24),variable=radioValue, value=7,command = validate).pack()
r8 = tk.Radiobutton(root,text = "新竹縣",font=(24),variable=radioValue, value=8,command = validate).pack()
r9 = tk.Radiobutton(root,text = "新竹市",font=(24),variable=radioValue, value=9,command = validate).pack()
def com(*args): # 處理事件， *args 表示可變引數  
    global mode
    mode = comboxlist.get()
    if(mode == '桃園'):
        mode = '%E6%A1%83%E5%9C%92%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '雲林'):
        mode = '%E9%9B%B2%E6%9E%97%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '臺南'):
        mode = '%E5%8F%B0%E5%8D%97%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '基隆'):
        mode = '%E5%9F%BA%E9%9A%86%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '臺北'):
        mode = '%E5%8F%B0%E5%8C%97%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '新北'):
        mode = '%E6%96%B0%E5%8C%97%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '臺中'):
        mode = '%E5%8F%B0%E4%B8%AD%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '嘉義縣'):
        mode = '%E5%98%89%E7%BE%A9%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '嘉義市'):
        mode = '%E5%98%89%E7%BE%A9%E5%B8%82-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '金門'):
        mode = '%E9%87%91%E9%96%80%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '宜蘭'):
        mode = '%E5%AE%9C%E8%98%AD%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '南投'):
        mode = '%E5%8D%97%E6%8A%95%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
    if(mode == '彰化'):
        mode = '%E5%BD%B0%E5%8C%96%E7%B8%A3-%E8%94%AC%E9%A3%9F%E9%A4%90%E5%BB%B3'
comvalue=tk.StringVar()
comboxlist=ttk.Combobox(root,textvariable=comvalue)
comboxlist["values"]=("桃園","雲林","臺南",'基隆','臺北','新北','臺中','嘉義縣','嘉義市','金門','宜蘭','南投','彰化')  
comboxlist.current(0)
comboxlist.bind("<<ComboboxSelected>>",com) # 繫結事件，（下拉列表框被選中時，繫結綁定的函式）  
comboxlist.pack()
acpre_frame = tk.Frame(root)
acpre_frame.pack(side=tk.TOP)
b1 = tk.Button(acpre_frame, text="爬取（若無特別設定將爬取上個暫存的縣市記錄）",font=(24),command = oas).pack(side=tk.LEFT)
acpre_label = tk.Label(acpre_frame, text='精準度',font=(24),fg = "#C13E43")
acpre_label.pack(side=tk.RIGHT)
root.mainloop()