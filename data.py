import numpy as np
import cv2 as cv
import time
import datetime
import os
import matplotlib.pyplot as plt

colour=((0, 205, 205),(154, 250, 0),(34,34,178),(211, 0, 148),(255, 118, 72),(137, 137, 139))#定义矩形颜色

videoadd=r"D:/shiyan/"#视频目录地址
videoname="cam3"#视频文件名
videosuf=".avi"#视频文件后缀
fadd=r"chart/"#表格存放地址

#k帧的预处理
K=50#设置为K帧处理一次表格图片
frametot=0#记录帧数
quick=0#加速帧变量
quickvalue=9#加速倍数（整数）
trans=10#帧数平移量
ttt=0#平移量计数

if not os.path.exists(videoadd+videoname+'_'+fadd):#如果表格目录不存在，则创建目录
    os.mkdir(videoadd+videoname+'_'+fadd)

cap = cv.VideoCapture(videoadd+videoname+videosuf) #参数为0是打开摄像头，文件名是打开视频
fps = cap.get(5)#获取视频的fps

fgbg = cv.createBackgroundSubtractorMOG2()#混合高斯背景建模算法

fourcc = cv.VideoWriter_fourcc(*'XVID')#设置保存图片格式
a,b=(int(cap.get(3)), int(cap.get(4)))#设置视频分辨率

out = cv.VideoWriter(videoadd+videoname+'_'+fadd+videoname+'_p1_video'+'.avi',fourcc, fps, (a,b))
    #分辨率要和原视频对应

anglelist=[]
Mmlist=[]
heightlist=[]
framelist=[]
for i in range(K):
    anglelist.append(1)
    Mmlist.append(1)
    heightlist.append(1)
    framelist.append(i+1)


while True:
    ret, frame = cap.read()  #读取图片
    quick+=1
    if quick % quickvalue != 0 :
        continue
    #帧数平移部分
    ttt+=1
    if ttt<=trans:
        continue

    if ret is False:        #判断有效读取
        break
    
    fgmask = fgbg.apply(frame) #得到二值图像

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # 形态学去噪
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, element)  # 开运算去噪

    contours, hierarchy = cv.findContours(fgmask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #寻找前景

    maxarea=10000000000000.0*1000000000000.0
    threshold=maxarea

    maxArea = -1.1
    threshold=800
    for cont in contours: # 取出最大面积
        Area = cv.contourArea(cont)
        maxArea = max(maxArea,Area)

    count=0
    for cont in contours:
        Area = cv.contourArea(cont)  # 计算轮廓面积
        if Area < maxArea or Area < threshold:  # 过滤面积小于最大面积和阈值
            continue
        if (cont.size // 2 < 6):# 绘制椭圆需要5个点,小于就取消
            continue
        count += 1  # 计数加一
        #print("{}-prospect:{}".format(count,Area),end="  ") #打印出每个前景的面积
        x, y, w, h = cv.boundingRect(cont)#提取矩形坐标
        retval = cv.fitEllipse(cont) #提取椭圆
        ma,Ma,angle=retval[1][0],retval[1][1],retval[2] #提取短轴，长轴，倾斜角度
        Mtom=Ma/ma
        #print("rate:{} angle:{} height:{}".format(Mtom,angle,height))#打印相关数据
        # 此部分用于学习前的数据处理操作
        #加入数据,范围在0到K
        anglelist[frametot % K] = angle
        Mmlist[frametot % K] = Mtom
        heightlist[frametot % K] = h
        #
        cv.rectangle(frame,(x,y),(x+w,y+h),colour[count%6],2)#原图上绘制矩形
        cv.rectangle(fgmask,(x,y),(x+w,y+h),(0xff, 0xff, 0xff), 2)  #黑白前景上绘制矩形
        cv.ellipse(frame, retval, colour[count%6], 2)#原图绘制椭圆
        cv.ellipse(fgmask, retval, (0xff, 0xff, 0xff), 2)#黑白前景绘制椭圆

        yy = 10 if y < 10 else y  # 防止编号到图片之外
        cv.putText(frame, str(count), (x, yy), cv.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)  # 在前景上写上编号

    if count>0:#如果有新的轮廓
        frametot+=1
        minhei=min(heightlist)
        if frametot%K==0:#到了K帧，处理一次图表
            for i in range(K):
                anglelist[i]=anglelist[i]*(b//1.5)/360+(b//6)+minhei
                Mmlist[i] = Mmlist[i] * (b // 1.5)/6 + minhei
            plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.axis('off')#去掉坐标轴
            plt.title(" ")#去掉标题
            plt.ylim(minhei, minhei + b // 1.5)#编辑
            #plt.ylim(0, 180)
            #plt.title("Angle every "+str(K)+" frames")
            plt.xlabel("frame")
            plt.ylabel("angle")
            plt.plot(framelist,anglelist, linewidth='2', color='#0000FF', linestyle='-')
            #plt.subplot(1, 1, 1)
            #plt.ylim(0, 6)
            #plt.title("Major axis/Minor axis every "+str(K)+" frames")
            plt.xlabel("frame")
            plt.ylabel("Major axis/Minor axis")
            plt.plot(framelist,Mmlist, linewidth='2', color='#00FF00', linestyle='-')
            #plt.subplot(1, 1, 1)
            #plt.ylim(minhei, minhei + b // 1.5)
            #plt.title("Height every "+str(K)+" frames")
            plt.xlabel("frame")
            plt.ylabel("Height")
            plt.plot(framelist,heightlist, linewidth='2', color='#FF0000', linestyle='-')
            plt.savefig(videoadd+videoname+'_'+ fadd + videoname+"_p"+str(frametot//K) + ".png")

    cv.putText(frame, "count:", (5, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1) #显示总数
    cv.putText(frame, str(count), (75, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    print("Running...")

    cv.imshow('frame', frame)#在原图上标注
    cv.imshow('frame2', fgmask)  # 以黑白的形式显示前景和背景
    out.write(frame)
    #类似处理轮廓的方法，裁剪视频
    if count>0:#如果有新的轮廓
        if frametot%K==0:#到了K帧
            out.release()  # 释放文件
            out = cv.VideoWriter(videoadd+videoname+'_'+fadd+videoname+"_p"+str((frametot//K)+1)+'_video'+ '.avi', fourcc,fps, (a, b))
    k = cv.waitKey(30)&0xff  #按esc退出
    if k == 27:
        break
    
out.release()#释放文件
cap.release()
cv.destroyAllWindows()#关闭所有窗口，原来代码写错

newvideo=videoadd+videoname+'_'+fadd+videoname+"_p"+str((frametot//K)+1)+'_video'+ '.avi'
if os.path.exists(newvideo): #如果多出来一个视频文件，则删除他
    os.remove(newvideo)