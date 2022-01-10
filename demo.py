# -*- coding: utf-8 -*-
# @Time    : 2022/1/4 9:56
# @Author  : Jiajun LUO
# @File    : demo.py
# @Software: PyCharm

import math

##计算2向量夹角
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]

    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


data = [[222.0,516.5686274509803], [280.8235294117647,479.3137254901961],
        [486.7058823529412,596.9607843137255],[471.01960784313724,689.1176470588235],
        [384.7450980392157,679.313725490196],[222.0,561.6666666666666]]


break_point = []
for i in range(1,len(data)-1):

    p1 = [data[i][0], data[i][1], data[i-1][0],data[i-1][1]]
    p2 = [data[i][0], data[i][1], data[i+1][0],data[i+1][1]]
   # print(p1, p2)
    ang = angle(p1, p2)
    print(ang)
    if ang < 165:
        break_point.append(data[i])
print(break_point)