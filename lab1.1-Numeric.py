import numpy as np
import copy
import matplotlib.pyplot as plt  

result=[]
with open('magic04.txt','r') as f:  #读取文件
    for line in f:
        line=list(map(str,line.split(','))) #逗号分割
        result.append(list(map(float,line[:10]))) #去除字母
D=np.array(result)
Dmean=D.mean(axis=0)#1.多元均值向量
print ("均值向量",Dmean) #输出
#2.计算样本协方差矩阵作为中心数据矩阵列之间的内积
Z=D-np.ones((D.shape[0],1),dtype=float)*Dmean #计算中心矩阵
Dcov1=np.dot(Z.T,Z)/D.shape[0]
Dcov2=np.cov(D.T) #3.计算样本协方差矩阵作为中心数据点之间的外积



#4.计算属性1和属性2的角度余弦
v1=Z[:,0]#属性一中心属性向量
v2=Z[:,1]#属性二中心属性向量
cos_angle=np.dot(v1,v2)/(np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))#计算向量夹角余弦值
angle=np.arccos(cos_angle)*360/2/np.pi#转化为角度
print ("向量夹角：",angle)
#散点图

plt.scatter(v1, v2, alpha=0.6)  # 绘制散点图，透明度为0.6
plt.title( 'scatter')
plt.xlabel( 'v1')
plt.ylabel( 'v2')
plt.show()
plt.savefig("scatter.jpg")

#normfun正态分布函数，mu: 均值，sigma:标准差，pdf:概率密度函数，np.exp():概率密度函数公式
def normfun(x,mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf
x = np.arange(-100, 200,1)
y = normfun(x, Dmean[0], v1.std())
plt.plot(x,y, color='g',linewidth = 3,alpha=0.6)
plt.title('probability density function')
plt.xlabel('v1')
plt.ylabel('Probability')
plt.savefig("pdf.jpg")  



Dvar=[] #存储方差
Dcov= copy.copy(Dcov1) #存储协方差 
for i in range(0,10):
    Dvar.append(Dcov1[i][i])
print ("属性",Dvar.index(max(Dvar))+1,"方差最大，值为：",max(Dvar))
print ("属性",Dvar.index(min(Dvar))+1,"方差最小，值为：",min(Dvar))
for i in range(0,10):
    Dcov[i][i]=np.mean(Dcov) #将对角线的值改为均值，即取最大最小值时不会选取到方差
re1=np.where(Dcov==np.max(Dcov))
re2=np.where(Dcov==np.min(Dcov))
print ("属性",re1[0][0]+1,"和属性",re1[0][1]+1,"的协方差值最大，值为：",np.max(Dcov))
print ("属性",re2[0][0]+1,"和属性",re2[0][1]+1,"的协方差值最小，值为：",np.min(Dcov))   
