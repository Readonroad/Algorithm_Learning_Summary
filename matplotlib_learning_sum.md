matplotlib是一个功能很强大的python画图工具，可以画出线图、散点图、等高线图、条形图、柱状图、3D图、图像动画等。

## matplotlib基本使用

导入模块matplotlib.pyplot和numpy
```python
import matplotlib.pyplot as plt
import numpy as np
```
### 画图基本操作--坐标轴设置
```python
#定义图像窗口
fig = plt.figure() #定义图像窗口figure(num=a,figsize=(b,c)) 指定窗口编号a和窗口大小(b,c)

#画图
plt.plot(x,y)       #画出曲线plt(x,y,color='red',linewidth=1.0,linestyle='-',label='line label')指定线条的颜色、宽度、类型以及标记线条

#设置坐标轴的范围和名称
plt.xlim((xa,xb))   #xlim()和ylim()设置对应坐标轴的范围
plt.xlabel('name_x') #xlabel()和ylabel()设置对应坐标轴的名称
plt.ylim((ya,yb))
plt.ylabel('name_y')

#设置坐标轴间隔和名称
new_ticks = np.linespace(-1,2,5)
plt.xticks(new_ticks)    #设置坐标轴间隔
plt.yticks([-2,-1.8,-1,1,22,3],['really bad','bad','normal','good','really good']) #对应y轴的刻度和名称

#获取当前坐标轴信息
ax = plt.gca() 
#设置坐标轴，top,bottom,left,right分别时四个边框
ax.spines['right'].set_color('none')  #spines设置边框;set_color设置边框颜色，默认白色，隐藏边框
ax.spines['top'].set_color('none')

ax.spines['bottom'].set_position(('data',0)) #设置边框位置为y=0，(位置所有属性：outward,axes,data)
ax.spines['left'].set_position(('data',0))

#调整坐标轴的位置
ax.xaxis.set_ticks_position('bottom') #设置x坐标轴的位置为bottom(所有位置：top,bottom,both,default,none)
ax.yaxis.set_ticks_position('left')

#显示图像
plt.show() 
```
### 画图基本操作--图例legend

#### 方式一
设置图例，首先要设置图中线条类型等信息，即在plot中添加label参数，用来标记图中的线条，再用legend自动添加图例
```python
#设置图例，首先要设置图中线条类型等信息，即在plot中添加label参数
plt.plot(x,y1,label = 'line 1')
plt.plot(x,y2,label = 'line 2')

#添加图例
plt.legend(loc = 'best') #best表示最佳位置，其他参数如:right,left,center,upper(lower) right,upper(lower)left, upper(lower) center
```
#### 方式二
指定plot（）返回值，通过返回值，可以修改plot中之前定义的lable信息
```python
#设置图例，保证调用plot时要返回对应的值,要求返回的变量要用逗号结尾，表示plt.plot()返回的是一个列表
line1,= plt.plot(x,y1,label = 'line 1')
line2, = plt.plot(x,y2,label = 'line 2')

#添加图例
plt.legend(handles = [line1,line2],labels=['line_1','line_2'],loc = 'best') #labels重新设定了线条的label信息
```
### 标注
图线中需要标注的地方，用annotation. 标注有两种方法，一种是plt中的annotate,一种是plt里面的text写标注。

#### 方法一：添加注释annotate
```python
#对图线中的（x0,y0)点进行标注
plt.annotate(r'$2x+1=%s$' %y0,  #标注信息
             xy=(x0,y0),        #数据的位置
             xycoords = 'data', #'data'表示基于数据的值选择标注位置
             xytext=(+30,-30),textcoords='offset points',fontsize =16, #标注的位置的描述和偏差值，字体大小
             arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2') #箭头类型设置
             )
```

#### 方法二：添加注释text
```python
#对图线中的（x0,y0)点进行标注
plt.text(-3.7,3,    #text的位置
         r'$this\ is\ the\ some\ text$',  #text的内容，空格前要加反义转换符'\'
         fontdict={'size':16,'color':'r'} #设置文本字体和颜色 
         )
```
### tick能见度
对图形中相互遮挡的地方，通过设置相关内容的透明度使图形更易于观察。
```python
ax = plt.gca()   #获取当前坐标轴信息
for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(12)   #设置字体大小
    #bbox设置目的的内容的透明度相关参数
    label.set_bbox(dict(facecolor='white',   # 调节box的前景色
                         edgecolor='None',   #设置边框，None为无边框
                         alpha=0.7,          #alpha设置透明度   
                         zorder=2)) 
plt.show()
```
## 线图plot
```python
#坐标轴
x = np.linspace(-1,2,50) #定义坐标轴的范围为(-1,1),共50个数据点
y =  2*x +1

#画图
plt.figure()        
plt.plot(x,y)       #画出曲线plt(x,y,color='red',linewidth=1.0,linestyle='-',label='line label')指定线条的颜色、宽度、类型以及标记线条

plt.xlim((-1,3))   #xlim()和ylim()设置对应坐标轴的范围
plt.xlabel('name_x') #xlabel()和ylabel()设置对应坐标轴的名称
plt.ylim((-2,5))
plt.ylabel('name_y')

#设置坐标轴信息
ax = plt.gca()   #获取当前坐标轴信息
ax.spines['right'].set_color('none')  #隐藏右边框
ax.spines['top'].set_color('none')    #隐藏上边框  

#调整坐标轴的位置
ax.xaxis.set_ticks_position('bottom') #设置x轴为下边框
ax.spines['buttom'].set_position(('data',0)) #设置边框位置为y=0，(位置所有属性：outward,axes,data)
ax.yaxis.set_ticks_position('left')  #设置y轴为上左边框
ax.spines['left'].set_position(('data',0))
plt.show() 显示图像
```
## 散点图scatter
```python
import matplotlib.pyplot as plt
import numpy as np

n =1024
X = np.random.normal(0,1,n) #生成服从均值为0,方差为1的n个数据点
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X)         #数据点颜色值
plt.scatter(X,Y,s= 75, c=T,marker='o',alpha=0.5)  #X,Y为数据location,s为大小，c为颜色，alpha为透明度。
plt.show()
```

## 柱状图bar
```python
import matplotlib.pyplot as plt
import numpy as np

#生成数据
n =12
X=np.arange(n)
Y1 = (1-X/float(n))*np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n))*np.random.uniform(0.5,1.0,n)

#画出柱状图，并加颜色
plt.bar(X,+Y1, facecolor='', edgecolor = '')#facecolor设置主体颜色，edgecolor设置边框颜色
plt.bar(X,-Y2)

#设置轴的范围以及刻度
plt.xlim(-5,n)
plt.xticks(())
plt.ylim(-1.25,1.25)
plt.yticks(())

#柱状图Y1上加数据
for x,y in zip(X,Y1):
    #ha:horizontal alignment 水平对准
    #va:vertical alignment 垂直对准
    plt.text(x+0.4, y+0.05,'%.2f' % y,ha='center',va='bottom')

#柱状图Y2上加数据
for x,y in zip(X,Y2):
    #ha:horizontal alignment 水平对准
    #va:vertical alignment 垂直对准
    plt.text(x+0.4, y+0.05,'%.2f' % y,ha='center',va='top')
   
plt.show()
```
## 等高线图Contours
等高线图数据集为三维点(x,y)和对应的高度值
```python
import matplotlib.pyplot as plt
import numpy as np

#定义高度值生成函数
def f(x,y):
    return (1-x/2+x**5 + y**3)*np.exp(-x**2-y**2)

#生成数据
n =256
x = np.linespace(-3,3,n)
y = np.linespace(-3,3,n)
X,Y = np.meshgrid(x,y)    #meshgrid()在二维平面中将x，y分别对应起来，编织成栅格

#颜色填充contourf
#位置参数X,Y,f(X,Y), 8表示登高线的密集程度，alpha为透明度，cmap表示将f(X,Y)的值对应到color map的暖色组中寻找对应的颜色
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plot.cm.hot)

#登高线绘制
#位置参数X,Y,f(X,Y), 8表示登高线的密集程度，登高线的颜色为black,线条宽度为0.5
C= plt.contour(X,Y,f(X,Y),8,colors = 'black',linewidth=0.5)

#登高线上添加高度数字
#clabel添加数据，inline控制是否将label画在线里面，字体大小为10
plt.clabel(C,inline = True, fontsize=10)
#坐标轴隐藏
plt.xticks(())
plt.yticks(())
```
## 图像
由矩阵画图像
```python
#interpolation表示插值方式，origin选择原点的位置
plt.imshow(a,interpolation='nearest', cmap ='bone',origin='lower')

#添加colorbar,shrink控制colorbar的长度为原图长度的比例
plt.colorbar(shrink = 0.92)
```
## 3D图像
```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  #额外添加一个模块，即Axes 3D坐标显示

#定义图像窗口
fig = plt.figure()
ax=Axes3D(fig)      #3D坐标轴

#数据点
x=np.arange(-4,4,0.25)
y=np.arange(-4,4,0.25)
X,Y = np.meshgrid(x,y)    #生成x-y网格
#高度值
Z=np.sin(np.sqrt(X**2+Y**2))

#画出三维曲面，rstride和cstride分别代表row和column的跨度，cmap表示填充的颜色
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))

#添加平面上的登高线,zdir表示登高线的值，'z'表示投影在xy平面;‘x'表示投影在yz平面
ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap=plt.get_cmap('rainbow'))
```

## 多图显示
### 多合一显示

使用plt.subplot(row,column,num)创建多个小图并显示,分别对应行列和图的个数

### 分格显示
方式1：subplot2grid
```python
import matplotlib.pyplot as plt

plt.figure()
#使用subplot2grid创建图像
#(3,3)表示将窗口划分为几行几列，（0，0）表示图像的起点，colspan和rowspan分别表示列和行的跨度，缺省情况下为1.
ax1 = plt.subplot2grid((3,3),(0,0),colspan =3)
ax1.plot()
```
方式2：gridspec
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.figrue()

#划分图像窗口，使用gridspec.GridSpec()将整个图像窗口分成几行几列
gs=gridspec.GridSpec(3,3)

#使用subplot指定对应的窗口
ax1=plt.subplot(gs[0,:])
ax2=plt.subplot(gs[1,:2])  
...
```

## 图中图
```python
import matplotlib as plt

fig = plt.figure()

#确定大图的位置
left,bottom,width,height=0.1,0.1,0.8,0.8   #这四个值是占整个figure坐标系的百分比
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plt(x,y)

#小图的位置
left,bottom,width,height=0.2,0.6,0.25,0.25  #这四个值是占整个figure坐标系的百分比
ax2 = fig.add_axes([left,bottom,width,height])
ax2.plt(x,y)
...
```
