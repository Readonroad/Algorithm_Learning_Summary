## Numpy库
Numpy是python的第三方库，支持多维数组和矩阵运算，运算是c代码实现，故而效率很高。矩阵属于多维数组。

### 数组上的基本操作
列表上的元素不能做数学运算，需要先转化为数组
```
#生成数组及数组属性
a = array(a)         #列表转化为数组
a = linespace(start,end,step) # 生成一组step的等间隔数组
b = sin(a)           #三角函数生成数组
a=arange()           #等间隔的数组，前闭后开
type(a)              #数组属性 Numpy.ndarray
a.dtype              #数组元素属性
a.itemsize           #数组元素所占字节
a.nbytes             #数组所有元素所占空间
a.shape or shape(a)  #查看数组的形状
a.reshape(m,n)       #改变数组的形状
a.size or size(a)    #数组数目
a.ndim               #数组维度

#数组支持的操作
a+1             #加1操作
a+a             #数组想加
a*a             #数组对应的元素相乘
a**a            #数组对应元素乘方
numpy.dot(a,b)  #矩阵乘法
a.fill(n)       #设定数组元素的值，如果该值与a元素类型不一致，将该值转化为对应的数组类型值

#数组元素的访问：下标方式，切片方式，花式索引方式
a[0]            #通过下标索引获取对应的元素
a[:2]           #切片方式获取元素，前闭后开，切片操作可以使用负索引，只能等间隔或连续操作数组元素
a[-2:]          #获取后两位元素，切片操作是引用机制，节约了空间，缺点是修改一个值会改变另一个值
a[b>0]          #b>0返回的是元素为布尔值的数组，通过布尔数组访问元素
a[b]            #b为给定的元素位置数组，花式索引可以指定元素的位置，实现任意位置操作
#where语句
where(a)        #返回数组中所有非零元素的索引，返回的是元组类型，可以通过加索引得到对应的数组
where(a>n)[0]   #返回满足条件的元素索引位置，也可以直接使用where()的返回值进行索引
```
### 修改数组操作
```
#复数数组操作
a.real       #查看数组实部
a.imag       #数组虚部，虚部是只读的，不能修改
array(a,dtype=float32) #创建数组实，可以指定数组元素的类型，不指定时，数组可以根据元素类型自动判断

array(a,dtype=object)   #任意类型的数组
#asarray不仅可以作用与数组，还可以将其他类型转化为数组，当转化的对象是数组时，不会产生新的对象，保证了效率
asarray(a, dtype=类型)  #转换原来数组a的类型
a.astype(类型)          #改变原来数组的类型，返回数组的一份拷贝
```

### 数组元素数学操作
两种方式：对象引用方法a.sum()或者函数方式sum(a)
```
a.sum(axis=0) or sum(a, axis=0)     #求和,指定维度求和，维度可以为负值，-1表示最后一维
a.prod()      or prod(a)            #求所有元素的积，指定维度求积
max()/min()/argmin()/argmax()/mean()/average()/std()/var()
clip()   #限制在某个范围，小于最小值的为最小值，大于最大值的为最大值
ptp()    #计算最大值和最小值的差
round(decimals=n)   #四舍五入,近似到指定精度
cumsum(axis=None)   #累加和
cumprod(axis=None) #累加积
```
### 数组排序
```
sort(a,axis=n)    #默认axis为最后一维，升序,如2维，默认对每行排序，a不变
argsort(a,axis=n) #返回对a排序后元素在数组a中的位置索引
a.sort(axis=n)    #与函数sort的区别在于，这里会改变a的值
a.argsort()与函数argsort()同
searchsorted(sorted_array,values) #保持第一个sorted_array数组排序性质不变，将values插入sorted_array中，返回插入sorted_array中的位置索引
```
### 修改数组形状
```
a.shape        #数组形状
a.shape = m,n  #赋值的方式修改数组的形状，两种方式绝不会改变原来数组a的形状，而是返回一个新的数组
a.reshape(m,n) 
a[newaxis,:]            #新增一个维度,根据newaxis插入的位置，返回不同形状的数组
a[:,newaxis]
a[:,newaxis,newaxis]    #新增多个维度
a.squeeze()             #去除数组中长度为1的维度，返回新的数组
a.transpose() or a.T    #数组转置,转置只是交换了数组轴的位置，对数组的另一种view,改变转置数组的值，也会对应修改原来数组的值

#数组连接
concatenate((a,a1,...,aN),axis=n） #将不同的数组按照顺序连接起来，axis=0表示行连接，列不变
array((a,b))        #若a和b维度一样，用array可以得到高一维的数组
vstack((a1,a2))     #行方向连接，列不变
hstack((a1,a2))     #列方向连接，行不变
dstack((a1,a2))     #高一维数组
a.swapaxes(axis1,axis2) #交换两个维度的职位
#flatten
a.flatten()      #将多维数组转化为1维数组，默认按行拼接，返回数组的复制，不会影响原来数组
a.flat           #返回所有元素的迭代器，通过下标可以得到对应的元素，修改其值，会改变原来数组a的值
a.flat[:]        #返回1位数组
a.ravel()        #返回1维数组，更高效，原来数组的引用，修改返回数组的值也会对应修改原来数组的值
aa = a.T
aa.ravel()       #先转置再ravel，此时修改返回数组的值，不会对应修改aa和a的值，因为aa是a的一个view
atleast_xd(a)    #x可以去1,2,3,即atleast_1d,atleast_2d,at_least_3d,保证数组a至少有x维
```
### 数组对角线
```
a.diagonal(offset=n)    #查看对角线的元素，offset偏移，查看次对角线，正值表示右偏，负值表示左偏
#花式索引获取对角线，可以通过赋值，修改对角线的值
i=[0,1,2]
a[i,i]  
```
### 数组与字符串转化
```
a.tolist()            #数组转为列表
a.tostring()          #转化为字符串，按照行读取数据
a.tostring(order='F') #表示用Fortran的格式，按照列读取数据
fromstring(s,dtype=np.uint8) #从字符串中读取数据，指定数据类型，返回的是一维数组
```
### 数组生成
```
arange(start,stop=None,step=1,dtype=None) #生成[start,stop),间隔为step的数组
range(start,stop=None,step=1,dtype=None) #生成[start,stop),间隔为step的列表，非numpy中的函数
linespace(start,stop,N)    #生成[start,stop]等间距的N个元素的数组
logspace(start,stop,N)     #产生N个对数等间距分布的数组，默认以10为底
#meshgrid可以设置轴排列的先后顺序，默认indexing='xy',笛卡尔坐标，2维数组，返回行列向量

#indexing='ij'矩阵坐标，返回列行向量
meshgrid(x,y)              #生成二维平面的网格，返回结果分别对应网格的二个维度
meshgrid(x,y,sparse=True)  #去除x和y中冗余的元素，返回单一的行向量和列向量

#r_和c_产生行向量或者列向量,r_和c_的用法相同
numpy.r_[0:1:.1]     #切片方式产生一个数组
numpy.r_[0:1:5j]     #复数5j表示生成5个值的数组 array([0,.25,.5,.75,1.])

#特殊数组
ones(shape,dtype=float64)   #产生一个指定形状的全1数组
zeros(shape, dtype=float64) #产生一个指定形状的全0数组
empty(shape,dtype=float64,order='C') #指定大小，order指定内存存储方式，按行存储。‘F'表示按列存储，填充值随机生成

#产生一个与a大小一样，类型一样的对应数组
empty_like(a)
zeros_like(a)
ones_like(a)

identity(n,dtype=float64)  #产生一个n*n的单位矩阵
```

### 矩阵
```
#生成矩阵
numpy.mat(a)   #将数据a转化为矩阵
numpy.mat('1,2,3;3,4,5')  #字符串转化为矩阵
numpy.mat('a,b;b,a')      #数组拼接为字符串，转化对矩阵
A*B        #矩阵乘法
A.I        #矩阵逆运算
A**4       #矩阵指数表示矩阵连乘
```

### 二元运算
```
add(a,b)         a+b
subtract(a,b)    a-b
multiply(a,b)    a*b
divide(a,b)      a/b
power(a,b)       a**b  对应元素相乘
remainder(a,b)   a%b
== equal / != not_equal 等等
```

### 数组广播机制
```

```
