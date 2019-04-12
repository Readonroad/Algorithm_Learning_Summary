python 爬虫网页：构造网址url，发起请求request,获取网页内容，筛选内容，保存结果
## requests库
python的第三方库，用来处理URL资源。

### 安装requests库
找到python安装路径下的Scripts文件夹，pip命令在该文件夹下。
```python
pip install requests
```
### 使用requests库
get/post操作中参数介绍
```
* url：就是目标网址，接收完整（带http）的地址字符串。
* headers：请求头，存储本地信息如浏览器版本，是一个字典。
* data：要提交的数据，字典。
* cookies：cookies，字典。
* timeout：超时设置，如果服务器在指定秒数内没有应答，抛出异常，用于避免无响应连接，整形或浮点             数。
* params：为网址添加条件数据，字典。
* proxies：ip代理时使用，字典类型。
* auth : 元组，支持http认证功能
```
```python
import requests
url =  "xxxx"
r = requests.get(url)
r_params = requests.get(url,params)  #params以字典的形式填入
r.status_code   #通过返回码可以查看get结果

r.text   #可以显示对应文本结果
r.encoding #requests自动检测编码，通过encoding可以查看对应的结果
```
使用http的其他操作，post/put/delete类似，需要传入对应的参数。

## BeautifulSoup4 
Beautiful Soup是一个可以从HTML或XML文件中提取数据的python库，能够通过你喜欢的转化器实现惯用的文档导航，查找，修改文档的方式。

参考文档：https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/
### 安装BeautifulSoup4库
找到python安装路径下的Scripts文件夹，pip命令在改文件夹下。
```python
pip install beautifulsoup4
#安装解析器,python 自带解析器 "html.parser"
pip install lxml (html5lib)
```
### 使用BeautifulSoup4库
```python
from bs4 import BeautifulSoup
#传入一个html/xml文档对象，或一段字符串，或一个文件句柄
#BeautifulSoup首先将文档转换为Unicode编码，然后选择最合适的解析器解析文档，可以手动指定对应的解析器
soup = BeautifulSoup(xmlfile) 
```
### BeautifulSoup对象
BeautifulSoup将html/xml文档转换为树形结构，每个节点都是python对象，共归纳为四种对象:Tag, NavigableString, BeautifulSoup, Comment.
#### tag   
 tag对象与html/xml原生文档中的tag相同
```python
#获取tag的名字
tag.name
#获取tag的属性
```
#### NavigableString   
包含在tag中的字符串，BeautifulSoup用NavigableString类来包装tag中的字符串
#### BeautifulSoup   
BeautifulSoup 对象表示的是一个文档的全部内容.大部分时候,可以把它当作 Tag 对象，但是因为BeautifulSoup 对象并不是真正的HTML或XML的tag，它没有name和attribute属性，所以BeautifulSoup 对象包含了一个值为 “[document]” 的特殊属性 .name
#### Comment   
该类是NavigableString 的子类，用来表示文档中的注释部分。

## robobrowser库
robobrowser是一个没有界面浏览器，运行在内训中，可以打开网页，点击链接和按钮，并且提交表单。它调用了python的requests和Beautifulsoup库

### robobrowser库的安装
```python
pip install robobrowser
```
### 使用robobrowser库
创建browser，打开网页
```python
from robobrowser import RoboBrowser
url = 'xxxx'
b = RoboBrowser(history = True)
b.open(url)   #创建browser,打开网页，默认方法为get，其他可选变量与requesets中的get类型。
```
获取内容的方法：find/select/find_all
```
#find和find_all是beautifulsoup中的方法,支持过滤器：字符串、正则表达式、列表、True(匹配任何值)、定义方法（只能接受一个元素参数）
find     返回页面上符合条件的1个元素
find_all 返回所有符合条件的tag集合
select   支持参数为css选择器，返回list
```
在网页上，点击链接：follow_link
```python
#用find/select/find_all方法过滤相应的链接，调用follow_link点击链接
from robobrowser import RoboBrowser
url = 'xxxx'
b = RoboBrowser(history = True)
b.open(url)   #创建browser,打开网页，

#y获取对应的链接
link= b.find(id = 'xxx').a
#跳转链接
b.follow_link(link)
```
表单操作：页面登录才能抓取内容
```python
#get_form    用来抓取form
#submit_form 用来提交from
#form[name].value 用来赋值

#coding: utf-8
from robobrowser import RoboBrowser

url = 'http://testerhome.com/account/sign_in/'
b = RoboBrowser(history=True)
b.open(url)

# 获取登陆表单
login_form = b.get_form(action='/account/sign_in')
print login_form

# 输入用户名和密码
login_form['user[login]'].value = 'your account'
login_form['user[password]'].value = 'your password'

# 提交表单
b.submit_form(login_form)

# 打印登陆成功的信息
print b.select('.alert.alert-success')[0].text
```
## Pandas库
Pandas是Python的一个模块，高性能、高效率、高水平的数据分析库。

中文教程：https://wizardforcel.gitbooks.io/pandas-official-tut-zh/content/

## dpkt库
Dpkt在创建和解析数据包上是一个非常棒的框架，解析原始PCAP文件格式，以及解析在PCAP文件中的数据包。
```python
#打开pcap文件，用Reader类读取文件
f = open('test.pcap')
pcap = dpkt.pcap.Reader(f)

#解析pacp对象的数据
for ts, buf in pcap:
    print ts, len(buf)  #打印时间邮戳和数据表每个记录的数据长度
    #使用dpkt中的类，可以将原始的buffer解析和编码为Python对象
    eth = dpkt.ethernet.Ethernet(buf)   #Ethernet类
    #dpkt中的Ethernet类得到的数据中可能还包含有更高层次的协议，如，IP,TCP等
    #通过eth.pack_hdr 可以看到对应的Ethernet封装的消息包
    ip = eth.data   #eth消息包中有对应的ip类对象，通过点取(.)可以得到对应的信息
    tcp = ip.data   #tcp类对象
    #如果tcp中有http会话，可以通过HTTP解析数据
    if tcp.dport == 80 and len(tcp.data) >0:
        http = dpkt.http.Request(tcp.data)   #dport为80表示是http请求，而不是response
        #http解析属性
        http.method
        http.url
        http.version
        http.headers['user-agent'] 

```
