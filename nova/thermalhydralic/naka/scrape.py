
'''
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "https://scml-svr.naka.qst.go.jp/index-e.php"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
'''

import mechanize
from bs4 import BeautifulSoup
from urllib.request import urlopen
import http.cookiejar
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

url = "https://scml-svr.naka.qst.go.jp/index-e.php"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")

cj = http.cookiejar.CookieJar()
br = mechanize.Browser()
br.set_handle_robots(False)
br.set_cookiejar(cj)
br.open('https://scml-svr.naka.qst.go.jp/index-e.php')

print(br.response().read())
print('\n\n\n')

br.select_form(nr=1)
br.form['userid'] = 'FG'
br.form['password'] = 'edmly70a'
br.submit()

br.follow_link('Experimental Data')

print(br.response().read())
