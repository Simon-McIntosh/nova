import numpy as np
from collections import OrderedDict
import urllib.request
import requests
auth = ('mcintos', 'sm085888?')

# https://user.iter.org/?uid=3C6V7B&action=get_document

# disruptions main page: 2PN3CN
# looking for: 3C6V7B


class node(object):

    def __init__(self, value, children=[]):
        self.value = value
        self.children = children

    def __str__(self, level=0):
        ret = '\n\t'*level+repr(self.value)
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'


class IDM:

    def __init__(self, uid):
        self.uid = uid  # main page

        self.get_root()

        #self.get_scenario('MD_UP_lin50ms_Cat.I')
        self.show()

    def get_root(self):
        self.root = []  # root directory
        self.items = IDM.get_items(self.uid)
        for item in self.items:
            self.root.append(node('{}, {}'.format(item, self.items[item])))

    def get_scenario(self, scenario):
        i = list(idm.items.keys()).index(scenario)
        files = IDM.get_files(scenario, self.items[scenario])
        self.root[i].children = [node('{} {}'.format(f, files[f]))
                                 for f in files]

    def get_items(uid):
        page = IDM.load_page(uid)
        items = IDM.get_uid(page, 'GridItem')
        return items

    def get_files(scenario, uid):
        page = IDM.load_page(uid)
        files = IDM.get_uid(page, 'get_document')
        IDM.strip_keys(scenario, files)
        return files

        #for item in items:
        #

    def show(self):
        for r in self.root:
            print(r)

    def load_page(ref):
        r = requests.get('https://user.iter.org/?uid={}'.format(ref),
                         auth=requests.auth.HTTPBasicAuth(*auth))
        page = r.content.decode()
        return page

    def get_uid(page, trigger):
        # trigger = GridItem or get_document
        links = OrderedDict()
        fid, index = 0, np.zeros(2, dtype=int)
        while fid >= 0:
            fid = page.find('uid=')
            text = page[fid:fid+150]
            if trigger in text:
                index[0] = text.find('>')+1
                index[1] = text.find('<')
                name, uid = text[index[0]:index[1]], text[4:10]
                links[name] = uid
            page = page[fid+11:]  # advance
        return links


    def strip_keys(scenario, files):
        for f in list(files):
            index = f.find(scenario)
            if index > 0:
                f_trim = f[:index-1]  # trim name
                files[f_trim] = files.pop(f)



    #def get_files(scenario)

    #'2PN3CN'

if __name__ == '__main__':
    #idm = IDM('2PN3CN')

    #idm = IDM('2MJTU8')

    page = IDM.load_page('2MJTU8')
    files = IDM.get_uid(page, 'get_document')

    print(files)

    page = IDM.load_page('2N9JKB')


'''
log = []
for i, link in enumerate(links):
    log.append(node(link))

    page = load_page(links[link])
    files = get_uid(page, 'get_document')
    strip_keys(scenario, files)
'''
