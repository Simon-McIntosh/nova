# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:44:42 2020

@author: mcintos
"""


from contextlib import contextmanager

@contextmanager
def switch(flag):
    flag['status'] = False
    yield flag
    flag['status'] = True
            
flag = {'status': True}
print(flag)
with switch(flag):
    print(flag['status'])
print(flag['status'])