from multiprocessing import Pool

from itertools import combinations_with_replacement as cr

a = cr(range(3),2)
print(list(a))
'''
def f(x):
    return x*x


if __name__ == '__main__':
    pool = Pool()
    L = pool.map(f,range(int(1e2)))
    pool.close()
    pool.join()
    print(sum(L))
'''