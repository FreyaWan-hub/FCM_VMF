

from scipy import optimize
import numpy as np

def func(x,y,p):
    """
    数据拟合函数 z=ax+by+c
    :param x: 自变量x 是list
    :param y: 自变量y 是list
    :param p: 拟合参数 a, b, c
    """
    x = np.array(x)
    y = np.array(y)
    a, b, c = p
    z = a*x+b*y+c
    return z

def residuals(p, z, x, y):
    """
    得到数据z和拟合函数之差
    """
    z = np.array(z)
    return z - func(x,y,p)


def fitting(x, y, z):
    plsq = optimize.leastsq(residuals, np.array([0, 0, 0]), args=(z, x, y))
    return plsq[0]








