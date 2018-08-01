# Autogenerated with SMOP 0.32-7-gcce8558
from smop.core import *
# 

    
@function
def distortion(A=None,y=None,*args,**kwargs):
    varargin = distortion.varargin
    nargin = distortion.nargin

    yn = y[:, 1]
    yn = dot(A, yn)
    yn[yn > 1] = 1
    yn[yn < - 1] = - 1
    new_sig = copy(yn)
    if size(y, 2) > 1:
        yn = y[:, 2]
        yn = dot(A, yn)
        yn[yn > 1] = 1
        yn[yn < - 1] = - 1
        new_sig[:, 2] = yn