import numpy as np
import math

X = np.array([[1,1,1],[1,0,1],[1,1,0],[1,2,2],[1,2,3],[1,3,2]])
y = np.array([[0],[0],[0],[1],[1],[1]])
a = 0.001
m = np.size(X,0)

def SH(s):
    stx = -1*np.dot(X,s)
    z = 1/(1+np.exp(stx))
    return z

def SJ(s):
    J = (-1/m)*np.dot(np.array([[1,1,1,1,1,1]]),
    y*np.log(SH(s))+(1-y)*np.log(1-SH(s)))
    return J[0,0]

def Ss(s):
    s0 = s[0,0]
    s0 = s0 - a * np.dot(np.array([[1,1,1,1,1,1]]),(SH(s)-y) * np.dot(X,np.array([[1],[0],[0]])))
    s1 = s[1,0]
    s1 = s1 - a * np.dot(np.array([[1,1,1,1,1,1]]),(SH(s)-y) * np.dot(X,np.array([[0],[1],[0]])))
    s2 = s[2,0]
    s2 = s2 - a * np.dot(np.array([[1,1,1,1,1,1]]),(SH(s)-y) * np.dot(X,np.array([[0],[0],[1]])))
    j = np.array(s)
    j[0,0]=s0
    j[1,0]=s1
    j[2,0]=s2
    return j

def diedai(s):
    J = SJ(s) 
    s = Ss(s)
    return J,s

def main():
    s = np.array([[1.0],[1.0],[1.0]])
    i = 0
    J = 1
    while(i<500):
        J,s=diedai(s)
        print("第",i,"次迭代\n")
        print("损失J：",J,"\n")
        print("参数s：",s,"\n")
        i +=1

main()