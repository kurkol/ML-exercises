import numpy as np

"""
X1   x2  x3   y
1    1   1    6
-1   2   3    12
4    0   2    10

设模型为y=s1x1+s2x2+s3x3
初始s为
S=[0
0,
0,]

s1=s1-a/3(Y-y)x1
s2=s2-a/3(Y-y)x2
s3=s3-a/3(Y-y)x3
"""
X = np.array([[1,1,1],[-1,2,3],[4,0,2]])
y = np.array([[6],[12],[10]])
a = 0.01
m = np.size(X,0)
x = np.array([[1],[2],[2]])

def dj(s):
    s0 = s[0,0]
    s0 = np.dot(np.array([[1,1,1]]), (np.dot(X,s) - y) * np.dot(X,np.array([[1],[0],[0]])))
    s1 = s[1,0]
    s1 = np.dot(np.array([[1,1,1]]), (np.dot(X,s) - y) * np.dot(X,np.array([[0],[1],[0]])))
    s2 = s[2,0]
    s2 = np.dot(np.array([[1,1,1]]), (np.dot(X,s) - y) * np.dot(X,np.array([[0],[0],[1]])))
    j = np.array(s)
    j[0,0]=s0
    j[1,0]=s1
    j[2,0]=s2
    return j

def SJ(s):
    J = np.dot(np.array([[1,1,1]]),(np.dot(X,s) - y)**2)
    return J

def diedai(s):
    J = SJ(s) 
    j = dj(s)
    s = s - (a/m) * j
    return J,j,s

def main():
    s = np.array([[0],[0],[0]])
    i=0
    J=1
    while(J>0.0001):
        J,j,s=diedai(s)
        print("第",i,"次迭代\n")
        print("损失J：",J,"\n")
        print("参数s：",s,"\n")
        i +=1
        if(i>=10000):
            break

main()