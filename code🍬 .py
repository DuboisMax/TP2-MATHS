import math, numpy as np, matplotlib.pyplot as plt
from time import time


#pivot Gauss
def pivot(A, p):
    n=A.shape[0]
    i = p
    for k in range(p + 1, n):
        if abs(A[k, p]) > abs(A[i, p]):
            i = k
    return i
def gauss(A, b):
    A1, b1 = A.copy(), b.copy()
    n = len(b) 
    for l in range(n - 1):
        piv = pivot(A, l)
        if piv != l: 
            Li = A[l, :].copy()
            A[l, :] = A[piv, :]
            A[piv, :] = Li
            Ep = b1[piv]
            b1[piv] = b1[l]
            b1[l] = Ep
        for k in range(l + 1, n): 
            fac = -A1[k, l] / A1[l, l]
            A1[k, :] += fac * A1[l, :]
            b1[k] = b1[k] + b1[l] * fac
    return A1, b1

def remontee(A, b):
    n = len(b)
    x = np.zeros([n]) 
    for i in range(n - 1, -1, -1):
        somme = np.dot(A[i, :], x)
        x[i] = (b[i] - somme) / A[i, i]
    return x
def solve_gauss(A, b):
    A1, b1 = gauss(A, b)
    return remontee(A1, b1)

#LU décomposition

def dec_lu_tridiag(A):
    n=np.shape(A)[0]
    L, U=np.zeros((n,n)),np.eye(n)    
    L[0,0]=A[0,0]
    L[1,0]=A[1,0]
    U[0,1]=A[0,1]/A[1,1]
    for i in range(1,n-1):
        L[i+1,i]=A[i+1,i]
        L[i,i]=A[i,i]-A[i,i-1]*U[i-1,i]
        U[i,i+1]=A[i-1,i]/L[i,i]
    L[n-1,n-1]=A[n-1,n-1]-A[n-1,n-2]*U[n-2,n-1]
    return L,U

def tridiag_matrice(a,b,c,n):
    #np.diag(a*np.ones(n),0)+np.diag(c*np.ones(n-1),1)+np.diag(b*np.ones(n-1),-1)
    return a*np.eye(n)+b*np.eye(n,k=-1)+c*np.eye(n,k=1)

n=100
h= 2/(n+1)
M = np.eye(n)*4*(1-1/h**2)+np.eye(n,k=-1)*(2/h**2-1/(2*h))+np.eye(n,k=1)*(2/h**2+1/2*h)
#print(dec_lu_tridiag(M))

def colonne_f(taille_vecteur,ci):
    colonne = [1-i**2 + 5*i for i in range(taille_vecteur)]
    colonne[0] -= ci[0]*( 2/ h**2 - 1/ (2*h) )
    colonne[-1] -= ci[1]*( 2/ h**2 + 1/ (2*h) )
    return np.array(colonne)

def descente(A, b):
    n = len(b)
    x = np.zeros([n]) 
    for i in range(n):
        somme = np.dot(A[i, :], x)
        x[i] = (b[i] - somme) / A[i, i]
    return x

L,U = dec_lu_tridiag(M)
vec_f = colonne_f(n,(0.05,0.05))

y =descente(L,vec_f)
#print(y)

def remontee(A,b):
    n = len(b)
    x = np.zeros([n]) 
    for i in range(n - 1, -1, -1):
        somme = np.dot(A[i, :], x)
        x[i] = (b[i] - somme) / A[i, i]
    return x

Y_lu=remontee(U,y)
Y_linalg=np.linalg.solve(M,vec_f)
#print(max(abs(Y_lu-Y_linalg)))

#décomposition Q&R
def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

Q,R = qr_decomposition(M)

b_new = np.dot(Q.T, vec_f)

Y_qr = np.linalg.solve(R, b_new)

#print("Vecteur solution x :", x_solution)
print(max(abs(Y_qr-Y_linalg)))

x = np.linspace(0,2,100)
plt.plot(x,Y_qr,x,Y_lu)
plt.show()

