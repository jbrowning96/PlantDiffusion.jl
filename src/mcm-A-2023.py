import math
import numpy as np
import sklearn as sk
import pandas as pd
import sympy as sp
import scipy
import scipy.fft
import matplotlib as plt
import time
import threading
#############################################################
'''
Global Parameters:
'''
K = 1
Q = 0.05
M = 1.2
A = 40
N = 4
E = 3.5
LAMBDA = 0.035
GAMMA = 20
DB = 6.25 * 10**(-4)
DW = 6.25 * 10**(-2)
DH = 0.05
S0 = 0.125
Z = 0
P = 500
R = 0.95
F = 0.1

'''
Nondimensional Parameters:
'''
q = Q/K
v = N/M
alpha = A/M
eta = E*K
gamma = GAMMA*K/M
p = LAMBDA*P/(M*N)
deltaB = DB/(M*S0**2)
deltaW = DW/(M*S0**2)
deltaH = DH*N/(M*LAMBDA*S0**2)
zeta = LAMBDA*Z/N
rho =  R

# b = B/K
# w = LAMBDA*W/N
# h = LAMBDA*H/N
# t = M * T

'''
Other Global Parameters:
'''
RES = 32
NL = 15
XMIN = 0
YMIN = 0
XMAX = 6
YMAX = 6
dx = (XMAX - XMIN) / RES
#############################################################


def initial_conditions(y, x):
  return .5 + math.cos(x) * math.sin(y)

def init(n):

    '''
    Initializes 3 matrices with the given initial conditions for B, W, and H.
    To use this one would go into b[i,j], w[i,j] and h[i,j] wherever values of
    b(i,j,t), w(i,j,t), and h(i,j,t) are needed.
    '''

    b = np.zeros((n,n))
    w = np.ones((n,n))
    h = np.zeros((n,n))

    vec_ic = np.vectorize(initial_conditions)

    b[:,:] = vec_ic(np.linspace(XMIN, XMAX, RES), np.linspace(YMIN, YMAX, RES))
    w[:,:] = vec_ic(np.linspace(XMIN, XMAX, RES), np.linspace(YMIN, YMAX, RES))
    h[:,:] = vec_ic(np.linspace(XMIN, XMAX, RES), np.linspace(YMIN, YMAX, RES))


    return b,w,h

def periodic_bc(u):
    u[0, :] = u[-1, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

    return u

def laplacian(u):
    # Second order finite difference with matrix passed to it
    Lu = np.zeros((RES,RES))
    for i in range(RES):
        for j in range(RES):
            Lu[i,j] = (u[i-1,j] + u[i,j-1] - 4*u[i,j] + u[(i+1)%RES,j] + u[i,(j+1)%RES]) / (dx**2)
    return Lu

def laplacianH(h):
    Lh = np.zeros((RES, RES))
    for i in range(RES):
      for j in range(RES):
        Lh[i, j] = 2 * ((h[(i+1)%RES][j] - h[i][j])**2 + (h[i][(j+1)%RES] - h[i][j])**2) / dx**2
    Lh = Lh + laplacian(h)
    return Lh

def infil(u):
    '''
    Pass in the b function, returns an array for all the infiltration rates at this time step.
    '''
    return alpha * ((u[:,:] + q * F)/(u[:,:] + q))

def dbdt(b, Lb, G_b):
    return np.multiply(G_b*b,(1-b)) - b + deltaB*Lb # type: ignore

def dwdt(b,w,h,I,Lw,G_w):
    return I*h - np.multiply(v*(1-rho*b),w) - G_w*w + deltaW*Lw # type: ignore

def dhdt(h,I, Lh):
    #print(p - I*h + deltaH*Lh)
    return p - I*h + deltaH*Lh

def disc_phi(b):
  phiDiscrete = np.zeros((RES,RES))
  for i in range(RES):
    for j in range(RES):
      phiDiscrete[i][j] = 1 + eta * b[i][j]
  return phiDiscrete

def gauss(y, x, phi):
  return 1 / (2 * math.pi) * np.exp( -1 * (x**2 + y**2) / (2 * phi**2)) # type: ignore
#
def get_phi_n(n_l, x_max):
  phi_n = np.zeros(n_l)
  for i in range(n_l):
    phi_n[i] = (x_max * (i+1))/(n_l)
  return phi_n

def fourier_gaussian(phi_n, n_l, x_0, y_0):
    X_and_Y = np.zeros((RES,RES))
    Gaussian = np.zeros((n_l,RES,RES//2+1))
    for l in range(n_l):
        for i in range(RES):
            for j in range(RES):
                X_and_Y[i,j] = gauss(x_0 + i/RES, y_0 + j/RES, phi_n[l])
        Gaussian[l] = scipy.fft.rfft2(X_and_Y) # type: ignore
    return Gaussian

def phiFunction(phi_j, phi_l, x_max, x_min, y_max, y_min):
	intr = math.sqrt(1 / 2 * ((1/phi_l)**2 + (1/phi_j)**2))
	return math.pi / (2 * ((1/phi_l)**2 + (1/phi_j)**2)) * ( scipy.special.erf(x_max * intr) - scipy.special.erf(x_min * intr)) * (scipy.special.erf(y_max * intr) - scipy.special.erf(y_min * intr)) # type: ignore

def getCoeffMatrixEff(n, n_l, disc_phi, phi_n, x_min, y_min, x_max, y_max):
  convolutionMatrix = np.zeros((n_l, n_l))
  b = np.zeros((n_l, n, n))
  AcoeffMatrix = np.zeros((n_l, n, n))
  for j in range(n_l):
    for l in range(n_l):
      convolutionMatrix[j][l] = phiFunction(phi_n[j], phi_n[l], x_max, x_min, y_max, y_min)
    vphiFunction = np.vectorize(phiFunction)
    b[j , :, :] = vphiFunction(disc_phi, phi_n[j], x_max, x_min, y_max, y_min)
  for i in range(n):
    for j in range(n):
      AcoeffMatrix[:, i, j] = np.linalg.solve(convolutionMatrix, np.squeeze(b[:, i, j]))
  return AcoeffMatrix

#SAME AS ABOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def gb_and_gw(n,n_l,w,b, Gaussian, A_coeff):
    W_fourier = scipy.fft.rfft2(w) # type: ignore
    B_fourier_dummy = np.zeros((n_l, n, n))
    B_fourier = np.zeros((n_l, n, n//2 + 1))
    G_b = np.zeros((n,n))
    G_b_dummy = np.zeros((n,n//2+1))
    G_w_dummy = np.zeros((n,n//2+1))
    for l in range(n_l):
        B_fourier_dummy[l] = np.multiply(A_coeff[l,:,:], b) # type: ignore
        B_fourier[l] = scipy.fft.rfft2(B_fourier_dummy[l]) # type: ignore
        G_w_dummy = np.add(G_w_dummy,np.multiply(Gaussian[l], B_fourier[l])) # type: ignore
    for i in range(n):
        for j in range(n):
            for l in range(n_l):
                G_b_dummy = np.add(G_b_dummy, A_coeff[l, i, j] * np.multiply(Gaussian[l], W_fourier)) #type: ignore
            G_b[i, j] = v * scipy.fft.irfft2(G_b_dummy)[i, j] #type: ignore
    G_w = gamma * scipy.fft.irfft2(G_w_dummy) #type: ignore
    return G_b, G_w


def evalI(i, outputMat, psi, phi, X_mat, Y_mat, dX, dY):
  temp_mat = np.zeros((RES, RES))
  X_prime = np.linspace(XMIN, XMAX, RES)
  Y_prime = np.linspace(YMIN, YMAX, RES)
  for j in range(RES):
    for k in range(RES):
      for l in range(RES):
        upper = -1 * (((X_mat[i] - X_prime[k])*dX)**2 + ((Y_mat[j] - Y_prime[l])*dY)**2) / (2 * phi[k][l])
        temp_mat[k][l] = np.exp(upper) * psi[k][l] #type: ignore
    outputMat[i][j] = scipy.integrate.simps(scipy.integrate.simps(temp_mat, X_prime), Y_prime) #type: ignore
    #print("i is" + str(i) + " j is " +str(j) + "eval 1")

def evalI2(i, outputMat, psi, phi, X_mat, Y_mat, dX, dY):
  temp_mat= np.zeros((RES, RES))
  X_prime = np.linspace(XMIN, XMAX, RES)
  Y_prime = np.linspace(YMIN, YMAX, RES)
  for j in range(RES):
    for k in range(RES):
      for l in range(RES):
        upper = -1 * (((X_mat[i] - X_prime[k])*dX)**2 + ((Y_mat[j] - Y_prime[l])*dY)**2) / (2 * phi[k][l])
        temp_mat[k][l] = np.exp(upper) * psi[i][j] #type: ignore
    outputMat[i][j] = scipy.integrate.simps(scipy.integrate.simps(temp_mat, X_prime), Y_prime) #type: ignore
    #print("i is" + str(i) + " j is " +str(j) + "eval 2")
		
def evalBoth(i, outputMat1, outputMat2, psi1, psi2, phi, X_mat, Y_mat, dX, dY):
  evalI(i, outputMat1, psi1, phi, X_mat, Y_mat, dX, dY)
  evalI2(i, outputMat2, psi2, phi, X_mat, Y_mat, dX, dY) 

########################################################################
tic = time.perf_counter()

B,W,H = init(RES)

phi_n = get_phi_n(NL, XMAX)

Gaussian = fourier_gaussian(phi_n, NL, XMIN, YMIN)

X_mat = np.linspace(XMIN, XMAX, RES)
Y_mat = np.linspace(XMIN, XMAX, RES)
dX = (XMAX - XMIN) / RES
dY = (YMAX - YMIN) / RES

loops = 1000
dt = 1/loops

B_all = np.zeros((loops, RES, RES))
W_all = np.zeros((loops, RES, RES))
H_all = np.zeros((loops, RES, RES))

# Start of loop:
for i in range(loops):
    B_all[i] = B
    W_all[i] = W
    H_all[i] = H
    print(i)
    '''
    periodic_bc(B)
    periodic_bc(W)
    periodic_bc(H)
    '''
    disc_phiB = disc_phi(B)

    Lb = laplacian(B)
    Lw = laplacian(W)
    Lh = laplacianH(H)

    #CoefficientMatrix = getCoeffMatrixEff(RES,NL, disc_phiB, phi_n, XMIN,YMIN,XMAX,YMAX)

    #G_b, G_w = gb_and_gw(RES,NL,B,W, Gaussian, CoefficientMatrix)

    G_w = np.zeros((RES, RES))
    G_b = np.zeros((RES, RES))
    for i in range(RES):
      threading.Thread(target=evalBoth(i, G_w, G_b, B, W, disc_phiB, X_mat, Y_mat, dX, dY)).start();
    I = infil(B)
    #G_b = 1.1
    #G_w = 1.1
    #I = 1

    Db = (dbdt(B, Lb, G_b)) * dt
    Dw = (dwdt(B,W,H,I,Lw,G_w)) * dt
    Dh = (dhdt(H,I,Lh)) * dt

    B = np.add(B,Db) #type: ignore
    W = np.add(W,Dw) #type: ignore
    H = np.add(H,Dh) #type: ignore
    print("B is ")
    print(B)
    print("\n")
    print("W is")
    print(W)
    print("\n")
    print("H is ")
    print(H)


toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

print("Woohoo!")