import math
import numpy as np
import sklearn as sk
import pandas as pd
import sympy as sp
import scipy
import scipy.fft
import matplotlib as plt
import time
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
RES = 100
dx = 1 / RES
NL = 10
XMIN = 0
YMIN = 0
XMAX = 1
YMAX = 1
#############################################################




def init(n):

    '''
    Initializes 3 matrices with the given initial conditions for B, W, and H.
    To use this one would go into b[i,j], w[i,j] and h[i,j] wherever values of
    b(i,j,t), w(i,j,t), and h(i,j,t) are needed.
    '''

    b = np.zeros((n,n))
    w = np.ones((n,n))
    h = np.zeros((n,n))

    x, y = np.meshgrid(np.linspace(0, 1, n, False), np.linspace(0, 1, n, False))

    mask = (0.45<x) & (x<0.55) & (0.45<y) & (y<0.55)


    b[mask] = 0.50
    w[mask] = 0.75
    h[mask] = 0.10

    return b,w,h

def periodic_bc(u):
    u[0, :] = u[-1, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

    return u

def laplacian(u):
    #Second order finite difference with matrix passed to it
    Lu = np.zeros((RES,RES))
    for i in range(RES):
        for j in range(RES):
            Lu[i,j] = (u[i-1,j] + u[i,j-1] - 4*u[i,j] + u[(i+1)%RES,j] + u[i,(j+1)%RES])/dx**2
    return Lu

def laplacianH(h):
    U = np.multiply(h,h)
    return laplacian(U)

def infil(u):
    '''
    Pass in the b function, returns an array for all the infiltration rates at this time step.
    '''
    return alpha * ((u[:,:] + q * F)/(u[:,:] + q))

def dbdt(b, Lb, G_b):
    return np.multiply(np.multiply(G_b,b),(1-b)) - b + deltaB*Lb

def dwdt(b,w,h,I,Lw,G_w):
    return np.multiply(I,h) - np.multiply(v*(1-rho*b),w) - np.multiply(G_w,w) + deltaH*Lw

def dhdt(h,I, Lh):
    return p - np.multiply(h,I) + deltaH*Lh

def disc_phi(b):
  phiDiscrete = np.zeros((RES,RES))
  for i in range(RES):
    for j in range(RES):
      phiDiscrete[i][j] = 1 + eta * b[i][j]
  return phiDiscrete

def gauss(y, x, phi):
  return 1 / (2 * math.pi) * np.exp( -1 * (x**2 + y**2) / (2 * phi**2))
#
def g_coefficient_mat(n_l, x_0, y_0, x_max, y_max):
  phi_n = np.zeros(n_l)
  for i in range(n_l):
    phi_n[i] = 1 + i/n_l
  convolutionMatrix = np.zeros((n_l,n_l))
  for j in range(n_l):
    for l in range(n_l):
        f = lambda y, x: gauss(y, x, phi_n[j]) * gauss(y, x, phi_n[l])
        convolutionMatrix[j][l] = scipy.integrate.dblquad(f, x_0, x_max, y_0, y_max)[0]
        # INTEGRATE RETURNS ERROR, WE ARE NOT USING IT YET!!!!!!!!!!!
  return convolutionMatrix, phi_n

def fourier_gaussian(phi_n, n_l, x_0, y_0):
    X_and_Y = np.zeros((RES,RES))
    Gaussian = np.zeros((n_l,RES,RES//2+1))
    for l in range(n_l):
        for i in range(RES):
            for j in range(RES):
                X_and_Y[i,j] = gauss(x_0 + i/RES, y_0 + j/RES, phi_n[l])
        Gaussian[l] = scipy.fft.rfft2(X_and_Y)
    return Gaussian

def CoefficientMatrixFunction(n, n_l, disc_phi, phi_n, x_0, y_0, x_max, y_max, convolutionMatrix):
    b_vecs = np.zeros(n_l)
    ACoefficientMatrix = np.empty([10,])
    disc_phi = np.reshape(disc_phi,n*n)
    k = 0
    for l in range(n*n*n_l):
        if l%n_l == 0 and l != 0:
            k += 1
        f = lambda y, x: gauss(y, x, disc_phi[k]) * gauss(y, x, phi_n[l%n_l])
        b_vecs[l%n_l] = scipy.integrate.dblquad(f, x_0, x_max, y_0, y_max)[0]

        if l % n_l == 9:
            dummy = np.linalg.solve(convolutionMatrix,b_vecs)
            ACoefficientMatrix = np.vstack((ACoefficientMatrix,dummy))
            ## b_vecs = np.zeros(n_l)

    ACoefficientMatrix = np.delete(ACoefficientMatrix,0,0)
    ACoefficientMatrix = np.reshape(ACoefficientMatrix,(n,n,n_l), 'C')
    return ACoefficientMatrix

#SAME AS ABOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def gb_and_gw(n,n_l,w,b, Gaussian, A_coeff):
    W_fourier = scipy.fft.rfft2(w)
    B_fourier_dummy = np.zeros((n_l, n, n))
    B_fourier = np.zeros((n_l, n, n//2 + 1))
    G_b_dummy = np.zeros((n,n//2+1))
    G_w_dummy = np.zeros((n,n//2+1))
    for l in range(n_l):
      B_fourier_dummy[l] = np.multiply(A_coeff[:,:,l], b)
      B_fourier[l] = scipy.fft.rfft2(B_fourier_dummy[l])
      G_b_dummy = np.add(G_b_dummy,np.matmul(A_coeff[:,:,l], np.multiply(Gaussian[l], W_fourier)))
      G_w_dummy = np.add(G_w_dummy,np.multiply(Gaussian[l], B_fourier[l]))
    G_b = v * scipy.fft.irfft2(G_b_dummy)
    G_w = gamma * scipy.fft.irfft2(G_w_dummy)
    return G_b, G_w


########################################################################

B,H,W = init(RES)

disc_phiB = disc_phi(B)

ConvolutionMatrix, phi_n = g_coefficient_mat(NL, XMIN, YMIN, XMAX, YMAX)

Gaussian = fourier_gaussian(phi_n, NL, XMIN, YMIN)


# Start of loop:
for i in range(5):

    CoefficientMatrix = CoefficientMatrixFunction(RES,NL, disc_phiB, phi_n, XMIN,YMIN,XMAX,YMAX, ConvolutionMatrix)

    G_b, G_w = gb_and_gw(RES,NL,B,W, Gaussian, CoefficientMatrix)

    I = infil(B)

    Lb = laplacian(B)
    Lw = laplacian(W)
    Lh = laplacianH(H)

    Db = dbdt(B, Lb, G_b)
    Dw = dwdt(B,W,H,I,Lw,G_w)
    Dh = dhdt(H,I,Lh)

    B = np.add(B,Db)
    W = np.add(W,Dw)
    H = np.add(H,Dh)

    print(B, W, H)
    print("\n")

print("Woohoo!")


def gb_and_gw(n,n_l,w,b, Gaussian, A_coeff):
    W_fourier = scipy.fft.rfft2(w)
    B_fourier_dummy = np.zeros((n_l, n, n))
    B_fourier = np.zeros((n_l, n, n//2 + 1))
    G_b_dummy = np.zeros((n,n//2+1))
    G_w_dummy = np.zeros((n,n//2+1))
    for l in range(n_l):
      B_fourier_dummy[l] = np.multiply(A_coeff[:,:,l], b)
      B_fourier[l] = scipy.fft.rfft2(B_fourier_dummy[l])
      G_w_dummy = np.add(G_w_dummy,np.multiply(Gaussian[l], B_fourier[l]))
	for i in range(n):
		for j in range(n):
			for l in range(n_l):
				G_b_dummy = np.add(G_b_dummy, A_coeff[i, j, l] * np.multiply(Gaussian[l], W_fourier))
			G_b[i, j] = v * scipy.fft.irfft2(G_b_dummy)[i, j]
    G_w = gamma * scipy.fft.irfft2(G_w_dummy)
    return G_b, G_w


def phiFunction(phi_j, phi_l, x_max, x_min, y_max, y_min):
	intr = math.sqrt(1 / 2 * ((1/phi_l)**2 + (1/phi_j)**2))
	return math.pi / (2 * ((1/phi_l)**2 + (1/phi_j)**2)) * ( scipy.special.erf(x_max * intr) - scipy.special.erf(x_min * intr)) * (scipy.special.erf(y_max * intr) - scipy.special.erf(y_min * intr))

def getCoeffMatrixEff(n, n_l, disc_phi, x_min, y_min, x_max, y_max):
	phi_n = np.zeros(n_l)
	for i in range(n_l):
		phi_n[n] = 1 + i/n_l
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
			AcoeffMatrix[:, i, j] = np.linalg.solve(M, b[i, j, :])
	return AcoeffMatrix

import math
import numpy as np
import sklearn as sk
import pandas as pd
import sympy as sp
import scipy
import scipy.fft
import matplotlib as plt
import time
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
RES = 100
NL = 5
XMIN = 0
YMIN = 0
XMAX = 6
YMAX = 6
dx = XMAX / RES
#############################################################




def init(n):

    '''
    Initializes 3 matrices with the given initial conditions for B, W, and H.
    To use this one would go into b[i,j], w[i,j] and h[i,j] wherever values of
    b(i,j,t), w(i,j,t), and h(i,j,t) are needed.
    '''

    b = np.zeros((n,n))
    w = np.ones((n,n))
    h = np.zeros((n,n))

    b[:,:] = 0.20
    w[:,:] = 0.10
    h[:,:] = 0.01


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
            Lu[i,j] = (u[i-1,j] + u[i,j-1] - 4*u[i,j] + u[(i+1)%RES,j] + u[i,(j+1)%RES]) /dx**2
    return Lu

def laplacianH(h):
    U = np.multiply(h,h)
    return laplacian(U)

def infil(u):
    '''
    Pass in the b function, returns an array for all the infiltration rates at this time step.
    '''
    return alpha * ((u[:,:] + q * F)/(u[:,:] + q))

def dbdt(b, Lb, G_b):
    return np.multiply(np.multiply(G_b,b),(1-b)) - b + deltaB*Lb

def dwdt(b,w,h,I,Lw,G_w):
    return np.multiply(I,h) - np.multiply(v*(1-rho*b),w) - np.multiply(G_w,w) + deltaH*Lw

def dhdt(h,I, Lh):
    return p - np.multiply(h,I) #+ deltaH*Lh

def disc_phi(b):
  phiDiscrete = np.zeros((RES,RES))
  for i in range(RES):
    for j in range(RES):
      phiDiscrete[i][j] = 1 + eta * b[i][j]
  return phiDiscrete

def gauss(y, x, phi):
  return 1 / (2 * math.pi) * np.exp( -1 * (x**2 + y**2) / (2 * phi**2))
#
def g_coefficient_mat(n_l, x_0, y_0, x_max, y_max):
  phi_n = np.zeros(n_l)
  for i in range(n_l):
    phi_n[i] = (x_max * (i+1))/(n_l)
  convolutionMatrix = np.zeros((n_l,n_l))
  for j in range(n_l):
    for l in range(n_l):
        f = lambda y, x: gauss(y, x, phi_n[j]) * gauss(y, x, phi_n[l])
        convolutionMatrix[j][l] = scipy.integrate.dblquad(f, x_0, x_max, y_0, y_max)[0]
        # INTEGRATE RETURNS ERROR, WE ARE NOT USING IT YET!!!!!!!!!!!
  return convolutionMatrix, phi_n

def fourier_gaussian(phi_n, n_l, x_0, y_0):
    X_and_Y = np.zeros((RES,RES))
    Gaussian = np.zeros((n_l,RES,RES//2+1))
    for l in range(n_l):
        for i in range(RES):
            for j in range(RES):
                X_and_Y[i,j] = gauss(x_0 + i/RES, y_0 + j/RES, phi_n[l])
        Gaussian[l] = scipy.fft.rfft2(X_and_Y)
    return Gaussian

def CoefficientMatrixFunction(n, n_l, disc_phi, phi_n, x_0, y_0, x_max, y_max, convolutionMatrix):
    b_vecs = np.zeros(n_l)
    ACoefficientMatrix = np.empty([n_l,])
    disc_phi = np.reshape(disc_phi,n*n)
    k = 0
    for l in range(n*n*n_l):
        if l%n_l == 0 and l != 0:
            k += 1
        f = lambda y, x: gauss(y, x, disc_phi[k]) * gauss(y, x, phi_n[l%n_l])
        b_vecs[l%n_l] = scipy.integrate.dblquad(f, x_0, x_max, y_0, y_max)[0]

        if l % n_l == (n_l-1):
            dummy = np.linalg.solve(convolutionMatrix,b_vecs)
            ACoefficientMatrix = np.vstack((ACoefficientMatrix,dummy))
            ## b_vecs = np.zeros(n_l)

    ACoefficientMatrix = np.delete(ACoefficientMatrix,0,0)
    ACoefficientMatrix = np.reshape(ACoefficientMatrix,(n,n,n_l), 'C')
    return ACoefficientMatrix

#SAME AS ABOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def gb_and_gw(n,n_l,w,b, Gaussian, A_coeff):
    W_fourier = scipy.fft.rfft2(w)
    B_fourier_dummy = np.zeros((n_l, n, n))
    B_fourier = np.zeros((n_l, n, n//2 + 1))
    G_b = np.zeros((n,n))
    G_b_dummy = np.zeros((n,n//2+1))
    G_w_dummy = np.zeros((n,n//2+1))
    for l in range(n_l):
        B_fourier_dummy[l] = np.multiply(A_coeff[:,:,l], b)
        B_fourier[l] = scipy.fft.rfft2(B_fourier_dummy[l])
        G_w_dummy = np.add(G_w_dummy,np.multiply(Gaussian[l], B_fourier[l]))
    for i in range(n):
        for j in range(n):
            for l in range(n_l):
                G_b_dummy = np.add(G_b_dummy, A_coeff[i, j, l] * np.multiply(Gaussian[l], W_fourier))
            G_b[i, j] = v * scipy.fft.irfft2(G_b_dummy)[i, j]
    G_w = gamma * scipy.fft.irfft2(G_w_dummy)
    return G_b, G_w

########################################################################
tic = time.perf_counter()

B,W,H = init(RES)

disc_phiB = disc_phi(B)

ConvolutionMatrix, phi_n = g_coefficient_mat(NL, XMIN, YMIN, XMAX, YMAX)

Gaussian = fourier_gaussian(phi_n, NL, XMIN, YMIN)


# Start of loop:
for i in range(10):
    '''
    periodic_bc(B)
    periodic_bc(W)
    periodic_bc(H)
    '''

    Lb = laplacian(B)
    Lw = laplacian(W)
    Lh = laplacianH(H)

    CoefficientMatrix = CoefficientMatrixFunction(RES,NL, disc_phiB, phi_n, XMIN,YMIN,XMAX,YMAX, ConvolutionMatrix)

    G_b, G_w = gb_and_gw(RES,NL,B,W, Gaussian, CoefficientMatrix)

    I = infil(B)

    Db = (dbdt(B, Lb, G_b)) /10
    Dw = (dwdt(B,W,H,I,Lw,G_w)) /10
    Dh = (dhdt(H,I,Lh)) /10

    B = np.add(B,Db)
    W = np.add(W,Dw)
    H = np.add(H,Dh)


toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

print("Woohoo!")