import numpy as np
from numpy import pi, sqrt, sin, cos, exp, log10, array, real, conj
import cuentitas
import random

# Rutina de integracion gaussiana en 2 dimensiones, el input es la cantidad de nodos a utilizar
def integracionGauss(n): 
# Precalculamos nodos y pesos (única vez para todas las integrales
    beta = np.zeros(n,dtype=float)
    for k in range(1,n+1):
        beta[k-1] = 0.5/np.sqrt(1-(2*(k))**(-2))

    m = n+1
    T_low = np.zeros((m,m))
    T_up = np.zeros((m,m))
    T = np.zeros((m,m))
    
    # defino T_low
    for i in range(0,m):
        for j in range(0,m):
            if i==j+1:
                T_low[i,j]=beta[i-1]

    # defino T_up
    for i in range(0,m):
        for j in range(0,m):
            if j==i+1:
                T_up[i,j]=beta[i]


    T = T_low + T_up        
    d_,V = np.linalg.eig(T)
    D = np.zeros((m,m))
 
    for i in range(0,m):
        for j in range(0,m):
            if i==j:
                D[i,j]=d_[i]

    W = (2*V[0,:]**2)
    Wt = np.kron(W,W)

    X,Y = np.meshgrid(d_,d_)
    return X,Y,Wt

# funcion para discriminar los datos de cada zona medida por cassini
# devuelve en el orden(angulo - observacion (3 s0 y 1 emis))
def medicion(zona):     #angulo - observacion
    if zona == 'VP1':
        out = [np.asarray([20.44, 18.79, 17.36]),np.asarray([-6.13, -5.71, -5.66, 0.927])]
    elif zona == 'VP2':
        out = [np.asarray([18.63, 17.19, 11.25]),np.asarray([-6.65, -6.17, -5.93, 0.937])]
    elif zona == 'HU0':
        out = [np.asarray([24.59, 15.76, 20.69]),np.asarray([-4.07, -3.27, -3.59, 0.917])]
    elif zona == 'HU1':
        out = [np.asarray([22.46, 15.33, 15.57]),np.asarray([-3.64, -3.12, -2.90, 0.921])]
    elif zona == 'HU2':
        out = [np.asarray([18.86, 21.10, 12.98]),np.asarray([-4.08, -9.34, -3.55, 0.914])]
    elif zona == 'DU1':
        out = [np.asarray([17.20, 20.10, 12.98]),np.asarray([-9.02, -9.55, -8.09, 0.943])]
    elif zona == 'DU2':
        out = [np.asarray([23.84, 11.21, 12.64]),np.asarray([-11.96, -6.82, -7.99, 0.954])]
    elif zona == 'DP1':
        out = [np.asarray([19.64, 17.76, 14.06]),np.asarray([-7.79, -8.34, -7.44, 0.947])]
    elif zona == 'DP2':
        out = [np.asarray([24.21, 13.78, 16.76]),np.asarray([-11.25, -8.29, -8.62, 0.937])]
    else:
      print('\n Inputs: VP1, VP2, HU0, HU1, HU2, DU1, DU2, DP1, DP2 \n')
    
    return out


# Actua como sigmaEmi (descripto mas abajo) pero sin la parte de emisividad
def sigma(ep1,ep2,d,s1,l1,s2,l2,angulo): #funcion de antes pero sin la parte tensorial
    landa = 0.025
    k0 = 2*np.pi/landa
    phi = np.pi
    ### s0
    phs = phi + np.pi
    thi = angulo  #cambiar segun la zona
    thi = thi*np.pi/180 #DU2 T008, T061, T021
    ths = thi
    k1 = k0*(sin(ths)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(ths)*sin(phs)-sin(thi)*sin(phi))
    
    a1f1 = cuentitas.a1VVF1(k0,thi,phi,ths,phs,ep1,ep2,d)
    a1f2 = cuentitas.a1VVF2(k0,thi,phi,ths,phs,ep1,ep2,d)

    w1 = cuentitas.w(s1,l1,k1,k2)
    w2 = cuentitas.w(s2,l2,k1,k2)
    w12 = cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2)
    
    aux=4*np.pi*k0**2*cos(ths)**2*(abs(a1f1)**2*w1+abs(a1f2)**2*w2+2*real(a1f1*conj(a1f2))*w12)
    
    s0s = 10*np.log10(aux) #db

    return s0s


def sigmaVV(ep1, ep2, d, s1, l1, s2, l2, angulo, landa):
    '''
    devulevle s0VV con funcion de parametros y landa
    '''
#     landa = 0.025
    k0 = 2*np.pi/landa
    phi = np.pi
    ### s0
    phs = phi + np.pi
    thi = angulo  #cambiar segun la zona
    thi = thi*np.pi/180 #DU2 T008, T061, T021
    ths = thi
    k1 = k0*(sin(ths)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(ths)*sin(phs)-sin(thi)*sin(phi))
    
    a1vvf1 = cuentitas.a1VVF1(k0,thi,phi,ths,phs,ep1,ep2,d)
    a1vvf2 = cuentitas.a1VVF2(k0,thi,phi,ths,phs,ep1,ep2,d)

    w1 = cuentitas.w(s1,l1,k1,k2)
    w2 = cuentitas.w(s2,l2,k1,k2)
    w12 = cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2)
    
    aux=4*np.pi*k0**2*cos(ths)**2*( abs(a1vvf1)**2*w1 + abs(a1vvf2)**2*w2 + 2*real(a1vvf1*conj(a1vvf2))*w12 )
    
    s0s = 10*np.log10(aux) #db

    return s0s



def sigmaHH(ep1, ep2, d, s1, l1, s2, l2, angulo, landa):
    '''
    devulevle s0HH con funcion de parametros y landa
    '''
#     landa = 0.025
    k0 = 2*np.pi/landa
    phi = np.pi
    ### s0
    phs = phi + np.pi
    thi = angulo  #cambiar segun la zona
    thi = thi*np.pi/180 #DU2 T008, T061, T021
    ths = thi
    k1 = k0*(sin(ths)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(ths)*sin(phs)-sin(thi)*sin(phi))
    
    a1hhf1 = cuentitas.a1HHF1(k0,thi,phi,ths,phs,ep1,ep2,d)
    a1hhf2 = cuentitas.a1HHF2(k0,thi,phi,ths,phs,ep1,ep2,d)
    
    w1 = cuentitas.w(s1,l1,k1,k2)
    w2 = cuentitas.w(s2,l2,k1,k2)
    w12 = cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2)
       
    aux=4*np.pi*k0**2*cos(ths)**2*( abs(a1hhf1)**2*w1 + abs(a1hhf2)**2*w2 + 2*real(a1hhf1*conj(a1hhf2))*w12 )
    
    
    s0s = 10*np.log10(aux) #db

    return s0s

def sigmaHV(ep1, ep2, d, s1, l1, s2, l2, angulo, landa):
    '''
    devulevle s0HV con funcion de parametros y landa
    '''
#     landa = 0.025
    k0 = 2*np.pi/landa
    phi = np.pi
    ### s0
    phs = phi + np.pi
    thi = angulo  #cambiar segun la zona
    thi = thi*np.pi/180 #DU2 T008, T061, T021
    ths = thi
    k1 = k0*(sin(ths)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(ths)*sin(phs)-sin(thi)*sin(phi))
    
    a1hvf1 = cuentitas.a1HVF1(k0,thi,phi,ths,phs,ep1,ep2,d)
    a1hvf2 = cuentitas.a1HVF2(k0,thi,phi,ths,phs,ep1,ep2,d)
    
    w1 = cuentitas.w(s1,l1,k1,k2)
    w2 = cuentitas.w(s2,l2,k1,k2)
    w12 = cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2)
       
    aux=4*np.pi*k0**2*cos(ths)**2*( abs(a1hvf1)**2*w1 + abs(a1hvf2)**2*w2 + 2*real(a1hvf1*conj(a1hvf2))*w12 )
    
    
    s0s = 10*np.log10(aux) #db

    return s0s


# Modelo de scattering con la totalidad de los parametros a ingresar
def S0VV_completo(k0,thi,phi,th,phs,ep1,ep2,d,s1,l1,s2,l2):
    
    k1 = k0*(sin(th)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(th)*sin(phs)-sin(thi)*sin(phi))
    
    return 4*np.pi*k0**2*cos(th)**2*(abs(cuentitas.a1VVF1(k0,thi,phi,th,phs,ep1,ep2,d))**2*cuentitas.w(s1,l1,k1,k2)+\
            abs(cuentitas.a1VVF2(k0,thi,phi,th,phs,ep1,ep2,d))**2*cuentitas.w(s2,l2,k1,k2)+\
            2*real(cuentitas.a1VVF1(k0,thi,phi,th,phs,ep1,ep2,d)*conj(cuentitas.a1VVF2(k0,thi,phi,th,phs,ep1,ep2,d)))*cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2))

# Canal HH
def S0HH_completo(k0,thi,phi,th,phs,ep1,ep2,d,s1,l1,s2,l2):
    
    k1 = k0*(sin(th)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(th)*sin(phs)-sin(thi)*sin(phi))
    
    return 4*np.pi*k0**2*cos(th)**2*(abs(cuentitas.a1HHF1(k0,thi,phi,th,phs,ep1,ep2,d))**2*cuentitas.w(s1,l1,k1,k2)+\
            abs(cuentitas.a1HHF2(k0,thi,phi,th,phs,ep1,ep2,d))**2*cuentitas.w(s2,l2,k1,k2)+\
            2*real(cuentitas.a1HHF1(k0,thi,phi,th,phs,ep1,ep2,d)*conj(cuentitas.a1HHF2(k0,thi,phi,th,phs,ep1,ep2,d)))*cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2))



# el input corresponde a los parametros usuales, anguloWR es el angulo thi incidente en forma de array (WR denota que es sin el ruido generado)
# X e Y son los nodos de integracion gaussiana, Wt es el vector que se genera, todos sacados de integracionGauss(n) y m es la cantidad de nodos
# el retorno: s0s es el valor del coeficiente de backscattering, me devuelve tantos como angulosWR le haya dado, emi es la emisividad calculada 
# en cada uno de los angulos anguloWR
def sigmaEmi(ep1,ep2,d,s1,l1,s2,l2,anguloWR,X,Y,Wt,m): 
    landa = 0.025
    k0 = 2*np.pi/landa
    phi = np.pi
    ### s0
    phs = phi + np.pi
    thi = anguloWR  #cambiar segun la zona
    thi = thi*np.pi/180 #DU2 T008, T061, T021
    ths = thi
    k1 = k0*(sin(ths)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(ths)*sin(phs)-sin(thi)*sin(phi))
    
    a1f1 = cuentitas.a1VVF1(k0,thi,phi,ths,phs,ep1,ep2,d)
    a1f2 = cuentitas.a1VVF2(k0,thi,phi,ths,phs,ep1,ep2,d)

    w1 = cuentitas.w(s1,l1,k1,k2)
    w2 = cuentitas.w(s2,l2,k1,k2)
    w12 = cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2)
    
    aux=4*np.pi*k0**2*cos(ths)**2*(abs(a1f1)**2*w1+abs(a1f2)**2*w2+2*real(a1f1*conj(a1f2))*w12)
    
    s0s = 10*np.log10(aux)

    ths_ = (X+1)*np.pi/4
    phs_ = (Y+1)*np.pi
    
    ths_1 = np.reshape(ths_, (1,m**2))
    phs_1 = np.reshape(phs_, (1,m**2))
    
    q = (np.sin(ths_1)*S0VV_completo(k0,thi,phi,ths_1,phs_1,ep1,ep2,d,s1,l1,s2,l2))
    aux = Wt*q
    I_gauss = (np.pi**2/4)*aux.sum()
 
    emi = 1 - (I_gauss/(4*np.pi))
    
    return [s0s,emi]




def sigmaEmiVV(ep1,ep2,d,s1,l1,s2,l2,anguloWR,X,Y,Wt,m,landa): 
#     landa = 0.025
    k0 = 2*np.pi/landa
    phi = np.pi
    ### s0
    phs = phi + np.pi
    thi = anguloWR  #cambiar segun la zona
    thi = thi*np.pi/180 #DU2 T008, T061, T021
    ths = thi
    k1 = k0*(sin(ths)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(ths)*sin(phs)-sin(thi)*sin(phi))
    
    a1vvf1 = cuentitas.a1VVF1(k0,thi,phi,ths,phs,ep1,ep2,d)
    a1vvf2 = cuentitas.a1VVF2(k0,thi,phi,ths,phs,ep1,ep2,d)

    w1 = cuentitas.w(s1,l1,k1,k2)
    w2 = cuentitas.w(s2,l2,k1,k2)
    w12 = cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2)
    
    aux=4*np.pi*k0**2*cos(ths)**2*(abs(a1vvf1)**2*w1+abs(a1vvf2)**2*w2+2*real(a1vvf1*conj(a1vvf2))*w12)
    
    s0s = 10*np.log10(aux)

    ths_ = (X+1)*np.pi/4
    phs_ = (Y+1)*np.pi
    
    ths_1 = np.reshape(ths_, (1,m**2))
    phs_1 = np.reshape(phs_, (1,m**2))
    
    q = (np.sin(ths_1)*S0VV_completo(k0,thi,phi,ths_1,phs_1,ep1,ep2,d,s1,l1,s2,l2))
    aux = Wt*q
    I_gauss = (np.pi**2/4)*aux.sum()
 
    emi = 1 - (I_gauss/(4*np.pi))
    
    return [s0s,emi]


def sigmaEmiHH(ep1,ep2,d,s1,l1,s2,l2,anguloWR,X,Y,Wt,m,landa): 
#     landa = 0.025
    k0 = 2*np.pi/landa
    phi = np.pi
    ### s0
    phs = phi + np.pi
    thi = anguloWR  #cambiar segun la zona
    thi = thi*np.pi/180 #DU2 T008, T061, T021
    ths = thi
    k1 = k0*(sin(ths)*cos(phs)-sin(thi)*cos(phi))
    k2 = k0*(sin(ths)*sin(phs)-sin(thi)*sin(phi))
    
    a1hhf1 = cuentitas.a1HHF1(k0,thi,phi,ths,phs,ep1,ep2,d)
    a1hhf2 = cuentitas.a1HHF2(k0,thi,phi,ths,phs,ep1,ep2,d)

    w1 = cuentitas.w(s1,l1,k1,k2)
    w2 = cuentitas.w(s2,l2,k1,k2)
    w12 = cuentitas.w_f1f2(s1,l1,s2,l2,k1,k2)
    
    aux=4*np.pi*k0**2*cos(ths)**2*(abs(a1hhf1)**2*w1+abs(a1hhf2)**2*w2+2*real(a1hhf1*conj(a1hhf2))*w12)
    
    s0s = 10*np.log10(aux)

    ths_ = (X+1)*np.pi/4
    phs_ = (Y+1)*np.pi
    
    ths_1 = np.reshape(ths_, (1,m**2))
    phs_1 = np.reshape(phs_, (1,m**2))
    
    q = (np.sin(ths_1)*S0HH_completo(k0,thi,phi,ths_1,phs_1,ep1,ep2,d,s1,l1,s2,l2))
    aux = Wt*q
    I_gauss = (np.pi**2/4)*aux.sum()
 
    emi = 1 - (I_gauss/(4*np.pi))
    
    return [s0s,emi]



#me devuelve array de 2d con 2*n elementos en total, n angulos entre 10 y 30 y n sigma0
def generador(n):
    angulos = []
    for i in range(n):
        angulos.append(random.uniform(10,30))
    
    res = []
    for j in range(len(angulos)):
        res.append(sigma(1.93,3,0.05,0.0017,0.015,0.001,0.015,angulos[j]))

    return np.asarray([np.asarray(angulos), np.asarray(res)])



def generador1E(n): 
    '''
    me devuelve array de 2d con 2*n+1 elementos en total, n angulos entre 10 y 30 en el primer casillero y (n       sigma0 + 1 emisividad) en el segundo
    '''
    m = 51
    X,Y,Wt = integracionGauss(m-1)
    angulos = []
    for i in range(n):
        angulos.append(random.uniform(10,30))
    
    s0 = []
    for j in range(len(angulos)):
        s0.append(sigma(1.93,5,0.05,0.0017,0.015,0.003,0.01,angulos[j]))
        #s0.append(sigma(1.93,3,0.05,0.0017,0.015,0.001,0.015,angulos[j])) original que usaba
        
    emi = sigmaEmi(1.93,5,0.05,0.0017,0.015,0.003,0.01,0.0001,X,Y,Wt,m)[1]
    #sigmaEmi(1.93,3,0.05,0.0017,0.015,0.001,0.015,0.0001,X,Y,Wt,m)[1]
    
    out = s0 + [emi]
    return np.asarray([np.asarray(angulos), np.asarray(out)], dtype=object)


def generadorVV(ep1, ep2, d, s1, l1, s2, l2, incAng, landa): 
    m = 51
    X,Y,Wt = integracionGauss(m-1)
    
    s0 = []
    for j in range(len(incAng)):
        for k in range(len(landa)):
#         s0.append(sigma(ep1,ep2,d,s1,l1,s2,l2,incAng[j]))
            s0.append(sigmaVV(ep1, ep2, d, s1, l1, s2, l2, incAng[j], landa[k]))
    
    emi = []
    for k in range(len(landa)):
        emi.append(sigmaEmiVV(ep1,ep2,d,s1,l1,s2,l2,0.0001,X,Y,Wt,m,landa[k])[1])
    #sigmaEmi(1.93,3,0.05,0.0017,0.015,0.001,0.015,0.0001,X,Y,Wt,m)[1]
    
    out = s0 + emi
    return np.asarray([np.asarray(incAng), np.asarray(out)], dtype=object)


def generadorHH(ep1, ep2, d, s1, l1, s2, l2, incAng, landa): 
    m = 51
    X,Y,Wt = integracionGauss(m-1)
    
    s0 = []
    for j in range(len(incAng)):
        for k in range(len(landa)):
#         s0.append(sigma(ep1,ep2,d,s1,l1,s2,l2,incAng[j]))
            s0.append(sigmaHH(ep1, ep2, d, s1, l1, s2, l2, incAng[j], landa[k]))
    
    emi = []
    for k in range(len(landa)):
        emi.append(sigmaEmiHH(ep1,ep2,d,s1,l1,s2,l2,0.0001,X,Y,Wt,m,landa[k])[1])
    #sigmaEmi(1.93,3,0.05,0.0017,0.015,0.001,0.015,0.0001,X,Y,Wt,m)[1]
    
    out = s0 + emi
    return np.asarray([np.asarray(incAng), np.asarray(out)], dtype=object)



def generadorHHVV(ep1, ep2, d, s1, l1, s2, l2, incAng, landa): 

    s0HH = generadorHH(ep1,ep2,d,s1,l1,s2,l2,incAng,landa)
    s0VV = generadorVV(ep1,ep2,d,s1,l1,s2,l2,incAng,landa)
        
    hh = len(s0HH[1])
    vv = len(s0VV[1])
    
    a = np.zeros(hh+vv)
    l = len(landa)
        
        
          #Ordeno primero s0HH y s0VV
    for i in range(0,hh-l):
        a[i] = s0HH[1][i]
            
    for j in range(0,vv-l):
        a[hh-l+j] = s0VV[1][j]
            
          #Ordeno emiHH y emiVV
    for m in range(0,l):
        a[hh+vv-2*l+m] = s0HH[1][hh-l+m]
            
    for k in range(0,l):
        a[hh+vv-l+k] = s0VV[1][vv-l+k]
        
    return np.asarray([np.asarray(incAng), np.asarray(a)], dtype=object)


def generadorHHVVlambda(ep1, ep2, d, s1, l1, s2, l2, incAng, landa): 

    s0HH = generadorHH(ep1,ep2,d,s1,l1,s2,l2,incAng,[landa[0]])
    s0VV = generadorVV(ep1,ep2,d,s1,l1,s2,l2,incAng,[landa[1]])
        
    hh = len(s0HH[1])
    vv = len(s0VV[1])
    
    a = np.zeros(hh+vv)
    
    a[0] = s0HH[1][0]
    a[1] = s0VV[1][0]
    a[2] = s0HH[1][1]
    a[3] = s0VV[1][1]
    
    return np.asarray([np.asarray(incAng), np.asarray(a)], dtype=object)

def generadorFran1Capa(ep, s, l, incAng, landa): 
    m = 51
    X,Y,Wt = integracionGauss(m-1)
    
    s0 = []
    for j in range(len(incAng)):
        for k in range(len(landa)):
#         s0.append(sigma(ep1,ep2,d,s1,l1,s2,l2,incAng[j]))
            s0.append(sigmaVV(1, ep, 0.1, 0.001, 0.01, s, l, incAng[j], landa[k]))
    
    emi = []
    for k in range(len(landa)):
        emi.append(sigmaEmiVV(1,ep,0.1,0.001,0.01,s,l,0.0001,X,Y,Wt,m,landa[k])[1])
    #sigmaEmi(1.93,3,0.05,0.0017,0.015,0.001,0.015,0.0001,X,Y,Wt,m)[1]
    
    out = s0 + emi
    return np.asarray([np.asarray(incAng), np.asarray(out)], dtype=object)

#me devuelve 2d array con n angulos en el primer lugar (entre 10 y 30) y 2n en el segundo con n sigma0 + n emisividades
def generadorEmisividad(n): 
    m = 51
    X,Y,Wt = integracionGauss(m-1)
    angulos = []
    for i in range(n):
        angulos.append(random.uniform(10,30))
    
    s0 = []
    emi = []
    for j in range(len(angulos)):
        s0.append(sigmaEmi(1.93,3,0.05,0.0017,0.015,0.001,0.015,angulos[j],X,Y,Wt,m)[0])
        emi.append(sigmaEmi(1.93,3,0.05,0.0017,0.015,0.001,0.015,angulos[j],X,Y,Wt,m)[1])
    
    out = s0+emi
    return np.asarray([np.asarray(angulos), np.asarray(out)], dtype=object)