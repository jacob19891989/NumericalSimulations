import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pylab

#Defining the Hamiltonian

def H(alpha):
    return 0.5*alpha**2

#Defining the function of the Upwind Scheme
def H_Up(alpha,beta,alpha0=0.0):
    if alpha>=alpha0 and beta>=alpha0:
        return H(alpha)
    elif alpha>=alpha0 and beta<=alpha0:
        return H(beta)+H(alpha)
    elif alpha<=alpha0 and beta>=alpha0:
        return 0.0
    else:
        return H(beta)
    
#Vectorizing the function H_Up
HUpVec=np.vectorize(H_Up,otypes=[np.float])

#The initial Lipschitz condition
def u0_L(x):
    return max(1-16*(x-0.5)**2,0.0)

#Vectorizing the function u0_L
u0LVector = np.vectorize(u0_L, otypes=[np.float])

#Exact solution of the initial condition u0_L
def Exactu0_L(x,t):
    if 1/4<=x and x<=3/4:
        return (np.absolute(x-0.5)-1/4)**2/(2*t)
    else:
        return 0.0
    
#Vectorizing Exactu0_L
UEu0L=np.vectorize(Exactu0_L,otypes=[np.float])

#Exact solution of the initial condition u0_SC
def Exactu0_SC(x,t):
    return min(np.absolute(x-0.5)**2/(2*t+1/16)-1,0)

#Vectorizing Exactu0_SC
UEu0SC=np.vectorize(Exactu0_SC,otypes=[np.float])

#The numerical solution of the First initial condition using Upwind Scheme
def Upu0_L(x,t):
    Vji=np.zeros((len(x),len(t)))
    Vji[:,0]=u0LVector(x)    
    for i,k in enumerate(t[1:]):
        DjVi=np.diff(Vji[:,i])/Deltax
        Vji[1:len(x)-1,i+1]=Vji[1:len(x)-1,i]-Deltat*HUpVec(DjVi[:len(DjVi)-1],DjVi[1:])
    return Vji

#The numerical solution of the Second intial condition using Upwind Scheme
def Upu0_SC(x,t):
    Vji=np.zeros((len(x),len(t)))
    Vji[:,0]=-u0LVector(x)
    for i,k in enumerate(t[1:]):
        DjVi=np.diff(Vji[:,i])/Deltax
        Vji[1:len(x)-1,i+1]=Vji[1:len(x)-1,i]-Deltat*HUpVec(DjVi[:len(DjVi)-1],DjVi[1:])
    return Vji

#This function is for the LF Scheme
def HLf(alpha,beta,Deltax,Deltat):
    return(H((alpha+beta)*0.5)-0.5*(Deltax/Deltat)*(beta-alpha))

#Vectorizing HLf
HLFVec=np.vectorize(HLf,otypes=[np.float])

#The numerical solution of the First initial condition using LF Scheme
def LFu0_L(x,t,Deltax,Deltat):
    Vji=np.zeros((len(x),len(t)))
    Vji[:,0]=u0LVector(x)
    for i,k in enumerate(t[1:]):
        DjVi=np.diff(Vji[:,i])/Deltax
        Vji[1:len(x)-1,i+1]=Vji[1:len(x)-1,i]-Deltat*HLFVec(DjVi[:len(DjVi)-1],DjVi[1:],Deltax,Deltat)
    return Vji

#The numerical solution of the second initial condition using LF Scheme
def LFu0_SC(x,t,Deltax,Deltat):
    Vji=np.zeros((len(x),len(t)))
    Vji[:,0]=-u0LVector(x)
    for i,k in enumerate(t[1:]):
        DjVi=np.diff(Vji[:,i])/Deltax
        Vji[1:len(x)-1,i+1]=Vji[1:len(x)-1,i]-Deltat*HLFVec(DjVi[:len(DjVi)-1],DjVi[1:],Deltax,Deltat)
    return Vji

#Discretization of space and time
Deltax=1/200
Deltat=Deltax/40
x=np.arange(start=0,stop=1,step=Deltax)
t=np.arange(start=0,stop=0.05,step=Deltat)

#Evolution in space of the Up Scheme for the first initial condition and t=0.049875

Vji=Upu0_L(x,t)
plt.plot(x,UEu0L(x,t[len(t)-1]),color="blue")
plt.plot(x,Vji[:,len(t)-1],'x',marker="o",color="red")
plt.show()
np.savetxt('Data/FirstUpSpace.dat',np.stack((x,Vji[:,len(t)-1]),axis=-1))
np.savetxt('Data/FirstUpTime.dat',np.stack((t,Vji[100]),axis=-1))
print('The error on the infinity norm for the Up Wind Scheme using the first initial condition and t=0.049875 is: ')
print(LA.norm(Vji[:,len(t)-1]-UEu0L(x,t[len(t)-1]),np.inf))

#Evolution in time of the Up Scheme for the first initial condition and x=0.5

plt.plot(t[1:],UEu0L(x[100],t[1:]),color="blue")
plt.plot(t,Vji[100],'x',marker="o",color="red")
plt.show()

#Evolution in space of the Up Scheme for the second intial condition with t=0.049875

Vji=Upu0_SC(x,t)
plt.plot(x,UEu0SC(x,t[len(t)-1]),color="blue")
plt.plot(x,Vji[:,len(t)-1],'x',marker="o",color="red")
plt.show()
np.savetxt('Data/SecondUpSpace.dat',np.stack((x,Vji[:,len(t)-1]),axis=-1))
np.savetxt('Data/SecondUpTime.dat',np.stack((t,Vji[160]),axis=-1))
print('The error on the infinity norm for the Up Wind Scheme using the second initial condition and t=0.049875 is: ')
print(LA.norm(Vji[:,len(t)-1]-UEu0SC(x,t[len(t)-1]),np.inf))

#Evolution in time of the Up wind Scheme for the second intial condition and x=0.8

plt.plot(t[1:],UEu0SC(x[160],t[1:]),color="blue")
plt.plot(t,Vji[160],'x',marker="o",color="red")
plt.show()

#Evolution in space of the LF Scheme for the first initial condition and t=0.049875

Vji=LFu0_L(x,t,Deltax,Deltat)
plt.plot(x,UEu0L(x,t[len(t)-1]),color="blue")
plt.plot(x,Vji[:,len(t)-1],'x',marker="o",color="red")
plt.show()
np.savetxt('Data/FirstLFSpace.dat',np.stack((x,Vji[:,len(t)-1]),axis=-1))
np.savetxt('Data/FirstLFTime.dat',np.stack((t,Vji[100]),axis=-1))
print('The error on the infinity norm for the LF Scheme using the first initial condition and t=0.049875 is: ')
print(LA.norm(Vji[:,len(t)-1]-UEu0L(x,t[len(t)-1]),np.inf))

#Evolution in time of the LF Scheme for the first initial condition and x=0.5

plt.plot(t[1:],UEu0L(x[100],t[1:]),color="blue")
plt.plot(t,Vji[100],'x',marker="o",color="red")
plt.show()

#Evolution in space of the LF Scheme for the second initial condition and t=0.049875

Vji=LFu0_SC(x,t,Deltax,Deltat)
plt.plot(x,UEu0SC(x,t[len(t)-1]),color="blue")
plt.plot(x,Vji[:,len(t)-1],'x',marker="o",color="red")
plt.show()
np.savetxt('Data/SecondLFSpace.dat',np.stack((x,Vji[:,len(t)-1]),axis=-1))
np.savetxt('Data/SecondLFTime.dat',np.stack((t,Vji[160]),axis=-1))
print('The error on the infinity norm for the LF Scheme using the second initial condition and t=0.049875 is: ')
print(LA.norm(Vji[:,len(t)-1]-UEu0SC(x,t[len(t)-1]),np.inf))

#Evolution in time of the LF Scheme for the second intial condition and x=0.8

plt.plot(t[1:],UEu0SC(x[160],t[1:]),color="blue")
plt.plot(t,Vji[160],'x',marker="o",color="red")
plt.show()
