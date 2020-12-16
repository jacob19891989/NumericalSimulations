import numpy as np  
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pylab

#The function f
def func_f(t):
    return np.sin(2*np.pi*t)

#The initial Lipschitz condition
def u0_L(x):
    return max(1-16*(x-0.25)**2,0.0)

#Vectorizing the function u0_L
u0LVector = np.vectorize(u0_L, otypes=[np.float])

#Exact solution of the initial condition u0_L
def Exactu0_L(x,t):
    return u0_L(-0.5*1/np.pi+x+0.5*np.cos(2*np.pi*t)/np.pi)

#Vectorizing Exactu0_L
UEu0L=np.vectorize(Exactu0_L,otypes=[np.float])


#Function that finds the nearest neighbors of an element in an array
def findN(values,vecx):
    for ii in values:
        if ii<vecx[0]: ii=vecx[0]
        elif ii>vecx[len(vecx)-1]: ii=vecx[len(vecx)-1]
        LB=vecx[vecx<=ii].max()
        UB=vecx[vecx>=ii].min()
        yield (ii,np.where(vecx==LB)[0],np.where(vecx==UB)[0])
        
#Lagrange interpolation
def InterPol(values,vecx,vecu,Deltax):
    result=np.zeros(len(values))
    kk=0
    for val,ii,jj in findN(values,vecx):
        result[kk]=1/Deltax*((vecx[jj]-val)*vecu[ii]+(val-vecx[ii])*vecu[jj])
        kk=kk+1
    return result
        
#The numerical solution using SL Scheme
def Upu0_L(x,t,Deltax):
    Vji=np.zeros((len(x),len(t)))
    Vji[:,0]=u0LVector(x)    
    for i,k in enumerate(t[1:]):        
        values=x-np.full(len(x),Deltat*func_f(k))
        Vji[:,i+1]=InterPol(values,x,Vji[:,i],Deltax)
    return Vji

#Discretization of space and time
Deltax=1/200
Deltat=Deltax/40
x=np.arange(start=0,stop=1,step=Deltax)
t=np.arange(start=0,stop=0.5,step=Deltat)

#Evolution in space of the 

Vji=Upu0_L(x,t,Deltax)

#t=0
plt.plot(x,Vji[:,0],'x',marker="o",color="red")
plt.plot(x,UEu0L(x,t[0]),color="blue")
plt.show()
#np.savetxt('Data/Spacet0.dat',np.stack((x,Vji[:,0]),axis=-1))
print('The error on the infinity norm for the SL Scheme: ')
print(LA.norm(Vji[:,0]-UEu0L(x,t[0]),np.inf))

#t=0.25
print(np.where(t==0.25)[0])
plt.plot(x,Vji[:,np.where(t==0.25)[0]],'x',marker="o",color="red")
plt.plot(x,UEu0L(x,t[np.where(t==0.25)[0]]),color="blue")
plt.show()
#np.savetxt('Data/Spacet0.25.dat',np.stack((x,Vji[:,2000]),axis=-1))
print('The error on the infinity norm is: ')
print(LA.norm(Vji[:,2000]-UEu0L(x,t[2000]),np.inf))

#t=0.5
print(t[len(t)-1])
plt.plot(x,Vji[:,len(t)-1],'x',marker="o",color="red")
plt.plot(x,UEu0L(x,t[len(t)-1]),color="blue")
plt.show()

#np.savetxt('Data/Spacet0.5.dat',np.stack((x,Vji[:,len(t)-1]),axis=-1))
print('The error on the infinity norm is: ')
print(LA.norm(Vji[:,len(t)-1]-UEu0L(x,t[len(t)-1]),np.inf))

#Evolution in time
print(x[10])
plt.plot(t,Vji[10],'x',marker="o",color="red")
plt.plot(t,UEu0L(x[10],t),color="blue")
plt.show()
#np.savetxt('Data/Timex10.dat',np.stack((t,Vji[10]),axis=-1))

print(x[50])
plt.plot(t[:2300],Vji[50,:2300],'x',marker="o",color="red")
plt.plot(t[:2300],UEu0L(x[50],t[:2300]),color="blue")
plt.show()
#np.savetxt('Data/Timex50.dat',np.stack((t,Vji[50]),axis=-1))

plt.plot(t,Vji[100],'x',marker="o",color="red")
plt.plot(t,UEu0L(x[100],t),color="blue")
plt.show()
#np.savetxt('Data/Timex100.dat',np.stack((t,Vji[100]),axis=-1))

#Error analysis
cuenta=0
error=np.zeros(9)
while cuenta<9:
    print('step: ',cuenta)
    Deltax=1/(100+100*cuenta)
    Deltat=Deltax/40
    x=np.arange(start=0,stop=1,step=Deltax)
    t=np.arange(start=0,stop=0.25,step=Deltat)
    Vji=Upu0_L(x,t,Deltax)
    error[cuenta]=LA.norm(Vji[:,len(t)-1]-UEu0L(x,t[len(t)-1]),np.inf)
    cuenta=cuenta+1

plt.plot(np.arange(1,10),error)
plt.show()
