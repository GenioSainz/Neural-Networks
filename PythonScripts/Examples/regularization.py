
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(59)

a = -3
b = 3
n = 9
N = 200

x  = np.linspace(a,b,N)
y  = x**2

xs = np.linspace(a,b,n)
ys = xs**2 + np.random.randn(n)

pol1  = np.polyfit(xs,ys,1)
yfit1 = np.polyval(pol1,x)

pol2  = np.polyfit(xs,ys,2)
yfit2 = np.polyval(pol2,x)

poln  = np.polyfit(xs,ys,n-1)
yfitn = np.polyval(poln,x)


## plot mean digits  https://matplotlib.org/stable/tutorials/introductory/customizing.html
##############################
plt.rcParams.update(plt.rcParamsDefault)
plt.close('all')

SMALL_SIZE  = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font',  size=SMALL_SIZE)

# TITLE
plt.rc('axes',titlesize=BIGGER_SIZE)
plt.rc('axes',titleweight='bold') 
# XY-LABELLS
plt.rc('axes',labelsize=SMALL_SIZE)  
# XY-TICKS
plt.rc('xtick',labelsize=SMALL_SIZE)   
plt.rc('ytick',labelsize=SMALL_SIZE)   
# LEGEND
plt.rc('legend',fontsize =SMALL_SIZE)
plt.rc('legend',framealpha=1)           
plt.rc('legend',loc='upper center')     
# LINES
plt.rc('lines',linewidth=2)
# GRID
plt.rc('axes' ,grid=True)


px2inch = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, axs = plt.subplots(1,3, constrained_layout=True,figsize=(1800*px2inch , 600*px2inch))

xlims = [a-1,b+1]
ylims = [np.floor(np.min(yfitn)),np.ceil(np.max(yfitn))+2]

rms0 = np.sum( (y-yfit1)**2/N )
txt0 = "Underfitting: Grade {}\n MSE {:.3f}".format(1,rms0)
## plot 0
###########
axs[0].plot(x,y,'r'    ,label='Mathematical Model')
axs[0].plot(x,yfit1,'b',label='Neural Net Model')
axs[0].plot(xs,ys,'go' ,label='Train Inputs')
axs[0].set_title(txt0)
axs[0].set_xlabel('Model is too simple')
axs[0].set_xlim(xlims)
axs[0].set_ylim(ylims)
axs[0].legend()


rms1 = np.sum( (y-yfit2)**2/N )
txt1 = "Optimal: Grade {}\n MSE: {:.3f}".format(2,rms1)
## plot 1
###########
axs[1].plot(x,y,'r'    ,label='True Mathematical Model')
axs[1].plot(x,yfit2,'b',label='Neural Network Model')
axs[1].plot(xs,ys,'go' ,label='Samples: Train Inputs')
axs[1].set_title(txt1)
axs[1].set_xlabel('Model generalises well')
axs[1].set_xlim(xlims)
axs[1].set_ylim(ylims)
axs[1].legend()


rms2 = np.sum( (y-yfitn)**2/N )
txt2 = "Overfitting: Grade {}\n MSE: {:.3f}".format(n-1,rms2)
## plot 2
###########
axs[2].plot(x,y,'r'    ,label='True Mathematical Model')
axs[2].plot(x,yfitn,'b',label='Neural Network Model')
axs[2].plot(xs,ys,'go' ,label='Samples: Train Inputs')
axs[2].set_title(txt2)
axs[2].set_xlabel('Model is too complex and even captures the inputs noise')
axs[2].set_xlim(xlims)
axs[2].set_ylim(ylims)
axs[2].legend()


plt.savefig('imgs/regularization.png',dpi=100)
plt.show()