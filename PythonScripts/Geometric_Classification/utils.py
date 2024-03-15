import matplotlib.pyplot as plt
import numpy as np
import string
import random

px2inch = 1/plt.rcParams['figure.dpi']

def random_str(length=5):
    letters = string.digits+string.ascii_letters+string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def eval_net(net,N=200):
    x12   = np.linspace(-1,1,N)
    X1,X2 = np.meshgrid(x12,x12)
    Zc    = [] # discrete   values
    Zd    = [] # continuous values
    for x1i,x2i in zip(X1.flatten(),X2.flatten()):
        x = np.array([x1i,x2i]).reshape(2,1)
        y = net.feedForward(x)
        Zc.append(np.max(y)   ) 
        Zd.append(np.argmax(y)) 

    Zc = np.array(Zc).reshape(N,N)
    Zd = np.array(Zd).reshape(N,N)

    return X1,X2,Zc,Zd

def get_train_data(Mxy,Nclass=2):
    #               x1     x2     y
    # row of Mxy    xcoord,ycoord,label
    train_data=[]
    for row in Mxy:
        x1,x2,label = row
        x = np.array([[x1],[x2]])
        y = np.zeros((Nclass,1))
        y[int(label)] = 1
        train_data.append( (x,y) )
    return train_data


def get_circles(Npts=100,k=0.5,Nrings=2):

    R      = np.linspace(0,1,Nrings+1)
    Mxy    = []
    for i in range(R.size-1):

        r1 = R[i]
        r2 = R[i+1]
    
        t = 2*np.pi*np.random.rand(Npts,1)
        r = (r2-r1)*( (1-k)*np.random.rand(Npts,1)) + r1

        x1 = r*np.cos(t)
        x2 = r*np.sin(t)
        y  = i*np.ones((Npts,1))

        Mxy.append( np.concatenate((x1,x2,y), axis=1) )

    Mxy        = np.concatenate( Mxy, axis=0)
    train_data = get_train_data(Mxy,Nclass=Nrings)

    return train_data,Mxy


def get_double_spiral(a=0.45,n_turns=2,pts_turn=250,knoise=0.015):

    def get_spiral(positive=True):

        t1        = np.pi/2
        t2        = n_turns*2*np.pi
        npts      = n_turns*pts_turn
        t         = np.linspace(t1,t2,npts).reshape(npts,1)
        r         = a/(2*np.pi)*t
        kr_linear = knoise*np.linspace(0,n_turns,npts).reshape(npts,1)

        if   positive: S=+1; y=np.ones_like(t)
        else:          S=-1; y=np.zeros_like(t)

        x1 = S*( r*np.cos(t) + kr_linear*np.random.randn(npts,1) )
        x2 = S*( r*np.sin(t) + kr_linear*np.random.randn(npts,1) )

        return np.concatenate((x1,x2,y), axis=1) 

    Maxy = get_spiral(positive=True)
    Mbxy = get_spiral(positive=False)
    Mxy  = np.concatenate( (Maxy,Mbxy), axis=0)

    train_data = get_train_data(Mxy,Nclass=2)
    
    return train_data, Mxy  


def subplot_check_rect(fig,ax,ind,cmap_points,Mxy,X1,X2,train_results,alphaMesh=0.9,sizeMarkers=8,cmap_con='jet',cmap_dis='Greys'):
      
      accuracy_array = train_results['accuracy_array']
      epoch_array    = train_results['epoch_array']
      cost_array     = train_results['cost_array']

      Zc = train_results['Zcd_end'][0]
      Zd = train_results['Zcd_end'][1]

      k = ind[0]
      pco0 = ax[k].pcolormesh(X1,X2,Zc,cmap=cmap_con, vmin=0, vmax=1,alpha=alphaMesh)
      ax[k].scatter(Mxy[:,0],Mxy[:,1],c=cmap_points,s=sizeMarkers)
      ax[k].set_aspect(1)
      fig.colorbar(pco0, ax=ax[k], shrink=1, location='top')
      ax[k].set_xticklabels([]); ax[k].set_xticks([])
      ax[k].set_yticklabels([]); ax[k].set_yticks([])

      k = ind[1]
      pco1 = ax[k].pcolormesh(X1,X2,Zd,cmap=cmap_dis,alpha=alphaMesh)
      ax[k].scatter(Mxy[:,0],Mxy[:,1],c=cmap_points,s=sizeMarkers)
      ax[k].set_aspect(1)
      fig.colorbar(pco1, ax=ax[k], shrink=1, location='top')
      ax[k].set_xticklabels([]); ax[k].set_xticks([])
      ax[k].set_yticklabels([]); ax[k].set_yticks([])

      k = ind[2]
      color = 'tab:blue'
      ax[k].plot(np.nan,np.nan, color='tab:red',label='Train Accuracy %')
      ax[k].plot( epoch_array,cost_array,color=color,label='Train Cost')
      ax[k].tick_params(axis='y', labelcolor=color)
      ax[k].set_xlabel('Epochs')
      ax[k].legend(loc=7)

      color = 'tab:red'
      ax2 = ax[k].twinx()
      ax2.plot( epoch_array,accuracy_array,color=color)
      ax2.tick_params(axis='y', labelcolor=color)
      ax2.set_ylim(0,105)

def subplot_check_square(ax,ind,cmap_points,Mxy,X1,X2,Zc,Zd,alphaMesh=0.75,sizeMarkers=8,cmap_con='jet',cmap_dis='Greys'):

      k = ind[0]
      mesh_con = ax[k].pcolormesh(X1,X2,Zc,cmap=cmap_con, vmin=0, vmax=1,alpha=alphaMesh)
      ax[k].scatter(Mxy[:,0],Mxy[:,1],c=cmap_points,s=sizeMarkers)
      ax[k].set_aspect(1)
      ax[k].set_xticklabels([]); ax[k].set_xticks([])
      ax[k].set_yticklabels([]); ax[k].set_yticks([])

      k = ind[1]
      mesh_dis = ax[k].pcolormesh(X1,X2,Zd,cmap=cmap_dis,alpha=alphaMesh)
      ax[k].scatter(Mxy[:,0],Mxy[:,1],c=cmap_points,s=sizeMarkers)
      ax[k].set_aspect(1)
      ax[k].set_xticklabels([]); ax[k].set_xticks([])
      ax[k].set_yticklabels([]); ax[k].set_yticks([])

      return  mesh_con,mesh_dis