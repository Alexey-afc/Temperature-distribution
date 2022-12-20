import inline as inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation



L = 1
N3=10# сколько отрезков до центра
N1=2*N3# деление по осям
T=2
N2=100 # деление времени

dx=2*L/N1
dy=2*L/N1
dt =T/(N2)
dd=dt/(dx**2)

tc=np.linspace(0,T,N2)
center=[]
def cen(u):
    for  k in range(0,N2,1):
        center.append(u[k,N3,N3])
    return center

u=np.empty((N2,N1+2,N1+2))


# for k in range(0,N2,1):
#     for i in range(0, N1 + 2, 1):
#         u[k, i, N1+1] = 0
#         u[k, i, 0] = 0
# for k in range(0,N2,1):
#     for i in range(0, N1 + 2, 1):
#         u[k, N1 + 1,i ] = 0
#         u[k, 0, i] = 0
for i in range(0, N1+2, 1):
    for j in range(0, N1+2, 1):
        u[0, i, j] = (1-(-L+i*dx)**2/(L**2))*(1-(-L+j*dy)**2/(L**2))
for k in range(0, N2-1, 1):
    for i in range(1, N1+1 , 1):
        for j in range(1, N1+1 , 1):
            u[k + 1, i, j] = dd * (u[k, i + 1, j] + u[k, i - 1, j] + u[k, i, j + 1] + u[k, i, j - 1] - 4 * u[k, i, j]) + u[k, i, j]

    for i in range(1, N1 + 1, 1):
        # u[k + 1, i, 0] = u[k + 1, i, 1]
        u[k + 1, i, N1 + 1] = u[k + 1, i, N1]
        u[k + 1, 0, i] = u[k + 1, 1, i]
        u[k + 1, N1 + 1, i] = u[k + 1, N1, i]



t1=[]
t2=[]
t=[]
for k in range(0, N2, 1):
    for i in range(0, N1+2 , 1):
        t1 = []
        for j in range(0, N1+2 , 1):
            t1.append(u[k,i,j])
        t2.append(t)
    t.append(t2)

def plotheatmap(u_k, k):
    plt.clf()
    plt.title(f"Temperature at t = {k * dt:.3f} unit time")


    plt.xlabel("x")
    plt.ylabel("y")


    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=1)
    plt.colorbar()

    return plt
def animate(k):
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=100, repeat=False)
anim.save("heat_equation_solution_circ.gif")
plt.xlabel('x')
plt.ylabel('y')
plt.show()



# o1=cen(u)
# plt.plot(tc,o1)
# x1=np.linspace(0,T,1000)
# y=np.exp(-4.5*x1)
# plt.plot(x1,y)
# plt.savefig('saved_figure.png')
plt.show()

