import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-1,1,50)
X,Y = np.meshgrid(x,x)
# fig = plt.figure()
# ax = Axes3D(fig)
ax = plt.subplot(111, projection='3d')
ax.plot_surface(X,Y,op.rosen([X, Y]))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
