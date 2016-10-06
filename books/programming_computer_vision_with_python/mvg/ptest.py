import sfm, camera
import pandas as pd
import numpy as np

points2D = [np.loadtxt('2D/00'+str(i+1)+'.corners').T for i in range(3)]
points3D = np.loadtxt('3D/p3d').T
corr = np.genfromtxt('2D/nview-corners',dtype='int',missing_values='*')
corr = corr[:,0]
ndx3D = np.where(corr>=0)[0] # missing values are -1
ndx2D = corr[ndx3D]

x = points2D[0][:,ndx2D] # view 1
x = np.vstack( (x,np.ones(x.shape[1])) )
X = points3D[:,ndx3D]
X = np.vstack( (X,np.ones(X.shape[1])) )
# estimate P
print sfm.compute_P(x,X)

