
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
# from mayavi import mlab
from numpy.linalg import norm



# LOADING and SAVING data

def load_camera_calibration(filename):
	# loads camera intrinsics

	if os.path.isfile(filename):

		fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
		sz = fs.getNode("imageSize")
		M = fs.getNode("cameraMatrix").mat()
		D = fs.getNode("distCoeffs").mat()
		if sz.isSeq(): # za Rokov format, ki lahko vsebuje sezname
			width = sz.at(0).real()
			height = sz.at(1).real()
		else:
			sz = sz.mat()
			height = int(sz[0][0])
			width = int(sz[1][0])
		return (width, height, M,D)
	else:
		print("calibration file not found!")

def save_lidar_calibration_data(filename, R, T, C): # save 
	print("saving lidar calibration file: ", filename)
	cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
	cv_file.write("R", R)
	cv_file.write("T", T)
	cv_file.write("C", C)
	cv_file.release()

# LIDAR PROJECTION

def get_colors(points, rnge=100, relative=False, cmap_name='Wistia', cmap_len=256, nonlinear_factor=1):
	colors = []
	dist = np.sqrt(points[0,:]**2+points[1,:]**2+points[2,:]**2)

	if relative:
		rnge = np.max(dist)

	f = dist/rnge
	f = np.clip(f, 0, 1)

	if nonlinear_factor!=1:
		f = np.power(f, nonlinear_factor)
	
	cmap = plt.get_cmap(cmap_name)
	sm = plt.cm.ScalarMappable(cmap=cmap)
	color_range = sm.to_rgba(np.linspace(0, 1, cmap_len))[:,0:3]

	idx = f*(cmap_len-1)

	for i in idx:
		clr = color_range[int(i),:]
		colors.append(clr)

	return np.array(colors)

def project_lidar_points(point_cloud, sz, R, t, M, D, rn=1e6, cmap_name='turbo_r', cmap_len=256, nonlinear_factor=1):
	pc = point_cloud[:,:3].copy().T
	# idx = (pc[2,:]>0) # remove points behind camera
	# pc = pc[:,idx]

	if pc.shape[1]==0:
		return pc, None, None

	colors = get_colors(pc, rn, cmap_name=cmap_name, cmap_len=cmap_len, nonlinear_factor=nonlinear_factor)

	pts, _ = cv2.projectPoints(pc, R, t, M, distCoeffs=D) # TODO change this for simpler projection
	pts = pts[:,0,:]
	mask = (pts[:,0]>0) & (pts[:,1]>0) & (pts[:,0]<sz[1]-1) & (pts[:,1]<sz[0]-1) # create mask for valid pixels

	return pts, colors, mask


# lidar processing

def get_mask(shape, rvec, tvec, M):

	# define target plane
	plane_w = -1.48
	plane_h = -1.05
	points = np.array([[0,0,0],[0,plane_h,0],[plane_w,plane_h,0],[plane_w,0,0]])
	points[:,0]+=0.365
	points[:,1]+=0.215

	points[:,0]*=-1
	points[:,1]*=-1

	corner_pts, _ = cv2.projectPoints(points, rvec, tvec, M, distCoeffs=None)
	corner_pts = corner_pts[:,0,:]

	mask = np.zeros(shape,dtype=np.uint8)
	cv2.fillPoly(mask, pts = [corner_pts.astype(np.int32)], color=(255,255,255))

	return mask

def get_grid(): # for calibration, not needed really
	# create grid

	objectPoints= []
	# grid_size = 0.06 # 6cm
	grid_size = 0.3
	# grid_size = 0.05
	# grid_size = 300
	rows, cols = 3, 6

	z = 0
	off = grid_size/2

	# objectPoints.append([0,0.0,0])

	for i in range(rows):
		row = []
		for j in range(cols):
			
			# objectPoints.append( ((2*j + i%2)*grid_size, i*grid_size, z) )
			if j%2==0:
				# p = (j*off,i*grid_size,z)
				p = (-j*off,i*(-grid_size),z)
				row.append(p)
				# row.insert(1, p)
			else:
				if i<rows-1:
					# p = (j*(off),i*(off*2)+off,z)
					p = (-j*(off),-(i*(off*2)+off),z)
					row.append(p)

		# print(i,j,row)
		# print(row[::2], row[1::2])
		if i<rows-1:
			row = row[::2]+row[1::2]
		# else:
		# 	pass

		# print(i,j,row,'\n')

		objectPoints.extend(row)


		

	# for i in range(cols):
	# 	for j in range(rows):
	# 		if j>0:
	# 			# objectPoints.append( (i*grid_size, (2*j + i%2)*grid_size, 0) )
	# 			# objectPoints.append( (i*grid_size, (2*j + i%2)*grid_size, z) )
	# 			objectPoints.append( ((2*j + i%2)*grid_size, i*grid_size, z) )
	# 		else:
	# 			if i%2==1:
	# 				# print(i,(i*grid_size, (2*j + i%2)*grid_size, 0))
					# objectPoints.append( (i*grid_size, (2*j + i%2)*grid_size, z) )

	objectPoints= np.array(objectPoints).astype('float32')

	# plt.scatter(objectPoints[:,0], objectPoints[:,1])
	# plt.gca().set_aspect('equal', 'box')
	# plt.show()

	objectPoints[:,0]*=-1
	objectPoints[:,1]*=-1

	return objectPoints

def get_edges(im, M, corners, c=1):

	grid = get_grid()

	_, rvec, tvec = cv2.solvePnP(grid, corners, M, None, flags=cv2.SOLVEPNP_IPPE)
	mask = get_mask(im.shape[:-1], rvec, tvec, M)

	n_blur = c
	n_blur+=1 if n_blur%2==0 else 0

	sigma = n_blur//5
	# sigma = n_blur//10

	edges = cv2.Canny(mask, threshold1=0, threshold2=100).astype(np.float32)
	edges = cv2.GaussianBlur(edges,(n_blur,n_blur), sigma, sigma)

	return edges/np.max(edges)

def calculate_gradient(params, edges, lidar_data, M, dt, da):

	scan = lidar_data[:,0:3]

	res = []
	
	for i, x in enumerate(params):
		p = params.copy()
		p[i]+=dt if i<3 else da
		R, T = params_to_input(p)
		s1 = get_score(R, T, scan, M, edges)

		p = params.copy()
		p[i]-=dt if i<3 else da
		# print(p)
		R, T = params_to_input(p)
		s2 = get_score(R, T, scan, M, edges)

		g = (s2-s1)/2

		res.append(g)

	return np.array(res)

def params_to_input(params):
	# ordering is x,y,z,p,y,r
	R = eulerAnglesToRotationMatrix(params[3:]) # TODO load this from yaml file
	T = np.array(params[0:3])

	return R, T

def eulerAnglesToRotationMatrix(theta):
	# Calculates Rotation Matrix given euler angles.
	theta = [np.radians(x) for x in theta]
	
	R_x = np.array([[1, 0, 0],
					[0, np.cos(theta[0]), -np.sin(theta[0]) ],
					[0, np.sin(theta[0]), np.cos(theta[0]) ]
					])		
					
	R_y = np.array([[np.cos(theta[1]), 0,np.sin(theta[1]) ],
					[0, 1,0],
					[-np.sin(theta[1]),0,np.cos(theta[1]) ]
					])
				
	R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
					[np.sin(theta[2]), np.cos(theta[2]), 0],
					[0, 0,1]
					])
					
	# R = np.dot(R_x, np.dot( R_y, R_z ))
	# R = R_x @ R_y @ R_z
	R = R_z @ R_y @ R_x

	return R

def get_score(R, T, pc, M, edges):
	# from utils import project_lidar_points
	# disp(pc)
	# print(R, T)
	# x = project_lidar_points(pc, edges.shape, R, T, M, np.array([]))
	# print(x)
	# if l2im is not None:
	# 	pts, _, mask = project_lidar_points(pc, edges.shape, R, T, M, np.array([]), l2im=l2im)
	# else:
	pts, _, mask = project_lidar_points(pc, edges.shape, R, T, M, np.array([]))

	pts = pts[mask, :]
	s = evaluate_solution_edges(edges, pts)

	# e = edges.copy()

	# for point in pts:
	# 	# print(point)
	# 	point = (int(point[0]), int(point[1]))
	# 	cv2.circle(e, point, 3, (255,0,0), cv2.FILLED, lineType=cv2.LINE_AA)

	# print(s)

	# cv2.imshow("a", e)
	# # key = cv2.waitKey(1) & 0xFF
	# key = cv2.waitKey(0) & 0xFF

	# if key==ord("q"):
		# return

	return s

def evaluate_solution_edges(edges, pts):

	pts = pts.astype(np.int32)

	inliers = list(map(lambda x: edges[x[1],x[0]], pts))

	# print(sum(inliers))
	return sum(inliers)