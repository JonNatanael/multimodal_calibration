
import cv2, os, glob
import numpy as np
import matplotlib.pyplot as plt
# from mayavi import mlab
from numpy.linalg import norm


# LOADING and SAVING data

def load_camera_calibration(filename):
	"""
	Loads camera intrinsics from YAML file

	Args:
		filename

	Returns:
		width: camera image width
		height: camera image height
		M: calibration matrix
		D: distortion parameters

	"""

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

	"""
	Saves camera-lidar data to YAML file

	Args:
		filename
		R: rotation matrix from lidar to camera
		T: translation matrix from lidar to camera
		C: transformation matrix from lidar to camera coordinate system
	"""

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
	grid_size = 0.3
	rows, cols = 3, 6

	z = 0
	off = grid_size/2

	for i in range(rows):
		row = []
		for j in range(cols):
			if j%2==0:
				p = (j*off,i*grid_size,z)
				# p = (-j*off,i*(-grid_size),z)
				row.append(p)
				# row.insert(1, p)
			else:
				if i<rows-1:
					p = (j*(off),i*(off*2)+off,z)
					# p = (-j*(off),-(i*(off*2)+off),z)
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

	# objectPoints[:,0]*=-1
	# objectPoints[:,1]*=-1

	return objectPoints

def find_corners(im, patternsize = (3,5)):

	ret, corners = cv2.findCirclesGrid(im, patternsize, flags=cv2.CALIB_CB_ASYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING)

	return corners if ret else None

def get_edges(im, M, corners, thickness=1):

	grid = get_grid()

	_, rvec, tvec = cv2.solvePnP(grid, corners, M, None, flags=cv2.SOLVEPNP_IPPE)
	mask = get_mask(im.shape[:-1], rvec, tvec, M)

	# n_blur = thickness
	thickness+=1 if thickness%2==0 else 0

	sigma = thickness//5
	# sigma = n_blur//10

	edges = cv2.Canny(mask, threshold1=0, threshold2=100).astype(np.float32)
	edges = cv2.GaussianBlur(edges,(thickness, thickness), sigma, sigma)

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
		R, T = params_to_input(p)
		s2 = get_score(R, T, scan, M, edges)

		g = (s2-s1)/2

		res.append(g)

	return np.array(res)

def params_to_input(params):
	# ordering is x,y,z,p,y,r
	R = eulerAnglesToRotationMatrix(params[3:])
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
					
	R = R_z @ R_y @ R_x

	return R

def get_score(R, T, pc, M, edges):
	'''
	Project point cloud using current solution and calculate the score with regard to edges image

	'''

	pts, _, mask = project_lidar_points(pc, edges.shape, R, T, M, np.array([]))

	pts = pts[mask, :]
	s = evaluate_solution_edges(edges, pts)

	return s

def evaluate_solution_edges(edges, pts):
	"""
	sums up the edge image values of projected lidar points

	Args:
		edges: image of target edges
		pts: projected lidar points coordinates (2D)

	Returns:
		score

	"""

	pts = pts.astype(np.int32)

	inliers = list(map(lambda x: edges[x[1],x[0]], pts))

	# print(sum(inliers))
	return sum(inliers)

def process_data(camera_name='zed', cfg=None, use_cache=True):
	"""
	Preprocesses image and LIDAR data, saves to cache

	Args:
		camera_name
		cfg: config data (usually in config.yaml)

	"""

	os.makedirs(cfg.GENERAL.cache_dir, exist_ok=True)
	os.makedirs(cfg.GENERAL.cache_dir+'/'+camera_name, exist_ok=True)
	# use_cache	

	images_list = sorted(glob.glob(f'{cfg.GENERAL.data_dir}/{camera_name}/*.png'))
	if not images_list:
		images_list = sorted(glob.glob(f'{cfg.GENERAL.data_dir}/{camera_name}/*.jpg'))

	lidar_list = sorted(glob.glob(f'{cfg.GENERAL.data_dir}/{camera_name}/*.npy'))

	data = []

	for i, (im_fn, lidar_fn) in enumerate(zip(images_list, lidar_list)):
	# for i, (im_fn, lidar_fn) in enumerate(zip(images_list[::5], lidar_list[::5])):


		name = im_fn.split('/')[-1][:-4]
		print(name)

		cache_name = f'{cfg.GENERAL.cache_dir}/{name}.npy'

		if os.path.exists(cache_name) and cfg.GENERAL.use_cache:
			d = np.load(cache_name, allow_pickle=True).item()
		else:
			im = cv2.imread(im_fn)
			im = cv2.undistort(im, cfg.M, cfg.D)
			print(f'{im.shape=}')
			lidar_raw = np.load(lidar_fn, allow_pickle=True).item()
			# print(f'{lidar_raw=}')
			try:
				lidar = lidar_raw['pc']
			except:
				# print(lidar_raw.keys())
				# print(lidar_raw['scans'])
				# input()
				# continue
				lidar = lidar_raw['scans'][0]
			# corners = lidar_raw['corners']

			if camera_name=='thermal_camera':
				corners = find_corners(255-im)
			elif 'stereo' in camera_name:
				corners = find_corners(im)
				if corners is None:
					corners = find_corners(255-im)
			else:
				corners = find_corners(im)


			# print(corners)
			if corners is None:
				continue

			# transform coordinate system
			lidar[:,:3] = (cfg.GEOMETRY.lidar2camera @ lidar[:,:3].T).T
			idx = (lidar[:,2]>0) # remove points behind camera
			lidar = lidar[idx,:]

			# process lidar to edges

			grid = get_grid()
			_, rvec, tvec = cv2.solvePnP(grid, corners, cfg.GEOMETRY.M, None, flags=cv2.SOLVEPNP_IPPE)
			print(tvec)
			target_distance = tvec[-1][0]
			print(target_distance)

			lidar_ = extract_lidar_edges(lidar, target_distance=target_distance)

			# get image edges
			edges = get_edges(im, cfg.GEOMETRY.M, corners, thickness=int(cfg.IMAGE.edge_thickness))		


			
			d = {'im': im, 'edges': edges, 'lidar_raw': lidar, 'lidar_edges': lidar_, 'corners': corners, 'target_distance': target_distance}
			np.save(cache_name, d)

		data.append(d)

	return data

def angle(v1, v2, acute=True):
	# v1 is your firsr vector
	# v2 is your second vector
		angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
		if (acute == True):
			return np.degrees(angle)
		else:
			return 2 * np.pi - angle

def extract_lidar_edges(pc, target_distance, distance_margin=1.5):
	"""
	Extracts lidar points with large difference in distance

	Args:
		pc: lidar point cloud

	Returns:
		numpy array of candidate points

	"""

	res = []

	azi_thr = 2

	# split into beams
	for beam_idx in range(16):
		beam = pc[pc[:,-1]==beam_idx,:]
		beam = beam[:,:3]
		# beam_ = beam.copy()

		for i, pt in enumerate(beam[:-1]):
			pt2 = beam[i+1]

			# print(pt, pt2, angle(pt, pt2))
			# input()

			azi = angle(pt, pt2)

			# dynamic threshold
			min_d = pt[-1] if pt[-1]<pt2[-1] else pt2[-1]
			dist_thr = min_d*0.5
			dist_thr = 0.5
			# dist_thr = 0.1
			# abs_dist_thr = 1

			# if azi>azi_thr:
			# 	continue

			if np.abs(pt[-1]-pt2[-1])>dist_thr:
				if pt[-1]<pt2[-1]: # take closest point
					if np.abs(pt[-1]-target_distance)<distance_margin:
						res.append(pt)
				else:
					if np.abs(pt2[-1]-target_distance)<distance_margin:
						res.append(pt2)

	return np.array(res)