
import numpy as np
from numpy.linalg import inv
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2, glob, os, struct, json, yaml
from datetime import datetime, timedelta, date
import time
# from datetime import timedelta
# from mayavi import mlab
# from ltf_parser import ltf_parser
from termcolor import colored
from types import SimpleNamespace
from functools import singledispatch

import math
from PIL import Image
# import urllib.request as urllib2
import timeit
from random import shuffle
import matplotlib.pyplot as plt

from utils import *

@singledispatch
def wrap_namespace(ob):
    return ob

@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})

@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]

calib_dir = 'calibrations'
data_dir = 'calib_data'

# TODO preprocess LIDAR point clouds and save to cache

def process_data(camera_name='zed'):

	images = sorted(glob.glob(f'{data_dir}/{camera_name}/*.png'))
	lidar = sorted(glob.glob(f'{data_dir}/{camera_name}/*.npy'))

	return images, lidar

def extract_lidar_edges(pc):

	# pc = extract_line_segments_from_scan(pc)

	res = []

	# split into beams
	for beam_idx in range(16):
		beam = pc[pc[:,-1]==beam_idx,:]
		beam = beam[:,:3]
		beam_ = beam.copy()

		segment = []

		for i, pt in enumerate(beam[:-1]):
			# print(pt)
			# input()
			pt2 = beam[i+1]

			# dynamic threshold
			min_d = pt[-1] if pt[-1]<pt2[-1] else pt2[-1]
			dist_thr = min_d*0.5

			if np.abs(pt[-1]-pt2[-1])>dist_thr:
				if pt[-1]<pt2[-1]: # take closest point
					res.append(pt)
				else:
					res.append(pt2)

			# if not segment:
			# 	segment.append(pt)
			# 	continue
			# else:

			# 	if np.abs(pt[-1]-pt2[-1])<dist_thr:
			# 		first = segment[0]
			# 		last = segment[-1]
			# 		if is_collinear(first, last, pt):
			# 			segment.append(pt)

			# 		# if pt[-1]<pt2[-1]: # take closest point
			# 		# 	res.append(pt)
			# 		# else:
			# 		# 	res.append(pt2)

			# 	else: # if z difference is large enough
			# 		res.append(segment[0])
			# 		res.append(segment[-1])
			# 		segment = []
			# 		# check if collinear



	# iterate over each beam

	# return edge points

	# print(res)

	return np.array(res)

class Loader(yaml.Loader):
    pass

def calibrate_extrinsics(camera_name='zed'):

	# TODO load config
	# with open("config.json") as f:
	# 	cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

	with open("config.yaml", 'r') as f:
		cfg = wrap_namespace(yaml.safe_load(f))
		# cfg = SimpleNamespace(**cfg)

	res_fn = f'{camera_name}_lidar.yaml'

	print(cfg)
	# print(cfg.OPTIMIZATION.a)

	# print(cfg.lidar2camera)

	cfg.GEOMETRY.lidar2camera = np.array(cfg.GEOMETRY.lidar2camera)
	cfg.OPTIMIZATION.lr = float(cfg.OPTIMIZATION.lr)

	# print(cfg)


	# return


	# C_lidar_zed = np.array([ 1, 0, 0, 0, 0, -1, 0, 1, 0 ]).reshape((3,3))

	# load intrinsics
	calib_fn = f'{calib_dir}/{camera_name}.yaml'
	width, height, M, D = load_camera_calibration(calib_fn)
	print(width, height, M, D)
	M*=0.5
	M[-1,-1]=1

	# load and preprocess LIDAR data
	images_list, lidar_list = process_data(camera_name)

	print(len(images_list))
	print(len(lidar_list))

	# preprocess data

	R = np.eye(3)
	T = np.zeros((1,3))

	R = np.array([ 9.9992722851089055e-01, 1.1602398976433012e-02,
	   -3.3048480329459775e-03, -1.0509123433775139e-02,
	   9.7226168168842508e-01, 2.3365954002575909e-01,
	   5.9241883142654153e-03, -2.3360780521717867e-01,
	   9.7231285980101578e-01 ]).reshape((3,3))
	T = np.array([ 7.0653076245761171e-02, -8.2157305047760609e-01,
	   -3.4356599116650351e-02 ])

	# radius = 3

	images_list = images_list[1:]
	lidar_list = lidar_list[1:]

	data = []

	# edge_thickness = 55
	n_samples = 2

	cache_dir = 'cache'

	os.makedirs(cache_dir, exist_ok=True)

	cv2.namedWindow("image", cv2.WINDOW_NORMAL)


	for i, (im_fn, lidar_fn) in enumerate(zip(images_list, lidar_list)):

		# print(im_fn)

		name = im_fn.split('/')[-1][:-4]
		cache_name = f'{cache_dir}/{name}.npy'

		if os.path.exists(cache_name):

			d = np.load(cache_name, allow_pickle=True).item()
			# print("cache")
		else:
			im = cv2.imread(im_fn)
			lidar_raw = np.load(lidar_fn, allow_pickle=True).item()
			lidar = lidar_raw['pc']
			c = lidar_raw['corners']

			lidar[:,:3] = (cfg.GEOMETRY.lidar2camera @ lidar[:,:3].T).T
			idx = (lidar[:,2]>0) # remove points behind camera
			lidar = lidar[idx,:]

			# process lidar to edges
			lidar_ = extract_lidar_edges(lidar)

			# get image edges
			edges = get_edges(im, M, c, c=int(cfg.IMAGE.edge_thickness))		

			
			d = {'im': im, 'edges': edges, 'lidar_raw': lidar, 'lidar_edges': lidar_, 'corners': c}
			np.save(cache_name, d)

		data.append(d)

		# if i==n_samples:
		# 	break

	# optimization loop

	params = cfg.INITIAL_PARAMETERS.ZED
	learning_rate_decay = np.exp(np.log(cfg.OPTIMIZATION.final_lr_perc)/cfg.OPTIMIZATION.n_iter)

	select_idx = 1

	print("LR", cfg.OPTIMIZATION.lr)

	for n in range(cfg.OPTIMIZATION.n_iter):

		gradients = []
		d = np.random.choice(data)
		# d = data[2]

		G = calculate_gradient(params, d['edges'], d['lidar_edges'], M, cfg.OPTIMIZATION.dt, cfg.OPTIMIZATION.da)

		params-=(cfg.OPTIMIZATION.lr*G)

		# print(G)

		# print(params)

		cfg.OPTIMIZATION.lr*=learning_rate_decay

		

		# DISPLAY

		if n%cfg.DISPLAY.display_interval==0:


			R, T = params_to_input(params)
			s = get_score(R, T, data[select_idx]['lidar_edges'], M, data[select_idx]['edges'])
			print(s)

			im_ = data[select_idx]['im']
			im = data[select_idx]['im']
			s = data[select_idx]['lidar_raw']
			a = data[select_idx]['lidar_edges']
			e = data[select_idx]['edges']
			c = data[select_idx]['corners']

			im_ = im_.astype(np.float32)/255
			e = cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)
			e = e.astype(np.float32)
			im_ = im_*(1-e)+e

			im_ = (im_*255).astype(np.uint8)

			for point in c:
				# print(point)
				pt = (int(point[0,0]),int(point[0,1]))
				try:
					cv2.circle(im_, pt, cfg.DISPLAY.point_size, (0,0,255), cv2.FILLED, lineType=cv2.LINE_AA)
				except Exception as e:
					print(e)

			pts_, colors_, mask_ = project_lidar_points(s, im.shape, R, T, M, np.array([]), rn=50)
			for point, clr in zip(pts_, colors_):
				# print(point)
				point = (int(point[0]), int(point[1]))
				clr = (int(clr[0]*255), int(clr[1]*255), int(clr[2]*255))

				try:
					# cv2.circle(im_, point, cfg.point_size, (255,0,0), cv2.FILLED, lineType=cv2.LINE_AA)
					cv2.circle(im_, point, cfg.DISPLAY.point_size, clr, cv2.FILLED, lineType=cv2.LINE_AA)
				except:
					pass

			pts_edges, _, mask_edges = project_lidar_points(a, im.shape, R, T, M, np.array([]))
			pts_edges = pts_edges[mask_edges, :]
			for point in pts_edges:
				point = (int(point[0]), int(point[1]))
				try:
					cv2.circle(im_, point, cfg.DISPLAY.point_size+1, (0,255,0), cv2.FILLED, lineType=cv2.LINE_AA)
				except:
					pass

			cv2.imshow("image", im_)

			key = cv2.waitKey(1) & 0xFF
			# key = cv2.waitKey(0) & 0xFF

			if key==ord("q"):
				return

	R, T = params_to_input(params)
	save_lidar_calibration_data(res_fn, R, T, cfg.GEOMETRY.lidar2camera)

	cv2.imshow("image", im_)
	key = cv2.waitKey(0) & 0xFF


	return

	# check data
	for d in data:
		im, edges, lidar, lidar_, c = d['im'], d['edges'], d['lidar_raw'], d['lidar_edges'], d['corners']
	
		pts, colors, mask = project_lidar_points(lidar, im.shape, R, T, M, np.array([]), rn=100, cmap_name='turbo_r', nonlinear_factor=0.5)
		if colors is not None:
			pts = pts[mask,:]
			colors = colors[mask]
			if pts is not None and pts.shape[1]!=0:
				for pt, color in zip(pts,colors):
					# print(pt)
					c = tuple([int(color[2]*255), int(color[1]*255), int(color[0]*255)][::-1])
					# try:
					# 	cv2.circle(im, (int(pt[0]),int(pt[1])), radius=radius, color=c, thickness=-1)
					# except Exception as e:
					# 	print(e)
					# 	pass

		if lidar_.any():
			pts_, colors_, mask_ = project_lidar_points(lidar_, im.shape, R, T, M, np.array([]), rn=100, cmap_name='turbo_r', nonlinear_factor=0.5)
			if colors_ is not None:
				pts_ = pts_[mask_,:]
				colors_ = colors_[mask_]
				if pts_ is not None and pts_.shape[1]!=0:
					for pt, color in zip(pts_,colors_):
						# print(pt)
						# c = tuple([int(color[2]*255), int(color[1]*255), int(color[0]*255)][::-1])
						try:
							cv2.circle(im, (int(pt[0]),int(pt[1])), radius=radius, color=(255, 0, 0), thickness=-1)
						except Exception as e:
							print(e)
							pass

		plt.clf()
		plt.imshow(im)
		plt.imshow(edges, alpha=0.5)

		plt.draw()
		plt.pause(0.01)
		plt.waitforbuttonpress()
		# break
	# plt.show()


if __name__=="__main__":
	# main()
	calibrate_extrinsics()