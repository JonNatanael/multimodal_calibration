import cv2, glob, os, yaml
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def calibrate_extrinsics(camera_name=None):

	# load config
	with open("config.yaml", 'r') as f:
		cfg = wrap_namespace(yaml.safe_load(f))

	if camera_name is None:
		camera_name = cfg.GENERAL.camera_name

	res_fn = f'calibrations/{camera_name}_lidar.yaml'

	cfg.GEOMETRY.lidar2camera = np.array(cfg.GEOMETRY.lidar2camera)

	# load intrinsics
	try:
		calib_fn = f'{cfg.GENERAL.calibration_dir}/{camera_name}.yaml'
		print(calib_fn)
		width, height, M, D = load_camera_calibration(calib_fn)
	except:
		print("calibration not found!")
		return

	params = getattr(cfg.INITIAL_PARAMETERS, camera_name.upper()).PARAMS
	cfg.M = M
	cfg.D = D
	cfg.height = height
	cfg.width = width

	cfg.GEOMETRY.M = M
	cfg.GEOMETRY.D = D

	# load and preprocess LIDAR data
	data = process_data(camera_name, cfg, use_cache=cfg.GENERAL.use_cache)

	if not data:
		print(f"no data found for {camera_name}")
		return

	if cfg.DISPLAY.show_progress:
		cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL )	
	
	# calculate learning rate decay based on final LR percent from config (LR falls from 100% to final_lr_perc% over the optimization runtime)
	learning_rate_decay = np.exp(np.log(cfg.OPTIMIZATION.final_lr_perc)/cfg.OPTIMIZATION.n_iter)

	print(f"Loaded {len(data)} image-lidar pairs")
	print(f"Starting optimization for {camera_name}")

	# optimization loop

	for n in range(cfg.OPTIMIZATION.n_iter):
		print(f"{n}/{cfg.OPTIMIZATION.n_iter}", end='\r')

		# minibatch
		N_samples = min(cfg.OPTIMIZATION.batch_size, len(data))
		d = np.random.choice(data, N_samples)
		G = 0
		for sample in d:
			# calculate gradients with current parameters
			G+=calculate_gradient(params, sample['edges'], sample['lidar_edges'], M, cfg.OPTIMIZATION.dt, cfg.OPTIMIZATION.da)

		# update parameters
		params-=(cfg.OPTIMIZATION.lr*G)

		cfg.OPTIMIZATION.lr*=learning_rate_decay

		# DISPLAY
		if cfg.DISPLAY.show_progress and n%cfg.DISPLAY.display_interval==0:

			R, T = params_to_input(params)

			test_idx = cfg.DISPLAY.image_index

			im = data[test_idx]['im'].copy()
			nm = data[test_idx]['name']
			im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
			im = cv2.resize(im, (int(width), int(height)), interpolation=cv2.INTER_AREA)

			s = data[test_idx]['lidar_raw']
			a = data[test_idx]['lidar_edges']
			e = data[test_idx]['edges']
			c = data[test_idx]['corners']

			im = im.astype(np.float32)/255
			e = cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)
			e = e.astype(np.float32)

			# overlay target edges
			im = im*(1-e)+e
			im = (im*255).astype(np.uint8)

			# draw all LIDAR points
			pts_, colors_, mask_ = project_lidar_points(s, im.shape, R, T, M, np.array([]), rn=cfg.DISPLAY.lidar_range)
			for point, clr in zip(pts_, colors_):
				point = (int(point[0]), int(point[1]))
				clr = (int(clr[0]*255), int(clr[1]*255), int(clr[2]*255))
				if point[0]>0 and point[1]>0 and point[0]<im.shape[1] and point[1]<im.shape[0]:
					cv2.circle(im, point, cfg.DISPLAY.point_size, clr, cv2.FILLED, lineType=cv2.LINE_AA)

			# draw LIDAR corner points
			pts_edges, _, mask_edges = project_lidar_points(a, im.shape, R, T, M, np.array([]))
			pts_edges = pts_edges[mask_edges, :]
			for point in pts_edges:
				point = (int(point[0]), int(point[1]))
				cv2.circle(im, point, cfg.DISPLAY.point_size+2, (0,0,255), cv2.FILLED, lineType=cv2.LINE_AA)

			# draw calibration target points
			for point in c:
				pt = (int(point[0,0]),int(point[0,1]))
				cv2.circle(im, pt, cfg.DISPLAY.point_size*3, (0,255,255), cv2.FILLED, lineType=cv2.LINE_AA)

			cv2.imshow("image", im)
			key = cv2.waitKey(1) & 0xFF
			if n==0: # wait on first iteration
				cv2.waitKey(0)
			if key==ord("q"):
				print("Calibration cancelled")
				return

	print(f"Calibration complete. Saving parameters to: {res_fn}")
	R, T = params_to_input(params)

	save_lidar_calibration_data(res_fn, R, T, cfg.GEOMETRY.lidar2camera)

	if cfg.DISPLAY.show_progress:
		cv2.imshow("image", im)
		key = cv2.waitKey(0) & 0xFF

if __name__=="__main__":
	calibrate_extrinsics()
