from detect import detect, ConvNetb

import time
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
from pathlib import Path
from datetime import datetime
from utils.gb_utils import *
from utils.gb_scoring import *

# Path to a csv file to open and store the outputs of the this script in. This will be created once every time this script is called. 
results_path = str('./results_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv')

# Setup lists of detection parameters to loop through
nms_thres_list = [0.1, 0.2, 0.3, 0.4, 0.5]
conf_thres_list = [0.990, 0.992, 0.994, 0.996, 0.998]
weights_list = ['/mnt/Data/AQM_training/01_development/weights/backup_9_class_london_10folds_2_200.pt', '/mnt/Data/AQM_training/01_development/weights/backup_9_class_london_10folds_2_400.pt', '/mnt/Data/AQM_training/01_development/weights/backup_9_class_london_10folds_2_600.pt', \
				 '/mnt/Data/AQM_training/01_development/weights/backup_9_class_london_10folds_2_800.pt', '/mnt/Data/AQM_training/01_development/weights/latest_9_class_london_10folds_2.pt'] 
				# 'weights/backup_9_class_london_10folds_2_200.pt', 'weights/backup_9_class_london_10folds_2_400.pt', 'weights/backup_9_class_london_10folds_2_600.pt', 'weights/backup_9_class_london_10folds_2_800.pt', 'weights/latest_9_class_london_10folds_2.pt'	  ]

# Path to the targets matfile - carefull that this does not contain imagery used to create any of the above weights files.
targets_path = 'utils/london_10folds_9class_test_2.mat'

# Create dataframes from the contents ofthe matfile. 
targs_df, stats_df = create_dfs(targets_path)

# Paths to each buffer shapefile.
major_roads_buffer_shp_path = '/mnt/server/AQM/AQM/01_Vehicle_Detection/Europe/London/02_City_Data/02_Roads/02_Buffered/02_Quantile_Derived_Buffers/combined_buffer_q_0.9_MajorRoads.shp'
minor_roads_buffer_shp_path = '/mnt/server/AQM/AQM/01_Vehicle_Detection/Europe/London/02_City_Data/02_Roads/02_Buffered/02_Quantile_Derived_Buffers/combined_buffer_q_0.9_Minor_unclipped.shp'
roads_buffer_q_90_shp_path =  '/mnt/server/AQM/AQM/01_Vehicle_Detection/Europe/London/02_City_Data/02_Roads/02_Buffered/02_Quantile_Derived_Buffers/combined_buffer_q_0.9_unclipped.shp'
roads_buffer_mean_shp_path = '/mnt/server/AQM/AQM/01_Vehicle_Detection/Europe/London/02_City_Data/02_Roads/02_Buffered/02_Quantile_Derived_Buffers/0.5/measured_buffer_mean.shp'

# Intersect each road buffer with the boundaries of each image, to reduce time when overlaying bounding boxes. 
major_buffer_img_intersect = img_roads_intersect ( stats_df, major_roads_buffer_shp_path )
minor_buffer_img_intersect = img_roads_intersect ( stats_df, minor_roads_buffer_shp_path )
buffer_q_90_img_intersect = img_roads_intersect ( stats_df, roads_buffer_q_90_shp_path )
buffer_mean_img_intersect = img_roads_intersect ( stats_df, roads_buffer_mean_shp_path )

# Create geodataframe from the ground truth objects. Set up a unique ID for each bbox. 
targs_gdf = dfs_to_gdfs(targs_df, stats_df,  major_buffer_img_intersect.crs)
targs_gdf['bbox_id'] = targs_gdf.index

# Overlay the ground truth bboxes onto the road buffers and drop any duplicates arrising when a bbox touches more than one road buffer.
road_targs_maj = gpd.overlay(targs_gdf, major_buffer_img_intersect, how='intersection').drop_duplicates('bbox_id')
road_targs_min = gpd.overlay(targs_gdf, minor_buffer_img_intersect, how='intersection').drop_duplicates('bbox_id')
road_targs_q90 = gpd.overlay(targs_gdf, buffer_q_90_img_intersect, how='intersection').drop_duplicates('bbox_id')
road_targs_mean = gpd.overlay(targs_gdf, buffer_mean_img_intersect, how='intersection').drop_duplicates('bbox_id')

class Options:
	"""
	Object to store detection parameters as attributes, that would normally be parsed from the command line.
	"""
	def __init__(self, targets, weights, conf_thres, nms_thres, classes, cfg, names):
		self.targets = targets
		self.weights = weights
		self.conf_thres = conf_thres
		self.nms_thres = nms_thres
		self.classes = classes
		self.cfg = cfg
		self.names = names
		self.source = targets
		self.batch_size = 1
		self.output = './output'
		self.img_size = 32*51
		self.iou_thres = 0.25
		self.plot_flag = False 
		self.secondary_classifier = False

for nms_thres in nms_thres_list:
	for conf_thres in conf_thres_list:
		for i, weights in enumerate(weights_list):
			kwargs1 = {'conf_thres':conf_thres, 'nms_thres':nms_thres, 'weights':weights, 'targets':targets_path, }
			kwargs2 = {'cfg':'cfg/c9_a30symmetric.cfg', 'names':'data/ldn_9.names', 'classes':9  }
			opt = Options(**kwargs1, **kwargs2)
			# opt is an object in the same format as opt in detect.py
			print('Commencing detections with the following parameters:')
			print(kwargs1)
			
			torch.cuda.empty_cache() 
			out_dir = detect(opt) # Run detections as normal
			torch.cuda.empty_cache()

			# Gather detections into a dataframe and convert into geodataframe. 
			dets_df = yolo_dets_to_df(out_dir)
			dets_df.reset_index(inplace=True, drop=True)
			dets_gdf = dfs_to_gdfs(dets_df, stats_df, major_buffer_img_intersect.crs)

			# Overlay Gt and detections Bboxes to assign IoU, true_pos, and a unique true_pos_id. 
			true_dets_gdf, true_targs_gdf = get_true_pos(dets_gdf, targs_gdf, opt.iou_thres)

			# Compute all relevant statistics including regarding movement, and add these to a 'scores' dataframe 
			scores_df = get_scores( true_dets_gdf, true_targs_gdf, stats_df, out_dir, opt )
			scores_df.loc[:, 'buffer_type'] = 'None'

			# Create new dataframes for the road buffered ground truths, including the true_pos information. 
			road_targs_maj = true_targs_gdf.loc[ true_targs_gdf.bbox_id.isin(road_targs_maj.bbox_id) ]
			road_targs_min = true_targs_gdf.loc[ true_targs_gdf.bbox_id.isin(road_targs_min.bbox_id) ]
			road_targs_q90 = true_targs_gdf.loc[ true_targs_gdf.bbox_id.isin(road_targs_q90.bbox_id) ]
			road_targs_mean = true_targs_gdf.loc[ true_targs_gdf.bbox_id.isin(road_targs_mean.bbox_id) ]

			print('Overlaying detections onto each road buffer...')
			# Create new dataframes for the road buffered detections, including the true_pos information. 
			road_dets_maj = gpd.overlay(true_dets_gdf, major_buffer_img_intersect, how='intersection')
			road_dets_min = gpd.overlay(true_dets_gdf, minor_buffer_img_intersect, how='intersection')
			road_dets_q90 = gpd.overlay(true_dets_gdf, buffer_q_90_img_intersect, how='intersection')
			road_dets_mean = gpd.overlay(true_dets_gdf, buffer_mean_img_intersect, how='intersection')

			# Compute all relevant statistics for the road buffered dataframes, including movement information. 
			scores_df_maj = get_road_scores( road_dets_maj, road_targs_maj, stats_df, 'major', out_dir, opt )
			scores_df_min = get_road_scores( road_dets_min, road_targs_min, stats_df, 'minor', out_dir, opt )
			scores_df_q90 = get_road_scores( road_dets_q90, road_targs_q90, stats_df, '0.9 quantile', out_dir, opt )
			scores_df_mean = get_road_scores( road_dets_mean, road_targs_mean, stats_df, 'mean', out_dir, opt )

			# Join all statistics/counts into a single df:
			scores_all = pd.concat( [ scores_df, scores_df_maj, scores_df_min, scores_df_q90, scores_df_mean ] )
			# Add Datetime
			scores_all.loc[:, 'datetime'] = out_dir.split('/')[-1]
			# Add 'epochs'. In fact, this is the last number that appears in the weights path after the final '_'. This might not be the number of epochs. 
			scores_all.loc[:, 'epochs'] = str(Path(opt.weights).stem).split('_')[-1]

			# Save scores into the detection folder.
			scores_all.to_csv(Path( out_dir, Path(opt.targets).stem + '_all' ).with_suffix('.txt'))
			
			#  Append the main results file so that all results from every loop are stored in one place.
			with open(results_path, 'a') as f:
				scores_all.to_csv(f, header=f.tell()==0)
			
			# Output the shapefiles in a parallel folder to the .txt files. These can be identified in the main results spreadsheet using the datetime. 
			gdf_to_shps(dets_gdf, out_dir+'\\shps' )

	


