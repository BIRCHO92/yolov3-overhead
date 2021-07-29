from pathlib import Path
import scipy.io
import numpy as np
import pandas as pd
import itertools

import re
import os
import cv2
import glob
import shutil

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import rasterio as rio
import geopandas as gpd
import shapely as sp

from rasterio.warp import transform_bounds

import pyproj

def img_roads_intersect (stats_df, road_buffer_shp_path ):
	"""
	Intersect the target image bounds (the georeferenced extend of each image) with the road buffer to generate a geodataframe containing only relevant road buffer data. 
	"""
	print('Reading road buffer shapefile into geodataframe...')
	buffer_gdf = gpd.read_file(road_buffer_shp_path)
	buffer_crs = buffer_gdf.crs.to_epsg()

	print('Using CRS : ', buffer_crs, ' for road buffer.')

	img_paths = stats_df['img_paths'].tolist()

	ldn_gdf = gpd.GeoDataFrame(geometry=[], crs=buffer_gdf.crs)

	# print('Warning, reading img_paths from glob, not stats_df')
	# imgs_path = '/content/drive/My Drive/George_Birchenough/04_London_Datasets/london_8bit_combined/london_8bit_ACOMP_30cm/images_RGB'
	# img_paths = Path(imgs_path).rglob('*.tif')

	print('Reading image bounds into geodataframe and setting CRS ...')
	for img_path in img_paths:
		with rio.open(img_path) as src:
			bounds = transform_bounds(src.crs, rio.crs.CRS.from_epsg(buffer_crs), src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3])
			geom = sp.geometry.box(*bounds)
			ldn_gdf.loc[img_path, 'geometry'] = geom

	buffer_gdf = buffer_gdf.loc[:, buffer_gdf.columns.intersection(['geometry', 'LEGEND', 'DESC_TERM', 'ROADNATURE', 'layer'])]
	ldn_gdf.reset_index(drop=True, inplace=True)

	print('Intersecting image bounds with road buffer to reduce computing time when intersecting bboxes..')
	buffer_img_intersect = gpd.overlay(ldn_gdf, buffer_gdf, how='intersection')

	return buffer_img_intersect

def dfs_to_gdfs(dfs, stats_df, buffer_crs = 'EPSG:32630'):
	""" 
	For a given dataframe with ground truths/detections in yolo detection format, including the image paths, return geodataframe. 
	"""    
	print('Using CRS : ', buffer_crs, ' dataset geodataframes.')

	img_paths = stats_df['img_paths'].tolist()
 
	# print('Warning, reading img_paths from glob, not stats_df')
	# imgs_path = '/content/drive/My Drive/George_Birchenough/04_London_Datasets/london_8bit_combined/london_8bit_ACOMP_30cm/images_RGB'
	# img_paths = Path(imgs_path).rglob('*.tif')

	gdf_out = pd.DataFrame()
	for img_path in img_paths:
		img_name = str ( Path( img_path ).stem )
		df = dfs.loc[ dfs['chip'] == img_name ].reset_index(drop=True).copy()
		df_in = df.copy()
		gdf_in = df_to_gdf(img_path, df, buffer_crs)
		gdf = gpd.GeoDataFrame(df_in, geometry = gdf_in.geometry)
		gdf_out = pd.concat([gdf_out, gdf])
	gdf_out.reset_index(drop=True, inplace = True)
	return gdf_out

def df_to_gdf(img_path, df, buffer_crs):
	""" For a given image, and its corresponding labels in a dataframe, output a geodataframe.
	"""
	if df.shape[0] == 0:
		return gpd.GeoDataFrame()
	elif img_path is not None:
		with rio.open(img_path) as src:
			crs = src.crs
			df.xmin, df.ymin = rio.transform.xy(src.transform, df.ymin, df.xmin)
			df.xmax, df.ymax = rio.transform.xy(src.transform, df.ymax, df.xmax)

	gdf = gpd.GeoDataFrame( df, 
		geometry=[sp.geometry.Polygon([(r.xmin, r.ymin), (r.xmax, r.ymin), (r.xmax, r.ymax), (r.xmin, r.ymax)]) for r in 
		df.itertuples()], crs=crs)
 
	gdf = gdf.to_crs(buffer_crs)
	
	return gdf

def gdf_to_shps(gdf, outputs_path ):
	"""
	Export a geoDataFrame to a series of shapefiles, one for each image present in the table. 

	"""
	Path(outputs_path).mkdir(exist_ok=True, parents=True)
	for name, group in gdf.groupby('chip'):
		out_path = Path(outputs_path, name).with_suffix('.shp')
		group.to_file(out_path)

def dets_dfs_to_shps(dets_dfs, stats_df, outputs_path=None, crs=None):
	""" 
	For a given folder of yolo detections, and a matfile containing the details of the images including the image paths,
	loop through the paths and run yolo_to_shp. 
	"""    
	print("Writing shapefiles...")
	Path(outputs_path).mkdir(exist_ok=True, parents=True)
	img_paths = stats_df['img_paths'].tolist()
	for img_path in img_paths:
		out_path = Path(outputs_path, Path(img_path).stem).with_suffix('.shp')
		img_name = str ( Path( img_path ).stem )
		dets_df = dets_dfs.loc[ dets_dfs['chip'] == img_name ].reset_index().copy()
		dets_df_to_shps(img_path, dets_df, out_path, crs)

def dets_df_to_shps(img_path, dets_df, out_path, crs=None):
	""" For a given image, and its corresponding labels in a dataframe, output the shapefile. 
	"""
	if dets_df.shape[0] == 0:
		return ValueError
	elif img_path is not None:
		with rio.open(img_path) as src:
			crs = src.crs
			dets_df.xmin, dets_df.ymin = rio.transform.xy(src.transform, dets_df.ymin, dets_df.xmin)
			dets_df.xmax, dets_df.ymax = rio.transform.xy(src.transform, dets_df.ymax, dets_df.xmax)

	dets_gdf = gpd.GeoDataFrame( 
		geometry=[sp.geometry.Polygon([(r.xmin, r.ymin), (r.xmax, r.ymin), (r.xmax, r.ymax), (r.xmin, r.ymax)]) for r in 
		dets_df.itertuples()], crs=crs)

	dets_gdf['object_class_id'] = dets_df['class_id']
	dets_gdf['confidence'] = dets_df['confidence']
	dets_gdf['true_pos'] = dets_df['true_pos']
	dets_gdf['IoU'] = dets_df['IoU']
	dets_gdf.to_file(out_path)

def yolo_dets_to_df(yolos_path):
	""" 
	Grab data from a path containing yolo format detections, and return a df containing all rows. 
	"""
	print("Loading detections into a dataframe ...")
	dets_df2 = pd.DataFrame()
	for yolo_path in Path(yolos_path).glob('*.txt'):
		yolo_path = Path(yolo_path)
		dets_df = yolo_det_to_df(yolo_path)
		dets_df2 = pd.concat([dets_df2, dets_df])
	return dets_df2

def yolo_det_to_df(yolo_path, headers=['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence', 'det_id']):
	""" 
	Grab data from yolo detection file and return a dataframe.
	"""
	try:
		dets_df = pd.read_csv(yolo_path, names=headers, delim_whitespace=True)
	except FileNotFoundError:
		return FileNotFoundError

	dets_df['chip'] = Path(yolo_path).stem 
	return dets_df


def vectorised_iou (bboxes1, bboxes2):
	""" Given two sets of bboxes length N and M,  compute the IoU matrix using numpy vector math. Return N x M matrix .
	"""
	x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
	x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)        
	xA = np.maximum(x11, np.transpose(x21))
	yA = np.maximum(y11, np.transpose(y21))
	xB = np.minimum(x12, np.transpose(x22))
	yB = np.minimum(y12, np.transpose(y22))
	interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)        
	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)        
	iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)        
	return iou


def get_true_pos( dets_df, targs_df,iou_thres ):
	""" Given dataframe for detections, targets (ground truths) and iou threshold, return dataframes with true positives highlighted.
	"""
	print("Comparing each BBox from detections, with each Bbox from ground-truths, IoU Threshold = ",iou_thres)
	dets_df2=pd.DataFrame() # Initialise dataframes for storing new data. 
	targs_df2=pd.DataFrame()
	grouped = targs_df.groupby('chip')  
	iterator=0
	for i, (name, group) in enumerate(grouped):
		# print("Image ",i+1," of ", len(grouped))
		gt_df = targs_df.loc[ targs_df['chip'] == name, : ].copy() # Ground truth data from the targets .mat file, for the current image.
		gt_df.reset_index(inplace=True)
		p_df = dets_df.loc[ dets_df['chip'] == name, : ].sort_values('confidence', ascending=False).copy() # Detection data, in descending confidence. 
		p_df.reset_index(inplace=True)
		# print(p_df.head())
		bboxes_p = np.array(p_df.loc[ : , ['xmin', 'ymin', 'xmax', 'ymax'] ] ) 
		bboxes_gt = np.array(gt_df.loc[ :, ['xmin', 'ymin', 'xmax', 'ymax'] ] )
		iou = vectorised_iou (bboxes_p, bboxes_gt) # Compute iou matrix. 

		# Find values in the IoU matrix which are above the threshold. Store the indices in an array. Row[0] is the indice of the prediction, row[1] is the ground truth.
		inds = np.array(np.nonzero(iou>iou_thres)).transpose() 
		print("True pos with IoU over ", iou_thres, " found = ", len(inds))
		if len(inds) is not 0:
			inds = pd.DataFrame(inds).drop_duplicates(0,keep='first').drop_duplicates(1,keep='first') # Drop duplicate values, where a ground truth is attributed to multiple predictins. Keep the more confident prediction.
			# print(" Dropped duplicates, now = ",len(inds))
			for j in range(len(inds)):
				iterator +=1
				p_df.loc[ inds.iloc[j,0].tolist() , 'true_pos' ] = 1 # Assign true_pos value of 1 in the dataframes for each true_pos indice
				gt_df.loc[ inds.iloc[j,1].tolist() , 'true_pos' ] = 1
				p_df.loc[ inds.iloc[j,0].tolist() , 'true_pos_id' ] = iterator # Assign unique value to each true_pos value to link predictions to ground truths
				gt_df.loc[ inds.iloc[j,1].tolist() , 'true_pos_id' ] = iterator
				p_df.loc[ inds.iloc[j,0].tolist() , 'IoU' ] = np.around( iou[inds.iloc[j,0],inds.iloc[j,1]], decimals = 2  ) # Assign IoU value to each prediction true_pos.
				gt_df.loc[ inds.iloc[j,1].tolist() , 'IoU' ] = np.around( iou[inds.iloc[j,0],inds.iloc[j,1]], decimals = 2  ) # Same IoU for matching grount truth pair.  
			
			dets_df2 = pd.concat([dets_df2, p_df]) # Add each image back into a main dataframe. 
			targs_df2 = pd.concat([targs_df2, gt_df]) 
		else:
			dets_df2 = pd.concat([dets_df2, p_df]) # If there are no true_pos, still add the image back into the dataframe. 
			targs_df2 = pd.concat([targs_df2, gt_df])
	  
	dets_df = dets_df2.set_index('index').sort_index().fillna(0) # Sort and index dataframes as they were before, and fill NaNs with 0s. 
	targs_df = targs_df2.set_index('index').sort_index().fillna(0)

	return dets_df, targs_df

def AP_images(dets_df, targs_df, stats_df, opt):
	"""Given detections and targets dataframes, with true positive counts, compute average precision via trapezium-rule integration of the cumulative recall/precision graph.
	Return list of average precision, precision and recall for each image.
	"""
	print(" Computing integrals to attain Average Precision ... ")

	average_precision=[]
	precision=[]
	recall=[]
	dataset=[]
	chips=[]
	gt_count=[]
	det_count=[]
	true_pos_count=[]
	move_stats_df = pd.DataFrame()

	# Loop through the images within the dataset.
	grouped = targs_df.groupby('chip')
	for i, (name, group) in enumerate(grouped):
		# print("Image ",i+1," of ", len(grouped))
		gt_df = targs_df.loc[ targs_df['chip'] == name, : ].copy() # Ground truth data from the targets .mat file, for the current image.
		p_df = dets_df.loc[ dets_df['chip'] == name, : ].sort_values('confidence', ascending=False).copy() # Detection data, in descending confidence. 
		p_df.reset_index(inplace=True)
		grd_true = len(gt_df)
		cum_rec=[]
		cum_pre=[]
		d_integral=[]
		# Loop through each prediction in order of descending confidence. 
		for i, pred in enumerate(p_df.index.tolist()):
			# Compute cumulative number of true-pos.
			cum_true = sum( p_df.loc[0:pred, 'true_pos'] )  
			# Compute cumulative precision = number of true_pos / number of predictions so far.
			cum_pre.append(cum_true/(i+1))
			# Compute cumulative recall = number of true_pos so far / total number of ground truths.
			cum_rec.append(cum_true/grd_true)
			if i == 0:  # Initialise values for integration.
				y1 = cum_pre[0]
				x1 = cum_rec[0]
			if i > 0: 
				# Integration of { precision = f(recall) } d (recall) , via trapezium rule.
				y2 = cum_pre[i] # Height of right edge of trapezium
				x2 = cum_rec[i] # Lower right corner of trapezium
				dx = x2 - x1 # Width of trapezium 
				d_integral.append(0.5 * (y1 + y2) * dx)  # Area of trapezium
				y1 = y2   # Height of left edge of trapezium 
				x1 = x2  # Lower left corner of trapezium
		average_precision.append(sum(d_integral))
		precision.append(y2) # Final values of cumulative variables is the actual variable. 
		recall.append(x2)
		gt_count.append(grd_true)
		det_count.append(len(p_df))

		true_pos_count.append(cum_true)
		chips.append(name) # Store name of image in a parallel list. 
		dataset_name = stats_df.loc[stats_df.index == name, 'img_paths'].tolist()[0].split('/')[-3] # Work out which dataset the image came from.
		dataset.append(dataset_name)

		# Get movement stats 
		gt_df = targs_df.loc[ targs_df['chip'] == name, : ].copy() # Ground truth data from the targets .mat file, for the current image.
		p_df = dets_df.loc[ dets_df['chip'] == name, : ].copy()
		move_stats_df_right = get_moving_true_pos(p_df, gt_df, opt)
		move_stats_df = pd.concat([move_stats_df, move_stats_df_right])

	# Store all the computed information in a dataframe and return it. 
	scores_dict = {'Dataset': dataset, \
			'Average Precision': average_precision, \
			'Precision': precision, \
			'Recall': recall, \
			'object_count': gt_count, \
			'detection_count': det_count, \
			'gt_count': gt_count, \
			'true_pos_count': true_pos_count, \
				}

	score_df = pd.DataFrame(scores_dict, index = chips)
	# Convert columns to multiindex, with zeros as the upper level, to preserve move_stats_df multiindex levels when joining the dataframes:
	score_df.columns = [np.zeros(len(score_df.columns)), score_df.columns]

	move_stats_df.index = chips
	score_df = pd.concat([score_df, move_stats_df], axis = 1)
	return score_df

def AP_aggregate(dets_df, targs_df, opt):
	"""Given detections and targets dataframes, with true positive counts, compute average precision via trapezium-rule integration of the cumulative recall/precision graph.
	Return average precision, recall and precision for the entire test set as a 1 row dataframe.
	"""
	gt_df = targs_df.copy()
	p_df = dets_df.sort_values('confidence', ascending=False).copy()
	p_df.reset_index(inplace=True)
	grd_true = len(gt_df) # Total number of ground truths in targets matfile. 
	cum_rec = []
	cum_pre = []
	d_integral=[]
	# Loop through each prediction in order of descending confidence. 
	for i, pred in enumerate(p_df.index.tolist()):
		# Compute cumulative number of true-pos.
		cum_true = sum( p_df.loc[0:pred, 'true_pos'] )  
		# Compute cumulative precision = number of true_pos / number of predictions so far.
		cum_pre.append(cum_true/(i+1))
		# Compute cumulative recall = number of true_pos so far / total number of ground truths.
		cum_rec.append(cum_true/grd_true)
		if i == 0: # Initialise values for integration.
			y1 = cum_pre[0]
			x1 = cum_rec[0]
		if i > 0: 
			# Integration of { precision = f(recall) } d (recall) , via trapezium rule.
			y2 = cum_pre[i] # Height of right edge of trapezium
			x2 = cum_rec[i] # Lower right corner of trapezium
			dx = x2 - x1 # Width of trapezium 
			d_integral.append(0.5 * (y1 + y2) * dx) # Area of trapezium
			y1 = y2 # Height of left edge of trapezium 
			x1 = x2 # Lower left corner of trapezium
	average_precision = sum(d_integral)
	precision = y2 # Final value of cumulative variables is the actual variable. 
	recall = x2

	gt_count = grd_true
	true_pos_count=cum_true

	move_stats_df = get_moving_true_pos(dets_df, targs_df, opt)

	score = {'Dataset':'Aggregate', \
		'Average Precision': average_precision, \
		'Precision': precision, \
		'Recall': recall, \
		'object_count': len(targs_df), \
		'detection_count': len(dets_df), \
		'gt_count': gt_count, \
		'true_pos_count': true_pos_count, \
	 }

	score_df = pd.DataFrame(score, index = [Path(opt.targets).stem])
	score_df.columns = [np.zeros(len(score_df.columns)), score_df.columns]

	print(move_stats_df)
	move_stats_df.index = score_df.index

	score_df = pd.concat([score_df, move_stats_df], axis = 1)
	return score_df

def create_dfs(targets_path):
	"""
	Pull targets in from .mat file into targs_df, and create stats_df (N-img long) from targets/dets information. 
	"""
	target = scipy.io.loadmat(targets_path, squeeze_me=True)
	targs_df = pd.DataFrame(target['targets'], columns = ['class_id', 'xmin', 'ymin', 'xmax', 'ymax'] )
	targs_df['img_id'] = target['id'].astype(int)
	targs_df['chip'] = target['chips']

	stats_df = pd.DataFrame([ target['img_paths'], target['uchips'] ]).transpose()
	stats_df.columns = ['img_paths', 'uchips']

	stats_df.set_index('uchips', inplace=True)
	stats_df = pd.concat([ stats_df, targs_df.groupby('chip').chip.count()], axis=1 ).rename(columns={'chip':'object_count'})
	# stats_df = pd.concat([ stats_df, dets_df.groupby('chip').chip.count()], axis=1 ).rename(columns={'chip':'detection_count'})
	return targs_df, stats_df

def get_scores( dets_df, targs_df, stats_df, out_dir, opt, buffer_type='None' ):

	print(" Computing integrals to attain Average Precision ... ")
	score_agg_df = AP_aggregate(dets_df, targs_df, opt)
	score_df = AP_images(dets_df, targs_df, stats_df, opt)

	scores_df = pd.concat([score_agg_df, score_df])

	scores_df['weights'] = Path(opt.weights).stem
	scores_df['nms-thres'] = opt.nms_thres
	scores_df['conf_thres'] = opt.conf_thres
	scores_df['name'] = Path(opt.targets).stem
	scores_df['iou-thres'] = opt.iou_thres
	scores_df['buffer_type'] = buffer_type
	scores_df['classes'] = opt.classes

	print(scores_df.to_string())
	scores_df.to_csv(Path( out_dir, Path(opt.targets).stem ).with_suffix('.txt'))
	return(scores_df)

def get_total_true_pos(dets_gdf, targs_gdf):
	# Find the 'union' of the unique 'true_pos_id's of both detections and ground truths. 
	# This will ensure our statistics include every true pair where either the detection or the target box touches a road buffer, even if the matching detection or target does not. 
	true_dets = dets_gdf.true_pos_id.unique() # Includes true_pos_id = 0 (the negatives)
	true_targs = targs_gdf.true_pos_id.unique() # Includes true_pos_id = 0 (the negatives)
	total_true_pos = len( np.unique( np.concatenate((true_dets,true_targs), axis=0 ) ) ) - 1  # -1 because when true_pos_id = 0, it is not a true pos. 

	# Compute total number of detections and ground truths for statistics.
	# Include parents/children of detections/grount truths that are outside the buffer, even though their assosciate is inside.
	# These are 'ghost' values as they were ommitted during intersection, but now must be added back into the calculation.  
	# There is one 'ghost' detection for every unmatched ground truth, and one ghost ground truth for every unmatched detection. 
	total_dets = len(dets_gdf) + ( total_true_pos - ( len(true_dets) - 1 ) ) # Add the 'ghost' children of detections back into the statistic. -1 because true_targs includes true_pos_id = 0 
	total_gts = len(targs_gdf) + ( total_true_pos - ( len(true_targs) - 1 ) ) # Add the 'ghost' parents of grount truths back into the statistic. -1 because true_dets includes tre_pos_id = 0. 

	prec = total_true_pos / total_dets
	rec = total_true_pos / total_gts

	return total_true_pos, total_dets, total_gts, prec, rec

def get_road_scores(road_dets, road_targs, stats_df, buffer_type, out_dir, opt):
	"""
	Compute recall and precision for the road buffered geodtaframes. No support for Average Precision.
	Output scores_df for each image as well as for the aggregate data. 
	"""
	# Remove duplicate values from the intersected dataframes, where one bbox intersets more than one road buffer polygon.
	# For true_pos, we do this by ensuring each true_pos_id appears only once.
	pos_dets = road_dets.loc[ road_dets['true_pos'] == 1 ].drop_duplicates('true_pos_id').copy()
	# For non true-pos boxes, when intersected, the original unique index is inherited into column[0]. Use this to ensure all duplicates are removed. 
	neg_dets = road_dets.loc[ road_dets['true_pos'] == 0 ].drop_duplicates(road_dets.columns[0]).copy()
	road_dets = pd.concat([pos_dets, neg_dets]).reset_index(drop=True)
	# Same process for targets:
	pos_targs = road_targs.loc[ road_targs['true_pos'] == 1 ].drop_duplicates('true_pos_id').copy()
	neg_targs = road_targs.loc[ road_targs['true_pos'] == 0 ].drop_duplicates(road_dets.columns[0]).copy()
	road_targs = pd.concat([pos_targs, neg_targs]).reset_index(drop=True)

	precision=[] # Initialise lists 
	recall=[]
	chips=[]
	dataset=[]
	average_precision=[]
	object_count=[]
	det_count=[]
	gt_count=[]
	true_pos_count=[]
	move_stats_df = pd.DataFrame()
	## Loop through the images within the dataset. ##

	grouped = road_targs.groupby('chip')
	for i, (name, group) in enumerate(grouped):
		# print("Image ",i+1," of ", len(grouped))
		dets_gdf = road_dets.loc[ road_dets['chip'] == name, : ].copy() # Detections data for the current image in the loop.
		targs_gdf = road_targs.loc[ road_targs['chip'] == name, : ].copy() # Ground truth data for the current image in the loop.

		total_true_pos, total_dets, total_gts, prec, rec = get_total_true_pos(dets_gdf, targs_gdf)

		precision.append(prec) 
		recall.append(rec)
		average_precision.append('NaN') # No support for integral in road buffer yet.
		chips.append(name) 
		dataset_name = stats_df.loc[stats_df.index == name, 'img_paths'].tolist()[0].split('/')[-3] # Work out which dataset the image came from.
		dataset.append(dataset_name)
		object_count.append(len(targs_gdf)) # Find the number of ground truths in the image ( within the road buffer)
		det_count.append(total_dets) # Number of detections (including 'ghost' children)
		gt_count.append(total_gts) # Number of Ground_truths (ncluding 'ghost' parents)
		true_pos_count.append(total_true_pos) # Number of true positives (union of dets+gts)

		# Get movement stats 
		move_stats_df_right = get_moving_true_pos(dets_gdf, targs_gdf, opt)
		move_stats_df = pd.concat([ move_stats_df, move_stats_df_right ])

	# Store the stats for each image in a dataframe.
	
	scores_dict_1 = {'Dataset': dataset, \
					'Average Precision':average_precision, \
					'Precision':precision, \
					'Recall': recall, \
					'object_count':len(road_targs), \
					'detection_count':det_count, \
					'gt_count':gt_count, \
					'true_pos_count':true_pos_count, \
					}

	scores_df = pd.DataFrame(scores_dict_1, index=chips)

	move_stats_df.index = chips

	## Now compute stats for the aggregate test dataset. ##
	total_true_pos, total_dets, total_gts, agg_precision, agg_recall = get_total_true_pos(road_dets, road_targs)

	agg_precision = total_true_pos / total_dets
	agg_recall = total_true_pos / total_gts

	agg_det_count = total_dets
	agg_gt_count = total_gts
	agg_true_pos_count = total_true_pos

	average_precision = 'Nan'

	# Store the aggregate data in a single-row dataframe, and then join it with the previous.
	scores_dict_2 = {'Dataset':'Aggregate', \
					'Average Precision':average_precision, \
					'Precision':agg_precision, \
					'Recall': agg_recall, \
					'object_count':len(road_targs), \
					'detection_count':agg_det_count, \
					'gt_count':agg_gt_count, \
					'true_pos_count':agg_true_pos_count, \
					}

	score_agg_df = pd.DataFrame(scores_dict_2, index = [ (Path(opt.targets).stem + '_' + buffer_type) ])

	move_stats_agg_df = get_moving_true_pos(road_dets, road_targs, opt)
	move_stats_agg_df.index = [ (Path(opt.targets).stem + '_' + buffer_type) ]

	scores_df = pd.concat([score_agg_df, scores_df])
	move_stats_df = pd.concat([move_stats_agg_df, move_stats_df])

	scores_df.columns = [np.zeros(len(scores_df.columns)), scores_df.columns]

	scores_df = pd.concat([scores_df, move_stats_df ], axis = 1)

	# Add ancilliary information from the detect.py parser 
	scores_df['weights'] = Path(opt.weights).stem
	scores_df['nms-thres'] = opt.nms_thres
	scores_df['conf_thres'] = opt.conf_thres
	scores_df['name'] = (Path(opt.targets).stem + '_' + buffer_type)
	scores_df['iou-thres'] = opt.iou_thres
	scores_df['buffer_type'] = buffer_type
	scores_df['classes'] = opt.classes
	# Print and save data, appending filename with the buffer_type for easy identification. 
	print(scores_df.to_string())
	scores_df.to_csv(Path( out_dir, Path(opt.targets).stem + '_' + buffer_type).with_suffix('.txt'))

	return scores_df

def get_moving_true_pos(dets_df, targs_df, opt):
	"""
	Convert numerical class_id to pull out the movement class. 
	Sort both targs and dets by their true_pos_id, and then compare the movement attribution to detirmine if it is correct. 
	Create df with multiindex collumns with counts of each movement class >> 0: ground truths, 1: raw detections, 2: true detections, and 3: true detections with true movement attribution. 
	"""
	if opt.classes == 6:
		class_map = ['static', 'moving', 'static', 'moving', 'static', 'moving']
		for num, movement_class in enumerate(class_map):
			dets_df.loc[ dets_df['class_id'] == num, 'movement_class' ] = movement_class
			targs_df.loc[ targs_df['class_id'] == num, 'movement_class' ] = movement_class
	elif opt.classes == 9:
		class_map = ['parked', 'static', 'moving', 'parked', 'static', 'moving', 'parked', 'static', 'moving']
		for num, movement_class in enumerate(class_map):
			dets_df.loc[ dets_df['class_id'] == num, 'movement_class' ] = movement_class
			targs_df.loc[ targs_df['class_id'] == num, 'movement_class' ] = movement_class
	elif opt.classes == 3:
		class_map = ['parked', 'static', 'moving']
		for num, movement_class in enumerate(class_map):
			dets_df.loc[ dets_df['class_id'] == num, 'movement_class' ] = movement_class
			targs_df.loc[ targs_df['class_id'] == num, 'movement_class' ] = movement_class
	else:
		print('Number of classes has no movement class map. Returning 0 for movement stats.')
		return 0, 0, 0, 0

	true_dets = dets_df.loc[ dets_df['true_pos'] == 1 ].copy().reset_index().set_index('true_pos_id').sort_index()
	true_targs = targs_df.loc[ targs_df['true_pos'] == 1 ].copy().reset_index().set_index('true_pos_id').sort_index()

	true_dets['true_movement'] = true_dets.eq(true_targs).movement_class
	true_targs['true_movement'] = true_dets.eq(true_targs).movement_class

	true_dets = true_dets.reset_index(drop=False).set_index('index')
	true_targs = true_targs.reset_index(drop=False).set_index('index')

	dets_df.loc[ true_dets.index, 'true_movement'] = true_dets['true_movement'] 
	targs_df.loc[ true_targs.index, 'true_movement'] = true_targs['true_movement'] 

	# true_pos_moving_true = sum( dets_df.loc[dets_df['movement_class'] == 'moving'].true_movement == True )
	# true_pos_moving = sum( dets_df.loc[dets_df['movement_class'] == 'moving'].true_pos == 1 )

	# Ground truths
	gt_parked = sum (targs_df.movement_class == 'parked' )
	gt_static = sum (targs_df.movement_class == 'static' )
	gt_moving = sum (targs_df.movement_class == 'moving' )
	# Detections
	dets_parked = sum (dets_df.movement_class == 'parked' )
	dets_static = sum (dets_df.movement_class == 'static' )
	dets_moving = sum (dets_df.movement_class == 'moving' )
	# True Detections
	tp_parked = sum (true_dets.movement_class == 'parked' )
	tp_static = sum (true_dets.movement_class == 'static' )
	tp_moving = sum (true_dets.movement_class == 'moving' )
	# True detections with true movement 
	tptm_parked = sum( dets_df.loc[dets_df['movement_class'] == 'parked'].true_movement == True )
	tptm_static = sum( dets_df.loc[dets_df['movement_class'] == 'static'].true_movement == True )
	tptm_moving = sum( dets_df.loc[dets_df['movement_class'] == 'moving'].true_movement == True )

	cols_1 = ['Ground Truth', 'Detection', 'True Detection', 'True Detection with True Movement']
	cols_1 = np.array([ [cols_1[i], cols_1[i], cols_1[i]] for i in range(len(cols_1))  ]).flatten()

	cols_2 = ['Parked', 'Static', 'Moving']
	cols_2 = np.array([cols_2, cols_2, cols_2, cols_2]).flatten()

	vals = [gt_parked, gt_static, gt_moving, dets_parked, dets_static, dets_moving, tp_parked, tp_static, tp_moving, tptm_parked, tptm_static, tptm_moving ]
	df = pd.DataFrame( columns=[cols_1, cols_2])
	df.loc[0] = vals
	return df
