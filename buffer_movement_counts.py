yolos_path = '/mnt/Data/AQM_training/01_development/labels_9_class'
images_path = '/mnt/Data/AQM_training/london_RGB_8bit_DRA-ACOMP_50cm/images'
road_buffer_shp_path = '/mnt/server/AQM/AQM/01_Vehicle_Detection/Europe/London/02_City_Data/02_Roads/02_Buffered/02_Measured_Buffers/combined_buffer_q_0.9.shp'

from gb_utils import *
from gb_scoring import *

def append_results(df):
	all_historic_results = './all_test_results_01.csv'
	with open(all_historic_results, 'a') as f:
		df.to_csv(f)


dets_df, stats_df = yolos_to_df(yolos_path, images_path)
print('Build DF with ', len(dets_df), ' objects, within ', len(stats_df), ' images')

dets_df = dets_df.dropna().copy()
dets_df.reset_index(drop=True, inplace=True)
print('Removed Nans. Left with ', len(dets_df), ' objects, within ', len(stats_df), ' images')
print('Rearranging classes into parked, static, moving.')
class_map = [0, 1, 2, 0, 1, 2, 0, 1, 2]
newArray = np.zeros(len(dets_df))
for k, v in enumerate(class_map):
	dets_df.loc[ dets_df['class_id']==k, 'class_id_3_moving' ] = v


buffer_gdf = gpd.read_file(road_buffer_shp_path)
buffer_crs = buffer_gdf.crs.to_epsg()

print(' Creating Geodataframes from Dataframes using crs from road buffer : ')
stats_df['img_paths'] = stats_df.img_path.tolist()
dets_gdf = dfs_to_gdfs(dets_df, stats_df, buffer_crs)
dets_gdf.reset_index(drop=True, inplace=True)

print('Overlaying boxes onto road buffer.')
road_dets = gpd.overlay(dets_gdf, buffer_gdf, how='intersection')

print('We now have this many objects: ', len(road_dets))
print ( 'This many are unique objects: ', len(road_dets['index'].unique()) )

print('Computing area of each intersected box.')
road_dets['box_area'] = road_dets.area
print('Removing duplicate boxes keeping the one with largest area')
print ( 'After .area, This many are unique objects: ', len(road_dets['index'].unique()) )

# Sort by intersecting area, then by buffer size, largest first.
road_dets = road_dets.sort_values(['box_area', 'max_buffer'], ascending = [False, False])
# Drop duplicates from th inherited index_1 so that each GT is only represented once.
road_dets = road_dets.drop_duplicates('index', keep = 'first')
print('After drop duplicates: ',len(road_dets) )
# Create dataframe with group counts 
dfg = road_dets.groupby(['LEGEND', 'class_id_3_moving']).size()
dfg = dfg.reset_index().set_index('LEGEND')
dfg = dfg.set_index([dfg.index, 'class_id_3_moving'])[0].unstack()
dfg.columns = ['parked', 'static', 'moving']
print('Saving CSV.')
append_results(dfg)

# Repeat but with only major/minor labels:
road_dets = road_dets.sort_values(['box_area', 'layer'], ascending = [False, True])

dfg = road_dets.groupby(['layer', 'class_id_3_moving']).size()
dfg = dfg.reset_index().set_index('layer')
dfg = dfg.set_index([dfg.index, 'class_id_3_moving'])[0].unstack()
dfg.columns = ['parked', 'static', 'moving']
append_results(dfg)

# Reset geometetry to centroids and repeat above analyses:

dets_gdf.geometry = dets_gdf.geometry.centroid

print('Overlaying box centroids onto road buffer.')
road_dets = gpd.overlay(dets_gdf, buffer_gdf, how='intersection')

print('We now have this many objects: ', len(road_dets))
print ( 'This many are unique objects: ', len(road_dets['index'].unique()) )

# Sort by intersecting area, then by buffer size, largest first.
road_dets = road_dets.sort_values(['max_buffer'], ascending = False)
# Drop duplicates from th inherited index_1 so that each GT is only represented once.
road_dets = road_dets.drop_duplicates('index', keep = 'first')
print('After drop duplicates: ',len(road_dets) )
# Create dataframe with group counts 
dfg = road_dets.groupby(['LEGEND', 'class_id_3_moving']).size()
dfg = dfg.reset_index().set_index('LEGEND')
dfg = dfg.set_index([dfg.index, 'class_id_3_moving'])[0].unstack()
dfg.columns = ['parked', 'static', 'moving']
print('Saving CSV.')
append_results(dfg)

# Repeat but with only major/minor labels:
road_dets = road_dets.sort_values(['layer'], ascending = True)

dfg = road_dets.groupby(['layer', 'class_id_3_moving']).size()
dfg = dfg.reset_index().set_index('layer')
dfg = dfg.set_index([dfg.index, 'class_id_3_moving'])[0].unstack()
dfg.columns = ['parked', 'static', 'moving']

append_results(dfg)
