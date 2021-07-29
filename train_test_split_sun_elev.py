import argparse
import time
from sys import platform

from gb_utils import *
from pathlib import Path
from datetime import datetime



parser = argparse.ArgumentParser()
# Get data configuration
parser.add_argument('--images', type=str, default='train_images/5.tif', help='path to images')
parser.add_argument('--sym_path', type=str, default='symlinks')
parser.add_argument('--matfile_name', type=str, default='test')
parser.add_argument('--sun_thres', type=str, default=35)
parser.add_argument('--valid_ratio', type=str, default=0.1)

opt = parser.parse_args()
print(opt)

def create(opt):

   # class_map = [0, 0, 1, 1, 2, 2]
   class_map = [0, 1, 2]
   root_paths = [opt.images]
     
   dets_df, stats_df = paths_to_df([opt.images])
   stats_df = paths_to_symlinks([opt.images], opt.sym_path, stats_df)
   stats_df = get_sun_elev(stats_df)
   t1, t2, v1, v2 = train_test_split(dets_df, stats_df, opt.sun_thres, opt.valid_ratio)
   

   dfs = (t1, t2, v1, v2)

   for i, df in enumerate(dfs):
      matfile_path = 'london_'+str(opt.matfile_name)+'_'+str(i)+'.mat'
      print('Writing ',matfile_path)
      df_to_mat(df, matfile_path, class_map)

create(opt)
