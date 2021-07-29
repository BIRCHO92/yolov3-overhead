from gb_utils import *
from pathlib import Path


def create():

  # Class mapping. Must be a list with the same length as the number of classes that were stored in the yolo-style .txt label files.
  # class_map can be used to amalgamate certain classes. 
  class_map = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Leave classes unchanged. 

  # List of parent directories within which images and labels are stored in their own named subdirectories. 
  root_paths = [ \
        # "/mnt/Data/AQM_training/london_RGB_8bit_ACOMP_30cm",
            "/content/drive/My Drive/George_Birchenough/04_London_Datasets/london_8bit_combined/london_8bit_DRA-ACOMP_30cm",
            "/content/drive/My Drive/George_Birchenough/04_London_Datasets/london_8bit_combined/london_8bit_DRA-ACOMP_50cm"]

  # Name of the subdiretories within each of the above directories, containing yolo style labels in .txt files.
  label_dirname = 'labels_9_class'
  # Name of the subdiretories within each of the above directories, containing .tif images with identical names as above labels. 
  img_dirname = 'images_RGB'

  # Path in which to save the resulting matfile. 
  matfile_path = "/home/george/xview-yolov3/utils/test_set_001.mat"

  dets_df, stats_df = paths_to_df(root_paths, label_dirname, img_dirname)
  df_tuple = (dets_df, stats_df)
  df_to_mat(df_tuple, matfile_path, class_map)

create()
