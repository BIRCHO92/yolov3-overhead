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

  # Name of the test - this will be used to create the name for each kfold matfile pair : test_name_train_1.mat, test_name_test_1.mat
  test_name = 'london_10folds_9class'

  # Number of folds to create. Eg if k = 10, then 10 train/test pairs will be created (20 matfiles in total)
  k = 10

  # Path to directory in which to store the kfolds matfiles. 
  matfile_dir_path = "/home/george/xview-yolov3/utils"

  dets_df, stats_df = paths_to_df(root_paths, label_dirname, img_dirname)
  dets_dict, stats_dict = kfolds(stats_df, dets_df, k)
  dfs_to_mat(dets_dict, stats_dict, matfile_dir_path, class_map, test_name)

create()
