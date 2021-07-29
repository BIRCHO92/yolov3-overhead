from pathlib import Path
import itertools
import re
import os
import pandas as pd
import numpy as np
import cv2
import scipy.io
import glob
import shutil
from astropy.coordinates import get_sun, AltAz, EarthLocation
from astropy.time import Time 

from pathlib import Path
import rasterio as rio
import geopandas as gpd
import pandas as pd
import shapely as sp

def shps_to_gdf(shapefiles_path, crs = 32630 ):
    """For a given folder of shapefiles, create single gdf with CRS as an input. Convert text labels to numerical. CRS defaults to 32360. """
    gdf = gpd.GeoDataFrame()
    for shp in Path(shapefiles_path).rglob('*.shp'):
        
        shp_path = Path(shp)
        # print('shp_path:', shp_path)
        gdf1 = gpd.read_file(shp_path)
        if gdf1.crs == None:
            print('Warning, naive CRS')
        gdf2 = gdf1.loc[ gdf1.geometry.dropna().index ] 
        if len(gdf1) != len(gdf2) :
            print( shp_path, ': Warning, NaNs present in geometry, removed ', len(gdf1) - len(gdf2), ' rows.' )
        gdf2 = gdf2.to_crs(crs)
        gdf2['chip'] = shp_path.stem
        gdf = pd.concat([gdf, gdf2])
    # Convert classes to unique ID integers
        # 0: small parked
        # 1: small static
        # 2: small moving
        # 3: medium parked
        # 4: medium static
        # 5: medium moving
        # 6: large parked
        # 7: large static
        # 8: large moving

    for i, combo in enumerate(list(itertools.product(['small', 'medium', 'large'], ['parked', 'static', 'moving']))):
        gdf.loc[(gdf['Class'] == combo[0]) & (gdf['Move_2'] == combo[1]), 'yolo_class'] = i
    gdf.reset_index(drop = True, inplace=True)
    return gdf

def shps_to_yolos(shapefiles_path, imgs_path, outputs_path):
        """For a given folder of shapefiles and images, converts labels to yolo format text files"""

        for shp in Path(shapefiles_path).glob('*.shp'):
                shp_path = Path(shp)
                print('shp_path:', shp_path)

                # Find image using regex - could have different order number or band tag :(
                pattern = re.compile('([^_]*)-[^_]*-[^_]*_[^_]*_[^_]*_[^_]*_(.*).shp')
                # Get capture groups of interest from shp name (datetime & tiles)
                groups = pattern.findall(shp_path.name)[0]
                pattern_to_find = re.compile('' + groups[0] + '-[^_]*-[^_]*_[^_]*_[^_]*_[^_]*_' + groups[1] + '.tif')
                match = None
                for file in os.listdir(imgs_path):
                        if pattern_to_find.match(file):
                                if not match:
                                        match = file
                                else:
                                        print("More than one match! skipping")
                                        continue  # Could handle better
                if match is None:
                        print("No matches! Skipping")
                        continue

                img_path = Path(imgs_path, match)
                print('img_path:', img_path)

                out_path = Path(outputs_path, img_path.stem).with_suffix('.txt')
                print('out_path:', out_path)
                shp_to_yolo(shp, img_path, out_path)

def shp_to_yolo(shapefile_path, img_path, output_path):
        """For a given shapefile, and image, converts labels to yolo format text files"""

        # Get img dimensions
        with rio.open(img_path) as src:
                dims = src.meta['width'], src.meta['height']
                transform = src.meta['transform']

        # Read shapefile into dataframe
        gdf = gpd.read_file(shapefile_path)

        # # Convert classes to unique ID integers
        # # 0: small static
        # # 1: small moving
        # # 2: medium static
        # # 3: medium moving
        # # 4: large static
        # # 5: large moving
        # for i, combo in enumerate(list(itertools.product(['small', 'medium', 'large'], ['static', 'moving']))):
        #     gdf.loc[(gdf['Class'] == combo[0]) & (gdf['Movement'] == combo[1]), 'yolo_class'] = i

        # Convert classes to unique ID integers
        # 0: small parked
        # 1: small static
        # 2: small moving
        # 3: medium parked
        # 4: medium static
        # 5: medium moving
        # 6: large parked
        # 7: large static
        # 8: large moving

        for i, combo in enumerate(list(itertools.product(['small', 'medium', 'large'], ['parked', 'static', 'moving']))):
                gdf.loc[(gdf['Class'] == combo[0]) & (gdf['Move_2'] == combo[1]), 'yolo_class'] = i

        # Convert geometry to yolo format:
        # coords as relative floats between 0-1
        gdf['float_minx'] = ((gdf.loc[:, 'geometry'].bounds.minx - transform[2]) / transform[0]) / dims[0]
        gdf['float_maxy'] = ((gdf.loc[:, 'geometry'].bounds.miny - transform[5]) / transform[4]) / dims[1]
        gdf['float_maxx'] = ((gdf.loc[:, 'geometry'].bounds.maxx - transform[2]) / transform[0]) / dims[0]
        gdf['float_miny'] = ((gdf.loc[:, 'geometry'].bounds.maxy - transform[5]) / transform[4]) / dims[1]

        gdf.loc[gdf['float_minx'] < 0, 'float_minx'] = 0
        gdf.loc[gdf['float_maxx'] > 1, 'float_maxx'] = 1
        gdf.loc[gdf['float_miny'] < 0, 'float_miny'] = 0
        gdf.loc[gdf['float_maxy'] > 1, 'float_maxy'] = 1

        # coords of centre, width, height
        gdf['width'] = gdf.loc[:, 'float_maxx'] - gdf.loc[:, 'float_minx']
        gdf['height'] = gdf.loc[:, 'float_maxy'] - gdf.loc[:, 'float_miny']
        gdf['centre_x'] = gdf.loc[:, 'float_minx'] + gdf.loc[:, 'width'] / 2
        gdf['centre_y'] = gdf.loc[:, 'float_miny'] + gdf.loc[:, 'height'] / 2

        # Write to output file
        Path(output_path.parent).mkdir(exist_ok=True)
        with open(output_path, 'w') as dest:
                np.savetxt(dest, gdf.loc[:, ('yolo_class', 'centre_x', 'centre_y', 'width', 'height')], '%g')

        print('Done')

def yolos_to_shps(yolos_path, images_path=None, outputs_path=None, crs=None, headers=['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence']):
        """ 
        For a given folder of yolo detections, and a folder of matching images, loop through the paths and run yolo_to_shp. 
        """
        Path(outputs_path).mkdir(exist_ok=True, parents=True)

        for yolo_path in Path(yolos_path).glob('*.txt'):
                yolo_path = Path(yolo_path)
                img_path = Path(images_path, yolo_path.stem).with_suffix('.tif')
                out_path = Path(outputs_path, yolo_path.stem).with_suffix('.shp')
                yolo_to_shp(yolo_path, img_path, out_path, crs)

def yolos_df_to_shps(yolos_path, targets_path, outputs_path=None, crs=None, headers=['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence']):
        """ 
        For a given folder of yolo detections, and a matfile containing the details of the images including the image paths,
         loop through the paths and run yolo_to_shp. 
        """    
        Path(outputs_path).mkdir(exist_ok=True, parents=True)
        img_paths = np.array(scipy.io.loadmat(targets_path, squeeze_me=True)['img_paths'])
        for img_path in img_paths:
            yolo_path = Path( yolos_path, Path(img_path).stem).with_suffix('.txt')
            out_path = Path(outputs_path, Path(yolo_path).stem).with_suffix('.shp')
            yolo_to_shp(yolo_path, img_path, out_path, crs)


def yolo_to_shp(yolo_path, image_path=None, out_path=None, crs=None, headers=['xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'confidence']):
        """ 
        For a given yolo detection .txt file and matching geo-referenced image .tif , produce the shapefile of the detection bboxes.
        """
        if crs is None and image_path is None:
                return AttributeError
        if crs is not None:
                if isinstance(crs, int):
                        crs = rio.crs.CRS.from_epsg(crs)
                else:
                        crs = rio.crs.CRS.from_string(crs)
        try:
                dets_df = pd.read_csv(yolo_path, names=headers, delim_whitespace=True)
        except FileNotFoundError:
                return FileNotFoundError

        if dets_df.shape[0] == 0:
                return ValueError
        elif image_path is not None:
                with rio.open(image_path) as src:
                        crs = src.crs
                        dets_df.xmin, dets_df.ymin = rio.transform.xy(src.transform, dets_df.ymin, dets_df.xmin)
                        dets_df.xmax, dets_df.ymax = rio.transform.xy(src.transform, dets_df.ymax, dets_df.xmax)

        dets_gdf = gpd.GeoDataFrame(
                geometry=[sp.geometry.Polygon([(r.xmin, r.ymin), (r.xmax, r.ymin), (r.xmax, r.ymax), (r.xmin, r.ymax)]) for r in
                                    dets_df.itertuples()], crs=crs)

        dets_gdf['object_class_id'] = dets_df['class_id']
        dets_gdf['confidence'] = dets_df['confidence']
        
        filename = Path(yolo_path).with_suffix('.shp').name
        if out_path is None:
                dets_gdf.to_file(Path(Path(yolo_path).parent, filename))
        else:
                dets_gdf.to_file(out_path)

def get_sun_elev(stats_df):
    
    stats_df['datetime'] = [ stats_df.index.tolist()[i].split('-')[0] for i in range(len(stats_df.index)) ]
    grouped = stats_df.groupby('datetime')
    
    loc = EarthLocation.of_address('London')
    for name, group in grouped:
        sun_time=Time.strptime(name, "%y%b%d%H%M%S")
        altaz = AltAz(obstime=sun_time, location=loc)
        zen_ang = get_sun(sun_time).transform_to(altaz)
        stats_df.loc[stats_df['datetime'] == name, 'sun_elev'] = zen_ang.alt.degree

    return stats_df    

def train_test_split_sun_elev(dets_df, stats_df, elev_thres, valid_ratio):
    print("Splitting up the dataset to create summer, winter based validation and training sets")
    stats_df.loc[ stats_df['sun_elev'] < elev_thres, 'season' ] = 'winter'
    stats_df.loc[ stats_df['sun_elev'] > elev_thres, 'season' ] = 'summer'

    wvalid = stats_df.loc[ stats_df['sun_elev'] < elev_thres, : ].sample(frac=valid_ratio*2, random_state = 1)
    svalid = stats_df.loc[ stats_df['sun_elev'] > elev_thres, : ].sample(frac=valid_ratio*2, random_state = 1)

    stats_df.loc[wvalid.index, 'dataset'] = 'valid'
    stats_df.loc[svalid.index, 'dataset'] = 'valid'

    stats_df.fillna(value={'dataset':'train'}, inplace=True)

    train_stats_1 = stats_df.drop ( stats_df.loc[ (stats_df['season'] == 'winter') & ( stats_df['dataset'] == 'valid' ),: ].index ).copy()
    valid_stats_1 = stats_df.drop( train_stats_1.index ).copy()

    train_stats_2 = stats_df.drop ( stats_df.loc[ (stats_df['season'] == 'summer') & ( stats_df['dataset'] == 'valid' ),: ].index ).copy()
    valid_stats_2 = stats_df.drop( train_stats_2.index ).copy()

    train_dets_1 = dets_df[ dets_df['chip'].isin ( train_stats_1.index ) ].copy()
    valid_dets_1 = dets_df.drop(train_dets_1.index).copy()

    train_dets_2 = dets_df[ dets_df['chip'].isin ( train_stats_2.index ) ]
    valid_dets_2 = dets_df.drop(train_dets_2.index)

    t1 = (train_dets_1, train_stats_1)
    t2 = (train_dets_2, train_stats_2)
    v1 = (valid_dets_1, valid_stats_1)
    v2 = (valid_dets_2, valid_stats_2)

    return t1, t2, v1, v2

def train_test_split(dets_df, stats_df, valid_ratio):
    print("Splitting up the dataset into validation and training sets, in a 10:90 ratio, randomly")

    valid = stats_df.sample(frac=valid_ratio, random_state = 1)
    
    stats_df.loc[valid.index, 'dataset'] = 'valid'
    stats_df.fillna(value={'dataset':'train'}, inplace=True)

    train_stats_1 = stats_df.drop ( stats_df.loc[stats_df['dataset'] == 'valid' ,: ].index ).copy()
    valid_stats_1 = stats_df.drop( train_stats_1.index ).copy()

    train_dets_1 = dets_df[ dets_df['chip'].isin ( train_stats_1.index ) ].copy()
    valid_dets_1 = dets_df.drop(train_dets_1.index).copy()

    t1 = (train_dets_1, train_stats_1)
    t2 = (train_dets_2, train_stats_2)
    v1 = (valid_dets_1, valid_stats_1)
    v2 = (valid_dets_2, valid_stats_2)

    return t1, t2, v1, v2

def kfolds(stats_df, dets_df, k_folds):
    """Take a N-img dataframe and return k-folds * N-img long dataframe, with n different train/test sets randomly sampled,
     in the form of dictionaries of stats and dets dataframes.  """
    df = stats_df.copy().reset_index()
    from sklearn.model_selection import KFold

    kf = KFold(n_splits = k_folds, shuffle = True, random_state = 2)
    kf.get_n_splits()
    i=0
    df_full = pd.DataFrame()
    for (train_index, test_index) in kf.split(df):
        i+=1
        df_train = df.loc[train_index, : ].copy()
        df_train.loc[:, 'dataset'] = 'train_'+str(i)

        df_test = df.loc[test_index, : ].copy()
        df_test.loc[:, 'dataset'] = 'test_'+str(i)

        df_full = pd.concat([df_full, df_train, df_test])

    grouped = df_full.groupby('dataset')
    names = ()
    stats_dict={}
    dets_dict={}
    for i, (name, group) in enumerate(grouped):
        stats_dict[name] = df_full.loc[ df_full['dataset'] == name, : ].copy()
        dets_dict[name] = dets_df[ dets_df['chip'].isin ( stats_dict[name].loc[:,'uchip'] ) ].copy()

    return dets_dict, stats_dict



# def paths_to_df(root_paths, headers=['class_id', 'height', 'width', 'centre_x', 'centre_y']):
#   print("Warning! Using incorrect yolo label format specific to London_combined_608 tiles!")

def paths_to_df(root_paths, label_dirname, img_dirname, headers=['class_id', 'centre_x', 'centre_y', 'width', 'height']):
    """
    For a list of directory paths, each containing named subdirectories containing images and yolo-style labels,
     create dataframes for image and bounding box data. 
    """
    dets_df1=pd.DataFrame(columns=headers)
    stats_df1=pd.DataFrame()
    for root_path in root_paths:
        print("Loading images and labels from ",root_path, "...")
        labels = os.path.join(root_path, label_dirname)
        imgs = os.path.join(root_path, img_dirname)
        dets_df2, stats_df2 = yolos_to_df(labels, imgs) 
        dets_df1 = pd.concat([dets_df1,dets_df2])
        stats_df1 = pd.concat([stats_df1, stats_df2])

    image_numbers = np.array(range(len(stats_df1)))+5000  
    stats_df1.loc[:,'image_numbers'] = image_numbers
    dets_df1 = dets_df1.reset_index(drop=True)
    stats_df1.set_index('uchip', inplace = True)

    # Drop any rows where the class is not valid.
    dets_df1 = dets_df1.drop( dets_df1.loc[ dets_df1.class_id.isna() ].index )

    return dets_df1, stats_df1

# def yolos_to_df(yolos_path, images_path, headers=['class_id', 'height', 'width', 'centre_x', 'centre_y']): # From 608 tiles - incorrect symlinks! 
#   print("Warning! Using incorrect yolo label format specific to London_combined_608 tiles!")

def yolos_to_df(yolos_path, images_path, headers=['class_id', 'centre_x', 'centre_y', 'width', 'height']): # Normal label mapping
    """ For parallel image/label folders, look for image from yolo path and run yolo_to_df function. """
    from pathlib import Path
    dets_df1=pd.DataFrame(columns=headers)
    stats_df1=pd.DataFrame()
    for yolo_path in Path(yolos_path).rglob('*.txt'):
        # print(yolo_path)
        try:
            yolo_path = Path(yolo_path)
            img_path = Path(images_path, yolo_path.stem).with_suffix('.tif')
            # print("   Img path = ",img_path)
            # print("   Label path = ",yolo_path)
        except:
            print('   filenotfounderror')
        dets_df2, stats_df2 = yolo_to_df(yolo_path, img_path) 
        dets_df1 = pd.concat([dets_df1,dets_df2])
        stats_df1 = pd.concat([stats_df1, stats_df2])
    return dets_df1, stats_df1

def yolo_to_df(yolo_path, image_path, headers=['class_id', 'centre_x', 'centre_y', 'width', 'height']):
# def yolo_to_df(yolo_path, image_path, headers=['class_id', 'height', 'width', 'centre_x', 'centre_y']):
    """ Take image data and Yolo label data and pull it into a dataframe"""
    try:
            dets_df = pd.read_csv(str(yolo_path), names=headers, delim_whitespace=True)
    except FileNotFoundError:
            return FileNotFoundError
    if dets_df.shape[0] == 0:
            return ValueError
    elif image_path is not None:
            img = cv2.imread(str(image_path))
            height,width,channels = np.shape(img)

            dets_df['xmin'] = ( dets_df.centre_x - dets_df.width / 2 ) * width
            dets_df['xmax'] = ( dets_df.centre_x + dets_df.width / 2 ) * width
            dets_df['ymin'] = ( dets_df.centre_y - dets_df.height / 2 ) * height
            dets_df['ymax'] = ( dets_df.centre_y + dets_df.height / 2 ) * height

            
            dets_df.loc[ dets_df['xmin'] < 0, 'xmin' ] = 0
            dets_df.loc[ dets_df['ymin'] < 0, 'ymin' ] = 0
            dets_df.loc[ dets_df['xmax'] > width, 'xmax' ] = width
            dets_df.loc[ dets_df['ymax'] > height, 'ymax' ] = height

            dets_df['chip'] = uchip = Path(image_path).stem

            stats = np.zeros((1,12))  # BGR mean and std, HSV mean and std
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            for j in range(3):
                stats[0,j + 0] = img[:, :, j].astype(np.float32).mean()
                stats[0,j + 3] = img[:, :, j].astype(np.float32).std()
                stats[0,j + 6] = hsv[:, :, j].mean()
                stats[0,j + 9] = hsv[:, :, j].std()
            stats_df=pd.DataFrame(stats, columns = ['bmean', 'gmean', 'rmean',\
                                                                                            'bstd', 'gstd', 'rstd',\
                                                                                            'hmean', 'smean', 'vmean',\
                                                                                            'hstd', 'sstd', 'vstd', ])
            stats_df['height'] = height
            stats_df['width'] = width      
            stats_df['uchip'] = uchip
            stats_df['img_path'] = str(image_path)
    return dets_df, stats_df

def dfs_to_mat(dets_dict, stats_dict, matfile_dir_path, class_map, test_name):
    """Pack the kfolds split datasets into tuple pairs for input to .mat generator.  """
    for name in dets_dict:
        df_tuple = ( dets_dict[name], stats_dict[name] )
        filename  = test_name + '_' + name
        print('Now creating fold name: ', filename)
        matfile_path = os.path.join(matfile_dir_path, filename+'.mat')
        df_to_mat(df_tuple, matfile_path, class_map)

def df_to_mat(df_tuple, matfile_path, class_map):
    """Take dataframe outputs from paths_to_df and write matfile for input to xview-yolov3 model"""
    dets_df, stats_df = df_tuple
    print("Loaded ", len(dets_df['chip'][:]), " objects." )
    stats_df.set_index('uchip', inplace=True)

    # Discard out-of-bounds boxes:
    dets_df = dets_df.loc[ (dets_df['centre_x'] > 0) & (dets_df['centre_y'] > 0)  , :]
    dets_df = dets_df.loc[ (dets_df['centre_x'] < 1) & (dets_df['centre_y'] < 1)  , :]

    dets_df = dets_df.loc[ (dets_df['centre_x'] - dets_df['width'] > 0) &
                                                (dets_df['centre_x'] + dets_df['width'] < 1)  , :]

    dets_df = dets_df.loc[ (dets_df['centre_y'] - dets_df['height'] > 0) &
                                                (dets_df['centre_y'] + dets_df['height'] < 1)  , :]

    print("Removed out of bounds objects. Now left with ", len(dets_df['chip'][:]), " objects." )

    print("Pull out box coordinates in xview-yolov3 format...")
    coords = np.array([dets_df.iloc[i,5:9] for i in range(len(dets_df.index))]).astype(int)
    coords[:,[1,2]]=coords[:,[2,1]]

    print("Create N_obj long list of image names for each bbox...")
    chips = np.array(dets_df['chip'][:])

    print("Create N_img long list of image dims..")
    shapes = np.array([ stats_df['height'], stats_df['width'] ]).transpose()

    print("Create N_img x 12 sized array of image stats...")
    stats = np.array( [ stats_df.iloc[i,:12] for i in range(len(stats_df.index)) ])

    print("Create N_img long list of unique chip names...")
    uchips = np.array( stats_df.index )

    # print("Create N_img long list of image sympaths...")
    # sympaths = np.array(stats_df.loc[:,'sympaths'])

    print("Create N_img long list of image actual paths...")
    img_paths = np.array(stats_df.loc[:,'img_path'])

    print("Create N_obj long list of class ID...")
    classes = np.array(dets_df['class_id']).astype(int)

    print("Create xview-yolov3 style array of obj class and bbox dims...")
    targets = np.vstack( (classes,coords.T)).T

    print("Update targets and classes")
    newArray = np.copy(targets)
    for k, v in enumerate(class_map):
        newArray[targets[:,0]==k,0] = v
    targets = newArray
    classes = targets[:,0] 

    chips_id = np.zeros(len(chips)).astype(int) # N_obj long list of numerical chip_id for each bbox

    print("Create N-img long list of numerical img number ...")
    image_numbers = np.array(stats_df.loc[:,'image_numbers'])

    for i, uchip in enumerate(uchips):
        bool_i = ( chips == uchip )
        chips_id[bool_i] = image_numbers[i]

    print("Create N_img list of image weights, constant in value, normalised for sum=1...")
    n_uchips = len(uchips) 
    image_weights = np.zeros(n_uchips) + 1/n_uchips # Constant image weights 

    print("Create class_mu, class_sigma, class_cov ... ") 

    h = coords[:,3]-coords[:,1].astype('float32')
    w = coords[:,2]-coords[:,0].astype('float32')

    Area = np.log(w*h)
    aspect_ratio = np.log(w/h)
    uc = np.unique(classes)
    n_uc = len(uc)

    class_mu = np.zeros((n_uc,4))
    class_sigma = np.zeros((n_uc,4))
    class_cov = np.zeros((n_uc,4,4))
    for i in range(n_uc):
        j = classes==uc[i]
        wj = np.log(w[j])
        hj = np.log(h[j])
        aj = Area[j]
        arj = aspect_ratio[j]
        data = [wj, hj, aj, arj]
        class_mu[i,:] = np.mean(data,1)
        class_sigma[i,:] = np.std(data,1)
        class_cov[i,:,:] = np.cov(data)

    targets = targets.astype('float32')
    chips_id = chips_id.astype('float32')
    image_numbers = image_numbers.astype('uint16')
    image_weights = image_weights.astype('<f8')

    print("Save matfile at ", matfile_path, "...")
    scipy.io.savemat(matfile_path, 
                                            {'coords': coords, 'chips': chips, 'classes': classes, 'shapes': shapes,
                                                'stats': stats, 'uchips': uchips, 'targets': targets, 
                                                'image_numbers':image_numbers, 'image_weights':image_weights, 
                                                'id':chips_id, 'wh': [w,h], 'class_sigma':class_sigma, 
                                                'class_mu':class_mu, 'class_cov':class_cov, 'class_map':class_map, 'img_paths':img_paths })
    return uchips, image_numbers 

def paths_to_symlinks(paths, sym_path, stats_df):
    """
    For a given list of image folders, look for subdir called 'images' and create a list of all images across those folders.
    Then create a folder with symlinks to those images, re-named numerically

    """

    os.makedirs(sym_path, exist_ok=True)
    shutil.rmtree(sym_path)
    os.makedirs(sym_path, exist_ok=True)

    img_numbers = stats_df.loc[:,'image_numbers']
    img_paths = np.array(stats_df.loc[:,'img_path'])
    newpaths=[]

    for img_path, img_number in zip(img_paths, img_numbers):
        newpath=os.path.join(sym_path,str(img_number)+'.tif')
        print(img_path)
        print(newpath)
        os.symlink(img_path, newpath)
        newpaths.append(newpath)

    stats_df.loc[:,'sympaths'] = newpaths
    print("Symlinks created: ", len(newpaths))
    return stats_df



 
