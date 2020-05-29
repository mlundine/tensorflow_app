# -*- coding: utf-8 -*-
"""

@author: Mark Lundine, mlundine@udel.edu
Any questions, feel free to email me, especially if you come across errors or problems.
I likely ran across the same errors at some point and can help solve them.


"""
import os
import numpy as np
from PIL import Image
import pandas as pd
import glob
from sklearn.preprocessing import minmax_scale
import itertools
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import shapefile
import tensorflow as tf
from osgeo import gdal
import io
import sys
import generate_tfrecord_mask_app as gtfma
import generate_tfrecord_app as gtfa
import train_mod
import export_inference_graph_mod


##write out necessary paths
root_mod = os.path.abspath(os.sep)
models_path_mod = os.path.join(root_mod, 'tensorflow_app', 'models')
research_path_mod = os.path.join(models_path_mod, 'research')
slim_path_mod = os.path.join(research_path_mod, 'slim')
set_py_path_mod = 'set PYTHONPATH=' + models_path_mod + r';' + research_path_mod + r';' + slim_path_mod
object_detection_path_mod = os.path.join(research_path_mod, 'object_detection')

# =============================================================================
# normalizeScale is fed an image, a low value, and a high value to rescale it to
# it will return the new rescaled image
# =============================================================================
def normalizeScale(image, low, high):
    image = image
    shape = np.shape(image)
    newImage = minmax_scale(image.ravel(), feature_range=(low,high)).reshape(shape)
    return newImage
# =============================================================================
# fix_and_reScale_DEMS is fed a file path for a folder holding DEMS as numpy arrays
# and then a file path to save the outputs
# it calls normalizeScale
# =============================================================================
def fix_and_reScale_DEMs(in_location, out_location):
    file = in_location 
    file = np.load(file)
    file = normalizeScale(file, 0, 255)
    new_p = Image.fromarray(file)
    new_p = new_p.convert("L")
    new_p.save(out_location)
    file = None
    new_p = None
# =============================================================================
# numpyToJPEG is fed a file path where the DEMS as numpy arrays are located
# a path to save the results
# it calls fix_and_reScale_DEMs
# =============================================================================   
def numpyToJPEG(inFolder, outFolder):
    for file in glob.glob(inFolder + '/*.npy'):
        name = os.path.splitext(os.path.basename(file))[0]
        outpath = outFolder + '\\' + name + '.jpeg'
        fix_and_reScale_DEMs(file, outpath) 
        

def numpy_rgb_to_jpeg(inFolder, outFolder):
    for file in glob.glob(inFolder + '/*.npy'):
        arr = np.load(file)
        arr = Image.fromarray(arr)
        name = os.path.splitext(os.path.basename(file))[0] + '.jpeg'
        outPath = os.path.join(outFolder, name)
        arr.save(outPath)
# =============================================================================
# translate_bboxes is fed the .csv file with the bounding box coordinates
# a file path to save a new .csv file with georeferenced coordinates for the 
# bounding boxes, a file with the geo-coordinates and resolution of the orignal DEM
# it outputs the .csv file with georeferenced bounding box coordinates
# =============================================================================  
def translate_bboxes(inFile, saveFile, coords_file, constant):
    chunk2 = pd.read_csv(inFile)
    geobox = []
    geobox.append(['file', 'xmin', 'ymin', 'xmax', 'ymax', 'score', 'label'])
    for chunk in pd.read_csv(coords_file, chunksize=1):
        res = chunk.iloc[0,5]
        #constant = r'C:\jpegs\'
        for i in range(len(chunk)):
            filename = os.path.splitext(os.path.basename(chunk.iloc[i,0]))[0]
            querystr = constant + '\\' + filename + '.jpeg'
            filtered = chunk2.query("file == @querystr")
            for j in range(len(filtered)):
                xmin = chunk.iloc[i,1]+res*filtered.iloc[j,3]
                ymin = chunk.iloc[i,2]-res*filtered.iloc[j,5]
                xmax = chunk.iloc[i,3]+res*filtered.iloc[j,4]
                ymax = chunk.iloc[i,4]-res*filtered.iloc[j,6]
                score = filtered.iloc[j,2]
                label = filtered.iloc[j,1]
                geobox.append([filename, xmin, ymin, xmax, ymax, score, label])
    np.savetxt(saveFile, geobox, delimiter=",", fmt='%s')

# =============================================================================
# Working but file does not have a spatial coordinate system, have to project in Arc
# =============================================================================
def pyshp_geobox_to_shapefiles(inFile, outFile):
    # funtion to generate a .prj file
    box = shapefile.Writer(outFile, shapeType=shapefile.POLYGON)
    box.field('score', 'N', decimal=8)
    box.field('label', 'C', size=20)
    for chunk in pd.read_csv(inFile, chunksize=1, engine='python'):
        score = float(chunk.iloc[0,5])
        x_min = float(chunk.iloc[0,1])
        y_min = float(chunk.iloc[0,2])
        x_max = float(chunk.iloc[0,3])
        y_max = float(chunk.iloc[0,4])
        label = chunk.iloc[0,6]
        box.poly([[[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]]])
        box.record(score, label)

def randomize_images(folder, train_path, test_path):
    types = ('/*.jpg', '/*.jpeg')
    image_list = []
    for ext in types:
        for im in glob.glob(folder + ext):
            image_list.append(im)
    image_list = np.array(image_list)
    np.random.shuffle(image_list)
    pct = 0
    for i in range(len(image_list)):
        if pct <= 0.8:
            im = Image.open(image_list[i])
            im.save(train_path + '\\' + os.path.basename(image_list[i]))
            im.close()
        else:
            im = Image.open(image_list[i])
            im.save(test_path + '\\' + os.path.basename(image_list[i]))
            im.close()
        pct = i/len(image_list)
    
def set_python_path():
    cmd2 = set_py_path_mod
    os.system(cmd2)
    
    
def make_protobufs():
    cmd2 = r'cd ' + research_path_mod
    cmd3 = r'protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto'
    os.system(cmd2 + r'&&' + cmd3)

def run_setup_py():
    cmd2 = r'cd ' + research_path_mod
    cmd3 = r'python setup.py build'
    cmd4 = r'python setup.py install'
    os.system(cmd2 + r'&&' + cmd3 + r'&&' + cmd4)
    
def check_setup():
    cmd2 = r'cd C:\tensorflow1\models\research\object_detection'
    cmd3 = r'jupyter notebook object_detection_tutorial.ipynb'
    os.system(cmd2 + r'&&' + cmd3)
    
def make_annotation_csvs(images_folder):
    def xml_to_csv(path):
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
        ## changed 'class' to 'label'
        column_name = ['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df


    def do_both():
        for folder in ['train','test']:
            image_path = os.path.join(images_folder, folder)
            xml_df = xml_to_csv(image_path)
            xml_df.to_csv((os.path.join(images_folder,folder + '_labels.csv')), index=None)
            print('Successfully converted xml to csv.')

    do_both()
    
def make_tf_records(images_folder):
    train_images = os.path.join(images_folder, 'train')
    test_images = os.path.join(images_folder, 'test')
    train_labels = os.path.join(images_folder, 'train_labels.csv')
    test_labels = os.path.join(images_folder, 'test_labels.csv')
    train_tfr = os.path.join(images_folder, 'frcnn_records', 'train.record')
    test_tfr = os.path.join(images_folder, 'frcnn_records', 'test.record')
    
    cmd2 = r'cd ' + object_detection_path_mod
    cmd3 = r'notepad generate_tfrecord_app.py'
    os.system(cmd2 + r'&&' + cmd3)
    gtfa.main(train_images, train_labels, train_tfr)
    gtfa.main(test_images, test_labels, test_tfr)

##this needs to be tested
def make_tf_records_mask(images_folder):
    train_images = os.path.join(images_folder, 'train')
    train_mask_images = os.path.join(images_folder, 'train_mask')
    test_images = os.path.join(images_folder, 'test')
    test_mask_images = os.path.join(images_folder, 'test_mask')
    train_labels = os.path.join(images_folder, 'train_labels.csv')
    test_labels = os.path.join(images_folder, 'test_labels.csv')
    train_tfr = os.path.join(images_folder, 'mrcnn_records', 'train.record')
    test_tfr = os.path.join(images_folder, 'mrcnn_records', 'test.record')

    cmd2 = r'cd ' + object_detection_path_mod
    cmd3 = r'notepad generate_tfrecord_mask_app.py'
    os.system(cmd2 + r'&&' + cmd3)
    gtfma.main(train_images, train_mask_images, train_labels, train_tfr)
    gtfma.main(test_images, test_mask_images, test_labels, test_tfr)


##THIS NEEDS TO BE SAVED TO THE PROJECT DIRECTORY IN THE TRAINING FOLDER BY USER
def make_label_map():
    labelMapPath = os.path.join(object_detection_path_mod, 'training', 'labelmap.pbtxt')
    cmd2 = r'notepad ' + labelMapPath
    os.system(cmd2)

#THIS NEEDS TO BE SAVED TO THE PROJECT DIRECTORY IN TRAINING FOLDER BY USER
def configure_training(model_type):
    if model_type == 'faster':
        configPath = os.path.join(object_detection_path_mod, 'training', 'faster_rcnn_inception_v2_pets.config')
        cmd2 = r'notepad ' + configPath
        os.system(cmd2)
    else:
        configPath = os.path.join(object_detection_path_mod, 'training', 'mask_rcnn_resnet101_atrous_coco.config')
        cmd2 = r'notepad ' + configPath
        os.system(cmd2)        
    
def train(project_path, model_type):
    if model_type == 'faster':
        trainingdir = os.path.join(project_path, 'frcnn_training')
        configpath = os.path.join(project_path, 'frcnn_training', 'faster_rcnn_inception_v2_pets.config')
        train_mod.main(trainingdir, configpath)
    else:
        trainingdir = os.path.join(project_path, 'mrcnn_training')
        configpath = os.path.join(project_path, 'mrcnn_training', 'mask_rcnn_resnet101_atrous_coco.config')
        train_mod.main(trainingdir, configpath)
       
def make_inference_graph(ckpt_num, project_path, model_type):
    if model_type == 'faster':
        trainingdir = os.path.join(project_path, 'frcnn_training', 'model.ckpt-'+str(ckpt_num))
        configpath = os.path.join(project_path, 'frcnn_training', 'faster_rcnn_inception_v2_pets.config')
        infgraphdir = os.path.join(project_path, 'frcnn_inference_graph')
        export_inference_graph_mod.main(configpath, trainingdir, infgraphdir)
    else:
        trainingdir = os.path.join(project_path, 'mrcnn_training', 'model.ckpt-'+str(ckpt_num))
        configpath = os.path.join(project_path, 'mrcnn_training', 'mask_rcnn_resnet101_atrous_coco.config')
        infgraphdir = os.path.join(project_path, 'mrcnn_inference_graph')
        export_inference_graph_mod.main(configpath, trainingdir, infgraphdir)

###add functionality to work on multiple classes
def p_r_curve(in_csv, test_csv, project_name):
    plt.ioff()
    ##Counting all of the test image annotations
    test_df = pd.read_csv(test_csv)
    result_df = pd.read_csv(in_csv)
    file_col = result_df['file'].copy(deep=True)
    for i in range(len(file_col)):
        file_col.iloc[i] = os.path.basename(result_df.file[i])
    result_df['file'] = file_col

    test_counter_df = pd.crosstab(test_df.filename, test_df.label).reset_index()

    labels = list(test_counter_df.columns[1:])
    dfCols = list(test_counter_df.columns[1:])
    testcols = list(s + '_test' for s in labels)
    dfCols.insert(0,'file')
    dfCols.extend(testcols)
    dfCols.append('threshold')
    true_pos = list(s + '_true_pos' for s in labels)
    false_pos = list(s + '_false_pos' for s in labels)
    false_neg = list(s + '_false_neg' for s in labels)
    dfCols.extend(true_pos)
    dfCols.extend(false_pos)
    dfCols.extend(false_neg)

    
    lst = np.arange(0,1,0.01)
    threshold = list(itertools.chain.from_iterable(itertools.repeat(x, len(test_counter_df)) for x in lst))
    
    p_r_df = pd.DataFrame(columns = dfCols, index = range(len(test_counter_df)*100))

    
    file_list = list(test_counter_df['filename'])*100
    p_r_df['file'] = file_list
    p_r_df['threshold'] = threshold
    
    for z in range(len(labels)):
        lab = labels[z]
        testlab = testcols[z]
        p_r_df[testlab] = list(test_counter_df[lab])*100
   
    counts = np.zeros((len(p_r_df)))
    for lab in labels:
        querystr = lab
        classFilter = result_df.query('label == @querystr')
        k=0
        for thresh in np.arange(0,1,0.01):
            threshquery = 'score >=' + str(thresh)
            threshFilter = classFilter.query(threshquery)
            result_count = threshFilter.groupby('file').file.count()
            for j in range(len(test_counter_df['filename'])):
                imagequery = test_counter_df['filename'][j]
                try:
                    rcQuery = result_count[imagequery]
                    count = rcQuery
                except:
                    count = 0
                
                counts[k] = count
                k=k+1
        p_r_df[lab] = counts
    
    for q in range(len(labels)):
        lab = labels[q]
        testlab = testcols[q]
        tp = lab + '_true_pos'
        fp = lab + '_false_pos'
        fn = lab + '_false_neg'
        
        resCounts = np.array(p_r_df[lab])
        testCounts = np.array(p_r_df[testlab])
        difference = resCounts - testCounts
        posIndices = np.nonzero(difference>=0)
        negIndices = np.nonzero(difference<0)
        
        true_positives = np.zeros((len(p_r_df),))
        false_negatives = np.zeros((len(p_r_df),))
        false_positives = np.zeros((len(p_r_df),))

        
        true_positives[posIndices] = testCounts[posIndices]
        true_positives[negIndices] = resCounts[negIndices]
        
        false_negatives[posIndices] = 0
        false_negatives[negIndices] = abs(difference[negIndices])
        false_positives[posIndices] = difference[posIndices]
        false_positives[negIndices] = 0
        p_r_df[tp] = true_positives
        p_r_df[fp] = false_positives
        p_r_df[fn] = false_negatives
        
        thresh_precision = np.zeros(len(np.arange(0,1,0.01)))
        thresh_recall = np.zeros(len(np.arange(0,1,0.01)))
        thresh_true_pos = np.zeros(len(np.arange(0,1,0.01)))
        thresh_false_neg = np.zeros(len(np.arange(0,1,0.01)))
        thresh_false_pos = np.zeros(len(np.arange(0,1,0.01)))
        g = 0
        for thresh in np.arange(0,1,0.01):
            threshquery = 'threshold ==' + str(thresh)
            filtered = p_r_df.query(threshquery)
            thresh_true_pos[g] = sum(filtered[tp])
            thresh_false_pos[g] = sum(filtered[fp])
            thresh_false_neg[g] = sum(filtered[fn])
            thresh_precision[g] = thresh_true_pos[g]/(thresh_true_pos[g]+thresh_false_pos[g])
            thresh_recall[g] = thresh_true_pos[g]/(thresh_true_pos[g] + thresh_false_neg[g])
            g=g+1
        plt.rcParams.update({'font.size':12})
        fig = plt.gcf()
        fig.set_size_inches(10,8)
        plt.plot(thresh_recall, thresh_precision, label=lab)
        plt.legend()
        dataDict = {'threshold':np.arange(0,1,0.01), 'true_pos':thresh_true_pos, 'false_pos':thresh_false_pos,
                    'false_neg':thresh_false_neg, 'precision':thresh_precision, 'recall':thresh_recall}
        prData = pd.DataFrame(dataDict)
        wd = os.getcwd()
        csv_save = os.path.join(wd, project_name, 'implementation', 'results', 'pr_curve', lab+r'.csv')
        prData.to_csv(csv_save,index=False)
    
    wd = os.getcwd()
    csv_save2 = os.path.join(wd, project_name, 'implementation', 'results', 'pr_curve', 'counts_by_image.csv')
    p_r_df.to_csv(csv_save2, index=False)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    pngPath = os.path.join(wd, project_name, 'implementation', 'results', 'pr_curve', 'prCurvePlot.png')
    plt.savefig(pngPath)
    return pngPath    
        
        
        
        
                
                

    
    

            

    
    
    
    
    
    
    
    
    
    
    
    
