# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:29:52 2020

@author: Mark Lundine
"""

###in progress
import tensorflow as tf
import os
import cv2
import numpy as np
import glob
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
import sys
from PIL import Image
import matplotlib.pyplot as plt

global root
root = os.path.abspath(os.sep)
global obj_det
obj_det = os.path.join(root, 'tensorflow_app', 'models', 'research', 'object_detection')



def detection_function(BATCH, PATH_TO_IMAGES, THRESHOLD, NUM_CLASSES, PROJECT, MODEL):
    root_mod = os.path.abspath(os.sep)
    obj_det_mod = os.path.join(root_mod, 'tensorflow_app', 'models', 'research', 'object_detection')
    
    
    
    
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    if MODEL == 'faster':
        PATH_TO_CKPT = os.path.join(PROJECT, 'frcnn_inference_graph', 'frozen_inference_graph.pb')    
    else:
        PATH_TO_CKPT = os.path.join(PROJECT, 'ssd_inference_graph', 'frozen_inference_graph.pb')   
    # Path to label map file
    if MODEL == 'faster':
        PATH_TO_LABELS = os.path.join(PROJECT, 'frcnn_training', 'labelmap.pbtxt')
    else:
        PATH_TO_LABELS = os.path.join(PROJECT, 'ssd_training', 'labelmap.pbtxt')
    #Path to save the images with detection results
    PATH_TO_SAVE_IMAGES = os.path.join(PROJECT, 'implementation', 'results', 'images')

    # Save path, this should end with .csv, cuz we are saving a spreadsheet 
    # with all of the bounding box coordinates
    SAVE_CSV = os.path.join(PROJECT, 'implementation', 'results', 'bounding_boxes', 'result_bbox.csv')    

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)
    
    # Define input and output tensors (i.e. data) for the object detection classifier
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    def detector_batch(path_to_images, path_to_save_csv, path_to_save_images, threshold):
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        final_box = []
        final_box.append(['file','label', 'score', 'x_min', 'x_max', 'y_min', 'y_max'])
        image_list = []
        types = ['/*.jpg', '/*jpeg']
        for ext in types:
            for imDude in glob.glob(path_to_images + ext):
                image_list.append(imDude)
        for file in image_list:
            print(file)
            image = cv2.imread(file)
            image_expanded = np.expand_dims(image, axis=0)
            
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            
            boxes_display = boxes[classes==1]
            scores_display = scores[classes==1]
            classes_display = classes[classes==1]
            ##Drawing the boxes and score on the image
            vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes_display),
            np.squeeze(classes_display).astype(np.int32),
            np.squeeze(scores_display),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=threshold)
        

            ## 'C:\detection_tests\test1.jpeg' becomes 'test1'
            image_name = os.path.splitext(os.path.basename(file))[0]
            # All the results have been drawn on image, save the image
            imSavePath = os.path.join(path_to_save_images, image_name + '.jpeg')
            cv2.imwrite(imSavePath, image)
            

            
            
            # Return bounding box coordinates as a .csv file
            min_score_thresh=0
            #bboxes = boxes[np.logical_and(classes == 1,scores>min_score_thresh)]
            #scores = scores[np.logical_and(classes ==1, scores>min_score_thresh)]
            bboxes = boxes[scores>min_score_thresh]
            scores = scores[scores>min_score_thresh]
            im_width, im_height = image.shape[0:2]
            # Close the image
            image = None
            j=1
            for box in bboxes:
                ymin, xmin, ymax, xmax = box
                final_box.append([file.replace('/','\\'), category_index.get(classes[j-1]).get('name'), scores[j-1], xmin*im_width, xmax *im_width, ymin*im_height, ymax*im_height])
                j=j+1
                
        np.savetxt(path_to_save_csv, final_box, delimiter=",", fmt='%s')
    
    def detector_single(path_to_image, path_to_save_csv, path_to_save_image, threshold):
        final_box = []
        final_box.append(['file','label', 'score', 'x_min', 'x_max', 'y_min', 'y_max'])
        
        image = cv2.imread(path_to_image)
        image_expanded = np.expand_dims(image, axis=0)
        
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
            
        ##Drawing the boxes and score on the image
        vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=threshold)   
            
        
        image_name = os.path.splitext(os.path.basename(path_to_image))[0]
        

        imSavePath = os.path.join(path_to_save_image, image_name + '.jpeg')
        # All the results have been drawn on image, save the image
        cv2.imwrite(imSavePath, image)
            

            
            
        # Return bounding box coordinates as a .csv file
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        min_score_thresh = 0
        bboxes = boxes[scores>min_score_thresh]
        scores = scores[scores>min_score_thresh]
        im_width, im_height = image.shape[0:2]
        # Close the image
        image = None
        j=1
        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            try:
                lab = category_index.get(classes[j-1]).get('name')
            except:
                lab = 'N/A'
            final_box.append([path_to_image, lab, scores[j-1], xmin*im_width, xmax *im_width, ymin*im_height, ymax*im_height])
            j=j+1
            
        np.savetxt(path_to_save_csv, final_box, delimiter=",", fmt='%s')
        
    
    if BATCH == 'batch':
        detector_batch(PATH_TO_IMAGES, SAVE_CSV, PATH_TO_SAVE_IMAGES, THRESHOLD)
        
    else:
        detector_single(PATH_TO_IMAGES, SAVE_CSV, PATH_TO_SAVE_IMAGES, THRESHOLD)
def detection_function_mask(BATCH, PATH_TO_IMAGES, THRESHOLD, NUM_CLASSES, PROJECT):
    root_mod = os.path.abspath(os.sep)
    obj_det_mod = os.path.join(root_mod, 'tensorflow_app', 'models', 'research', 'object_detection')
    
    
    
    
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(PROJECT, 'mrcnn_inference_graph', 'frozen_inference_graph.pb')    
    
    # Path to label map file
    PATH_TO_LABELS = os.path.join(PROJECT, 'mrcnn_training', 'labelmap.pbtxt')
    
    #Path to save the images with detection results
    PATH_TO_SAVE_IMAGES = os.path.join(PROJECT, 'implementation', 'results', 'images')
    PATH_TO_SAVE_MASKS = os.path.join(PROJECT, 'implementation', 'results', 'masks')

    # Save path, this should end with .csv, cuz we are saving a spreadsheet 
    # with all of the bounding box coordinates
    SAVE_CSV = os.path.join(PROJECT, 'implementation', 'results', 'bounding_boxes', 'result_bbox.csv')    

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    def run_mask_inference(image, graph):
        with graph.as_default():
            with tf.Session() as sess:
              # Get handles to input and output tensors
              ops = tf.get_default_graph().get_operations()
              all_tensor_names = {output.name for op in ops for output in op.outputs}
              tensor_dict = {}
              for key in [
                      'num_detections', 'detection_boxes', 'detection_scores',
                      'detection_classes', 'detection_masks'
                  ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                              tensor_name)
              if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
              image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        
              # Run inference
              output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: np.expand_dims(image, 0)})
        
              # all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
    
    def single_mask():
        image = cv2.imread(PATH_TO_IMAGES)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        output_dict = run_mask_inference(image, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
                                                           image,
                                                           output_dict['detection_boxes'],
                                                           output_dict['detection_classes'],
                                                           output_dict['detection_scores'],
                                                           category_index,
                                                           instance_masks=output_dict.get('detection_masks'),
                                                           use_normalized_coordinates=True,
                                                           line_thickness=8,
                                                           min_score_thresh=THRESHOLD)

        image_name = os.path.splitext(os.path.basename(PATH_TO_IMAGES))[0]
        savePath = os.path.join(PATH_TO_SAVE_IMAGES, image_name + '.jpeg')
        cv2.imwrite(savePath, image)

        boxes = np.squeeze(output_dict['detection_boxes'])
        scores = np.squeeze(output_dict['detection_scores'])
        classes = np.squeeze(output_dict['detection_classes'])
        threshold = 0
        bboxes = boxes[scores>threshold]
        scores = scores[scores>threshold]
        im_width, im_height = image.shape[0:2]
        final_box = []
        final_box.append(['file','label', 'score', 'x_min', 'x_max', 'y_min', 'y_max'])
        j=1
        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            final_box.append([PATH_TO_IMAGES, category_index.get(classes[j-1]).get('name'), scores[j-1], xmin*im_width, xmax*im_width, ymin*im_height, ymax*im_height])
            j = j+1
        np.savetxt(SAVE_CSV, final_box, delimiter=",", fmt='%s')

    def batch_mask():
        image_list = []
        types = ['/*.jpg', '/*jpeg']
        for ext in types:
            for imDude in glob.glob(PATH_TO_IMAGES + ext):
                image_list.append(imDude)
        for file in image_list:        
            image = cv2.imread(file)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            output_dict = run_mask_inference(image, detection_graph)
            vis_util.visualize_boxes_and_labels_on_image_array(
                                                               image,
                                                               output_dict['detection_boxes'],
                                                               output_dict['detection_classes'],
                                                               output_dict['detection_scores'],
                                                               category_index,
                                                               instance_masks=output_dict.get('detection_masks'),
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8,
                                                               min_score_thresh=THRESHOLD)

            image_name = os.path.splitext(os.path.basename(file))[0]
            savePath = os.path.join(PATH_TO_SAVE_IMAGES, image_name + '.jpeg')
            cv2.imwrite(savePath, image)

            boxes = np.squeeze(output_dict['detection_boxes'])
            scores = np.squeeze(output_dict['detection_scores'])
            classes = np.squeeze(output_dict['detection_classes'])
            threshold = 0
            bboxes = boxes[scores>threshold]
            scores = scores[scores>threshold]
            im_width, im_height = image.shape[0:2]
            final_box = []
            final_box.append(['file','label', 'score', 'x_min', 'x_max', 'y_min', 'y_max'])
            j=1
            for box in bboxes:
                ymin, xmin, ymax, xmax = box
                final_box.append([file, category_index.get(classes[j-1]).get('name'), scores[j-1], xmin*im_width, xmax*im_width, ymin*im_height, ymax*im_height])
                j = j+1
        np.savetxt(SAVE_CSV, final_box, delimiter=",", fmt='%s')        
        
    
    if BATCH == 'batch':
        batch_mask()
        
    else:
        single_mask()    
    
def main(project, batch, path_to_ims, threshold, numberOfClasses, mask):
    threshold = float(threshold)
    numberOfClasses = int(numberOfClasses)
    if mask == 'faster':
        a=detection_function(batch, path_to_ims, threshold, numberOfClasses, project, 'faster')
    elif mask == 'ssd':
        a = detection_function(batch, path_to_ims, threshold, numberOfClasses, project, 'ssd')
    else:
        a = detection_function_mask(batch, path_to_ims, threshold, numberOfClasses, project)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
