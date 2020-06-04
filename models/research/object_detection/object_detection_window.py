# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
#import sys
from mss import mss
from PIL import Image
import pyautogui
import win32gui
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
global root
root = os.path.abspath(os.sep)
global obj_det
obj_det = os.path.join(root, 'tensorflow_app', 'models', 'research', 'object_detection')


def screenshot(window_title=None):
    if window_title:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            x, y, x1, y1 = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
            im = pyautogui.screenshot(region=(x, y, x1, y1))
            return np.array(im)
        else:
            print('Window not found!')
    
def detection_function_box(THRESHOLD, NUM_CLASSES, PROJECT, MODEL, window_title):
    root_mod = os.path.abspath(os.sep)
    obj_det_mod = os.path.join(root_mod, 'tensorflow_app', 'models', 'research', 'object_detection')

    if MODEL == 'faster':
        PATH_TO_CKPT = os.path.join(PROJECT, 'frcnn_inference_graph', 'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(PROJECT, 'frcnn_training', 'labelmap.pbtxt')
    else:
        PATH_TO_CKPT = os.path.join(PROJECT, 'ssd_inference_graph', 'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(PROJECT, 'ssd_training', 'labelmap.pbtxt')
    ## Load the label map.
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

    # Initialize feed
    #bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}
    #sct = mss()

    while(True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = screenshot(window_title)          
        #frame = np.flip(frame[:, :, :3], 2) 
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=THRESHOLD)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()
    
def detection_function_mrcnn(THRESHOLD, NUM_CLASSES, PROJECT, window_title):
    root_mod = os.path.abspath(os.sep)
    obj_det_mod = os.path.join(root_mod, 'tensorflow_app', 'models', 'research', 'object_detection')

    PATH_TO_CKPT = os.path.join(PROJECT, 'mrcnn_inference_graph', 'frozen_inference_graph.pb')

    PATH_TO_LABELS = os.path.join(PROJECT, 'mrcnn_training', 'labelmap.pbtxt')

    ## Load the label map.
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

    # Initialize feed
    #bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}
    #sct = mss()

    while(True):

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = screenshot(window_title)         
##        frame = np.flip(frame[:, :, :3], 2) 
##        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        # Perform the actual detection by running the model with the image as input
        output_dict = run_mask_inference(frame, detection_graph)
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=THRESHOLD)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()



def main(MODEL, THRESHOLD, NUM_CLASSES, PROJECT, window_title):
    if MODEL == 'faster':
        detection_function_box(THRESHOLD, NUM_CLASSES, PROJECT, 'faster', window_title)
    elif MODEL == 'ssd':
        detection_function_box(THRESHOLD, NUM_CLASSES, PROJECT, 'ssd', window_title)
    else:
        detection_function_mrcnn(THRESHOLD, NUM_CLASSES, PROJECT, window_title)

