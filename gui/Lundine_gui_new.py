# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:53:26 2020

@author: Mark Lundine
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import sys
import warnings
warnings.filterwarnings('ignore')

global root
root = os.path.abspath(os.sep)
global pythonpath1
pythonpath1 = os.path.join(root, 'tensorflow_app', 'models')
sys.path.append(pythonpath1)
global pythonpath2
pythonpath2 = os.path.join(pythonpath1, 'research')
sys.path.append(pythonpath2)
global pythonpath3
pythonpath3 = os.path.join(pythonpath2, 'slim')
sys.path.append(pythonpath3)
global set_py_path
set_py_path = 'set PYTHONPATH=' + pythonpath1 + r';' + pythonpath2 + r';' + pythonpath3
global object_detection_folder
object_detection_folder = os.path.join(pythonpath2, 'object_detection')
sys.path.append(object_detection_folder)
global yolov5folder
yolov5folder = os.path.join(root, 'tensorflow_app', 'yolov5')
sys.path.append(yolov5folder)
global yolomodels
yolomodels = os.path.join(yolov5folder, 'models')
sys.path.append(yolomodels)
global yoloutils
yoloutils = os.path.join(yolov5folder, 'utils')
sys.path.append(yoloutils)
global yolodata
yolodata = os.path.join(yolov5folder, 'data')
sys.path.append(yolodata)
import detection_functions_app as dtf
import gdal_functions_app as gdfa
import object_detection_tf
import object_detection_real_time
import object_detection_screen
import object_detection_window
from PIL.ImageQt import ImageQt
import matplotlib.pyplot as plt
global yolotrain
yolotrain = os.path.join(yolov5folder, r'lundine_yolo_train.py')
global yolodetect 
yolodetect = os.path.join(yolov5folder, r'Lundine_yolo_detect.py')




## Contains all of the widgets for the GUI   
class Window(QMainWindow):
    
    ## All of the button actions are functions
    ## Initializing the window
    def __init__(self):
        super(Window, self).__init__()
        
        sizeObject = QDesktopWidget().screenGeometry(-1)
        global screenWidth
        screenWidth = sizeObject.width()
        global screenHeight
        screenHeight = sizeObject.height()
        global bw1
        bw1 = int(screenWidth/15)
        global bw2
        bw2 = int(screenWidth/50)
        global bh1
        bh1 = int(screenHeight/15)
        global bh2
        bh2 = int(screenHeight/20)

        

        self.setWindowTitle("TensorFlow Object Detection GUI")
        self.setGeometry(50, 50, 10 + int(screenWidth/2), 10 + int(screenHeight/2))

        self.home()




    
    ## Clicking the exit button hides all of the buttons above it
    def exit_buttons(self, buttons):
        for button in buttons:
            button.hide()
    
    ## Converts annotation xmls to csvs and then csvs to tf records
    def convertAnnotations_button(self, button, modelButton):
        print(annotation_images_dir)
        dtf.make_annotation_csvs(annotation_images_dir)
        if str(modelButton.currentText()) == 'Faster R-CNN':
            dtf.make_tf_records(annotation_images_dir, 'faster')
        elif str(modelButton.currentText()) == 'Mask R-CNN':
            dtf.make_tf_records_mask(annotation_images_dir)
        elif str(modelButton.currentText()) == 'Yolov5':
            dtf.make_yolo_records(annotation_images_dir, yolo_labels)
        else:
            dtf.make_tf_records(annotation_images_dir, 'ssd')
        button.setEnabled(False)

    
    ## Opens up notepad so you can edit the label map
    def makeLabelMap_button(self, button):
        dtf.make_label_map()
        button.setEnabled(False)
    
    ## Opens ups notepad so you can edit the config file
    def configTraining_button(self,button, modelButton):
        if str(modelButton.currentText()) == 'Faster R-CNN':
            dtf.configure_training('faster')
        elif str(modelButton.currentText()) == 'Mask R-CNN':
            dtf.configure_training('mask')
        else:
            dtf.configure_training('ssd')
        button.setEnabled(False)
    
    ## Starts the training, must be terminated with ctrl+c in the anaconda prompt
    def startTraining_button(self, modelButton):
        if str(modelButton.currentText()) == 'Faster R-CNN':
            dtf.train(project_dir, 'faster')
        elif str(modelButton.currentText()) == 'Mask R-CNN':
            dtf.train(project_dir, 'mask')
        elif str(modelButton.currentText()) == 'Yolov5':
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(self,"Select Weights", "","All Files (*);;Weights (*.pt)", options=options)
            if fileName:
                weights = fileName
                options2 = QFileDialog.Options()
                options2 |= QFileDialog.DontUseNativeDialog
                fileName2, _ = QFileDialog.getOpenFileName(self,"Select Training Yaml", "","All Files (*);;Yaml (*.yaml)", options=options2)
                if fileName2:
                    data = fileName2
                    batch = 5
                    #TODO epochs slider
                    epochs = 300
                    cmd0 = r'activate yolov5 & '
                    cmd1 = r'python ' + yolotrain + ' --weights '
                    cmd2 = weights + ' --data ' + data + ' --epochs ' + str(epochs) + ' --batch-size ' + str(batch) + ' --project ' + yolo_dir
                    os.system(cmd0 + cmd1 + cmd2)


            #main(weights, data, epochs, batch_size, project)
            
##            cd = 'cd C:/tensorflow_app/yolov5 & '
##            cmd0 = 'activate yolov5 &'
##            cmd1 = 'python C:/tensorflow_app/yolov5/train.py --img 640 --batch 5 --epochs 20 --data C:/tensorflow_app/gui/euchre/euchre.yaml'
##            cmd2 = ' --weights yolov5s.pt'
##            os.system(cd+cmd0+cmd1+cmd2)
##
##            ##python C:/tensorflow_app/gui/yolov5/train.py --img 640 --batch 5 --epochs 3000 --data ____.yaml --weights yolov5s.pt
        else:
            dtf.train(project_dir, 'ssd')
        
    
    ## Exports the inference graph, needs the highest checkpoint number
    def exportGraph_button(self, ckpt, modelButton):
        if str(modelButton.currentText()) == 'Faster R-CNN':
            dtf.make_inference_graph(ckpt, project_dir, 'faster')
        elif str(modelButton.currentText()) == 'Mask R-CNN':
            dtf.make_inference_graph(ckpt, project_dir, 'mask')
        else:
            dtf.make_inference_graph(ckpt, project_dir, 'ssd')
    ## downloads all of the necessary libraries and makes tensorflow1
    ## and gdal1 anaconda envs, once clicked, setup is disabled and check setup is enabled
    def setup_button(self, button_item, goButton):
        goButton.hide()
        if button_item == 'New Project':
            name_widget = QLineEdit(self)
            name_widget.move(bw1,int(1.5*bw1))
            name_widget.resize(bw1,bw1/2)
            name_widget.show()
            ok_button = QPushButton('OK', self)
            ok_button.move(int(2*bw1),int(1.5*bw1))
            ok_button.resize(int(bw1/2),int(bw1/2))
            ok_button.show()
            ok_button.clicked.connect(lambda: self.setup_button2(name_widget.text(), ok_button, name_widget))
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            folderName = str(QFileDialog.getExistingDirectory(self, "Select Project"))
            if folderName:
                projectName = os.path.basename(folderName)
                self.setup_button2(projectName)

    
    
    
    
    def setup_button2(self, project_name, okBut=None, name_line=None):
        topName = QLabel('Project: ' + project_name, self)
        topName.resize(int(2*bw1),int(bw1/4))
        topName.move(int(bw1*4),0)
        topName.show()
        
        if name_line != None:
            name_line.hide()
        if okBut != None:
            okBut.hide()
            okBut.setEnabled(False)
            
        #writing out the dir strings
        wd = os.getcwd()
        #one project
        global project_dir
        project_dir = os.path.join(wd, project_name)
        #one folder for annotation images
        global annotation_images_dir
        annotation_images_dir = os.path.join(project_dir, 'images')
        #one folder for training images
        global train_dir 
        train_dir = os.path.join(project_dir, 'images', 'train')
        global train_mask_dir
        train_mask_dir = os.path.join(project_dir, 'images', 'train_mask')
        #one folder for testing images
        global test_dir 
        test_dir = os.path.join(project_dir, 'images', 'test')
        global test_mask_dir
        test_mask_dir = os.path.join(project_dir, 'images', 'test_mask')
        #two folders for tfrecords, mask and faster
        global frcnn_records
        global mrcnn_records
        global ssd_records
        frcnn_records = os.path.join(project_dir, 'images', 'frcnn_records')
        mrcnn_records = os.path.join(project_dir, 'images', 'mrcnn_records')
        ssd_records = os.path.join(project_dir, 'images', 'ssd_records')
        #one folder for implementation
        global implementation_dir 
        implementation_dir = os.path.join(project_dir, 'implementation')
        #one folder for converting tiffs to jpegs
        global jpeg_dir 
        jpeg_dir = os.path.join(implementation_dir, 'jpegs')
        #one folder for converting tiffs to numpy arrays
        global npy_dir
        npy_dir = os.path.join(implementation_dir, 'numpy_arrays')
        #one folder to save results (bboxes, images, geobboxes, shapefile)
        global result_dir 
        result_dir = os.path.join(implementation_dir, 'results')
        #one folder for result images
        global result_images_dir 
        result_images_dir = os.path.join(result_dir, 'images')
        #one folder for bboxes
        global bbox_dir 
        bbox_dir = os.path.join(result_dir, 'bounding_boxes')
        #one folder for geobboxes
        global geobbox_dir 
        geobbox_dir = os.path.join(result_dir, 'geo_bounding_boxes')
        #one folder for masks
        global mask_results
        mask_results = os.path.join(result_dir, 'masks')
        #one folder for pr curve
        global pr_dir 
        pr_dir = os.path.join(result_dir, 'pr_curve')
        #two folders for inference graphs, mask and faster
        global frcnn_inference_dir
        global mrcnn_inference_dir
        global ssd_inference_dir
        frcnn_inference_dir = os.path.join(project_dir, 'frcnn_inference_graph')
        mrcnn_inference_dir = os.path.join(project_dir, 'mrcnn_inference_graph')
        ssd_inference_dir = os.path.join(project_dir, 'ssd_inference_graph')
        #two folders for training, mask and faster
        global frcnn_training_dir
        global mrcnn_training_dir
        global ssd_training_dir
        global yolo_dir
        global yolo_labels
        frcnn_training_dir = os.path.join(project_dir, 'frcnn_training')
        mrcnn_training_dir = os.path.join(project_dir, 'mrcnn_training')
        ssd_training_dir = os.path.join(project_dir, 'ssd_training')
        yolo_dir = os.path.join(project_dir, 'yolodata')
        yolo_labels = os.path.join(project_dir, 'labels')
        
        
        try:
            #making them if they don't exist
            os.makedirs(annotation_images_dir)
            os.makedirs(train_dir)
            os.makedirs(train_mask_dir)
            os.makedirs(test_dir)
            os.makedirs(test_mask_dir)
            os.makedirs(implementation_dir)
            os.makedirs(jpeg_dir)
            os.makedirs(npy_dir)
            os.makedirs(result_dir)
            os.makedirs(bbox_dir)
            os.makedirs(geobbox_dir)
            os.makedirs(pr_dir)
            os.makedirs(frcnn_inference_dir)
            os.makedirs(mrcnn_inference_dir)
            os.makedirs(ssd_inference_dir)
            os.makedirs(frcnn_training_dir)
            os.makedirs(mrcnn_training_dir)
            os.makedirs(ssd_training_dir)
            os.makedirs(mrcnn_records)
            os.makedirs(frcnn_records)
            os.makedirs(ssd_records)
            os.makedirs(mask_results)
            os.makedirs(result_images_dir)
            os.makedirs(yolo_dir)
            os.makedirs(yolo_labels)
            
        except:
            pass


    ##TO DO  
    ## runs detection on a single image, displays the result in gui window
    ## gets threshold and number of classes from the gui sliders
    def singleImage_button(self, thresh, classes, button, modelButton):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Image", "","All Files (*);;Images (*.jpeg *.jpg)", options=options)
        if fileName:
            if str(modelButton.currentText()) == 'Faster R-CNN':
                object_detection_tf.main(project_dir, 'single', fileName, float(thresh), int(classes), 'faster')
                detect_im = os.path.join(result_images_dir, os.path.splitext(os.path.basename(fileName))[0])            
                label = QLabel(self)
                pixmap = QPixmap(detect_im)
                scaleFac = 1
                logical1 = int(bw1*7) + (pixmap.width()/scaleFac) >= screenWidth-int(bw1*3)
                logical2 = int(bw1/2) + (pixmap.height()/scaleFac) >= screenHeight-int(bw1/2)
                while (logical1 or logical2):
                    scaleFac = scaleFac + 1
                    logical1 = int(bw1*7) + (pixmap.width()/scaleFac) >= screenWidth-int(bw1*3)
                    logical2 = int(bw1/2) + (pixmap.height()/scaleFac) >= screenHeight-int(bw1/2)
                small_pixmap = pixmap.scaled(int(pixmap.width()/scaleFac), int(pixmap.height()/scaleFac))
                label.setPixmap(small_pixmap)
                label.move(int(bw1*7),int(bw1/2))
                label.resize(int(pixmap.width()/scaleFac),int(pixmap.height()/scaleFac))
                label.show()
                buttons = [label, button]
                button.clicked.connect(lambda: self.exit_buttons(buttons))
            elif str(modelButton.currentText()) == 'SSD Mobilenet':
                object_detection_tf.main(project_dir, 'single', fileName, float(thresh), int(classes), 'ssd')
                detect_im = os.path.join(result_images_dir, os.path.splitext(os.path.basename(fileName))[0])            
                label = QLabel(self)
                pixmap = QPixmap(detect_im)
                scaleFac = 1
                logical1 = int(bw1*7) + (pixmap.width()/scaleFac) >= screenWidth-int(bw1*3)
                logical2 = int(bw1/2) + (pixmap.height()/scaleFac) >= screenHeight-int(bw1/2)
                while (logical1 or logical2):
                    scaleFac = scaleFac + 1
                    logical1 = int(bw1*7) + (pixmap.width()/scaleFac) >= screenWidth-int(bw1*3)
                    logical2 = int(bw1/2) + (pixmap.height()/scaleFac) >= screenHeight-int(bw1/2)
                small_pixmap = pixmap.scaled(int(pixmap.width()/scaleFac), int(pixmap.height()/scaleFac))
                label.setPixmap(small_pixmap)
                label.move(int(bw1*7),int(bw1/2))
                label.resize(int(pixmap.width()/scaleFac),int(pixmap.height()/scaleFac))
                label.show()
                buttons = [label, button]
                button.clicked.connect(lambda: self.exit_buttons(buttons))
            elif str(modelButton.currentText()) == 'Yolov5':
                options = QFileDialog.Options()
                options |= QFileDialog.DontUseNativeDialog
                weights, _ = QFileDialog.getOpenFileName(self,"Select Weights", "","All Files (*);;Weights (*.pt)", options=options)
                if weights:
                    #weights, source, conf, project
                    cmd0 = 'activate yolov5 & '
                    cmd1 = 'python ' + yolodetect + ' --weights '
                    cmd2 = weights + ' --source ' + fileName + ' --conf ' + str(thresh) + ' --project ' + yolo_dir + ' --view-img --save-txt --save-conf'
                    os.system(cmd0 + cmd1 + cmd2)
                    buttons = [button]
                    button.clicked.connect(lambda: self.exit_buttons(buttons))

            else:
                object_detection_tf.main(project_dir, 'single', fileName, float(thresh), int(classes), 'mask')
                detect_im = os.path.join(result_images_dir, os.path.splitext(os.path.basename(fileName))[0])            
                label = QLabel(self)
                pixmap = QPixmap(detect_im)
                scaleFac = 1
                logical1 = int(bw1*7) + (pixmap.width()/scaleFac) >= screenWidth-int(bw1*3)
                logical2 = int(bw1/2) + (pixmap.height()/scaleFac) >= screenHeight-int(bw1/2)
                while (logical1 or logical2):
                    scaleFac = scaleFac + 1
                    logical1 = int(bw1*7) + (pixmap.width()/scaleFac) >= screenWidth-int(bw1*3)
                    logical2 = int(bw1/2) + (pixmap.height()/scaleFac) >= screenHeight-int(bw1/2)
                small_pixmap = pixmap.scaled(int(pixmap.width()/scaleFac), int(pixmap.height()/scaleFac))
                label.setPixmap(small_pixmap)
                label.move(int(bw1*7),int(bw1/2))
                label.resize(int(pixmap.width()/scaleFac),int(pixmap.height()/scaleFac))
                label.show()
                buttons = [label, button]
                button.clicked.connect(lambda: self.exit_buttons(buttons))
    #TO DO
    ## runs detection on a batch of images
    ## gets the threshold and number of classes from the gui sliders
    def batch_button(self, thresh, classes, button, modelButton):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folderName:
            if str(modelButton.currentText()) == 'Faster R-CNN':
                object_detection_tf.main(project_dir, 'batch', folderName, float(thresh), int(classes), 'faster')
                buttons = [button]
                button.clicked.connect(lambda: self.exit_buttons(buttons))
            elif str(modelButton.currentText()) == 'SSD Mobilenet':
                object_detection_tf.main(project_dir, 'batch', folderName, float(thresh), int(classes), 'ssd')
                buttons = [button]
                button.clicked.connect(lambda: self.exit_buttons(buttons))
            elif str(modelButton.currentText()) == 'Yolov5':
                options = QFileDialog.Options()
                options |= QFileDialog.DontUseNativeDialog
                weights, _ = QFileDialog.getOpenFileName(self,"Select Weights", "","All Files (*);;Weights (*.pt)", options=options)
                if weights:
                    cmd0 = 'activate yolov5 & '
                    cmd1 = 'python ' + yolodetect + ' --weights '
                    cmd2 = weights + ' --source ' + folderName + ' --conf ' + str(thresh) + ' --project ' + yolo_dir + ' --save-txt --save-conf'
                    os.system(cmd0 + cmd1 + cmd2)
            else:
                object_detection_tf.main(project_dir, 'batch', folderName, float(thresh), int(classes), 'mask')
                buttons = [button]
                button.clicked.connect(lambda: self.exit_buttons(buttons))
    def videoCam_button(self, thresh, classes, button, modelButton):
        if str(modelButton.currentText()) == 'Faster R-CNN':
            object_detection_real_time.main('faster', thresh, classes, project_dir)
            buttons = [button]
            button.clicked.connect(lambda: self.exit_buttons(buttons))
        elif str(modelButton.currentText()) == 'SSD Mobilenet':
            object_detection_real_time.main('ssd', thresh, classes, project_dir)
            buttons = [button]
            button.clicked.connect(lambda: self.exit_buttons(buttons))
        elif str(modelButton.currentText()) == 'Yolov5':
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            weights, _ = QFileDialog.getOpenFileName(self,"Select Weights", "","All Files (*);;Weights (*.pt)", options=options)
            if weights:
                cmd0 = 'activate yolov5 & '
                cmd1 = 'python ' + yolodetect + ' --weights '
                cmd2 = weights + ' --source ' + '0' + ' --conf ' + str(thresh) + ' --project ' + yolo_dir
                os.system(cmd0 + cmd1 + cmd2)
        else:
            object_detection_real_time.main('mask', thresh, classes, project_dir)
    def screenCap_button(self, thresh, classes, top, left, width, height, button, modelButton):
        if str(modelButton.currentText()) == 'Faster R-CNN':
            object_detection_screen.main('faster', thresh, classes, project_dir, top, left, width, height)
        elif str(modelButton.currentText()) == 'SSD Mobilenet':
            object_detection_screen.main('ssd', thresh, classes, project_dir, top, left, width, height)
        elif str(modelButton.currentText()) == 'Yolov5':
                ##insert
            #TODO SCREENCAP YOLO
            print('yolo')
        else:
            object_detection_screen.main('mask', thresh, classes, project_dir, top, left, width, height)


    def windowGrabber_button(self, thresh, classes, windowName, button, modelButton):
        if str(modelButton.currentText()) == 'Faster R-CNN':
            object_detection_window.main('faster', thresh, classes, project_dir, windowName)
        elif str(modelButton.currentText()) == 'SSD Mobilenet':
            object_detection_window.main('ssd', thresh, classes, project_dir, windowName)
        elif str(modelButton.currentText()) == 'Yolov5':
            ###insert
            #TODO WINDOW GRABBER YOLO
            print('yolo')
        else:
            object_detection_window.main('mask', thresh, classes, project_dir, windowName)
    ## Converts geotiffs to jpegs
    def convertImages_button(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        npyFolder = npy_dir
        jpgFolder = jpeg_dir
        if folderName:
            gdfa.gdal_tifToNumpy(folderName, npyFolder)
            dtf.numpyToJPEG(npyFolder, jpgFolder)
            
    def convertImages2_button(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        npyFolder = npy_dir
        jpgFolder = jpeg_dir
        if folderName:
            gdfa.gdal_rgb_tif_to_npy(folderName, npyFolder)
            dtf.numpy_rgb_to_jpeg(npyFolder, jpgFolder)
    
    ## gets the coordinates of the four corners of a folder of geotiffs and saves to csv
    def getCoords_button(self):
        outSheet = os.path.join(geobbox_dir, 'raster_coords.csv')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = str(QFileDialog.getExistingDirectory(self, "Select Raster Image Directory"))
        if folderName:
            gdfa.gdal_get_coords_and_res(folderName, outSheet)
    
    ## Converts bounding box detection coordinates to geographic coordinates
    ## using the csv output from getCoords_button()
    #TODO CONVERT COORDS YOLO
    def convertCoords_button(self):
        rasterCoords = os.path.join(geobbox_dir, 'raster_coords.csv')
        saveSheet = os.path.join(geobbox_dir, 'result_geobbox.csv')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select bounding box csv", "","All Files (*);;CSVs (*.csv)", options=options)
        if fileName:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            folderName = str(QFileDialog.getExistingDirectory(self, "Select JPEG File Directory"))
            if folderName:
                folderName = folderName.replace('/', '\\')
                print(folderName)
                dtf.translate_bboxes(fileName, saveSheet, rasterCoords, folderName)
    
    ## Makes a shapefile using the georeferenced bounding box coordinates
    ## does not define the projection
    def makeShape_button(self):
        shapeFolder = os.path.join(result_dir, 'gis', 'geobox_shape')
        try:
            os.makedirs(shapeFolder)
        except:
            pass
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select geo bounding box csv", "","All Files (*);;CSVs (*.csv)", options=options)
        if fileName:
            dtf.pyshp_geobox_to_shapefiles(fileName, shapeFolder)
    
    ## Outputs a precision recall curve as well as a csv with all of the data
    ## used to construct the curve
    def getPR_button(self, button):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select bounding box csv (not georeferenced)", "","All Files (*);;CSVs (*.csv)", options=options)
        if fileName:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName2, _ = QFileDialog.getOpenFileName(self,"Select test label csv", "","All Files (*);;CSVs (*.csv)", options=options)
            if fileName2:
                image = dtf.p_r_curve(fileName, fileName2, project_dir)
                label = QLabel(self)
                pixmap = QPixmap(image)
                label.setPixmap(pixmap)
                label.move(800,100)
                label.resize(int(pixmap.width()),int(pixmap.height()))
                label.show()
                buttons = [label, button]
                button.clicked.connect(lambda: self.exit_buttons(buttons))
    
    ## Opens up labelimg for annotating images
    def launch_labelimg_button(self):
        wd = os.getcwd()
        labelimgdir = os.path.join(wd, 'labelimg')
        cmd1 = r'cd ' + labelimgdir
        cmd2 = r'labelimg.exe'
        os.system(cmd1 + r'&&' + cmd2)
    
    ## Randomizes the images that go into training and testing
    def setup_data_button(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderName = str(QFileDialog.getExistingDirectory(self, "Select jpeg image directory"))
        if folderName:
            dtf.randomize_images(folderName, train_dir, test_dir)
    
    ## Main window buttons
    ## All the functions below are butttons that contain more buttons
    def annotating_button(self, modelButton):
        convertImages = QPushButton('Covert Geotiffs to JPEGS (single band)', self)
        convertImages.resize(int(bw1*4),bw1)
        convertImages.move(int(3*bw1),bw1)
        convertImages.show()
        
        convertImages2 = QPushButton('Convert Geotiffs to JPEGS (RGB)', self)
        convertImages2.resize(int(bw1*4),bw1)
        convertImages2.move(int(3*bw1),2*bw1)
        convertImages2.show()
        
        setup_data = QPushButton('Set up training and test data', self)
        setup_data.resize(int(bw1*4),bw1)
        setup_data.move(int(3*bw1),3*bw1)
        setup_data.show()
        
        launch_labelimg = QPushButton('Launch Labelimg', self)
        launch_labelimg.resize(int(bw1*4),bw1)
        launch_labelimg.move(int(3*bw1),4*bw1)
        launch_labelimg.show()
        
        exitFunc = QPushButton('Exit', self)
        exitFunc.resize(int(bw1*4),bw1)
        exitFunc.move(int(3*bw1),5*bw1)
        exitFunc.show()
        
        buttons = [convertImages, convertImages2, setup_data, launch_labelimg, exitFunc]
        
        ##Actions
        convertImages.clicked.connect(self.convertImages_button)
        convertImages2.clicked.connect(self.convertImages2_button)
        setup_data.clicked.connect(self.setup_data_button)
        launch_labelimg.clicked.connect(self.launch_labelimg_button)
        exitFunc.clicked.connect(lambda: self.exit_buttons(buttons))
        
    def training_button(self, modelButton):
        convertAnnotations = QPushButton('1. Convert Annotations to TfRecords or Yolo Records', self)
        convertAnnotations.resize(int(bw1*4),bw1)
        convertAnnotations.move(int(3*bw1),bw1)
        convertAnnotations.show()
        
        makeLabelMap = QPushButton('2. Make Label Map (skip for yolo)', self)
        makeLabelMap.resize(int(bw1*4),bw1)
        makeLabelMap.move(int(3*bw1),2*bw1)
        makeLabelMap.show()
        
        configTraining = QPushButton('3. Configure Training (skip for yolo)', self)
        configTraining.resize(int(bw1*4),bw1)
        configTraining.move(int(3*bw1),3*bw1)
        configTraining.show()
        
        startTraining = QPushButton('4. Start Training', self)
        startTraining.resize(int(bw1*4),bw1)
        startTraining.move(int(3*bw1),bw1*4)
        startTraining.show()
        
        exportGraph = QPushButton('5. Export Inference Graph (skip for yolo)', self)
        exportGraph.resize(int(bw1*4),bw1)
        exportGraph.move(int(3*bw1),5*bw1)
        exportGraph.show()
        
        ckptSlider = QSpinBox(self)
        ckptSlider.setMinimum(1)
        ckptSlider.setMaximum(100000)
        ckptSlider.setValue(40000)
        ckptSlider.move(int(7*bw1),int(5.35*bw1))
        ckptSlider.show()
        
        exitFunc = QPushButton('Exit', self)
        exitFunc.resize(int(bw1*4),bw1)
        exitFunc.move(int(3*bw1),6*bw1)
        exitFunc.show()
        
        buttons = [convertAnnotations, makeLabelMap, configTraining, startTraining, exportGraph, ckptSlider, exitFunc]
        

        convertAnnotations.clicked.connect(lambda: self.convertAnnotations_button(convertAnnotations, modelButton))
        makeLabelMap.clicked.connect(lambda: self.makeLabelMap_button(makeLabelMap))
        configTraining.clicked.connect(lambda: self.configTraining_button(configTraining, modelButton))
        startTraining.clicked.connect(lambda: self.startTraining_button(modelButton))
        exportGraph.clicked.connect(lambda: self.exportGraph_button(ckptSlider.value(), modelButton))
        exitFunc.clicked.connect(lambda: self.exit_buttons(buttons))
        
    def implementation_button(self, modelButton):
        single_image = QPushButton('Single Image', self)
        single_image.resize(bw1, bw1)
        single_image.move(int(bw1*1.5),0)
        single_image.show()
        
        batch = QPushButton('Batch of Images', self)
        batch.resize(bw1, bw1)
        batch.move(int(bw1*1.5),bw1)
        batch.show()

        videoCam = QPushButton('Video Camera', self)
        videoCam.resize(bw1, bw1)
        videoCam.move(int(bw1*1.5),2*bw1)
        videoCam.show()

        screenCap = QPushButton('Screen Capture', self)
        screenCap.resize(bw1, bw1)
        screenCap.move(int(bw1*1.5),3*bw1)
        screenCap.show()
        
        threshLab = QLabel('Threshold', self)
        threshLab.move(int(bw1*1.75),int(4*bw1))
        threshLab.show()
        threshSlider = QDoubleSpinBox(self)
        threshSlider.move(int(bw1*1.5),int(bw1*4.25))
        threshSlider.setMinimum(0.00)
        threshSlider.setMaximum(0.99)
        threshSlider.setValue(0.60)
        threshSlider.show()
        
        numClassesLab = QLabel('Number of Classes', self)
        numClassesLab.resize(bw1,int(bw1/4))
        numClassesLab.move(int(1.6*bw1),int(4.5*bw1))
        numClassesLab.show()
        numClasses = QSpinBox(self)
        numClasses.move(int(bw1*1.5),int(bw1*4.75))
        numClasses.setMinimum(1)
        numClasses.show()

        topIntLab = QLabel('Top Coord.', self)
        topIntLab.move(int(2.75*bw1), int(3*bw1))
        topIntLab.show()
        
        topInt = QSpinBox(self)
        topInt.move(int(2.5*bw1), int(3.25*bw1))
        topInt.setMinimum(0)
        topInt.setMaximum(100)
        topInt.setValue(0)
        topInt.show()
        
        leftIntLab = QLabel('Left Coord.', self)
        leftIntLab.move(int(3.75*bw1), int(3*bw1))
        leftIntLab.show()
        
        leftInt = QSpinBox(self)
        leftInt.move(int(3.5*bw1),int(3.25*bw1))
        leftInt.setMinimum(0)
        leftInt.setMaximum(100)
        leftInt.setValue(0)
        leftInt.show()
        
        widthIntLab = QLabel('Width', self)
        widthIntLab.move(int(4.75*bw1), int(3*bw1))
        widthIntLab.show()
        
        widthInt = QSpinBox(self)
        widthInt.move(int(4.5*bw1), int(3.25*bw1))
        widthInt.setMinimum(100)
        widthInt.setMaximum(1000)
        widthInt.setValue(800)
        widthInt.show()
        
        heightIntLab = QLabel('Height', self)
        heightIntLab.move(int(5.75*bw1),int(3*bw1))
        heightIntLab.show()
        
        heightInt = QSpinBox(self)
        heightInt.move(int(5.5*bw1),int(3.25*bw1))
        heightInt.setMinimum(100)
        heightInt.setMaximum(1000)
        heightInt.setValue(800)
        heightInt.show()

        windowGrabber = QPushButton('Window Capture', self)
        windowGrabber.resize(bw1,bw1)
        windowGrabber.move(int(1.5*bw1), int(5.1*bw1))
        windowGrabber.show()

        windowName = QLineEdit(self)
        windowName.move(int(2.5*bw1), int(5.1*bw1))
        windowName.resize(bw1,int(bw1/2))
        windowName.show()

        
        exitFunc = QPushButton('Exit', self)
        exitFunc.resize(bw1, bw1)
        exitFunc.move(int(bw1*1.5),int(6.1*bw1))
        exitFunc.show()
        
        buttons = [single_image, batch, threshLab, threshSlider, numClassesLab, numClasses, videoCam, screenCap, topIntLab,
                   topInt, leftIntLab, leftInt, widthIntLab, widthInt, heightIntLab, heightInt, windowGrabber, windowName, exitFunc]
        
        ##Actions
        exitFunc.clicked.connect(lambda: self.exit_buttons(buttons))
        single_image.clicked.connect(lambda: self.singleImage_button(threshSlider.value(), numClasses.value(), exitFunc, modelButton))
        batch.clicked.connect(lambda: self.batch_button(threshSlider.value(), numClasses.value(), exitFunc, modelButton))
        videoCam.clicked.connect(lambda: self.videoCam_button(threshSlider.value(), numClasses.value(), exitFunc, modelButton))
        screenCap.clicked.connect(lambda: self.screenCap_button(threshSlider.value(), numClasses.value(),
                                                                topInt.value(), leftInt.value(), widthInt.value(),
                                                                heightInt.value(), exitFunc, modelButton))
        windowGrabber.clicked.connect(lambda: self.windowGrabber_button(threshSlider.value(), numClasses.value(), windowName.text(),
                                                                        exitFunc, modelButton))
                                                                        
        
    def output_results_button(self, modelButton):
        getCoords = QPushButton('Get raster coordinates and resolution', self)
        getCoords.resize(4*bw1,bw1)
        getCoords.move(3*bw1,bw1)
        getCoords.show()
        
        convertCoords = QPushButton('Convert detection coordinates to geographic coordinates', self)
        convertCoords.resize(4*bw1,bw1)
        convertCoords.move(3*bw1,2*bw1)
        convertCoords.show()
        
        makeShape = QPushButton('Make Shapefile', self)
        makeShape.resize(4*bw1,bw1)
        makeShape.move(3*bw1,3*bw1)
        makeShape.show()
        
        getPR = QPushButton('Get PR Curve', self)
        getPR.resize(4*bw1,bw1)
        getPR.move(3*bw1,4*bw1)
        getPR.show()
    
        exitFunc = QPushButton('Exit', self)
        exitFunc.resize(4*bw1,bw1)
        exitFunc.move(3*bw1,5*bw1)
        exitFunc.show()
        
        buttons = [getCoords, convertCoords, makeShape, getPR, exitFunc]
        
        ##Actions
        exitFunc.clicked.connect(lambda: self.exit_buttons(buttons))
        getCoords.clicked.connect(self.getCoords_button)
        convertCoords.clicked.connect(self.convertCoords_button)
        makeShape.clicked.connect(self.makeShape_button)
        getPR.clicked.connect(lambda: self.getPR_button(exitFunc))

    ## This is the main window 
    def home(self):
        ##button for mask or box
        maskBox = QComboBox(self)
        maskBox.addItem('Faster R-CNN')
        maskBox.addItem('Mask R-CNN')
        maskBox.addItem('SSD Mobilenet')
        maskBox.addItem('Yolov5')
        maskBox.resize(bw1,int(bw1/2))
        maskBox.move(0, bw1)
        
        ##Button for set up
        setUp = QComboBox(self)
        setUp.addItem('New Project')
        setUp.addItem('Existing Project')
        setUp.resize(bw1,int(bw1/2))
        setUp.move(0,int(bw1*1.5))
        setUpGo = QPushButton('Go', self)
        setUpGo.resize(int(bw1/4),int(bw1/2))
        setUpGo.move(bw1,int(1.5*bw1))
        
        
        annotating = QPushButton("1. Annotating", self)
        annotating.resize(bw1, bw1)
        annotating.move(0,3*bw1)
        
        training = QPushButton("2. Training", self)
        training.resize(bw1, bw1)
        training.move(0,4*bw1)
        
        implementation = QPushButton("3. Implementation", self)
        implementation.resize(bw1, bw1)
        implementation.move(0,5*bw1)
        
        output_results = QPushButton("4. Output Results", self)
        output_results.resize(bw1, bw1)
        output_results.move(0,6*bw1)
        
        
        ##Actions
        setUpGo.clicked.connect(lambda: self.setup_button(str(setUp.currentText()), setUpGo))
        annotating.clicked.connect(lambda: self.annotating_button(maskBox))
        training.clicked.connect(lambda: self.training_button(maskBox))
        implementation.clicked.connect(lambda: self.implementation_button(maskBox))
        output_results.clicked.connect(lambda: self.output_results_button(maskBox))
        

        self.show()

## Function outside of the class to run the app   
def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

## Calling run to run the app
run()
