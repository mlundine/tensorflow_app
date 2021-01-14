# tensorflow_app

**TensorFlow Graphical User Interface Guide**

![bayexample](/read_me_images/implement_gui.png)

**By: Mark Lundine**

**Setting Up with Anaconda (has yolov5 capabilities)**

Download this repository, unzip it somewhere on your device, with the name "tensorflow_app", not "tensorflow_app-master".

There are two conda environments containing all of the needed package versions located in tensorflow_app/envs.

The first one to use is tensorflowappgpu.yml or tensorflowappmac.yml for macs or tensorflowappcpu for the cpu version.

Open up Anaconda prompt on Windows or terminal on macs and run:

cd wherever_you_placed_it/tensorflow_app/envs

conda env create --file tensorflowappgpu.yml  or conda activate tensorflowappmac or conda env create --file tensorflowappcpu.yml

conda env create --file yolov5.yml or conda env create --file yolov5mac

conda activate tensorflowappgpu or conda activate tensorflowappmac or conda activate tensorflowappcpu

cd wherever_you_placed_it/tensoflow_app/gui

python Lundine_gui_new.py  (On Windows)

python Lundine_gui_new_mac.py   (On Macs)

Then the GUI will run.

Now go to downloading pretrained tensorflow models, unless you only want to make a yolov5 model, then go to Making a New Project.


**I. Installation and Setup for Executable (No yolov5 capabilities)**

Download this repository, and unzip it to your C-Drive, with the name "tensorflow_app".

Download the most recent executable under releases, GPU if you are using a GPU, CPU if you are using your CPU.

Place the executable inside C:/tensorflow_app/gui

To start the app, go to C:/tensorflow_app/gui and double click on TensorflowApplication.exe

After a few seconds (or even a minute, give it some time, it has a lot of stuff to unpack and your antivirus might be checking it), the GUI will appear.


**Downloading Pretrained Tensorflow Models**

In a web browser, go to [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).

![modelzoopic](/read_me_images/MODELZOO.png)

Download these three models (or just one): 

faster_rcnn_inception_v2_coco

mask_rcnn_resnet101_atrous_coco

ssd_mobilenet_v1_ppn_coco

Use 7-zip to unzip these files into C:\tensorflow_app\models\research\object_detection


**Making a New Project**


![GUI Home](/read_me_images/GUI_home.PNG)

To make a new project, leave this button as is, and hit go.  This will bring up a dialog box where you can give your project a name.  This is a folder name so only use numbers, letters, and underscores.  No spaces or emojis.

![New Project](/read_me_images/new_project_naming.PNG)

Once you have a good name, hit OK.  You should see at the top of the window Project Name: ______.

This will make a new project directory in /tensorflow_app/gui

You can also open an existing project by changing the New Project button to Existing project, and then hitting go.  You will then navigate to your project folder
C:/tensorflow_app/gui.

![Project Folder](/read_me_images/project_in_dir.PNG)

You can choose which model you would like to train on.  Faster RCNN and SSD Mobilenet give bounding boxes as outputs, with Faster RCNN being more accurate but has slower inference speeds compared to SSD, and Mask RCNN gives bounding boxes and segmentation masks.  Mask RCNN has the slowest inference speeds.

![modelTypes](/read_me_images/GUI_home_models.PNG)

**II. Annotating**

![Annotating](/read_me_images/annotating_gui.PNG)

If you have all of the photos you want to annotate as .jpgs, you are ready to start annotating. If you are using geotiffs with a single band (like digital elevation models), hit Convert Geotiffs to JPEGs (single band). This will open a dialog box which you can use to navigate to the folder containing all of your geotiffs. If your geotiffs are RGB, then use Convert Geotiffs to JPEGs. The new images will save in the folder C:\tensorflow_app\gui\YourProject\implementation\jpegs. It will also convert your images to numpy arrays and save these to the folder C:\tensorflow_app\gui\YourProject\implementation\numpy_arrays.

For Faster R-CNN, leave the changeable button on this. For Mask R-CNN, change it to Mask R-CNN.  

Next, we need to randomize which images go into training (80% of images) and testing (20% of images). Click Set up training and testing data, and navigate to the folder containing all of your jpegs (if you used either of the previous buttons, the folder is C:\tensorflow_app\gui\YourProject\implementation\jpegs.

This will randomly select 80% of the photos for training and 20% of the photos for testing, saving each chunk in C:\tensorflow_app\gui\YourProject\images\train and C:\tensorflow_app\gui\YourProject\images\test.

Next, you can start annotating. Click launch Labelimg. In this app, go to Open Directory, and select the train folder. Annotate each of your objects with a bounding box and a label. After each image is completely annotated, click save. This will create an xml file containing the coordinates for the annotations. Hit next image, and repeat. Keep in mind, every time you annotate an object, it should have exactly the same label (ex: only &#39;dog&#39;, not &#39;dog&#39; &#39;DOG&#39; &#39;Dog&#39;). Once you are done with the train folder, move onto the test folder. Once you are completely finished annotating all of the train and test photos, exit Labelimg. You can also exit the annotating section of the GUI at this point by hitting the Exit button.

If you are trying to use Mask RCNN, you will also need to make png masks of your annotations.  These are images that you need to segment by the classes you are detecting.  For example if you were trying to detect cats and dogs, you would need to make a new image where all of the pixels that have dogs are given a certain value, all of the pixels with cats are given another value, and the rest of the pixels are zero.  These would be placed in the train_mask and test_mask folders.  You would also need to annotate each image with bounding boxes.  So your entire annotation set would be a train folder of jpeg images, with bounding boxes made in Labelimg, a test folder of jpeg images, with bounding boxes made in LabelImg, a train_mask folder, with binary PNG masks, and a test_mask folder, with binary PNG masks.  The mask images should have the same name as their corresponding jpeg images, the only difference is that the masks should be PNGs instead of jpegs.

In the future I will make some functions that can automate making the mask annotations.  

**III. Training**

**Setting Up Yolo Training**

In .../tensorflow_app/yolov5, open the dataset.yaml file in a text editor.

edit the path to train and val

train: wherever_you_placed_it/tensorflow_app/gui/project_name/images/train

val: wherever_you_placed_it/tensorflow_app/gui/project_name/images/test

Save this yml file to  wherever_you_placed_it/tensorflow_app/gui/project_name/yolovdata.

Then go to Training a).

![Training](/read_me_images/training.PNG)

**Training a)**

You should be finished with all annotations before using this section of the GUI. First, hit Convert Annotations to TfRecords or Yolo Records.  This will save two csvs containing the annotations for the test and train datasets in C:\tensorflow_app\gui\YourProject\images.  It will also save two tf record files in C:\tensorflow_app\gui\YourProject\images\frcnn_records (or mrcnn_records if you are using Mask R-CNN).

If you are using yolov5, skipt to Training d).

**Training b)**

Next, hit Make Label Map. This will bring up Notepad. Modify the label map to match your objects (so change name and value). Each object should have a unique integer id and a unique string name. Check your train_labels.csv file to see what integer id each label was given.  If you are only building a one class detector, then the id for your class will be 1.  Save the file to C:/tensorflow_app/gui/YourProject/frcnn_training (or mrcnn_training if you are using Mask RCNN, SSD_training if you are using SSD Mobilenet) making sure the extension is .pbtxt. Once it is saved, close the Notepad window. Double check in the /frcnn_training folder that the extension is pbtxt. If it has .txt at the end, just edit the name and delete the txt.  Ignore Windows when it warns about changing the extension.

Here is an example labelmap from a euchre deck detector with six classes.

![labelmap](/read_me_images/labelmap.png)

**Training c)**

Next, hit configure training. This will bring up Notepad. There are a few changes you need to make.  For the changes that require filepaths, USE FORWARD SLASHES '/'.  When you copy the path in file explorer, Windows will make them backslashes '\'.  Make sure you change them to forward slashes:

num_classes: 6

Change this to match the number of classes your detector will have.

fine_tune_checkpoint : "fullfilepath/to/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

If you are using Mask RCNN it would be to the Mask RCNN model you downloaded.

This filepath should be the full path to the model you downloaded and put in the object detection folder.  

input_path: "filepath/to/train.record"

This filepath should point to C:/tensorflow_app/gui/YourProject/images/frcnn_records/train.record.

label_map_path: "filepath/to/labelmap.pbtxt"

This filepath should point to C:/tensorflow_app/gui/YourProject/frcnn_training/labelmap.pbtxt"

input_path: "filepath/to/test.record"

This filepath should point to C:/tensorflow_app/gui/YourProject/images/frcnn_records/test.record.

label_map_path: "filepath/to/labelmap.pbtxt"

This filepath should point to C:/tensorflow_app/gui/YourProject/frcnn_training/labelmap.pbtxt"

num_examples: 67

Change this to match the number of images in your test folder.

Here is a screenshot of a config file with areas that need edits outlined in red.

![config](/read_me_images/config.png)

Save this config file to /frcnn_training (or /mrcnn_training if you are doing Mask RCNN, ssd_training if you are doing SSD Mobilenet) as a .config file, not a .txt file, and then close the Notepad window. Double check the extension (.config not .txt). 

**Training d)**

For yolo models, you should choose a maximum image dimension to train on. The default is 640 pixels.  But if all your training images are smaller, say 300x300, change this to 300.  If they are larger (ex: 1200x1200) you could change this to 1200.  Keep in mind making this too large might max out your GPU memory.

Also for yolo models, you should decide a number of epochs to train for.  The default is 400.

Next, hit start training. For yolo models, it will ask you to choose a weights file, pick the yolov5s.pt in ../tensorflow_app/yolov5.  Then it will ask for your model's yaml file.

For yolo models, it will save training data to wherever_you_placed_it/tensorflow_app/gui/project_name/yolodata/train.

The weights file you can use for implementation is either best.pt or last.pt located in


wherever_you_placed_it/tensorflow_app/gui/project_name/yolodata/train/weights.


If you want to resume yolo training, just choose the last.pt as the weights file when you hit Start Training.  Otherwise, you are now ready to try detection with your yolov5 model.


**Training e)**
In the /frcnn_training folder, you will start to see checkpoint files appear.  These will be updated every couple of minutes.  Try to train for at least 40,000 steps.  This might take a full day.  Once you see it has trained for at least 40,000 steps, quit the GUI.  You can do this by hitting the X in the top right, and then let Windows shut the program down.  Go back to the /frcnn_training folder and find the checkpoint file with the highest number. Remember this number. Go back to the GUI, and change the slider value to that number and then hit export inference graph. Once this is done, hit Exit in the GUI. Then hit the Implementation button.

**IV. Implementation**


![Implementation](/read_me_images/implementation.PNG)


Yolo implementation will ask for you to give it a weights file, this should be the .pt file you get after training (either best.pt or last.pt in the train/weights folder).

First, change the threshold value to what you want your detector to run on (ex: 0.60 means the detector will only mark detections it is at least 60% confident in). Also change the number of classes to the number of classes your detector has.  Double check you have the correct number of classes before running single image or batch of images, otherwise, some errors will likely arise.

If you are using a yolo detector, you should specify the max image size to feed the detector, the default is 640.  If you are feeding it a 300x300, change this slider to 300.  If you are feeding it a 1200x1200, change the slider to 1200.  Making this too large could max out your GPU.  Also if you have a rectangular image, say 1280x800 and you want to feed yolo the default size 640, it will maintain the aspect ratio of the original image, feeding it a 640x400 image.

If you want to run on a single image, hit single image, and then navigate to the image (.jpg for Faster R-CNN, Mask R-CNN, ssd, .jpg, .tif, or .png for Yolov5). Once you hit open, the detection will execute and you should see the image with bounding boxes appear in the GUI. It will also save this image to C:/tensorflow_app/gui/YourProject/implementation/results/images.

If you want to run on a batch of images, change threshold to the level you want displayed on the ouput images and then hit Batch of Images and then navigate to the folder with all of those images (.jpg for Faster R-CNN, Mask R-CNN and ssd, .jpg, .tif, or .png for Yolov5). Once this folder is opened, the detector will start and save all of the images to C:/tensorflow_app/gui/YourProject/implementation/results/images.

It will also save the bounding boxes, labels, and thresholds to a .csv file in

C:/tensorflow_app/gui/YourProject/implementation/results/bounding_boxes.  This will include all detections with thresholds greater than zero.


I have also added implementation functions to run on your computer's webcam if it has one, and a function to run on a portion of your computer's screen. TFor the screen function, you need to input coordinates for what region you want the detector to run on.  The top coordinate is how many pixels, going from the top of your screen downward that you want the detection region to start.  The left coordinate is how many pixels from the left boundary that you want the detection region to start.  And then height and with are the height and width in pixels that you want the detection region to span.  

The latest function I have added is Window Capture.  This can be used to run a detector on a specific window open on your computer.  You just need to type the exact name of that window in the text box, then click the Window Capture button.  You will then see I window open showing the bounding boxes for objects it sees in the window you told it to look at.

The webcam, the screen capture, and the window capture functions do not output anything currently.  Expect some updates to add outputs for these functions soon.


The screen portion and window capture functions are not available for Yolo models yet.


![screendetection](/read_me_images/screenDetect.png)


Now you can hit Exit, and then go to Output Results.

**V. Output Results**


![Ouputs](/read_me_images/outputresults.PNG)


If your original images were in a projected geographic coordinate system, hit Get raster coordinates and resolution. This will get the resolution and the four corner coordinates of each image and save it to a .csv file in geo\_bounding\_boxes.  You have to point it to the folder of the original rasters as geotiffs.

Next, hit Convert detection coordinates to geographic coordinates. This will change the bounding box coordinates for each image from local coordinates to geographic coordinates.  You will need to show it the bounding box csv file made during implementation, and also the folder of jpegs that the detection was run on.

For yolo models, it will ask for the labels folder which can be found in detect/labels for the batch of images you ran the detector on.  It will then ask for the folder containing the geotiffs you got the raster coordinates and resolution from.

Next, hit Make Shapefile. This will make a Shapefile that can be opened in GIS software to display the detections. You will need to define the projection in the GIS software. This will save to /results/gis.

The next button is Get PR Curve.  This should be used after running the detector on all of the images in the test folder. So do that, and make sure it saves the result_bbox. 

Next, hit Get PR Curve. In the first dialog box, select the result_bbox.csv sitting in results/bounding_boxes.

Then, in the second dialog box, select the test_labels.csv in YourProject\images.

This will save a .csv containing your precision and recall data as well as make a plot of the precision and recall which will display in the GUI. It will save the .csv file with the data and a .png of the plot to

C:\tensorflow1\models\research\object\_detection\implementation\results\prdata.



I have included some screenshots of the what directories should look like when using this app.

![projectfolder](/read_me_images/project_folder.png)

If you trained a detector, the training folder should look like this:

![trainingFolder](/read_me_images/training_folder.png)

If you exported the inference graph, the inference_graph folder should look like this:

![infGraphFolder](/read_me_images/inf_graph_folder.png)

The implementation folder looks like this:

![impFolder](/read_me_images/imp_folder.png)

The results folder looks like this:

![resultFolder](/read_me_images/results_folder.png)


**VI. Libaries used**

Standard Python library that comes with Anaconda Python 3 installation

numpy, all the numerical methods

pillow, image processing library

opencv-python, image and video processing library

gdal (best installed with conda install gdal), geographic data library

pyqt5, all of the GUI related elements

matplotlib, plotting library

protobuf, making protbuf files

lxml, parsing xml files

Cython, speeding python code with C types

contextlib2, tensorflow related stuff

pyshp, making shapefiles

sklearn, a couple signal processing things

mss, screen captures

pyautogui, screen captures

win32gui, window captures

tensorflow==1.13.1 or tensorflow-gpu==1.13.1, object detection api

pyinstaller to make freeze the code

pytorch, yolov5 

yolov5 code from ultralytics

