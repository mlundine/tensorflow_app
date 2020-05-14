# tensorflow1
TensorFlow Object Detection GUI Scripts

**TensorFlow Graphical User Interface Guide (Windows)**

**By: Mark Lundine**

**I. Installation and Setup**

**Step A) Anaconda already installed**

If you already have Anaconda Python 3.x downloaded, you need to do the following:

Find the folder:

C:\Users\YourName\Anaconda3

Copy this path to your clipboard.

Go to Windows SearchEdit the system environmental variables

Click on Environment Variables

Click on Path in the bottom panel, and then click Edit..

Click on New

Paste C:\Users\YourName\Anaconda3 into the new space, and then hit Ok.

You can now skip to Step C).

**Step B) Anaconda not already installed**

Go to [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/) and download Anaconda Python 3.x, 64-bit version.

Step through the installation process for Anaconda with all of the default and recommended settings, until it asks if you want to add Anaconda to your path. Check the box that asks if you want to add it to your path, which would be the Alternative Approach in the below image.


**Step C) Downloading master folder, libraries, and creating new Anaconda environments**

Download this repository, and unzip it to your C-Drive, with the name "tensorflow1".

In a web browser, go to [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

Download these two models: 

faster_rcnn_inception_v2_coco

mask_rcnn_resnet101_atrous_coco

Use 7-zip to unzip these files into C:\tensorflow1\models\research\object_detection

Open Anaconda Prompt on your computer, you can do this by searching Anaconda Prompt in your search bar.

In the terminal, type:

Idle

Then hit enter. This will bring up a Python console. Go to FileOpen

Open the file C:\tensorflow1\gui\ **Lundine\_gui.py**

Now go to RunRun Module

This will bring up the GUI.

Click the button Set Up to finishing installing all of the necessary Python libraries, as well as to create two new Anaconda environments: tensorflow1 and gdal1.  Each time the Anaconda Prompt asks for a y or n, type y and hit enter.

The Set Up is finished once the button is disabled.

Next, hit Check Set Up. This will bring up a Jupyter Notebook in your default internet browser.

Run through each section of this script by hitting Run at the top of the window. There is a section that downloads a pre-trained network in this code, so wait for that section to finish before running the remaining sections. You can tell a block of code is currently executing if you see In[\*]. So for each block you run, wait until the asterisk disappears before running the next block. If everything is set up correctly, eventually you will see some pictures labeled with bounding boxes at the bottom of the Jupyter notebook. Once you can confirm this works, go to the Anaconda Prompt and hit Ctrl+C. This will end the GUI. Close all Python windows and the Anaconda Prompt, then re-open Anaconda Prompt, type idle, open the gui python file, and then run the module again.

**II. Annotating**

If you have all of the photos you want to annotate as .jpgs, you are ready to start annotating. If you are using geotiffs with a single band (like digital elevation models), hit Convert Geotiffs to JPEGs (single band). This will open a dialog box which you can use to navigate to the folder containing all of your geotiffs. If your geotiffs are RGB, then use Convert Geotiffs to JPEGs. The new images will save in the folder C:\tensorflow1\models\research\object\_detection\implementation\jpegs. It will also convert your images to numpy arrays and save these to the folder C:\tensorflow1\models\research\object\_detection\implementation\numpy\_arrays.

Next, we need to randomize which images go into training (80% of images) and testing (20% of images). Click Set up training and testing data, and navigate to the folder containing all of your jpegs (if you used either of the previous buttons, the folder is C:\tensorflow1\models\research\object\_detection\implementation\jpegs.

This will randomly select 80% of the photos for training and 20% of the photos for testing, saving each chunk in C:\tensorflow1\models\research\object\_detection\train and C:\tensorflow1\models\research\object\_detection\test.

Next, you can start annotating. Click launch Labelimg. In this app, go to Open Directory, and select the train folder. Annotate each of your objects with a bounding box and a label. After each image is completely annotated, click save. This will create an xml file containing the coordinates for the annotations. Hit next image, and repeat. Keep in mind, every time you annotate an object, it should have exactly the same label (ex: only &#39;dog&#39;, not &#39;dog&#39; &#39;DOG&#39; &#39;Dog&#39;). Once you are done with the train folder, move onto the test folder. Once you are completely finished annotating all of the train and test photos, exit Labelimg. You can also exit the annotating section of the GUI at this point by hitting the Exit button.

**III. Training**

You should be finished with all annotations before using this section of the GUI. First, hit Convert Annotations to TfRecords.

This will bring up a Notepad window.  You need to edit the section that says "TO DO".  This is your label map, which should have a unique integer for each class/label.  The else None statement needs to stay. Once this is complete, save the file, then close the Notepad window.

Next, hit Make Label Map. This will bring up Notepad. Modify the label map to match your objects. Each object should have a unique integer id and a unique string name. Save the file, making sure the extension is .pbtxt. Once it is save, close the Notepad window.

Next, hit configure training. This will bring up Notepad. There are two changes you need to make:

Line 9: num\_classes: 6

Change this to match the number of classes your detector will have.

Line 132: num\_examples: 67

Change this to match the number of images in your test folder.

Save this config file, and then close the Notepad window. Next, hit start training. You will see lots of text begin to appear in the Anaconda Prompt terminal. Eventually it will start training, specifying a step number and a loss value. Keep training until the loss value starts to level out, which usually takes at least 40,000 steps. This will take at least 24 hours and it will use all of your CPU, so leave the computer alone while it is training. Make sure auto updates are not enabled because it will interrupt the training and restart the computer. Once you are done training, hit Ctrl+C in the Anaconda Prompt. This will stop training and also quit out of the GUI. Close all of the Python windows and the Anaconda Prompt. Then restart the Anaconda Prompt, type idle, open the Lundine\_gui file, and run the module again. Hit Training. Next, navigate to C:\tensorflow1\models\research\object\_detection\training, and find the checkpoint file with the highest number. Remember this number. Go back to the GUI, and change the slider value to that number and then hit export inference graph. Once this is done, hit Exit in the GUI. Then hit the Implementation button.

**IV. Implementation**

First, change the threshold value to what you want your detector to run on (ex: 0.60 means the detector will only mark detections it is at least 60% confident in). Also change the number of classes to the number of classes your detector has.

If you want to run on a single image, hit single image, and then navigate to the image (.jpg). Once you hit open, the detection will execute and you should see the image with bounding boxes appear in the GUI. It will also save this image to C:\tensorflow1\models\research\object\_detection\implementation\results\images.

If you want to run on a batch of images, change threshold to 0.00 and then hit Batch of Images and then navigate to the folder with all of those images (.jpg). Once this folder is opened, the detector will start and save all of the images to C:\tensorflow1\models\research\object\_detection\implementation\results\images.

It will also save the bounding boxes, labels, and thresholds to a .csv file in

C:\tensorflow1\models\research\object\_detection\implementation\results\bounding\_boxes.

Now you can hit Exit, and then go to Output Results.

**V. Output Results**

If your original images were in a projected geographic coordinate system, first, close the GUI and Python windows. Then type in the Anaconda Prompt, activate gdal1. Then type idle and open up the GUI script and run it. Go to Output Results and then hit Get raster coordinates and resolution. This will get the resolution and the four corner coordinates of each image and save it to a .csv file in geo\_bounding\_boxes.

Next, hit Convert detection coordinates to geographic coordinates. This will change the bounding box coordinates for each image from local coordinates to geographic coordinates.

Next, exit the GUI and the Python windows and type activate tensorflow1 in the Anaconda Prompt. Then reopen the GUI and go back to Output Results.

The Make Shapefile button only works for a single class detector

Next, hit Make Shapefile. This will make a Shapefile that can be opened in GIS software to display the detections. You will need to define the projection in the GIS software. This will save to

C:\tensorflow1\models\research\object\_detection\implementation\results\gis

Next, hit Get PR Curve. In the first dialog box, select the result\_bbox.csv sitting in

C:\tensorflow1\models\research\object\_detection\implementation\results\bounding\_boxes.

Then, in the second dialog box, select the test\_labels.csv in

C:\tensorflow1\models\research\object\_detection\images.

This will save a .csv containing your precision and recall data as well as make a plot of the precision and recall which will display in the GUI. It will save the .csv file with the data and a .png of the plot to

C:\tensorflow1\models\research\object\_detection\implementation\results\prdata.
