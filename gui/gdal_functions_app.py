# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:12 2020

@author: Mark Lundine
"""
import sys
import os
import gdal
import numpy as np
import glob
# =============================================================================
# get coords and res will make a spreadsheet of the coordinates and resolution for a folder
# need to specify the folder with the DEMs and a .csv file path to save the DEMs' coordinates and resolutions
# of DEMs, using arcpy.  
# =============================================================================
def gdal_get_coords_and_res(folder, saveFile):
    myList = []
    myList.append(['file', 'xmin', 'ymin', 'xmax', 'ymax', 'xres', 'yres'])
    for dem in glob.glob(folder + '/*.tif'):
        src = gdal.Open(dem)
        xmin, xres, xskew, ymax, yskew, yres  = src.GetGeoTransform()
        xmax = xmin + (src.RasterXSize * xres)
        ymin = ymax + (src.RasterYSize * yres)
        myList.append([dem, xmin, ymin, xmax, ymax, xres, -yres])
        src = None
    np.savetxt(saveFile, myList, delimiter=",", fmt='%s')
  
# =============================================================================
# converts geotiffs to numpy arrays        
# =============================================================================
def gdal_tifToNumpy(inFolder, outFolder):
    for dem in glob.glob(inFolder + '/*.TIF'):
        inRas = gdal.Open(dem)
        arr = np.array(inRas.GetRasterBand(1).ReadAsArray())
        name = os.path.splitext(os.path.basename(dem))[0]
        outPath = os.path.join(outFolder, name)
        np.save(outPath, arr, allow_pickle=True, fix_imports=True)
        inRas = None
        arr = None
        
def gdal_rgb_tif_to_npy(inFolder, outFolder):
    for image in glob.glob(inFolder + '/*.TIF'):
        inRas = gdal.Open(image)
        height, width = np.shape(inRas.GetRasterBand(1).ReadAsArray())
        rgb = np.zeros((height,width,3), 'uint8')
        rgb[...,0] = inRas.GetRasterBand(1).ReadAsArray()
        rgb[...,1] = inRas.GetRasterBand(2).ReadAsArray()
        rgb[...,2] = inRas.GetRasterBand(3).ReadAsArray()
        name = os.path.splitext(os.path.basename(image))[0]
        outPath = os.path.join(outFolder, name)
        np.save(outPath, rgb, allow_pickle=True, fix_imports=True)
        inRas = None
        rgb = None
        height = None
        width = None
        
##if __name__ == "__main__":
##    func = sys.argv[1]
##    if func == 'tifToNumpy':
##        inPath = sys.argv[2]
##        outPath = sys.argv[3]
##        gdal_tifToNumpy(inPath, outPath)
##    elif func == 'tifToNumpyRGB':
##        inPath = sys.argv[2]
##        outPath = sys.argv[3]
##        gdal_rgb_tif_to_npy(inPath, outPath)
##    elif func == 'getCoords':
##        inPath = sys.argv[2]
##        outCsv = sys.argv[3]
##        gdal_get_coords_and_res(inPath, outCsv)
