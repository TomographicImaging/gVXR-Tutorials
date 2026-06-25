# -*- coding: utf-8 -*-
#
#  Copyright 2025 United Kingdom Research and Innovation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#   Authored by:    Franck Vidal (UKRI-STFC)


import os
import numpy as np

import urllib.request
import progressbar
import zipfile

from gvxrPython3 import gvxr

pbar = None;

def downloadLungman():

    # where to save the data
    lungman_path = os.path.join("output_data", "lungman");
    mesh_path = os.path.join(lungman_path, "MESHES");
    CT_path = os.path.join(lungman_path, "CT");

    if not os.path.exists(mesh_path):
        os.makedirs(mesh_path);    

    if not os.path.exists(CT_path):
        os.makedirs(CT_path);    

    # Download the data from [Zenodo](https://zenodo.org/records/10782644).
    zip_file_url = "https://zenodo.org/records/10782644/files/lungman_data.zip?download=1";

    zip_fname = os.path.join(lungman_path, "lungman_data.zip");

    if not os.path.exists(zip_fname):
        def show_progress(block_num, block_size, total_size):
            global pbar
            if pbar is None:
                pbar = progressbar.ProgressBar(maxval=total_size)
                pbar.start()
        
            downloaded = block_num * block_size
            if downloaded < total_size:
                pbar.update(downloaded)
            else:
                pbar.finish()
                pbar = None
                
        print("Download the file (%s) from %s\n" % (zip_fname, zip_file_url))
        urllib.request.urlretrieve(zip_file_url, zip_fname, show_progress)

    return zip_fname, lungman_path, mesh_path, CT_path

def extractLungmanSTL(zip_fname, lungman_path):
    stl_fname_set = [];

    with zipfile.ZipFile(zip_fname) as z:
        for fname in z.namelist():
            if ".stl" in fname:
                stl_fname = os.path.join(lungman_path, fname);
                stl_fname_set.append(stl_fname);
                
                if not os.path.exists(stl_fname):
                    print("Extract %s" % stl_fname);
                    with open(stl_fname, 'wb') as f:
                        f.write(z.read(fname));

    return stl_fname_set;


def extractFilesFromZipFile(zip_fname, input_path, output_path, fnames):
    DICOM_fname_set = [];

    for DICOM_slice in fnames:

        input_DICOM_fname = os.path.join(input_path, DICOM_slice);
        output_DICOM_fname = os.path.join(output_path, DICOM_slice + ".dcm");
        DICOM_fname_set.append(output_DICOM_fname);

        with zipfile.ZipFile(zip_fname) as z:
            if not os.path.exists(output_DICOM_fname):
                print("Extract %s" % input_DICOM_fname);
                with open(output_DICOM_fname, 'wb') as f:
                    f.write(z.read(input_DICOM_fname));

    return DICOM_fname_set;

def extractLungmanCT(zip_fname, CT_path, DICOM_fnames):
    DICOM_path = "CD2/DICOM/ST000000/SE000003/";
    return extractFilesFromZipFile(zip_fname, DICOM_path, CT_path, DICOM_fnames);

def extractLungmanDX(zip_fname, lungman_path):
    DICOM_path = "CD3/DICOM/ST000000/SE000000/";
    DICOM_fnames = ["DX000000"];
    return extractFilesFromZipFile(zip_fname, DICOM_path, lungman_path, DICOM_fnames);

def loadLungmanMeshes(mesh_path):
    gvxr.removePolygonMeshesFromXRayRenderer();

    gvxr.emptyMesh("lungman");

    geometry_set = {
        "bronchioles": {"HU": -419.57144, "Colour" : [0, 240, 240, 0.08]},
        "bronchus": {"HU": -40.36795, "Colour" : [0, 62, 186, 0.4]},
        "trachea": {"HU": -914.32916, "Colour" : [170, 85, 255, 0.4]},
        "diaphram": {"HU": -12.778751, "Colour" : [255, 85, 127, 1]},
        "skin": {"HU": -12.121676, "Colour" : [125, 125, 125, 0.17]},
        "heart": {"HU": 28.384626, "Colour" : [255, 0, 0, 1]},
        "sheets_low": {"HU": -158.2706, "Colour" : [193, 193, 193, 1]},
        "sheets_med": {"HU": 203.39578, "Colour" : [193, 193, 193, 1]},
        "sheets_high": {"HU": 324.9135, "Colour" : [193, 193, 193, 1]},
        "tumours_630HU": {"HU": -658.61346, "Colour" : [138, 0, 0, 1]},
        "tumours_100HU": {"HU": 83.32481, "Colour" : [255, 85, 0, 1]},
        "spine-hard-650": {"HU": 857.8602, "Colour" : [255, 255, 127, 1]},
        "spine-soft-650": {"HU": 375.58865, "Colour" : [255, 255, 127, 1]},
        "scaps-hard-550": {"HU": 709.09717, "Colour" : [255, 255, 127, 1]},
        "scaps-soft-550": {"HU": 372.82138, "Colour" : [255, 255, 127, 1]},
        "sternum-hard-550": {"HU": 789.6037, "Colour" : [255, 255, 127, 1]},
        "sternum-soft-550": {"HU": 378.79736, "Colour" : [255, 255, 127, 1]},
        "clavicle-hard-700": {"HU": 778.28, "Colour" : [255, 255, 127, 1]},
        "clavicle-soft-700": {"HU": 261.89047, "Colour" : [255, 255, 127, 1]},
    }

    for label in geometry_set:
        if "sheet" not in label:
            gvxr.loadMeshFile(label, os.path.join(mesh_path, label + ".stl"), "mm", True, "root");
            gvxr.setHounsfieldUnit(label, round(geometry_set[label]["HU"]));
            gvxr.setColour(label,
                geometry_set[label]["Colour"][0] / 255.0,
                geometry_set[label]["Colour"][1] / 255.0,
                geometry_set[label]["Colour"][2] / 255.0,
                geometry_set[label]["Colour"][3]);


def crop_image(img):
    """
    Crop an image to remove the possible empty border around the data
    :param img: the input image
    :return: the image after border removal
    """
    if np.max(img) != np.min(img):

        # Crop on the left if needed
        min_i = 0
        stop = False
        while not stop:

            if img[:, min_i].sum() == 0:
                min_i += 1
            else:
                stop = True

        # Crop on the right if needed
        max_i = img.shape[1] - 1
        stop = False
        while not stop:

            if img[:, max_i].sum() == 0:
                max_i -= 1
            else:
                stop = True

        # Crop on the top if needed
        min_j = 0
        stop = False
        while not stop:

            if img[min_j, :].sum() == 0:
                min_j += 1
            else:
                stop = True

        # Crop on the bottom if needed
        max_j = img.shape[0] - 1
        stop = False
        while not stop:

            if img[max_j, :].sum() == 0:
                max_j -= 1
            else:
                stop = True

        cropped = img[max(0, min_j - 1):max_j + 2, max(0, min_i - 1):max_i + 2]
    else:
        cropped = np.copy(img)

    return cropped


def pad_image(img, target_image_shape):
    """
    Zero padding of an image
    :param img: the image to be padded
    :param target_image: the target image size
    :return: the padded image
    """
    extra_x_high = (target_image_shape[1] - img.shape[1]) // 2
    extra_y_high = (target_image_shape[0] - img.shape[0]) // 2

    if not (target_image_shape[1] - img.shape[1]) % 2:
        extra_x_low = extra_x_high
    else:
        extra_x_low = extra_x_high + 1

    if not (target_image_shape[0] - img.shape[0]) % 2:
        extra_y_low = extra_y_high
    else:
        extra_y_low = extra_y_high + 1

    return np.pad(img,
                  ((extra_y_low, extra_y_high), (extra_x_low, extra_x_high)),
                  mode='constant',
                  constant_values=0)