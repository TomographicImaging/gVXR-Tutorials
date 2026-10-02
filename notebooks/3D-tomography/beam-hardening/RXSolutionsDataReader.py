#   Copyright 2025 UKRI-STFC

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# Authors:
# Franck Vidal (URKI-STFC)


import numpy as np
import os, sys
from pathlib import Path
from xml.etree import ElementTree
from tifffile import imread

from cil.framework import (
    AcquisitionGeometry,
)  # , AcquisitionData, ImageData, ImageGeometry, DataOrder
from cil.io.TIFF import TIFFStackReader


class RXSolutionsDataReader(object):
    """
    Create a reader for data acquired with a RX Solutions device

    Parameters
    ----------
    file_name: str
        file name to read

    normalise: bool, default=True
        normalises loaded projections by detector white level (I_0)
    """

    def __init__(
        self,
        file_name: str = None,
        pixel_pitch_in_mm: tuple[float, float] | float = None,
        first_angle: float = None,
        last_angle: float = None,
        last_angle_included: bool = False,
        normalise: bool = True,
        mode: str = "bin",
        roi=None,
    ):

        # Initialise class attributes
        self.file_name = file_name
        self.pixel_pitch_in_mm = pixel_pitch_in_mm
        self.first_angle = first_angle
        self.last_angle = last_angle
        self.last_angle_included = last_angle_included
        self.normalise = normalise
        self.mode = mode
        self.roi = roi

        self._ag = None  # The acquisition geometry object
        self.tiff_directory_path = None

        # The file name is set
        if file_name is not None:
            # Initialise the instance
            self.set_up(
                file_name=file_name,
                pixel_pitch_in_mm=pixel_pitch_in_mm,
                first_angle=first_angle,
                last_angle=last_angle,
                last_angle_included=last_angle_included,
                normalise=normalise,
                mode=mode,
                roi=roi,
            )

    def set_up(
        self,
        file_name: str = None,
        pixel_pitch_in_mm: tuple[float, float] | float = None,
        first_angle: float = None,
        last_angle: float = None,
        last_angle_included: bool = False,
        normalise: bool = True,
        mode: str = "bin",
        roi=None,
    ):
        """Set up the reader

        Parameters
        ----------
        file_name: str
            file name to read

        normalise: bool, default=True
            normalises loaded projections by detector white level (I_0)
        """

        # Save the attributes
        self.file_name = file_name
        self.first_angle = first_angle
        self.last_angle = last_angle
        self.last_angle_included = last_angle_included
        self.mode = mode
        self.roi = roi

        if isinstance(pixel_pitch_in_mm, float):
            self.pixel_pitch_in_mm = [pixel_pitch_in_mm, pixel_pitch_in_mm]
        elif pixel_pitch_in_mm is not None:
            self.pixel_pitch_in_mm = [pixel_pitch_in_mm[0], pixel_pitch_in_mm[1]]
        else:
            self.pixel_pitch_in_mm = pixel_pitch_in_mm

        self.normalise = normalise

        # Error check
        # Check a file name was provided
        if file_name is None:
            raise ValueError(
                "Path to unireconstruction.xml or geom.csv (or geometry.csv) file is required."
            )

        # Error check
        # Check if the file exists
        file_name = os.path.abspath(file_name)
        if not (os.path.isfile(file_name)):
            raise FileNotFoundError("{}".format(file_name))

        # Error check
        # Check the file name without the path
        file_type = self.__get_file_type(file_name)
        if (
            file_type != "unireconstruction.xml"
            and file_type != "geom.csv"
            and file_type != "geometry.csv"
        ):
            raise TypeError(
                'This reader can only process "unireconstruction.xml", "geom.csv" or "geometry.csv" files. Got {}'.format(
                    file_type
                )
            )

        # Get the directory path
        directory_path = Path(os.path.dirname(file_name))

        # Look for projections
        self.tiff_directory_path = directory_path / "Proj"
        if not os.path.isdir(self.tiff_directory_path):
            raise ValueError(
                f"The projection directory '{self.tiff_directory_path}' does not exist"
            )

        # Traditional orbital geometry
        if file_type == "unireconstruction.xml":
            self.__set_up_orbital()
        # Per-projection geometry
        elif file_type == "geom.csv" or file_type == "geometry.csv":
            self.__set_up_flexible()
        # Error check
        else:
            raise ValueError('Cannot read "' + file_type + '".')

    def __get_file_type(self, file_name):
        return os.path.basename(file_name).lower()

    def __get_scaling_factors(self):
        scaling_factor_axis_1 = 1
        scaling_factor_axis_2 = 1

        if self.mode == "bin" and self.roi is not None:
            if "axis_1" in self.roi.keys():
                if self.roi["axis_1"][0] is not None:
                    raise ValueError('roi["axis_1"][0] must be None.')
                if self.roi["axis_1"][1] is not None:
                    raise ValueError('roi["axis_1"][1] must be None.')

                scaling_factor_axis_1 = self.roi["axis_1"][2]

            if "axis_2" in self.roi.keys():
                if self.roi["axis_2"][0] is not None:
                    raise ValueError('roi["axis_2"][0] must be None.')
                if self.roi["axis_2"][1] is not None:
                    raise ValueError('roi["axis_2"][1] must be None.')

                scaling_factor_axis_2 = self.roi["axis_2"][2]

        return [scaling_factor_axis_1, scaling_factor_axis_2]

    def __set_up_orbital(self):

        # Error check
        file_type = self.__get_file_type(self.file_name)
        if file_type != "unireconstruction.xml":
            raise TypeError(
                'This method can only process "unireconstruction.xml" files. Got {}'.format(
                    file_type
                )
            )

        # Error check
        if self.pixel_pitch_in_mm is None:
            raise ValueError(
                'The pixel pitch in mm must be known when using "unireconstruction.xml".'
            )

        # Error check
        if self.first_angle is None:
            raise ValueError('The first angle must be known when using "unireconstruction.xml".')

        # Error check
        if self.last_angle is None:
            raise ValueError('The last angle must be known when using "unireconstruction.xml".')

        scaling_factor_axis_1, scaling_factor_axis_2 = self.__get_scaling_factors()
        self.pixel_pitch_in_mm[0] *= scaling_factor_axis_2
        self.pixel_pitch_in_mm[1] *= scaling_factor_axis_1

        # Open the XML file
        tree = ElementTree.parse(self.file_name)

        # Find the conebeam profile
        profile = tree.find("conebeam/profile")
        assert profile is not None

        # Get the number of projections
        number_of_projections = int(profile.attrib["images"])

        # Look for the name of projection images
        image_file_names = [image for image in self.tiff_directory_path.rglob("*.tif")]
        if len(image_file_names) != number_of_projections:
            raise IOError(
                "There are "
                + str(len(image_file_names))
                + " TIFF files in the projection directory. We expected "
                + str(number_of_projections)
                + ' based on the "unireconstruction.xml" file.'
            )

        # Find the acquisition information
        acquisition_info = tree.find("conebeam/acquisitioninfo")
        if acquisition_info is None:
            acquisition_info = tree.find("conebeam/acquisitionInfo")
        assert acquisition_info is not None

        # Find the acquisition geometry
        conf_geo = acquisition_info.find("geometry")
        assert conf_geo is not None

        # Get the SDD and SOD
        source_to_detector = float(conf_geo.attrib["sdd"])
        source_to_object = float(conf_geo.attrib["sod"])
        object_to_detector = source_to_detector - source_to_object

        # Create the acquisition geometry
        # self._ag = AcquisitionGeometry.create_Cone3D(
        #     source_position=[-source_to_object, 0, 0],
        #     detector_direction_x=[0, -1,  0],
        #     detector_direction_y=[0, 0, 1],
        #     detector_position=[object_to_detector, 0, 0],
        #     rotation_axis_position=[0, 0, 0],
        #     units='mm')

        # Find the conebeam correction
        correction = tree.find("conebeam/correction")

        # Get the offsets
        offsetX = 0.0
        offsetY = 0.0

        if correction is not None:
            offsetX = float(correction.attrib["offsetX"])
            offsetY = float(correction.attrib["offsetY"])

            if offsetX > 1e-9 or offsetY > 1e-9:
                print(
                    f"offsetX ({offsetX}) and offsetY ({offsetY}) are not compatible with the reader in "
                    "the orbital geometry mode. Offsets will be ignored. If the reconstruction is not correct, "
                    "consider loading the CSV file instead of the XML file. This way the flexible geometry mode will "
                    "be used.",
                    file=sys.stderr,
                )

        self._ag = AcquisitionGeometry.create_Cone3D(
            source_position=[0, -source_to_object, 0],
            rotation_axis_position=[0, 0, 0],
            detector_direction_x=[1, 0, 0],
            detector_direction_y=[0, 0, 1],  # Not working => q.rotate([0, 0, 1])
            detector_position=[
                -offsetX * self.pixel_pitch_in_mm[0],
                object_to_detector,
                -offsetY * self.pixel_pitch_in_mm[1],
            ],
            units="mm",
        )

        # Set the angles of rotation
        self._ag.set_angles(
            np.linspace(
                self.first_angle,
                self.last_angle,
                number_of_projections,
                endpoint=self.last_angle_included,
            )
        )

        # Read the first projection to extract its size in nmber of pixels
        first_projection_data = imread(image_file_names[0])
        projection_shape = first_projection_data.shape

        # self._ag.set_panel(detector_number_of_pixels, pixel_spacing_mm)
        self._ag.set_labels(["angle", "vertical", "horizontal"])

        # Panel is width x height
        self._ag.set_panel(
            [
                ##!! To double-check
                # axis_1 and 2 vs projection_shape[0] and [1]
                projection_shape[1] // scaling_factor_axis_2,
                projection_shape[0] // scaling_factor_axis_1,
            ],
            self.pixel_pitch_in_mm,
            origin="top-left",
        )

    def __set_up_flexible(self):

        # Error check
        file_type = self.__get_file_type(self.file_name)
        if file_type != "geom.csv" and file_type != "geometry.csv":
            raise TypeError(
                'This method can only process "geom.csv" (or "geometry.csv") files. Got {}'.format(
                    file_type
                )
            )

        # Error check
        if self.mode is not None and self.mode != "bin":
            raise ValueError('mode should be equal to None or "bin" only.')

        meta_data = np.genfromtxt(
            self.file_name,
            delimiter=";",
            skip_header=2,
            usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        )

        proj_files = np.genfromtxt(
            self.file_name, delimiter=";", skip_header=2, dtype=str, usecols=(0), encoding=None
        )

        subset_meta_data = meta_data[np.where(np.char.find(proj_files, "Proj\\") >= 0)]

        if subset_meta_data.size > 0:
            meta_data = subset_meta_data

        # Get the number of projections
        number_of_projections = meta_data.shape[0]

        # Look for the name of projection images
        image_file_names = [image for image in self.tiff_directory_path.rglob("*.tif")]
        if len(image_file_names) != number_of_projections:
            raise IOError(
                "There are "
                + str(len(image_file_names))
                + " TIFF files in the projection directory. We expected "
                + str(number_of_projections)
                + ' based on the "'
                + file_type
                + '" file.'
            )

        # Read the first projection to extract its size in nmber of pixels
        first_projection_data = imread(image_file_names[0])
        projection_shape = np.array(first_projection_data.shape)
        # Format the metadata
        source_position_set = meta_data[:, :3]
        detector_position_set = meta_data[:, 3:6]
        detector_direction_y_set = (
            (meta_data[:, 6:9] - detector_position_set) / projection_shape[0] * 2
        )
        detector_direction_x_set = (
            (meta_data[:, 9:] - detector_position_set) / projection_shape[1] * 2
        )

        # The pixel size in mm is the norm of the vectors in detector_direction_x_set and detector_direction_y_set
        pixel_pitch_in_mm = np.linalg.norm(
            (
                detector_direction_x_set[0, 0],
                detector_direction_x_set[0, 1],
                detector_direction_x_set[0, 2],
            )
        )

        pixel_pitch_direction_x_in_mm = np.linalg.norm(
            (
                detector_direction_x_set[0, 0],
                detector_direction_x_set[0, 1],
                detector_direction_x_set[0, 2],
            )
        )
        pixel_pitch_direction_y_in_mm = np.linalg.norm(
            (
                detector_direction_y_set[0, 0],
                detector_direction_y_set[0, 1],
                detector_direction_y_set[0, 2],
            )
        )

        detector_direction_x_set /= pixel_pitch_direction_x_in_mm
        detector_direction_y_set /= pixel_pitch_direction_y_in_mm

        # Roughly recentre the data on the Y-axis
        Y = 0.5 * np.mean(detector_direction_y_set[:, 1]) + 0.5 * np.mean(
            source_position_set[:, 1]
        )
        source_position_set[:, 1] -= Y
        detector_position_set[:, 1] -= Y

        # More finely recentre the data on the Z-axis (previously Y-axis)
        sod = (np.sum(source_position_set**2, axis=1) ** 0.5).mean()
        sdd = (np.sum((source_position_set - detector_position_set) ** 2, axis=1) ** 0.5).mean()
        alpha = sod / sdd
        Y = alpha * np.mean(source_position_set[:, 1]) + (1 - alpha) * np.mean(
            source_position_set[:, 1]
        )
        source_position_set[:, 1] -= Y
        detector_position_set[:, 1] -= Y

        # Retrieve the offsets from the XML file
        dir_path = os.path.dirname(os.path.realpath(self.file_name))
        XML_path = dir_path + "/" + "unireconstruction.xml"

        offsetX = 0.0
        offsetY = 0.0

        if os.path.exists(XML_path):
            # Open the XML file
            tree = ElementTree.parse(XML_path)

            # Find the conebeam correction
            correction = tree.find("conebeam/correction")

            # Get the offsets
            if correction is not None:
                offsetX = float(correction.attrib["offsetX"])
                offsetY = float(correction.attrib["offsetY"])

        # Axes transformation
        # X->Y
        # Y->Z
        # Z->X
        source_position_set = np.roll(source_position_set, 1, axis=1)
        detector_position_set = np.roll(detector_position_set, 1, axis=1)
        detector_direction_y_set = np.roll(detector_direction_y_set, 1, axis=1)
        detector_direction_x_set = np.roll(detector_direction_x_set, 1, axis=1)

        # Apply the offsets
        detector_position_set -= detector_direction_x_set * offsetX * pixel_pitch_direction_x_in_mm
        detector_position_set -= detector_direction_y_set * offsetY * pixel_pitch_direction_y_in_mm

        # # Rotation matrix for 90 degrees around the z-axis (clockwise)
        # # cos(-90°) = 0, sin(-90°) = -1
        # rotation = np.array([
        #     [0, 1, 0],
        #     [-1, 0, 0],
        #     [0, 0, 1]
        # ])
        #
        # # Rotate the acquisition geometry so that it matches the orbital geometry
        # source_position_set = source_position_set @ rotation.T
        # detector_position_set = detector_position_set @ rotation.T
        # detector_direction_y_set = detector_direction_y_set @ rotation.T
        # detector_direction_x_set = detector_direction_x_set @ rotation.T

        # Apply the scaling factors
        scaling_factor_axis_1, scaling_factor_axis_2 = self.__get_scaling_factors()
        self.pixel_pitch_in_mm = [
            pixel_pitch_in_mm * scaling_factor_axis_2,
            pixel_pitch_in_mm * scaling_factor_axis_1,
        ]

        # Create the acquisition geometry
        self._ag = AcquisitionGeometry.create_Cone3D_Flex(
            source_position_set,
            detector_position_set,
            detector_direction_x_set,
            detector_direction_y_set,
            volume_centre_position=[0, 0, 0],
            units="mm",
        )

        # self._ag.set_panel(detector_number_of_pixels, pixel_spacing_mm)
        self._ag.set_labels(["projection", "vertical", "horizontal"])

        # Panel is width x height
        self._ag.set_panel(
            [
                ##!! To double-check
                # axis_1 and 2 vs projection_shape[0] and [1]
                projection_shape[1] // scaling_factor_axis_2,
                projection_shape[0] // scaling_factor_axis_1,
            ],
            self.pixel_pitch_in_mm,
            origin="top-left",
        )

    def read(self):
        """
        Reads projections and returns AcquisitionData with corresponding geometry,
        arranged as ['angle', horizontal'] if a single slice is loaded
        and ['vertical, 'angle', horizontal'] if more than 1 slice is loaded.
        """

        # Check a file name was provided
        if self.tiff_directory_path is None:
            raise ValueError("The reader was not set properly.")

        # Create the TIFF reader
        reader = TIFFStackReader(file_name=self.tiff_directory_path, mode=self.mode, roi=self.roi)

        ad = reader.read_as_AcquisitionData(self._ag)

        if self.normalise:
            white_level = np.max(ad.array)
            ad.array[ad.array < 1] = 1

            # cast the data read to float32
            ad = ad / np.float32(white_level)

        return ad

    def load_projections(self):
        """alias of read for backward compatibility"""
        return self.read()

    def get_geometry(self):
        """
        Return AcquisitionGeometry object
        """

        return self._ag
