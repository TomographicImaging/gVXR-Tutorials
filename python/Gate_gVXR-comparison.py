#!/usr/bin/env python3
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


import argparse, sys
import logging



def getData(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    phantom_path = data_path / "pediatric_phantom_data"
    if not os.path.exists(phantom_path):
        os.mkdir(phantom_path)

    phantom_ZIP_path = phantom_path / "Pediatric phantom.zip"
    if not os.path.exists(phantom_ZIP_path):
        urllib.request.urlretrieve("https://drive.uca.fr/f/384a08b5f73244cf9ead/?dl=1", phantom_ZIP_path)

        with zipfile.ZipFile(phantom_ZIP_path,"r") as zip_ref:
            print("Extract ZIP file", phantom_ZIP_path, "in", phantom_path)
            zip_ref.extractall(phantom_path)

    return phantom_path / "Pediatric phantom" / "Pediatric_model.mhd"

def initGVXR():
    gvxr.createOpenGLContext()
    gvxr.setSourcePosition(*source_focus_point_position_in_mm, "mm");
    gvxr.usePointSource();

    # gvxr.setMonoChromatic(energy_in_keV, "keV", number_of_photons_per_pixel);

    spectrum = loadSpekpySpectrum(kvp, 
        filters=filtration,
        th_in_deg=12,
        max_number_of_energy_bins=50,
    )

    gvxr.setNumberOfPhotonsPerPixelAtSDD(number_of_photons_per_pixel)
            
    gvxr.enablePoissonNoise()

    gvxr.setDetectorPosition(*detector_position_in_mm, "mm");
    gvxr.setDetectorUpVector(0, 1, 0);
    gvxr.setDetectorRightVector(1, 0, 0);
    gvxr.setDetectorNumberOfPixels(detector_cols, detector_rows);
    gvxr.setDetectorPixelSize(pixel_size_in_mm[0], pixel_size_in_mm[1], "mm");

    if use_scintillation:
        gvxr.setScintillator("Gd2O2S DRZ-Plus", scintillator_thickness_in_mm, "mm")

    gvxr.makeCuboid("Ti90Al6V4Box", *Ti90Al6V4Box_size_in_cm, "cm")
    gvxr.translateNode("Ti90Al6V4Box", *Ti90Al6V4Box_translation_in_cm, "cm")
    gvxr.addPolygonMeshAsInnerSurface("Ti90Al6V4Box")
    gvxr.setMixture("Ti90Al6V4Box", "Ti90Al6V4")
    gvxr.setDensity("Ti90Al6V4Box", 4.43, "g/cm3")

    gvxr.makeCuboid("BrassBox", *BrassBox_size_in_cm, "cm")
    gvxr.translateNode("BrassBox", *BrassBox_translation_in_cm, "cm")
    gvxr.addPolygonMeshAsInnerSurface("BrassBox")
    gvxr.setMixture("BrassBox", ["Cu", "Zn"], [0.666666, 0.333334])
    gvxr.setDensity("BrassBox", 8.565, "g/cm3")

    gvxr.makeCuboid("WhiteMetalBox", *WhiteMetalBox_size_in_cm, "cm")
    gvxr.translateNode("WhiteMetalBox", *WhiteMetalBox_translation_in_cm, "cm")
    gvxr.addPolygonMeshAsInnerSurface("WhiteMetalBox")
    gvxr.setMixture("WhiteMetalBox", ["Sn", "Sb", "Cu", "Pb"], [0.8, 0.11, 0.03, 0.06])
    gvxr.setDensity("WhiteMetalBox", 7.40, "g/cm3")

    gvxr.makeCuboid("CarbonSteelBox", *CarbonSteelBox_size_in_cm, "cm")
    gvxr.translateNode("CarbonSteelBox", *CarbonSteelBox_translation_in_cm, "cm")
    gvxr.addPolygonMeshAsInnerSurface("CarbonSteelBox")
    gvxr.setMixture("CarbonSteelBox", 
        ["Mn", "C", "S", "P", "Fe"], 
        [0.0075, 0.00175, 0.00025, 0.0002, 0.9903])
    gvxr.setDensity("CarbonSteelBox", 7.87, "g/cm3")

    gvxr.makeCuboid("aluminiumbox", *aluminiumbox_size_in_cm, "cm")
    gvxr.translateNode("aluminiumbox", *aluminiumbox_translation_in_cm, "cm")
    gvxr.addPolygonMeshAsInnerSurface("aluminiumbox")
    gvxr.setElement("aluminiumbox", "Al")

    return spectrum


def runGVXR():
    gvxr_attenuation_image = np.array(gvxr.computeXRayImage())
    gvxr_flat_image = np.array(gvxr.getWhiteImage())
    return gvxr_attenuation_image, gvxr_flat_image

def initGate(spectrum, output_dir, fname):
    sim = gate.Simulation()

    sim.g4_verbose = False
    sim.g4_verbose_level = 1
    sim.visu = False
    sim.visu_type = "vrml"
    sim.random_engine = "MersenneTwister"
    sim.random_seed = "auto"
    if os.name != "nt":
        sim.number_of_threads = multiprocessing.cpu_count()
    else:
        sim.number_of_threads = 1
    print(f"Use {sim.number_of_threads} threads")
    sim.progress_bar = True
    sim.output_dir = output_dir

    sim.volume_manager.add_material_database(data_path / "GateMaterials.db")

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"

    # world
    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    # world.material = "G4_AIR"
    world.material = "Vacuum"

    # CBCT gantry source
    gantry = sim.add_volume("Box", "CBCT_gantry")
    gantry.size = [0.2 * m, 0.2 * m, 0.2 * m]
    # gantry.material = "G4_AIR"
    gantry.material = "Vacuum"
    gantry.color = [0, 1, 1, 1]
    gantry.translation = gantry_position_in_mm * mm

    # CBCT detector plane
    detector_plane = sim.add_volume("Box", "CBCT_detector_plane")
    detector_plane.size = [409.6 * mm, 409.6 * mm, scintillator_thickness_in_mm * mm]

    # detector_plane.size = [409.6 * mm, 409.6 * mm, 0.6 * mm]
    if use_scintillation:
        detector_plane.material = "Gd2O2S-DRZ-Plus"
    else:
        detector_plane.material = "G4_AIR"
    # detector_plane.material = "CsI"
    detector_plane.color = [1, 0, 0, 1]
    detector_plane.translation = detector_position_in_mm * mm

    # actor
    if use_scintillation:
        detector_actor = sim.add_actor("DoseActor", "detector_actor")
    else:
        detector_actor = sim.add_actor("FluenceActor", "detector_actor")
    detector_actor.attached_to = detector_plane
    # detector_actor.output_filename = "cbct.mhd"
    detector_actor.output_filename = fname
    detector_actor.spacing = pixel_size_in_mm * mm
    detector_actor.size = [detector_cols, detector_rows, 1]
        
    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option1"
    sim.physics_manager.set_production_cut("world", "all", 10 * mm)

    # source
    source = sim.add_source("GenericSource", "mysource")
    source.attached_to = gantry.name
    source.particle = "gamma"

    # source.energy.mono = energy_in_keV * keV

    source.energy.type = "spectrum_discrete"
    source.energy.spectrum_energies = np.array(spectrum[1], dtype=np.single) * float(keV)
    source.energy.spectrum_weights = np.array(spectrum[2], dtype=np.single) / np.sum(spectrum[2])

    source.position.type = "box"
    source.position.size = source_size_in_mm * mm
    source.direction.type = "focused"
    source.direction.focus_point = source_focus_point_position_in_mm * mm
    source.n = total_number_of_photons / sim.number_of_threads

    return sim


def addPhantom(sim, use_dummy):
    Ti90Al6V4Box = sim.add_volume("Box", "Ti90Al6V4Box")
    Ti90Al6V4Box.size = Ti90Al6V4Box_size_in_cm * cm
    Ti90Al6V4Box.translation = Ti90Al6V4Box_translation_in_cm * cm
    Ti90Al6V4Box.color = [0, 0, 1, 1]  # this is RGBa (a=alpha=opacity), so blue here

    BrassBox = sim.add_volume("Box", "BrassBox")
    BrassBox.size = BrassBox_size_in_cm * cm
    BrassBox.translation = BrassBox_translation_in_cm * cm
    BrassBox.color = [0, 0, 1, 1]  # this is RGBa (a=alpha=opacity), so blue here

    WhiteMetalBox = sim.add_volume("Box", "WhiteMetalBox")
    WhiteMetalBox.size = WhiteMetalBox_size_in_cm * cm
    WhiteMetalBox.translation = WhiteMetalBox_translation_in_cm * cm
    WhiteMetalBox.color = [0, 0, 1, 1]  # this is RGBa (a=alpha=opacity), so blue here

    CarbonSteelBox = sim.add_volume("Box", "CarbonSteelBox")
    CarbonSteelBox.size = CarbonSteelBox_size_in_cm * cm
    CarbonSteelBox.translation = CarbonSteelBox_translation_in_cm * cm
    CarbonSteelBox.color = [0, 0, 1, 1]  # this is RGBa (a=alpha=opacity), so blue here

    aluminiumbox = sim.add_volume("Box", "aluminiumbox")
    aluminiumbox.size = aluminiumbox_size_in_cm * cm
    aluminiumbox.translation = aluminiumbox_translation_in_cm * cm
    aluminiumbox.color = [0, 0, 1, 1]  # this is RGBa (a=alpha=opacity), so blue here

    if not use_dummy:
        Ti90Al6V4Box.material = "Ti6Al4V"
        BrassBox.material = "Brass"
        WhiteMetalBox.material = "WhiteMetal"
        CarbonSteelBox.material = "CarbonSteel"
        aluminiumbox.material = "Aluminium"
    else:
        Ti90Al6V4Box.material = "Vacuum"
        BrassBox.material = "Vacuum"
        WhiteMetalBox.material = "Vacuum"
        CarbonSteelBox.material = "Vacuum"
        aluminiumbox.material = "Vacuum"


# this first line is required at the beginning of all scripts
if __name__ == "__main__":

    try:
        parser=argparse.ArgumentParser(description="Compare simulations performed with Gate and gVXR")
        parser.add_argument("--particles", type=int, default=5000000, help="Total number of particles")
        parser.add_argument("--SOD", type=float, default=1000, help="source-object distance in mm")
        parser.add_argument("--SDD", type=float, default=1000+536, help="source-detector distance in mm")

        parser.add_argument('--scintillation', action='store_true', help="Use scintillation")
        parser.add_argument('--no-scintillation', dest='scintillation', action='store_false', help="Do not use scintillation")
        parser.set_defaults(scintillation=True)

        args=parser.parse_args()

        import os
        from time import time
        # from datetime import timedelta

        import urllib, zipfile # Download and extract the phantom data
        import numpy as np
        import opengate as gate
        from scipy.spatial.transform import Rotation
        from pathlib import Path
        import multiprocessing

        import SimpleITK as sitk
        # from tifffile import imread, imwrite
        import xraylib as xrl

        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        import matplotlib
        # matplotlib.use('TkAGG')   # generate postscript output by default
        # matplotlib.use("Qt5Agg")

        plt.rcParams['figure.figsize'] = [13, 7]
        plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

        font = {'family' : 'serif',
                'size'   : 10
            }
        matplotlib.rc('font', **font)

        # Uncomment the line below to use LaTeX fonts
        # matplotlib.rc('text', usetex=True)


        from gvxrPython3 import gvxr
        from gvxrPython3.utils import loadSpekpySpectrum

        kvp = 160
        filtration = [["Cu", 0.4, "mm"]]

        # units
        m = gate.g4_units.m
        cm = gate.g4_units.cm
        nm = gate.g4_units.nm
        cm3 = gate.g4_units.cm3
        keV = gate.g4_units.keV
        MeV = gate.g4_units.MeV
        mm = gate.g4_units.mm
        Bq = gate.g4_units.Bq
        gcm3 = gate.g4_units.g / cm3



        SOD_in_mm = args.SOD
        SDD_in_mm = args.SDD
        ODD_in_mm = SDD_in_mm - SOD_in_mm
        

        source_focus_point_position_in_mm   = np.array([0, 0, SOD_in_mm])
        gantry_position_in_mm   = np.array([0, 0, SOD_in_mm + 60])
        detector_position_in_mm = np.array([0, 0, -ODD_in_mm])

        Ti90Al6V4Box_size_in_cm = np.array([7.0, 7.0, 0.5])
        Ti90Al6V4Box_translation_in_cm = np.array([4.0, 4.0, 0.0])

        BrassBox_size_in_cm = np.array([7.0, 7.0, 0.4])
        BrassBox_translation_in_cm = np.array([-4.0, 4.0, 0.0])

        WhiteMetalBox_size_in_cm = np.array([7.0, 7.0, 0.3])
        WhiteMetalBox_translation_in_cm = np.array([4.0, -4.0, 0.0])

        CarbonSteelBox_size_in_cm = np.array([7.0, 7.0, 0.2])
        CarbonSteelBox_translation_in_cm = np.array([-4.0, -4.0, 0.0])

        aluminiumbox_size_in_cm = np.array([20.0, 20.0, 1.5])
        aluminiumbox_translation_in_cm = np.array([0.0, 0.0, -5.0])

        source_size_in_mm = np.array([16, 16, 1e-6])

        detector_cols = 100
        detector_rows = 100

        detector_size_in_mm = np.array([
            (abs(detector_position_in_mm[2]) + abs(source_focus_point_position_in_mm[2])) * source_size_in_mm[0] / abs(gantry_position_in_mm[2] - source_focus_point_position_in_mm[2]),
            (abs(detector_position_in_mm[2]) + abs(source_focus_point_position_in_mm[2])) * source_size_in_mm[1] / abs(gantry_position_in_mm[2] - source_focus_point_position_in_mm[2])
        ])

        pixel_size_in_mm = np.array([detector_size_in_mm[0] / detector_cols, detector_size_in_mm[1] / detector_rows, 10])

        use_scintillation = args.scintillation

        if use_scintillation:
            scintillator_thickness_in_mm = 0.21
        else:
            scintillator_thickness_in_mm = 1e-6

        # energy_in_keV = 60
        total_number_of_photons = args.particles
        number_of_photons_per_pixel = max(1,round(total_number_of_photons / (detector_cols * detector_rows)))

        if use_scintillation:
            output_dir = "./output_data/Gate_gVXR-comparison/with_scintillation"
        else:
            output_dir = "./output_data/Gate_gVXR-comparison/without_scintillation"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gate_flat_image_fname = "gate_flat_image-" + str(total_number_of_photons) + "photons.mha"
        gate_attenuation_image_fname = "gate_attenuation_image-" + str(total_number_of_photons) + "photons.mha"
        gvxr_flat_image_fname = "gvxr_flat_image-" + str(total_number_of_photons) + "photons.mha"
        gvxr_attenuation_image_fname = "gvxr_attenuation_image-" + str(total_number_of_photons) + "photons.mha"



        data_path = Path("../data")
        phantom_volume_file_path = getData(data_path)

        spectrum = initGVXR()
        gvxr_start_time = time()
        gvxr_attenuation_image, gvxr_flat_image = runGVXR()
        gvxr_stop_time = time()
        # gvxr.renderLoop()
        # exit()
        gvxr_flat_image[gvxr_flat_image<1e-6] = 1e-6

        sitk_image = sitk.GetImageFromArray(gvxr_flat_image.astype(np.single))
        sitk.WriteImage(sitk_image, os.path.join(output_dir, gvxr_flat_image_fname))

        sitk_image = sitk.GetImageFromArray(gvxr_attenuation_image.astype(np.single))
        sitk.WriteImage(sitk_image, os.path.join(output_dir, gvxr_attenuation_image_fname))

        sim = initGate(spectrum, output_dir, gate_attenuation_image_fname)
        addPhantom(sim, use_dummy=False)
        gate_start_time1 = time()
        sim.run(True)
        gate_stop_time1 = time()

        sim = initGate(spectrum, output_dir, gate_flat_image_fname)
        addPhantom(sim, use_dummy=True)
        gate_start_time0 = time()
        sim.run(True)
        gate_stop_time0 = time()

        execution_time_gate = gate_stop_time0 - gate_start_time0 + gate_stop_time1 - gate_start_time1
        execution_time_gvxr = gvxr_stop_time - gvxr_start_time

        execution_time_fname = os.path.join(output_dir, "execution_time.dat")
        if os.path.exists(execution_time_fname):
            file_exists = True
        else:
            file_exists = False

        speedup = execution_time_gate / execution_time_gvxr
        with open(execution_time_fname, "a") as myfile:
            if not file_exists:
                myfile.write("total_number_of_photons,execution time Gate in sec,execution time gVXR in sec,speedup\n")

            myfile.write("%i    &    %f    &    %f    &    %i\\\\\n" % (total_number_of_photons,execution_time_gate, execution_time_gvxr,speedup))


        print("********************************************************************************")
        print("Total number of particles:", total_number_of_photons)
        print("Execution time with Gate [in sec]:", "{:.2f}".format(execution_time_gate))
        print("Execution time with Gate [in sec]:", "{:.2f}".format(execution_time_gvxr))
        print("Speedup:", round(speedup))
        print("********************************************************************************")
        # ct image
        # patient = sim.add_volume("Image", "patient")
        # patient.image = data_path / "patient-2mm.mhd"
        # # patient.image = phantom_volume_file_path
        # if sim.visu:
        #     # if the visualisation is enabled we load a very crude ct
        #     # otherwise the visualisation is too slow
        #     patient.image = data_path / "patient-20mm.mhd"
        # patient.material = "G4_AIR"  # material used by default
        # f1 = data_path / "Schneider2000MaterialsTable.txt"
        # f2 = data_path / "Schneider2000DensitiesTable.txt"
        # tol = 0.2 * gcm3
        # (
        #     patient.voxel_materials,
        #     materials,
        # ) = gate.geometry.materials.HounsfieldUnit_to_material(sim, tol, f1, f2)
        # patient.rotation = Rotation.from_euler("y", -90, degrees=True).as_matrix()



        # # motion of the spect, create also the run time interval
        # motion = sim.add_actor("MotionVolumeActor", "Move")
        # motion.attached_to = patient.name
        # motion.translations = []
        # motion.rotations = []
        # n = 90
        # sec = gate.g4_units.second
        # sim.run_timing_intervals = []
        # gantry_rotation = -90
        # start_t = 0
        # end = 1 * sec / n
        # initial_rot = Rotation.from_euler("X", 90, degrees=True)
        # for r in range(n):
        #     t, rot = gate.geometry.utility.get_transform_orbiting(
        #         [0, 0, 0], "Z", gantry_rotation
        #     )
        #     rot = Rotation.from_matrix(rot)
        #     rot = rot * initial_rot
        #     rot = rot.as_matrix()
        #     motion.translations.append(t)
        #     motion.rotations.append(rot)
        #     sim.run_timing_intervals.append([start_t, end])
        #     gantry_rotation += 90.0 / n
        #     start_t = end
        #     end += 1 * sec / n
            
            
        # # Try to run for 2 hours
        # runtime_in_sec_for_500000 = 9
        # target_runtime_in_hours = 2
        # target_runtime_in_sec = target_runtime_in_hours * 60 * 60
        # source.n = ((target_runtime_in_sec*500000)/runtime_in_sec_for_500000) / sim.number_of_threads

        # statistics
        stats = sim.add_actor("SimulationStatisticsActor", "stats")
        stats.track_types_flag = True
        stats.output_filename = "stats-" + str(total_number_of_photons) + "photons.txt"
        stats.write_to_disk = True

        # print(f"Run {len(sim.run_timing_intervals)} intervals: {sim.run_timing_intervals}")

        # # check actor priority: the MotionVolumeActor must be first
        # l = [l for l in sim.actor_manager.user_info_actors.values()]
        # sorted_actors = sorted(l, key=lambda d: d.priority)
        # print(f"Actors order: ", [[l.name, l.priority] for l in sorted_actors])

        if use_scintillation:
            gate_flat_image_fname = Path(gate_flat_image_fname).stem + "_edep.mha"
            gate_attenuation_image_fname = Path(gate_attenuation_image_fname).stem + "_edep.mha"

        if os.path.exists(os.path.join(output_dir, gate_flat_image_fname)) and \
            os.path.exists(os.path.join(output_dir, gate_attenuation_image_fname)) and \
            os.path.exists(os.path.join(output_dir, gvxr_flat_image_fname)) and \
            os.path.exists(os.path.join(output_dir, gvxr_attenuation_image_fname)):

            gate_flat_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dir, gate_flat_image_fname)))[0]
            gate_attenuation_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(output_dir, gate_attenuation_image_fname)))[0]
            

            # plt.figure()
            # plt.imshow(gvxr_attenuation_image[0:17])
            # plt.show()

            plot_fname = "compare-" + str(total_number_of_photons) + "photons"
            plt.figure(figsize= (20,10))

            plt.suptitle("Image simulated with Gate and gVirtualXRay using\n" + str(total_number_of_photons) + " photons in total")#, y=1.02)

            plt.subplot(221)

            selection = gate_attenuation_image[0:17]

            gate_normalised_image = gate_attenuation_image / gate_flat_image
            gate_normalised_image[gate_attenuation_image < 1e-4] = 1e-4
            # gate_normalised_image = gate_attenuation_image / np.mean(gate_attenuation_image[0:17])

            plt.imshow(gate_normalised_image, cmap="gray", vmin=0, vmax=1)
            plt.plot([0, gate_normalised_image.shape[1]-1], [0, gate_normalised_image.shape[0]-1], color="blue")#, linestyle='dashed')
            plt.plot([gate_normalised_image.shape[1]-1, 0], [0, gate_normalised_image.shape[0]-1], color="red")#, linestyle='dashed')
            plt.colorbar(orientation='horizontal')
            # plt.title("Gate (" + str(timedelta(seconds=gate_stop_time - gate_start_time)) + ")")
            plt.title("Gate (" + "{:.2f}".format(execution_time_gate) + " seconds with " + str(sim.number_of_threads) + " threads/cores)")

            plt.subplot(222)
            gvxr_normalised_image = gvxr_attenuation_image / gvxr_flat_image
            # gvxr_normalised_image = gvxr_attenuation_image / np.mean(gvxr_attenuation_image[0:17])

            plt.imshow(gvxr_normalised_image, cmap="gray", vmin=0, vmax=1)
            plt.plot([0, gvxr_normalised_image.shape[1]-1], [0, gvxr_normalised_image.shape[0]-1], color="cyan")#, linestyle='dotted')
            plt.plot([gvxr_normalised_image.shape[1]-1, 0], [0, gvxr_normalised_image.shape[0]-1], color="magenta")#, linestyle='dashed')
            plt.colorbar(orientation='horizontal')
            # plt.title("gVXR (" + str(timedelta(seconds=gvxr_stop_time - gvxr_start_time)) + ")")
            plt.title("gVXR (" + "{:.2f}".format(execution_time_gvxr) + " seconds)")

            plt.subplot(223)
            relative_error = gate_normalised_image - gvxr_normalised_image
            plt.imshow(relative_error, vmin=-1, vmax=1)
            plt.colorbar(orientation='horizontal')
            plt.title("Signed error")
            
            plt.subplot(224)
            plt.plot(np.diag(gate_normalised_image), label="Gate", color="blue")#, linestyle='dashed')
            plt.plot(np.diag(gvxr_normalised_image), label="gVXR", color="cyan")#, linestyle='dotted')
            plt.plot(np.diag(np.flipud(gate_normalised_image), 1), label="Gate", color="red")
            plt.plot(np.diag(np.flipud(gvxr_normalised_image), 1), label="gVXR", color="magenta")

            plt.title("Digonal intensity profile")
            # plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, plot_fname + ".pdf"))
            plt.savefig(os.path.join(output_dir, plot_fname + ".png"))
            # plt.close()
            # plt.show()

            plot_fname = os.path.join(output_dir, "spectrum-" + str(kvp) + "kV-filtration_" + "{:.2f}".format(filtration[0][1]) + filtration[0][2] + "_of_" + filtration[0][0])
            plt.figure(figsize= (20,10))
            # plt.title("Beam spectrum")
            plt.bar(spectrum[1], spectrum[2], width=1)
            plt.xlabel('Energy in keV')
            plt.ylabel('Probability distribution of photons per keV')
            plt.tight_layout()
            plt.savefig(plot_fname + ".pdf")
            plt.savefig(plot_fname + ".png")
            plt.close()
            # plt.show()




            energy_response = np.array(gvxr.getEnergyResponse("keV"))

            if energy_response.shape[0]:
                plot_fname = os.path.join(output_dir, "energy-response-" + "{:.2f}".format(gvxr.getScintillatorThickness("um")) + "um-of-" + gvxr.getScintillatorMaterial())
                # plt.title("Beam spectrum")
                plt.figure(figsize= (15,10))
                plt.scatter(energy_response[:,0], energy_response[:,1], label="gVXR", marker="x")
                plt.xlabel('Incident energy in keV')
                plt.ylabel('Relative detector energy response')

                mat = "Gd2O2S"
                rho = 4.76
                thickness = scintillator_thickness_in_mm / 10.0 #cm

                energy_range = np.logspace(0., 2.48, num=1000)
                response = [E*(xrl.CS_Energy_CP(mat, E)/xrl.CS_Total_CP(mat, E))*(1.-np.exp(-xrl.CS_Total_CP(mat, E) * float(rho) * thickness)) for E in energy_range]
                plt.plot(energy_range, energy_range,color="red",linewidth=1.,linestyle="--")
                plt.plot(energy_range, response,color="blue",linewidth=2.,linestyle="-", label="JML's code")
                plt.xlim(0.,300.)
                plt.ylim(0.,40.)

                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_fname + ".pdf")
                plt.savefig(plot_fname + ".png")
                plt.close()
                # plt.show()
                # exit()



        # print output statistics
        print(stats)
    except Exception as error:
        # handle the exception
        logging.exception(error)

