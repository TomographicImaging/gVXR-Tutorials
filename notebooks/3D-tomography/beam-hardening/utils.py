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


import numpy as np
from gvxrPython3 import gvxr
from gvxrPython3.utils import (
    applyFiltration,
    loadSpectrum,
)


def setPolySpectrum(
    tube_voltage_kV: float,
    filters=None,
    tube_angle_in_deg: float = 12,
    mAs=None,
    unit="keV",
) -> dict:
    """Create a polychromatic spectrum.

    Create a polychromatic spectum using ``gvxr.utils.loadSpectrum``

    Parameters
    ----------
    tube_voltage_kV: float
        The tube voltage value in kV units.
    filters: list, default=None
        A list of filters used in the spectrum creation. Each filter is
        represented by a list containing ``[<material>, <thickness>, <unit>]``.
    tube_angle_in_deg: float, default=12
        The X-ray tube angle used for the creation of the spectrum.
    mAs
        The exposure in milliampere second.
    unit: str, default="keV"
        The unit of energy.

    Returns
    -------
    A dictionary with the keys being the energy bin number and the value being
    the number of photons at that energy.
    """
    gvxr.clearFiltration()
    gvxr.setVoltage(tube_voltage_kV, "kV")
    gvxr.setTubeAngle(tube_angle_in_deg)

    if mAs:
        gvxr.setmAs(mAs)
    else:
        gvxr.setmAs(-1)

    if filters:
        applyFiltration(filters)

    energy_bins = gvxr.getEnergyBins(unit)
    photon_count = np.array(gvxr.getPhotonCountsPerCm2At1m(), dtype=np.single)
    photon_count /= (photon_count * energy_bins).sum()

    return loadSpectrum(energy_bins, photon_count, unit, False)


def makeHollowCylinder(
    label,
    number_of_sectors,
    height,
    outer_radius,
    inner_radius,
    unit_of_length,
    *,
    parent="root",
):
    # Outer cylinder
    gvxr.makeCylinder(
        label,
        number_of_sectors,
        height,
        outer_radius,
        unit_of_length,
        parent,
    )

    # Inner cylinder
    height_with_buffer = height + 0.01 * height
    gvxr.makeCylinder(
        "inner-cylinder",
        number_of_sectors,
        height_with_buffer,
        inner_radius,
        unit_of_length,
        parent,
    )

    gvxr.subtractMesh(label, "inner-cylinder")
