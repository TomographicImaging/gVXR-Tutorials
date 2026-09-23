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

from collections.abc import Iterator
from operator import attrgetter
from typing import NamedTuple

import numpy as np
import numpy.polynomial.polynomial as poly
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


def transmission_to_absorption(data, tol=1e-9):
    data[data < tol] = tol
    return -np.log(data)


def find_optimal_stepwedge_size(
    data: np.ndarray, material: str, tolerance: float, max_iterations: int = 50
):
    """Binary search to find the optimal stepwedge height.

    Parameters
    ----------
    data: ndarray
        The data of the sinorgam corresponding to the reconstruction.
    """
    detector_length, detector_width = gvxr.getDetectorSize("mm")

    lower_bound = 0
    upper_bound = gvxr.getSourceDetectorDistance("mm")

    target = data.max()

    for _ in range(max_iterations):
        gvxr.removePolygonMeshesFromSceneGraph()
        mid_value = (upper_bound + lower_bound) * 0.5

        gvxr.makeCuboid("bin_search_couboid", mid_value, detector_length, detector_width, "mm")
        gvxr.addPolygonMeshAsOuterSurface("bin_search_couboid")
        gvxr.setElement("bin_search_couboid", material)

        projection = (
            np.array(gvxr.computeXRayImage(), dtype=np.single)
            / gvxr.getTotalEnergyWithDetectorResponse()
        )

        neg_log_projection = transmission_to_absorption(projection)
        max_projection = (
            neg_log_projection.max()
        )  # Should always be equal to 1 (unless the sample is larger)

        if abs(max_projection - target) <= tolerance:
            return mid_value

        if max_projection < target - tolerance:
            lower_bound = mid_value

        else:
            upper_bound = mid_value

    return -1


class PolynomialFit(NamedTuple):
    """
    Named tuple containing Polynomial fit related metrics.

    Attributes
    ----------
    order : int
        The order of the polynomial.
    coefficients : np.ndarray
        The array of polynomial coefficients.
    curve : np.ndarray
        The curve values themselves (y-axis points for the input x-values).
    rmse : float
        The root mean square error value of the fit against the input values.
    """

    order: int
    coefficients: np.ndarray
    curve: np.ndarray
    rmse: float


def _get_poly_fit_metrics(
    x: np.ndarray,
    y: np.ndarray,
    max_order: int,
) -> Iterator[PolynomialFit]:
    for order in range(1, max_order + 1):
        coefficients = poly.polyfit(x, y, order)
        fit_values = poly.polyval(x, coefficients)
        rmse = float(np.sqrt(np.mean((y - fit_values) ** 2)))

        yield PolynomialFit(order, coefficients, fit_values, rmse)


def get_optimal_poly_fit(
    x: list | np.ndarray,
    y: list | np.ndarray,
    max_order: int,
) -> PolynomialFit:
    """Return the curve optimally fitting inputted curve.

    The optimal curve is obtained by comparing several polynomial fits (based
    on the ``max_order`` value input) and outputting one with the least root
    mean square error.

    Parameters
    ----------
    x: ndarray
        x-coordinates of the polynomial curve to be fitted.
    y: ndarray
        y-coordinates of the polynomial curve to be fitted.
    max_order: int
        The maximum order of the polynomial to fit against
    """
    x = np.asarray(x)
    y = np.asarray(y)

    return min(_get_poly_fit_metrics(x, y, max_order), key=attrgetter("rmse"))
