import logging

import numpy as np
from cil.framework import AcquisitionData, DataProcessor

log = logging.getLogger(__name__)


class BeamHardeningCorrector(DataProcessor):
    r"""Beam Hardening Correction Processor.

    Processor to correct artifacts caused as a result of Beam hardening

    Parameters
    ----------
    polynomial_coefficients: numpy.ndarray

    linear_attenuation_coefficient: float

    path_length_max: float

    path_length_precision: float, default = 0.0

    constant_bias: float, default = 0.0

    Returns
    -------
    ImageData of the reconstruction with beam hardening corrected.
    """

    # Understand that the user may not be aware of the max path length in anything
    # apart from centre slice too (can we get from data?)
    #
    # Maybe set it as None by default and estimate max path length by estimating
    # path length of slice with
    #
    # Add options for strength - weak, medium, strong
    # Add option to determine automatic optimal linear curve automatically
    def __init__(
        self,
        polynomial_coefficients: np.ndarray,
        linear_attenuation_coefficient: float,
        path_length_max: float,
        path_length_precision: float = 0.01,
        *,
        constant_bias: float = 0.0,
    ):
        # TODO: check if LAC or more suitable way to arrive at it.
        kwargs = {
            "polynomial_coefficients": polynomial_coefficients,
            "linear_attenuation_coefficient": linear_attenuation_coefficient,
            "path_length_max": path_length_max,
            "path_length_precision": path_length_precision,
            "constant_bias": constant_bias,
        }
        super().__init__(**kwargs)

    def check_input(self, data):
        if self.polynomial_coefficients is None:
            poly_coeff_msg = "Please provide polynomial coefficient list"
            raise ValueError(poly_coeff_msg)

        if not isinstance(data, AcquisitionData):
            aq_format_msg = f"Expected AcquistionData, found {type(data)}"
            raise TypeError(aq_format_msg)

        return True

    def process(self, out=None):
        data = self.get_input()

        if out is None:
            out = data.copy()
            arr = out.as_array()
        else:
            out.fill(data.as_array())
            arr = out.as_array()

        number_of_samples = int(self.path_length_max / self.path_length_precision)

        poly_x_values = np.linspace(0, self.path_length_max, number_of_samples)
        poly_y_values = np.polynomial.polynomial.polyval(
            poly_x_values, self.polynomial_coefficients
        )

        true_x_value = np.interp(arr, poly_y_values, poly_x_values)

        np.multiply(self.linear_attenuation_coefficient, true_x_value, out=arr)
        arr += self.constant_bias

        out.fill(arr)

        return out
