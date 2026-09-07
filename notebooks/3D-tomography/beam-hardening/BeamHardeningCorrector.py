import logging

import numpy as np
from cil.framework import DataProcessor

log = logging.getLogger(__name__)


class BeamHardeningCorrector(DataProcessor):
    r"""Beam Hardening Correction Processor.

    Processor to correct artifacts caused as a result of Beam hardening

    Parameters
    ----------
    polynomial_coefficients: numpy.ndarray

    linear_attenuation_coefficient: float

    monochromatic_energy

    max_path_length: float

    precision: float

    constant_bias: float, default = 0.0


    Returns
    -------
    ImageData of the reconstruction with beam hardening corrected.
    """

    # Understand that the user may not be aware of the max path length in anything
    # apart from centre slice too (can we get from data?)
    def __init__(
        self,
        polynomial_coefficients: np.ndarray,
        linear_attenuation_coefficient: float,
        max_path_length: float,
        precision: float,
        *,
        constant_bias: float = 0.0,
    ):
        # TODO: check if LAC or more suitable way to arrive at it.
        kwargs = {
            "polynomial_coefficients": polynomial_coefficients,
            "linear_attenuation_coefficient": linear_attenuation_coefficient,
            "max_path_length": max_path_length,
            "precision": precision,
            "constant_bias": constant_bias,
        }
        super().__init__(**kwargs)

    def check_input(self, data):
        # make sure to check whether data is aq or recon
        return True

    def process(self, out=None):
        data = self.get_input()

        if out is None:
            out = data.copy()
            arr = out.as_array()
        else:
            out.fill(data.as_array())
            arr = out.as_array()

        number_of_samples = int(self.max_path_length / self.precision)

        poly_x_values = np.linspace(0, self.max_path_length, number_of_samples)
        poly_y_values = np.polynomial.polynomial.polyval(
            poly_x_values, self.polynomial_coefficients
        )

        # true_x_value = np.interp(data, poly_y_values, poly_x_values)
        true_x_value = np.interp(arr, poly_y_values, poly_x_values)

        # TODO add constant_bias() here
        arr = self.linear_attenuation_coefficient * true_x_value + self.constant_bias
        # np.multiply(self.linear_attenuation_coefficient, true_x_value, out=arr)

        out.fill(arr)

        return out
