from numba import jit
import numpy as np


@jit(nopython=True)
def numba_get_optimal_b(F: np.array, S: np.array, K: np.array, R: np.array, g_bold: np.array, s_bold: np.array,
                        lambda_S: float, lambda_K: float, lambda_R: float,
                        lambda_contrast: float,
                        T_contrast: np.array,
                        u_contrast: np.array) -> np.array:
    b_opt = np.linalg.inv(F + lambda_S * S + lambda_K * K + lambda_R * R + lambda_contrast * T_contrast).dot(g_bold - lambda_R * s_bold - lambda_contrast * u_contrast)
    return b_opt
