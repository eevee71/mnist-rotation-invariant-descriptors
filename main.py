from data.dataloader import load_MNIST
from src.moment_transforms import MomentTransform

def main():
    data, targets = load_MNIST()
    mt = MomentTransform()
    # test
    density_map, x_centered, y_centered = mt.prepare_density_center(data)
    invariants = mt.covariance_invariants(density_map, x_centered, y_centered)
    x_norm, y_norm = mt.normalization(x_centered, y_centered, invariants[:, 0])

    U = mt.gaussian_polynomial_moment_matrix(density_map, x_norm, y_norm)
    T = mt.hermite_to_monomial_matrix()
    coeffs, index = mt.homogeneous_coefficients(U, T)
    print(coeffs[0], index[0], coeffs.shape)

if __name__ == '__main__':
    main()