import numpy as np
from numba import njit, prange

@njit(parallel=True)
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm

@njit
def dot_product(v1, v2):
    return np.dot(v1, v2)

@njit
def matrix_vector_multiply(M, v):
    return np.dot(M, v)

@njit
def power_iteration(M, num_simulations):
    b_k = np.ones(M.shape[1])
    for _ in range(num_simulations):
        b_k = matrix_vector_multiply(M, b_k)
        b_k = normalize(b_k)
    eigenvalue = dot_product(matrix_vector_multiply(M, b_k), b_k)
    eigenvector = b_k
    return eigenvalue, eigenvector

class PCA_SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None

    def mean(self, X):
        return np.mean(X, axis=0)
    
    def center_data(self, X, mean):
        return X - mean

    def transpose(self, X):
        return np.transpose(X)

    def multiply(self, A, B):
        return np.dot(A, B)

    def covariance_matrix(self, X):
        return np.cov(X, rowvar=False)

    def svd(self, A):
        eigenvalues, eigenvectors = [], []
        for _ in prange(A.shape[0]):
            eigenvalue, eigenvector = power_iteration(A, 100)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            A = self.subtract(A, self.outer_product(eigenvector, eigenvector, eigenvalue))
        return np.array(eigenvalues), np.array(eigenvectors)
    
    def subtract(self, A, B):
        return A - B
    
    def outer_product(self, v1, v2, scalar):
        return np.outer(v1, v2) * scalar
    
    def fit(self, X):
        mean = self.mean(X)
        X_centered = self.center_data(X, mean)
        cov_matrix = self.covariance_matrix(X_centered)
        eigenvalues, eigenvectors = self.svd(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[sorted_indices[:self.n_components]]
        self.explained_variance_ = eigenvalues[sorted_indices[:self.n_components]]
        return self
    
    def transform(self, X):
        mean = self.mean(X)
        X_centered = self.center_data(X, mean)
        return self.multiply(X_centered, self.transpose(self.components_))
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def custom_pca(scaled_data, n_components=2):
    pca = PCA_SVD(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    explained_variances = pca.explained_variance_
    return principal_components, explained_variances
