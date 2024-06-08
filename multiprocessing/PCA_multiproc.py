import multiprocessing as mp
import math

def mean(X):
    return [sum(col) / len(col) for col in zip(*X)]

def center_row(row, mean):
    return [x - m for x, m in zip(row, mean)]

def row_multiply(A_row, B):
    return [sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)]

def cov_matrix_element(X_transposed, i, j, n_samples):
    return sum(X_transposed[i][k] * X_transposed[j][k] for k in range(n_samples)) / (n_samples - 1)

def normalize(v):
    norm = math.sqrt(sum(x ** 2 for x in v))
    return [x / norm for x in v]

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def matrix_vector_multiply(M, v):
    return [sum(M_row[j] * v[j] for j in range(len(v))) for M_row in M]

class PCA_SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None

    def mean(self, X):
        return mean(X)

    def center_data(self, X, mean):
        with mp.Pool() as pool:
            X_centered = pool.starmap(center_row, [(row, mean) for row in X])
        return X_centered

    def transpose(self, X):
        return list(map(list, zip(*X)))

    def multiply(self, A, B):
        with mp.Pool() as pool:
            result = pool.starmap(row_multiply, [(A_row, B) for A_row in A])
        return result

    def covariance_matrix(self, X):
        n_samples = len(X)
        X_transposed = self.transpose(X)
        cov_matrix = [[0] * len(X_transposed) for _ in range(len(X_transposed))]
        
        with mp.Pool() as pool:
            results = [pool.apply_async(cov_matrix_element, (X_transposed, i, j, n_samples)) for i in range(len(X_transposed)) for j in range(len(X_transposed))]
            for idx, res in enumerate(results):
                i = idx // len(X_transposed)
                j = idx % len(X_transposed)
                cov_matrix[i][j] = res.get()
                
        return cov_matrix

    def svd(self, A):
        eigenvalues, eigenvectors = [], []
        for _ in range(len(A)):
            eigenvalue, eigenvector = self._power_iteration(A, 100)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            A = self.subtract(A, self.outer_product(eigenvector, eigenvector, eigenvalue))
        return eigenvalues, eigenvectors

    def _power_iteration(self, M, num_simulations):
        b_k = [1.0] * len(M)
        for _ in range(num_simulations):
            b_k = matrix_vector_multiply(M, b_k)
            b_k = normalize(b_k)
        eigenvalue = dot_product(matrix_vector_multiply(M, b_k), b_k)
        eigenvector = b_k
        return eigenvalue, eigenvector

    def subtract(self, A, B):
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def outer_product(self, v1, v2, scalar):
        return [[v1[i] * v2[j] * scalar for j in range(len(v2))] for i in range(len(v1))]

    def fit(self, X):
        mean_value = self.mean(X)
        print("Mean found")
        X_centered = self.center_data(X, mean_value)
        print("Centered data found")
        cov_matrix = self.covariance_matrix(X_centered)
        print("Covariance matrix found")
        eigenvalues, eigenvectors = self.svd(cov_matrix)
        print("SVD found")
        sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)

        sum_of_variances = sum(eigenvalues)
        for i, value in enumerate(eigenvalues):
            percent = value / sum_of_variances * 100
            print(f'PC {i + 1} (of all): {percent:.2f}%')

        self.components_ = [eigenvectors[i] for i in sorted_indices[:self.n_components]]
        self.explained_variance_ = [eigenvalues[i] for i in sorted_indices[:self.n_components]]
        return self

    def transform(self, X):
        mean_value = self.mean(X)
        X_centered = self.center_data(X, mean_value)
        return self.multiply(X_centered, self.transpose(self.components_))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def custom_pca(scaled_data, n_components=2):
    pca = PCA_SVD(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    explained_variances = pca.explained_variance_
    return principal_components, explained_variances

def list_shape(lst):
    if isinstance(lst, list):
        if not lst:
            return (0,)
        elif isinstance(lst[0], list):
            return (len(lst), len(lst[0]))
        else:
            return (len(lst),)
    else:
        return ()