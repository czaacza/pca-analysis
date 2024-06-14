class PCA_SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        self.X_centered_ = None

    def mean(self, X):
        return [sum(col) / len(col) for col in zip(*X)]

    def center_data(self, X, mean):
        return [[x - m for x, m in zip(row, mean)] for row in X]

    def transpose(self, X):
        return list(map(list, zip(*X)))

    def multiply(self, A, B):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    def covariance_matrix(self, X):
        n_samples = len(X)
        X_transposed = self.transpose(X)
        cov_matrix = [[0] * len(X_transposed) for _ in range(len(X_transposed))]
        for i in range(len(X_transposed)):
            for j in range(len(X_transposed)):
                cov_matrix[i][j] = sum(X_transposed[i][k] * X_transposed[j][k] for k in range(n_samples)) / (n_samples - 1)
        return cov_matrix

    def svd(self, A):
        def normalize(v):
            norm = sum(x ** 2 for x in v) ** 0.5
            return [x / norm for x in v]

        def dot_product(v1, v2):
            return sum(x * y for x, y in zip(v1, v2))

        def matrix_vector_multiply(M, v):
            return [sum(M_row[j] * v[j] for j in range(len(v))) for M_row in M]

        def power_iteration(M, num_simulations):
            b_k = [1.0] * len(M)
            for _ in range(num_simulations):
                b_k = matrix_vector_multiply(M, b_k)
                b_k = normalize(b_k)
            eigenvalue = dot_product(matrix_vector_multiply(M, b_k), b_k)
            eigenvector = b_k
            return eigenvalue, eigenvector

        eigenvalues, eigenvectors = [], []
        for _ in range(len(A)):
            eigenvalue, eigenvector = power_iteration(A, 100)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            A = self.subtract(A, self.outer_product(eigenvector, eigenvector, eigenvalue))
        return eigenvalues, eigenvectors

    def subtract(self, A, B):
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def outer_product(self, v1, v2, scalar):
        return [[v1[i] * v2[j] * scalar for j in range(len(v2))] for i in range(len(v1))]

    def fit(self, X):
        self.mean_ = self.mean(X)
        self.X_centered_ = self.center_data(X, self.mean_)
        cov_matrix = self.covariance_matrix(self.X_centered_)
        eigenvalues, eigenvectors = self.svd(cov_matrix)
        sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)

        self.components_ = [eigenvectors[i] for i in sorted_indices[:self.n_components]]
        self.explained_variance_ = [eigenvalues[i] for i in sorted_indices[:self.n_components]]
        return self

    def transform(self):
        return self.multiply(self.X_centered_, self.transpose(self.components_))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform()

def custom_pca(scaled_data, n_components=2):
    pca = PCA_SVD(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    explained_variances = pca.explained_variance_
    return principal_components, explained_variances