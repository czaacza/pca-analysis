from mpi4py import MPI
import math

class PCA_SVD_MPI:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        self.X_centered_ = None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def chunk_indices(self, n, n_chunks):
        chunk_size = (n + n_chunks - 1) // n_chunks
        for i in range(0, n, chunk_size):
            yield range(i, min(i + chunk_size, n))

    def process_chunk_covariance(self, X_transposed, chunk, n_samples):
        partial_cov_matrix = [[0] * len(X_transposed) for _ in range(len(X_transposed))]
        for i in chunk:
            for j in range(len(X_transposed)):
                partial_cov_matrix[i][j] = sum(X_transposed[i][k] * X_transposed[j][k] for k in range(n_samples)) / (n_samples - 1)
        return partial_cov_matrix

    def custom_pca_mpi(self, X):
        def transpose(matrix):
            return list(map(list, zip(*matrix)))

        def perform_svd(matrix):
            def normalize(v):
                norm = math.sqrt(sum(x ** 2 for x in v))
                return [x / norm for x in v]

            def dot_product(v1, v2):
                return sum(x * y for x, y in zip(v1, v2))

            def matrix_vector_multiply(M, v):
                return [sum(M_row[j] * v[j] for j in range(len(v))) for M_row in M]

            def subtract(A, B):
                return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

            def outer_product(v1, v2, scalar):
                return [[v1[i] * v2[j] * scalar for j in range(len(v2))] for i in range(len(v1))]

            def power_iteration(M, num_simulations):
                b_k = [1.0] * len(M)
                for _ in range(num_simulations):
                    b_k = matrix_vector_multiply(M, b_k)
                    b_k = normalize(b_k)
                eigenvalue = dot_product(matrix_vector_multiply(M, b_k), b_k)
                eigenvector = b_k
                return eigenvalue, eigenvector

            eigenvalues, eigenvectors = [], []
            for _ in range(len(matrix)):
                eigenvalue, eigenvector = power_iteration(matrix, 100)
                eigenvalues.append(eigenvalue)
                eigenvectors.append(eigenvector)
                matrix = subtract(matrix, outer_product(eigenvector, eigenvector, eigenvalue))
            return eigenvalues, eigenvectors

        if self.rank == 0:
            self.mean_ = [sum(col) / len(col) for col in zip(*X)]
        else:
            self.mean_ = None
        self.mean_ = self.comm.bcast(self.mean_, root=0)

        if self.rank == 0:
            chunks = [X[i::self.size] for i in range(self.size)]
        else:
            chunks = None
        local_chunk = self.comm.scatter(chunks, root=0)

        X_centered_local = [[x - m for x, m in zip(row, self.mean_)] for row in local_chunk]
        gathered_centered_data = self.comm.gather(X_centered_local, root=0)

        if self.rank == 0:
            X_centered = [item for sublist in gathered_centered_data for item in sublist]
            X_transposed = transpose(X_centered)
            n_samples = len(X_centered)
        else:
            X_transposed = None
            n_samples = None

        n_samples = self.comm.bcast(n_samples, root=0)
        X_transposed = self.comm.bcast(X_transposed, root=0)

        chunks = list(self.chunk_indices(len(X_transposed), self.size))
        local_cov_matrix = self.process_chunk_covariance(X_transposed, chunks[self.rank], n_samples)
        gathered_cov_matrices = self.comm.gather(local_cov_matrix, root=0)

        if self.rank == 0:
            cov_matrix_size = len(X_transposed)
            cov_matrix = [[0] * cov_matrix_size for _ in range(cov_matrix_size)]
            for partial_cov_matrix in gathered_cov_matrices:
                for i in range(cov_matrix_size):
                    for j in range(cov_matrix_size):
                        cov_matrix[i][j] += partial_cov_matrix[i][j]

            for i in range(cov_matrix_size):
                for j in range(cov_matrix_size):
                    cov_matrix[i][j] /= (n_samples - 1)

            eigenvalues, eigenvectors = perform_svd(cov_matrix)
            sorted_indices = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
            self.components_ = [eigenvectors[i] for i in sorted_indices[:self.n_components]]
            self.explained_variance_ = [eigenvalues[i] for i in sorted_indices[:self.n_components]]
        else:
            self.components_ = None
            self.explained_variance_ = None

        self.components_ = self.comm.bcast(self.components_, root=0)
        self.explained_variance_ = self.comm.bcast(self.explained_variance_, root=0)

        if self.rank == 0:
            return self.components_, self.explained_variance_
