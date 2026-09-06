/// In-place Cyclic Jacobi eigendecomposition for real symmetric positive-definite matrices.
///
/// Factorizes a symmetric matrix $C \in \mathbb{R}^{D \times D}$ into:
/// $$C = B D^2 B^T$$
/// where $B$ is an orthogonal matrix of eigenvectors (columns), and $D$ is the diagonal matrix
/// of standard deviations (square root of eigenvalues).
///
/// Operates directly on contiguous memory with zero heap allocations during the iterative sweeps,
/// providing a pure-Swift, cross-platform eigenvalue solver independent of LAPACK.
package struct JacobiEigensystem: Sendable {
    package let dim: Int
    /// Orthogonal eigenvector matrix $B$ (each column is an eigenvector).
    package var B: FlatMatrix
    /// Standard deviations $D = \sqrt{\lambda_i}$.
    package var D: [Double]
    /// Eigenvalues $\lambda_i = D_i^2$.
    package var eigenvalues: [Double]

    package init(dim: Int) {
        self.dim = dim
        self.B = FlatMatrix.identity(dim: dim)
        self.D = [Double](repeating: 1.0, count: dim)
        self.eigenvalues = [Double](repeating: 1.0, count: dim)
    }

    /// Decomposes the symmetric positive-definite matrix $C$ into $B$ and $D$.
    ///
    /// - Parameters:
    ///   - matrix: Symmetric matrix to factorize.
    ///   - eps: Numerical floor for eigenvalues to guarantee positive-definiteness ($D_i \ge \sqrt{\text{eps}}$).
    ///   - maxSweeps: Maximum full cyclic Jacobi sweeps (default: 50).
    package mutating func decompose(
        matrix: FlatMatrix,
        eps: Double = 1e-12,
        maxSweeps: Int = 50
    ) {
        precondition(matrix.dim == dim, "Matrix dimension must match eigensystem dimension")

        // Work on a copy of the input matrix
        var A = matrix
        A.enforceSymmetry()

        // Initialize eigenvector matrix B as identity
        B = FlatMatrix.identity(dim: dim)

        let n = dim
        if n == 1 {
            let val = max(eps, A[0, 0])
            eigenvalues[0] = val
            D[0] = val.squareRoot()
            B[0, 0] = 1.0
            return
        }

        // Cyclic Jacobi sweeps
        for _ in 0..<maxSweeps {
            // Compute norm of off-diagonal elements
            var offDiagSum = 0.0
            for p in 0..<n {
                let rowP = p * n
                for q in (p + 1)..<n {
                    offDiagSum += abs(A.buffer[rowP + q])
                }
            }

            if offDiagSum < 1e-15 {
                break
            }

            let threshold = offDiagSum / Double(n * (n - 1))

            for p in 0..<n {
                for q in (p + 1)..<n {
                    let apq = A[p, q]
                    if abs(apq) <= threshold * 1e-3 && abs(apq) < 1e-15 {
                        A[p, q] = 0.0
                        A[q, p] = 0.0
                        continue
                    }

                    let app = A[p, p]
                    let aqq = A[q, q]
                    let theta = (aqq - app) / (2.0 * apq)

                    let t: Double
                    if theta >= 0.0 {
                        t = 1.0 / (theta + (theta * theta + 1.0).squareRoot())
                    } else {
                        t = -1.0 / (-theta + (theta * theta + 1.0).squareRoot())
                    }

                    let c = 1.0 / (t * t + 1.0).squareRoot()
                    let s = t * c
                    let tau = s / (1.0 + c)

                    // Update A_pp and A_qq
                    A[p, p] = app - t * apq
                    A[q, q] = aqq + t * apq
                    A[p, q] = 0.0
                    A[q, p] = 0.0

                    // Update remaining elements of A in rows/cols p and q
                    for r in 0..<n {
                        if r != p && r != q {
                            let arp = A[r, p]
                            let arq = A[r, q]
                            A[r, p] = arp - s * (arq + tau * arp)
                            A[p, r] = A[r, p]
                            A[r, q] = arq + s * (arp - tau * arq)
                            A[q, r] = A[r, q]
                        }
                    }

                    // Update eigenvector columns in B
                    for r in 0..<n {
                        let brp = B[r, p]
                        let brq = B[r, q]
                        B[r, p] = brp - s * (brq + tau * brp)
                        B[r, q] = brq + s * (brp - tau * brq)
                    }
                }
            }
        }

        // Extract eigenvalues and enforce positive-definiteness
        for i in 0..<n {
            let rawVal = A[i, i]
            let clampedVal = rawVal < eps ? eps : rawVal
            eigenvalues[i] = clampedVal
            D[i] = clampedVal.squareRoot()
        }

        // Sort eigenvalues in ascending order matching NumPy's eigh convention
        var indices = Array(0..<n)
        indices.sort { eigenvalues[$0] < eigenvalues[$1] }

        var sortedEigenvalues = [Double](repeating: 0.0, count: n)
        var sortedD = [Double](repeating: 0.0, count: n)
        var sortedB = FlatMatrix(dim: n)

        for (newIdx, oldIdx) in indices.enumerated() {
            sortedEigenvalues[newIdx] = eigenvalues[oldIdx]
            sortedD[newIdx] = D[oldIdx]
            for r in 0..<n {
                sortedB[r, newIdx] = B[r, oldIdx]
            }
        }

        self.eigenvalues = sortedEigenvalues
        self.D = sortedD
        self.B = sortedB
    }
}
