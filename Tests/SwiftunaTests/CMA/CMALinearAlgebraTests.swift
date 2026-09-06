//
//  CMALinearAlgebraTests.swift
//  SwiftunaTests
//

import Testing

@testable import Swiftuna

@Suite("CMA-ES Linear Algebra & Eigensystem Tests")
struct CMALinearAlgebraTests {

    @Test("FlatMatrix in-place outer product and vector multiplication")
    func testFlatMatrixOperations() {
        var mat = FlatMatrix.identity(dim: 3)
        #expect(mat.trace == 3.0)

        // Multiply identity by vector
        let v = [1.0, 2.0, 3.0]
        let y = mat.multiply(vec: v)
        #expect(y == v)

        // Add outer product v * v^T with multiplier 2.0
        mat.addOuterProduct(vec: v, multiplier: 2.0)

        // mat[0, 0] was 1.0, now 1.0 + 2.0 * 1.0 * 1.0 = 3.0
        #expect(abs(mat[0, 0] - 3.0) < 1e-12)
        // mat[0, 1] was 0.0, now 0.0 + 2.0 * 1.0 * 2.0 = 4.0
        #expect(abs(mat[0, 1] - 4.0) < 1e-12)
        // mat[1, 2] was 0.0, now 0.0 + 2.0 * 2.0 * 3.0 = 12.0
        #expect(abs(mat[1, 2] - 12.0) < 1e-12)

        // Check quadratic form v^T M v
        let qf = mat.quadraticForm(vec: v)
        // v^T I v = 1 + 4 + 9 = 14
        // v^T (2 v v^T) v = 2 * (14)^2 = 392
        // Total = 406
        #expect(abs(qf - 406.0) < 1e-12)
    }

    @Test("Cyclic Jacobi eigenvalue decomposition accurately reconstructs matrix")
    func testJacobiDecomposition() {
        let dim = 3
        var mat = FlatMatrix(dim: dim)
        // Symmetric positive definite matrix
        // [ 4  1  2 ]
        // [ 1  5  3 ]
        // [ 2  3  6 ]
        mat[0, 0] = 4.0
        mat[0, 1] = 1.0
        mat[0, 2] = 2.0
        mat[1, 0] = 1.0
        mat[1, 1] = 5.0
        mat[1, 2] = 3.0
        mat[2, 0] = 2.0
        mat[2, 1] = 3.0
        mat[2, 2] = 6.0

        var jacobi = JacobiEigensystem(dim: dim)
        jacobi.decompose(matrix: mat)

        // 1. Eigenvalues must be positive and strictly sorted
        #expect(jacobi.eigenvalues.count == 3)
        #expect(jacobi.eigenvalues[0] < jacobi.eigenvalues[1])
        #expect(jacobi.eigenvalues[1] < jacobi.eigenvalues[2])
        for val in jacobi.eigenvalues {
            #expect(val > 0.0)
        }

        // 2. Eigenvectors B must be orthogonal: B^T * B == I
        for i in 0..<dim {
            for j in 0..<dim {
                var dot = 0.0
                for r in 0..<dim {
                    dot += jacobi.B[r, i] * jacobi.B[r, j]
                }
                let expected = (i == j) ? 1.0 : 0.0
                #expect(abs(dot - expected) < 1e-10, "B is not orthogonal at (\(i), \(j))")
            }
        }

        // 3. Reconstruction check: B * diag(D^2) * B^T == mat
        var reconstructed = FlatMatrix(dim: dim)
        for i in 0..<dim {
            for j in 0..<dim {
                var sum = 0.0
                for k in 0..<dim {
                    sum += jacobi.B[i, k] * jacobi.eigenvalues[k] * jacobi.B[j, k]
                }
                reconstructed[i, j] = sum
            }
        }

        for i in 0..<dim {
            for j in 0..<dim {
                let diff = abs(reconstructed[i, j] - mat[i, j])
                #expect(diff < 1e-10, "Reconstruction mismatch at (\(i), \(j)): diff=\(diff)")
            }
        }
    }

    @Test("NumPy MT19937 PRNG reproduces NumPy RandomState exactly")
    func testNumpyPRNG() {
        var rng = NumpyMT19937PRNG(seed: 42)
        let expected = [
            0.4967141530112327,
            -0.13826430117118466,
            0.6476885381006925,
            1.5230298564080254,
            -0.23415337472333597,
        ]

        for (idx, expVal) in expected.enumerated() {
            let actual = rng.nextGaussian()
            let diff = abs(actual - expVal)
            #expect(diff < 1e-12, "PRNG mismatch at sample \(idx): actual=\(actual), expected=\(expVal)")
        }
    }
}
