/// A dense $N \times N$ square matrix backed by a single contiguous array buffer.
///
/// Designed specifically for CMA-ES covariance and eigenbasis computations to guarantee
/// zero heap churn and SIMD-friendly contiguous memory access.
package struct FlatMatrix: Sendable, Equatable, Codable {
    package let dim: Int
    package var buffer: ContiguousArray<Double>

    package init(dim: Int, initialValue: Double = 0.0) {
        self.dim = dim
        self.buffer = ContiguousArray(repeating: initialValue, count: dim * dim)
    }

    package init(dim: Int, buffer: ContiguousArray<Double>) {
        precondition(buffer.count == dim * dim, "Buffer count must match dim * dim")
        self.dim = dim
        self.buffer = buffer
    }

    /// Creates an identity matrix $I_N$.
    package static func identity(dim: Int) -> FlatMatrix {
        var mat = FlatMatrix(dim: dim, initialValue: 0.0)
        for i in 0..<dim {
            mat[i, i] = 1.0
        }
        return mat
    }

    /// Creates a zero matrix.
    package static func zeros(dim: Int) -> FlatMatrix {
        FlatMatrix(dim: dim, initialValue: 0.0)
    }

    @inlinable
    package subscript(r: Int, c: Int) -> Double {
        get {
            buffer[r * dim + c]
        }
        set {
            buffer[r * dim + c] = newValue
        }
    }

    /// In-place scalar multiplication: $M \leftarrow \alpha M$.
    @inlinable
    package mutating func scale(_ scalar: Double) {
        for i in 0..<buffer.count {
            buffer[i] *= scalar
        }
    }

    /// In-place accumulation of an outer product: $M \leftarrow M + \alpha (v v^T)$.
    @inlinable
    package mutating func addOuterProduct(vec: [Double], multiplier: Double) {
        precondition(vec.count == dim, "Vector dimension must match matrix dimension")
        for i in 0..<dim {
            let vi = vec[i]
            let rowOffset = i * dim
            let scaleVi = multiplier * vi
            for j in 0..<dim {
                buffer[rowOffset + j] += scaleVi * vec[j]
            }
        }
    }

    /// In-place symmetric outer product accumulation: $M \leftarrow M + \alpha (v v^T)$.
    /// Exploits symmetry $M[i, j] = M[j, i]$.
    @inlinable
    package mutating func addOuterProductSymmetric(vec: [Double], multiplier: Double) {
        precondition(vec.count == dim, "Vector dimension must match matrix dimension")
        for i in 0..<dim {
            let vi = vec[i]
            let rowOffset = i * dim
            let scaleVi = multiplier * vi
            for j in i..<dim {
                let val = scaleVi * vec[j]
                buffer[rowOffset + j] += val
                if i != j {
                    buffer[j * dim + i] += val
                }
            }
        }
    }

    /// In-place symmetry enforcement: $M \leftarrow \frac{1}{2}(M + M^T)$.
    @inlinable
    package mutating func enforceSymmetry() {
        for i in 0..<dim {
            let rowI = i * dim
            for j in (i + 1)..<dim {
                let rowJ = j * dim
                let avg = (buffer[rowI + j] + buffer[rowJ + i]) * 0.5
                buffer[rowI + j] = avg
                buffer[rowJ + i] = avg
            }
        }
    }

    /// Matrix-vector multiplication: $y = M x$.
    @inlinable
    package func multiply(vec: [Double]) -> [Double] {
        precondition(vec.count == dim, "Vector dimension must match matrix dimension")
        var result = [Double](repeating: 0.0, count: dim)
        for i in 0..<dim {
            var sum = 0.0
            let rowOffset = i * dim
            for j in 0..<dim {
                sum += buffer[rowOffset + j] * vec[j]
            }
            result[i] = sum
        }
        return result
    }

    /// Transposed matrix-vector multiplication: $y = M^T x$.
    @inlinable
    package func multiplyTransposed(vec: [Double]) -> [Double] {
        precondition(vec.count == dim, "Vector dimension must match matrix dimension")
        var result = [Double](repeating: 0.0, count: dim)
        for j in 0..<dim {
            let xj = vec[j]
            let rowOffset = j * dim
            for i in 0..<dim {
                result[i] += buffer[rowOffset + i] * xj
            }
        }
        return result
    }

    /// Computes the quadratic form $v^T M v$.
    @inlinable
    package func quadraticForm(vec: [Double]) -> Double {
        precondition(vec.count == dim, "Vector dimension must match matrix dimension")
        var sum = 0.0
        for i in 0..<dim {
            let vi = vec[i]
            let rowOffset = i * dim
            for j in 0..<dim {
                sum += vi * buffer[rowOffset + j] * vec[j]
            }
        }
        return sum
    }

    /// The trace of the matrix (sum of diagonal elements).
    package var trace: Double {
        var sum = 0.0
        for i in 0..<dim {
            sum += self[i, i]
        }
        return sum
    }
}
