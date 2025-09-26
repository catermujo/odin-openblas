package openblas

import lapack "./f77"

// ===================================================================================
// POSITIVE DEFINITE MATRIX EQUILIBRATION
// ===================================================================================

// Standard equilibration proc group
m_equilibrate_positive_definite :: proc {
	m_equilibrate_positive_definite_f32_c64,
	m_equilibrate_positive_definite_f64_c128,
}

// Equilibration with balancing proc group
m_equilibrate_positive_definite_balanced :: proc {
	m_equilibrate_positive_definite_balanced_f32_c64,
	m_equilibrate_positive_definite_balanced_f64_c128,
}

apply_positive_definite_scaling :: proc {
	apply_positive_definite_scaling_f32_c64,
	apply_positive_definite_scaling_f64_c128,
}

remove_positive_definite_scaling :: proc {
	remove_positive_definite_scaling_f32_c64,
	remove_positive_definite_scaling_f64_c128,
}

apply_vector_scaling :: proc {
	apply_vector_scaling_f32_c64,
	apply_vector_scaling_f64_c128,
}

// ===================================================================================
// STANDARD EQUILIBRATION IMPLEMENTATION
// ===================================================================================

// Compute equilibration scaling factors for positive definite matrix (f32/complex64)
// Computes diagonal scaling S such that S*A*S has unit diagonal
m_equilibrate_positive_definite_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f32, // Pre-allocated scaling factors (size n)
) -> (
	scond: f32,
	amax: f32,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := A.rows
	lda := A.stride

	when T == f32 {
		lapack.spoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex64 {
		lapack.cpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// Compute equilibration scaling factors for positive definite matrix (f64/complex128)
// Computes diagonal scaling S such that S*A*S has unit diagonal
m_equilibrate_positive_definite_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f64, // Pre-allocated scaling factors (size n)
) -> (
	scond: f64,
	amax: f64,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := A.rows
	lda := A.stride

	when T == f64 {
		lapack.dpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex128 {
		lapack.zpoequ_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// ===================================================================================
// BALANCED EQUILIBRATION IMPLEMENTATION
// ===================================================================================

// Compute balanced equilibration scaling factors for positive definite matrix (f32/complex64)
// Uses improved algorithm for better numerical stability
m_equilibrate_positive_definite_balanced_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f32, // Pre-allocated scaling factors (size n)
) -> (
	scond: f32,
	amax: f32,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f32 || T == complex64 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := A.rows
	lda := A.stride

	when T == f32 {
		lapack.spoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex64 {
		lapack.cpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// Compute balanced equilibration scaling factors for positive definite matrix (f64/complex128)
// Uses improved algorithm for better numerical stability
m_equilibrate_positive_definite_balanced_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix
	S: []f64, // Pre-allocated scaling factors (size n)
) -> (
	scond: f64,
	amax: f64,
	info: Info,
	ok: bool, // Ratio of smallest to largest scaling factor// Maximum absolute element value
) where T == f64 || T == complex128 {
	// Validate inputs
	assert(A.rows == A.cols, "Matrix must be square")
	assert(len(S) >= A.rows, "S array too small")

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)

	when T == f64 {
		lapack.dpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	} else when T == complex128 {
		lapack.zpoequb_(&n, raw_data(A.data), &lda, raw_data(S), &scond, &amax, &info)
	}

	return scond, amax, info, info == 0
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if equilibration is recommended based on condition scale
is_equilibration_needed :: proc(scond: $T) -> bool where is_float(T) {
	// Matrix is considered equilibrated if condition of scale factors is not too bad
	return scond < 0.1 || 1.0 / scond < 0.1
}

// Check if balanced equilibration is recommended based on condition scale
is_balanced_equilibration_needed :: proc(scond: $T) -> bool where is_float(T) {
	// Balanced equilibration has stricter criteria
	return scond < 0.25 || 1.0 / scond < 0.25
}

// Apply equilibration scaling to matrix (f32/complex64)
// Applies D*A*D scaling where D = diag(S)
// For symmetric positive definite matrices, only need to scale upper or lower triangle
apply_positive_definite_scaling_f32_c64 :: proc(
	A: ^Matrix($T),
	S: []f32,
	uplo := MatrixRegion.Upper,
) where T == f32 ||
	T == complex64 {
	assert(len(S) == A.rows && A.rows == A.cols, "Invalid scaling factors or non-square matrix")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	// Two approaches possible:
	// 1. Column-wise then row-wise BLAS scal (2n BLAS calls)
	// 2. Direct element-wise scaling (1 pass through matrix)
	//
	// For symmetric matrices, direct scaling is likely more efficient:
	// - Only processes half the elements (n*(n+1)/2)
	// - Better cache locality (single pass)
	// - Avoids BLAS call overhead for small matrices
	//
	// For very large matrices, column/row BLAS approach might win due to optimized BLAS,
	// but equilibration is typically done once before factorization, so not critical path.

	// Apply scaling: A_scaled = D * A * D where D = diag(S)
	// For symmetric matrices, only scale the stored triangle
	if uplo == .Upper {
		for i in 0 ..< A.rows {
			for j in i ..< A.cols {
				val := matrix_get(A, i, j)
				scaled_val: T
				when T == f32 {
					scaled_val = val * S[i] * S[j]
				} else when T == complex64 {
					scale := S[i] * S[j]
					scaled_val = complex(real(val) * scale, imag(val) * scale)
				}
				matrix_set(A, i, j, scaled_val)
				// For symmetric matrix, also set the lower triangle if not on diagonal
				if i != j {
					matrix_set(A, j, i, scaled_val)
				}
			}
		}
	} else {
		for i in 0 ..< A.rows {
			for j in 0 ..= i {
				val := matrix_get(A, i, j)
				scaled_val: T
				when T == f32 {
					scaled_val = val * S[i] * S[j]
				} else when T == complex64 {
					scale := S[i] * S[j]
					scaled_val = complex(real(val) * scale, imag(val) * scale)
				}
				matrix_set(A, i, j, scaled_val)
				// For symmetric matrix, also set the upper triangle if not on diagonal
				if i != j {
					matrix_set(A, j, i, scaled_val)
				}
			}
		}
	}
}

// Apply equilibration scaling to matrix (f64/complex128)
// Applies D*A*D scaling where D = diag(S)
// For symmetric positive definite matrices, only need to scale upper or lower triangle
apply_positive_definite_scaling_f64_c128 :: proc(
	A: ^Matrix($T),
	S: []f64,
	uplo := MatrixRegion.Upper,
) where T == f64 ||
	T == complex128 {
	assert(len(S) == A.rows && A.rows == A.cols, "Invalid scaling factors or non-square matrix")
	assert(uplo == .Upper || uplo == .Lower, "uplo must be Upper or Lower")

	// Apply scaling: A_scaled = D * A * D where D = diag(S)
	// For symmetric matrices, only scale the stored triangle
	if uplo == .Upper {
		for i in 0 ..< A.rows {
			for j in i ..< A.cols {
				val := matrix_get(A, i, j)
				scaled_val: T
				when T == f64 {
					scaled_val = val * S[i] * S[j]
				} else when T == complex128 {
					scale := S[i] * S[j]
					scaled_val = complex(real(val) * scale, imag(val) * scale)
				}
				matrix_set(A, i, j, scaled_val)
				// For symmetric matrix, also set the lower triangle if not on diagonal
				if i != j {
					matrix_set(A, j, i, scaled_val)
				}
			}
		}
	} else {
		for i in 0 ..< A.rows {
			for j in 0 ..= i {
				val := matrix_get(A, i, j)
				scaled_val: T
				when T == f64 {
					scaled_val = val * S[i] * S[j]
				} else when T == complex128 {
					scale := S[i] * S[j]
					scaled_val = complex(real(val) * scale, imag(val) * scale)
				}
				matrix_set(A, i, j, scaled_val)
				// For symmetric matrix, also set the upper triangle if not on diagonal
				if i != j {
					matrix_set(A, j, i, scaled_val)
				}
			}
		}
	}
}

// Remove equilibration scaling from solution vector (f32/complex64)
// Note: For equilibration, this applies x_original = D * x_scaled
// where the solution x was computed with the scaled system
remove_positive_definite_scaling_f32_c64 :: proc(
	x: ^Vector($T),
	S: []f32,
) where T == f32 ||
	T == complex64 {
	assert(len(S) == x.len, "Invalid scaling factors")

	// Apply element-wise scaling: x[i] = S[i] * x[i]
	// Manual loop is needed since BLAS scal applies uniform scaling
	for i in 0 ..< x.len {
		val := vector_get(x, i)
		when T == f32 {
			vector_set(x, i, val * S[i])
		} else when T == complex64 {
			vector_set(x, i, complex(real(val) * S[i], imag(val) * S[i]))
		}
	}
}

// Remove equilibration scaling from solution vector (f64/complex128)
// Note: For equilibration, this applies x_original = D * x_scaled
// where the solution x was computed with the scaled system
remove_positive_definite_scaling_f64_c128 :: proc(
	x: ^Vector($T),
	S: []f64,
) where T == f64 ||
	T == complex128 {
	assert(len(S) == x.len, "Invalid scaling factors")

	// Apply element-wise scaling: x[i] = S[i] * x[i]
	// Manual loop is needed since BLAS scal applies uniform scaling
	for i in 0 ..< x.len {
		val := vector_get(x, i)
		when T == f64 {
			vector_set(x, i, val * S[i])
		} else when T == complex128 {
			vector_set(x, i, complex(real(val) * S[i], imag(val) * S[i]))
		}
	}
}


// Apply equilibration scaling to vector (f32/complex64)
apply_vector_scaling_f32_c64 :: proc(v: ^Vector($T), S: []f32) where T == f32 || T == complex64 {
	assert(len(S) >= v.len, "S array too small")
	for i in 0 ..< v.len {
		val := vector_get(v, i)
		when T == f32 {
			vector_set(v, i, val * S[i])
		} else when T == complex64 {
			vector_set(v, i, complex(real(val) * S[i], imag(val) * S[i]))
		}
	}
}

// Apply equilibration scaling to vector (f64/complex128)
apply_vector_scaling_f64_c128 :: proc(v: ^Vector($T), S: []f64) where T == f64 || T == complex128 {
	assert(len(S) >= v.len, "S array too small")
	for i in 0 ..< v.len {
		val := vector_get(v, i)
		when T == f64 {
			vector_set(v, i, val * S[i])
		} else when T == complex128 {
			vector_set(v, i, complex(real(val) * S[i], imag(val) * S[i]))
		}
	}
}
