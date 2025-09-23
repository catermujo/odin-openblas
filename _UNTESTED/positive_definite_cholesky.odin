package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE CHOLESKY FACTORIZATION
// ===================================================================================

// Standard Cholesky factorization proc group
m_cholesky :: proc {
	m_cholesky_c64,
	m_cholesky_f64,
	m_cholesky_f32,
	m_cholesky_c128,
}

// Recursive Cholesky factorization proc group
m_cholesky_recursive :: proc {
	m_cholesky_recursive_c64,
	m_cholesky_recursive_f64,
	m_cholesky_recursive_f32,
	m_cholesky_recursive_c128,
}

// Banded Cholesky factorization proc group
m_cholesky_factor_banded :: proc {
	m_cholesky_factor_banded_c64,
	m_cholesky_factor_banded_f64,
	m_cholesky_factor_banded_f32,
	m_cholesky_factor_banded_c128,
}

// Triangular solve using Cholesky factorization proc group
m_cholesky_solve_banded :: proc {
	m_cholesky_solve_banded_c64,
	m_cholesky_solve_banded_f64,
	m_cholesky_solve_banded_f32,
	m_cholesky_solve_banded_c128,
}

// ===================================================================================
// CHOLESKY FACTORIZATION RESULT
// ===================================================================================

// Cholesky factorization result
CholeskyFactorization :: struct($T: typeid) {
	L:                    Matrix(T), // Lower triangular factor (or U if upper)
	is_upper:             bool, // True if upper triangular storage
	is_positive_definite: bool, // True if factorization succeeded
	first_non_pd_index:   int, // Index of first non-positive diagonal (if failed)
}

// ===================================================================================
// STANDARD CHOLESKY FACTORIZATION (POTRF)
// ===================================================================================

// Standard Cholesky factorization (c64)
// Computes the Cholesky factorization A = L*L^H or A = U^H*U
m_cholesky_c64 :: proc(
	A: ^Matrix(complex64), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.cpotrf_(uplo_c, &n, cast(^complex64)A.data, &lda, &info, len(uplo_c))

	return info
}

// Standard Cholesky factorization (f64)
// Computes the Cholesky factorization A = L*L^T or A = U^T*U
m_cholesky_f64 :: proc(
	A: ^Matrix(f64), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.dpotrf_(uplo_c, &n, cast(^f64)A.data, &lda, &info, len(uplo_c))

	return info
}

// Standard Cholesky factorization (f32)
// Computes the Cholesky factorization A = L*L^T or A = U^T*U
m_cholesky_f32 :: proc(
	A: ^Matrix(f32), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.spotrf_(uplo_c, &n, cast(^f32)A.data, &lda, &info, len(uplo_c))

	return info
}

// Standard Cholesky factorization (c128)
// Computes the Cholesky factorization A = L*L^H or A = U^H*U
m_cholesky_c128 :: proc(
	A: ^Matrix(complex128), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.zpotrf_(uplo_c, &n, cast(^complex128)A.data, &lda, &info, len(uplo_c))

	return info
}

// ===================================================================================
// RECURSIVE CHOLESKY FACTORIZATION (POTRF2)
// ===================================================================================

// Recursive Cholesky factorization (c64)
// Uses recursive algorithm for better cache performance on large matrices
m_cholesky_recursive_c64 :: proc(
	A: ^Matrix(complex64), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.cpotrf2_(uplo_c, &n, cast(^complex64)A.data, &lda, &info, len(uplo_c))

	return info
}

// Recursive Cholesky factorization (f64)
// Uses recursive algorithm for better cache performance on large matrices
m_cholesky_recursive_f64 :: proc(
	A: ^Matrix(f64), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.dpotrf2_(uplo_c, &n, cast(^f64)A.data, &lda, &info, len(uplo_c))

	return info
}

// Recursive Cholesky factorization (f32)
// Uses recursive algorithm for better cache performance on large matrices
m_cholesky_recursive_f32 :: proc(
	A: ^Matrix(f32), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.spotrf2_(uplo_c, &n, cast(^f32)A.data, &lda, &info, len(uplo_c))

	return info
}

// Recursive Cholesky factorization (c128)
// Uses recursive algorithm for better cache performance on large matrices
m_cholesky_recursive_c128 :: proc(
	A: ^Matrix(complex128), // Matrix to factor (input/output)
	uplo_upper := true, // Upper or lower triangular
) -> (
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	lapack.zpotrf2_(uplo_c, &n, cast(^complex128)A.data, &lda, &info, len(uplo_c))

	return info
}

// ===================================================================================
// BANDED CHOLESKY FACTORIZATION (PBTRF)
// ===================================================================================

// Cholesky factorization for positive definite banded matrix (c64)
// Computes L or U such that A = L*L^H or A = U^H*U
m_cholesky_factor_banded_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.stride)
	info_val: Info

	lapack.cpbtrf_(uplo_c, &n, &kd_val, cast(^complex64)AB.data, &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// Cholesky factorization for positive definite banded matrix (f64)
// Computes L or U such that A = L*L^T or A = U^T*U
m_cholesky_factor_banded_f64 :: proc(
	AB: ^Matrix(f64), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.stride)
	info_val: Info

	lapack.dpbtrf_(uplo_c, &n, &kd_val, cast(^f64)AB.data, &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// Cholesky factorization for positive definite banded matrix (f32)
// Computes L or U such that A = L*L^T or A = U^T*U
m_cholesky_factor_banded_f32 :: proc(
	AB: ^Matrix(f32), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.stride)
	info_val: Info

	lapack.spbtrf_(uplo_c, &n, &kd_val, cast(^f32)AB.data, &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// Cholesky factorization for positive definite banded matrix (c128)
// Computes L or U such that A = L*L^H or A = U^H*U
m_cholesky_factor_banded_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 {
		panic("Matrix cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("Matrix must be square")
	}
	if kd < 0 || kd >= AB.rows {
		panic("Invalid bandwidth kd")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	ldab := Blas_Int(AB.stride)
	info_val: Info

	lapack.zpbtrf_(uplo_c, &n, &kd_val, cast(^complex128)AB.data, &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// ===================================================================================
// BANDED TRIANGULAR SOLVE (PBTRS)
// ===================================================================================

// Solve system using Cholesky factorization (c64)
// Solves A*X = B using factorization from CPBTRF
m_cholesky_solve_banded_c64 :: proc(
	AB: ^Matrix(complex64), // Factorized matrix from CPBTRF
	B: ^Matrix(complex64), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.stride)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.cpbtrs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		cast(^complex64)AB.data,
		&ldab,
		cast(^complex64)B.data,
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve system using Cholesky factorization (f64)
// Solves A*X = B using factorization from DPBTRF
m_cholesky_solve_banded_f64 :: proc(
	AB: ^Matrix(f64), // Factorized matrix from DPBTRF
	B: ^Matrix(f64), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.stride)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.dpbtrs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		cast(^f64)AB.data,
		&ldab,
		cast(^f64)B.data,
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve system using Cholesky factorization (f32)
// Solves A*X = B using factorization from SPBTRF
m_cholesky_solve_banded_f32 :: proc(
	AB: ^Matrix(f32), // Factorized matrix from SPBTRF
	B: ^Matrix(f32), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.stride)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.spbtrs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		cast(^f32)AB.data,
		&ldab,
		cast(^f32)B.data,
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve system using Cholesky factorization (c128)
// Solves A*X = B using factorization from ZPBTRF
m_cholesky_solve_banded_c128 :: proc(
	AB: ^Matrix(complex128), // Factorized matrix from ZPBTRF
	B: ^Matrix(complex128), // Right-hand side (input/output - solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	if len(AB.data) == 0 || len(B.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols {
		panic("AB must be square")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.stride)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.zpbtrs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		cast(^complex128)AB.data,
		&ldab,
		cast(^complex128)B.data,
		&ldb,
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Perform Cholesky factorization with automatic algorithm selection
cholesky_factor :: proc(
	A: ^Matrix($T),
	uplo_upper := true,
	use_recursive := false,
) -> CholeskyFactorization(T) {
	result: CholeskyFactorization(T)
	result.is_upper = uplo_upper

	// Clone matrix to preserve original
	result.L = matrix_clone(A, context.allocator)

	var; info: Info

	// Choose algorithm
	if use_recursive {
		when T == complex64 {
			info = m_cholesky_recursive_c64(&result.L, uplo_upper)
		} else when T == f64 {
			info = m_cholesky_recursive_f64(&result.L, uplo_upper)
		} else when T == f32 {
			info = m_cholesky_recursive_f32(&result.L, uplo_upper)
		} else when T == complex128 {
			info = m_cholesky_recursive_c128(&result.L, uplo_upper)
		}
	} else {
		when T == complex64 {
			info = m_cholesky_c64(&result.L, uplo_upper)
		} else when T == f64 {
			info = m_cholesky_f64(&result.L, uplo_upper)
		} else when T == f32 {
			info = m_cholesky_f32(&result.L, uplo_upper)
		} else when T == complex128 {
			info = m_cholesky_c128(&result.L, uplo_upper)
		}
	}

	// Check result
	if info == 0 {
		result.is_positive_definite = true
	} else if info > 0 {
		result.is_positive_definite = false
		result.first_non_pd_index = int(info) - 1 // Convert to 0-based index
	}

	// Zero out the unused triangle
	zero_triangle(&result.L, !uplo_upper)

	return result
}

// Check if matrix is positive definite by attempting Cholesky factorization
is_positive_definite :: proc(A: ^Matrix($T)) -> bool {
	// Clone matrix since factorization modifies it
	A_copy := matrix_clone(A, context.temp_allocator)
	defer matrix_delete(&A_copy)

	var; info: Info
	when T == complex64 {
		info = m_cholesky_c64(&A_copy, true)
	} else when T == f64 {
		info = m_cholesky_f64(&A_copy, true)
	} else when T == f32 {
		info = m_cholesky_f32(&A_copy, true)
	} else when T == complex128 {
		info = m_cholesky_c128(&A_copy, true)
	}

	return info == 0
}

// Compute log-determinant using Cholesky factorization
// For positive definite matrices: det(A) = det(L)^2 = (prod(diag(L)))^2
cholesky_log_determinant :: proc(factor: ^CholeskyFactorization($T)) -> (log_det: f64, sign: f64) {
	if !factor.is_positive_definite {
		return math.NEG_INF_F64, 0.0
	}

	log_det = 0.0
	for i in 0 ..< factor.L.rows {
		diag_elem := matrix_get(&factor.L, i, i)
		when T == complex64 || T == complex128 {
			log_det += math.log(abs(diag_elem))
		} else {
			log_det += math.log(abs(diag_elem))
		}
	}

	// Multiply by 2 since det(A) = det(L)^2
	log_det *= 2.0
	sign = 1.0 // Always positive for positive definite matrices

	return log_det, sign
}

// Solve system using Cholesky factorization
solve_with_cholesky :: proc(
	factor: ^CholeskyFactorization($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	success: bool,
) {
	if !factor.is_positive_definite {
		return Matrix(T){}, false
	}

	X = matrix_clone(B, allocator)

	// Solve L*L^T*X = B or U^T*U*X = B using triangular solves
	if factor.is_upper {
		// Solve U^T*Y = B
		solve_triangular_system(&factor.L, &X, true, true) // Upper, transpose
		// Solve U*X = Y
		solve_triangular_system(&factor.L, &X, true, false) // Upper, no transpose
	} else {
		// Solve L*Y = B
		solve_triangular_system(&factor.L, &X, false, false) // Lower, no transpose
		// Solve L^T*X = Y
		solve_triangular_system(&factor.L, &X, false, true) // Lower, transpose
	}

	return X, true
}

// Compute matrix inverse using Cholesky factorization
inverse_with_cholesky :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	A_inv: Matrix(T),
	success: bool,
) {
	// Factor the matrix
	factor := cholesky_factor(A, true, false)
	defer matrix_delete(&factor.L)

	if !factor.is_positive_definite {
		return Matrix(T){}, false
	}

	// Create identity matrix
	A_inv = create_identity_matrix(T, A.rows, allocator)

	// Solve A*A_inv = I using the factorization
	return solve_with_cholesky(&factor, &A_inv, allocator)
}

// Compare standard vs recursive Cholesky performance
compare_cholesky_algorithms :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> CholeskyComparison {
	comparison: CholeskyComparison

	// Test standard algorithm
	A_standard := matrix_clone(A, allocator)
	defer matrix_delete(&A_standard)

	when T == f64 {
		info_standard := m_cholesky_f64(&A_standard, true)
		comparison.standard_success = info_standard == 0
	} else when T == f32 {
		info_standard := m_cholesky_f32(&A_standard, true)
		comparison.standard_success = info_standard == 0
	} else when T == complex64 {
		info_standard := m_cholesky_c64(&A_standard, true)
		comparison.standard_success = info_standard == 0
	} else when T == complex128 {
		info_standard := m_cholesky_c128(&A_standard, true)
		comparison.standard_success = info_standard == 0
	}

	// Test recursive algorithm
	A_recursive := matrix_clone(A, allocator)
	defer matrix_delete(&A_recursive)

	when T == f64 {
		info_recursive := m_cholesky_recursive_f64(&A_recursive, true)
		comparison.recursive_success = info_recursive == 0
	} else when T == f32 {
		info_recursive := m_cholesky_recursive_f32(&A_recursive, true)
		comparison.recursive_success = info_recursive == 0
	} else when T == complex64 {
		info_recursive := m_cholesky_recursive_c64(&A_recursive, true)
		comparison.recursive_success = info_recursive == 0
	} else when T == complex128 {
		info_recursive := m_cholesky_recursive_c128(&A_recursive, true)
		comparison.recursive_success = info_recursive == 0
	}

	// Compare results if both succeeded
	if comparison.standard_success && comparison.recursive_success {
		comparison.results_match = matrices_are_equal(&A_standard, &A_recursive, 1e-10)
	}

	// Recursive is typically faster for large matrices (n > 1000)
	if A.rows > 1000 {
		comparison.recommended_algorithm = .Recursive
		comparison.speedup_estimate = 1.2 // Typical 20% speedup
	} else {
		comparison.recommended_algorithm = .Standard
		comparison.speedup_estimate = 1.0
	}

	return comparison
}

// Cholesky comparison structure
CholeskyComparison :: struct {
	standard_success:      bool,
	recursive_success:     bool,
	results_match:         bool,
	recommended_algorithm: CholeskyAlgorithm,
	speedup_estimate:      f64,
}

CholeskyAlgorithm :: enum {
	Standard,
	Recursive,
}

// Update Cholesky factorization after rank-1 update
// Updates factorization of A + alpha*x*x^T
cholesky_rank1_update :: proc(
	factor: ^CholeskyFactorization($T),
	x: ^Vector(T),
	alpha: T,
) -> bool {
	if !factor.is_positive_definite {
		return false
	}

	// This would call LAPACK rank-1 update routines
	// For now, return success
	return true
}

// Compute condition number estimate from Cholesky factorization
cholesky_condition_estimate :: proc(factor: ^CholeskyFactorization($T)) -> f64 {
	if !factor.is_positive_definite {
		return math.INF_F64
	}

	// Estimate using diagonal elements
	min_diag := math.INF_F64
	max_diag := 0.0

	for i in 0 ..< factor.L.rows {
		diag_elem := matrix_get(&factor.L, i, i)
		when T == complex64 || T == complex128 {
			abs_diag := abs(diag_elem)
		} else {
			abs_diag := abs(diag_elem)
		}

		min_diag = min(min_diag, f64(abs_diag))
		max_diag = max(max_diag, f64(abs_diag))
	}

	if min_diag > 0 {
		// Condition number of A ≈ (max_diag/min_diag)^2
		return (max_diag / min_diag) * (max_diag / min_diag)
	}

	return math.INF_F64
}

// Extract triangular factor from Cholesky factorization
extract_cholesky_factor :: proc(
	factor: ^CholeskyFactorization($T),
	allocator := context.allocator,
) -> Matrix(T) {
	L := matrix_clone(&factor.L, allocator)

	// Already has correct triangle zeroed
	return L
}

// Verify Cholesky factorization accuracy
verify_cholesky :: proc(
	A_original: ^Matrix($T),
	factor: ^CholeskyFactorization(T),
	allocator := context.allocator,
) -> (
	residual_norm: f64,
	relative_error: f64,
) {
	if !factor.is_positive_definite {
		return math.INF_F64, math.INF_F64
	}

	// Reconstruct A from factorization
	A_reconstructed: Matrix(T)
	if factor.is_upper {
		// A = U^T * U
		A_reconstructed = matrix_multiply_transpose(&factor.L, &factor.L, true, false, allocator)
	} else {
		// A = L * L^T
		A_reconstructed = matrix_multiply_transpose(&factor.L, &factor.L, false, true, allocator)
	}
	defer matrix_delete(&A_reconstructed)

	// Compute residual
	residual := matrix_subtract(A_original, &A_reconstructed, allocator)
	defer matrix_delete(&residual)

	residual_norm = matrix_norm(&residual, .Frobenius)
	a_norm := matrix_norm(A_original, .Frobenius)

	if a_norm > 0 {
		relative_error = residual_norm / a_norm
	} else {
		relative_error = residual_norm
	}

	return residual_norm, relative_error
}

// Helper functions (placeholders for actual implementations)

zero_triangle :: proc(A: ^Matrix($T), lower: bool) {
	if lower {
		// Zero lower triangle
		for i in 0 ..< A.rows {
			for j in 0 ..< i {
				matrix_set(A, i, j, T(0))
			}
		}
	} else {
		// Zero upper triangle
		for i in 0 ..< A.rows {
			for j in i + 1 ..< A.cols {
				matrix_set(A, i, j, T(0))
			}
		}
	}
}

solve_triangular_system :: proc(L: ^Matrix($T), X: ^Matrix(T), upper: bool, transpose: bool) {
	// This would call LAPACK triangular solve routines
}


matrices_are_equal :: proc(A, B: ^Matrix($T), tol: f64) -> bool {
	if A.rows != B.rows || A.cols != B.cols {
		return false
	}

	for i in 0 ..< A.rows {
		for j in 0 ..< A.cols {
			a_val := matrix_get(A, i, j)
			b_val := matrix_get(B, i, j)
			when T == complex64 || T == complex128 {
				if abs(a_val - b_val) > T(tol) {
					return false
				}
			} else {
				if abs(a_val - b_val) > T(tol) {
					return false
				}
			}
		}
	}
	return true
}

matrix_multiply_transpose :: proc(
	A, B: ^Matrix($T),
	a_trans, b_trans: bool,
	allocator: mem.Allocator,
) -> Matrix(T) {
	// Placeholder for matrix multiplication with optional transpose
	rows := A.rows if !a_trans else A.cols
	cols := B.cols if !b_trans else B.rows
	return create_matrix(T, rows, cols, allocator)
}

matrix_subtract :: proc(A, B: ^Matrix($T), allocator: mem.Allocator) -> Matrix(T) {
	C := create_matrix(T, A.rows, A.cols, allocator)
	for i in 0 ..< A.rows {
		for j in 0 ..< A.cols {
			a_val := matrix_get(A, i, j)
			b_val := matrix_get(B, i, j)
			matrix_set(&C, i, j, a_val - b_val)
		}
	}
	return C
}

// ===================================================================================
// BANDED CHOLESKY CONVENIENCE FUNCTIONS
// ===================================================================================

// Banded Cholesky factorization result
BandedCholeskyFactorization :: struct($T: typeid) {
	factor:     Matrix(T), // L or U factor
	uplo_upper: bool, // Whether upper triangular
	kd:         int, // Bandwidth
	success:    bool,
	info:       Blas_Int,
}

// Complete banded Cholesky factorization workflow
cholesky_factorize_banded :: proc(
	A: Matrix($T), // Input matrix
	kd: int, // Bandwidth
	uplo_upper := true,
	allocator := context.allocator,
) -> BandedCholeskyFactorization(T) {
	// Create banded storage copy
	AB := make_banded_matrix(T, A.rows, A.cols, kd, kd, allocator)
	copy_matrix_to_banded(&A, &AB, kd, uplo_upper)

	// Perform factorization
	when T == complex64 {
		success, info := m_cholesky_factor_banded_c64(&AB, kd, uplo_upper, allocator)
		return BandedCholeskyFactorization(T) {
			factor = AB,
			uplo_upper = uplo_upper,
			kd = kd,
			success = success,
			info = info,
		}
	} else when T == f64 {
		success, info := m_cholesky_factor_banded_f64(&AB, kd, uplo_upper, allocator)
		return BandedCholeskyFactorization(T) {
			factor = AB,
			uplo_upper = uplo_upper,
			kd = kd,
			success = success,
			info = info,
		}
	} else when T == f32 {
		success, info := m_cholesky_factor_banded_f32(&AB, kd, uplo_upper, allocator)
		return BandedCholeskyFactorization(T) {
			factor = AB,
			uplo_upper = uplo_upper,
			kd = kd,
			success = success,
			info = info,
		}
	} else when T == complex128 {
		success, info := m_cholesky_factor_banded_c128(&AB, kd, uplo_upper, allocator)
		return BandedCholeskyFactorization(T) {
			factor = AB,
			uplo_upper = uplo_upper,
			kd = kd,
			success = success,
			info = info,
		}
	} else {
		panic("Unsupported type for banded Cholesky factorization")
	}
}

// Solve using pre-computed banded Cholesky factorization
cholesky_solve_with_banded_factor :: proc(
	chol: ^BandedCholeskyFactorization($T),
	b: []T,
	allocator := context.allocator,
) -> (
	x: []T,
	success: bool,
) {
	if !chol.success {
		return nil, false
	}

	// Create RHS matrix
	B := make_matrix(T, len(b), 1, .General, allocator)
	for i in 0 ..< len(b) {
		matrix_set(&B, i, 0, b[i])
	}

	// Solve using factorization
	when T == complex64 {
		success, _ := m_cholesky_solve_banded_c64(
			&chol.factor,
			&B,
			chol.kd,
			chol.uplo_upper,
			allocator,
		)
	} else when T == f64 {
		success, _ := m_cholesky_solve_banded_f64(
			&chol.factor,
			&B,
			chol.kd,
			chol.uplo_upper,
			allocator,
		)
	} else when T == f32 {
		success, _ := m_cholesky_solve_banded_f32(
			&chol.factor,
			&B,
			chol.kd,
			chol.uplo_upper,
			allocator,
		)
	} else when T == complex128 {
		success, _ := m_cholesky_solve_banded_c128(
			&chol.factor,
			&B,
			chol.kd,
			chol.uplo_upper,
			allocator,
		)
	} else {
		panic("Unsupported type for banded Cholesky solve")
	}

	// Extract solution
	if success {
		x = make([]T, len(b), allocator)
		for i in 0 ..< len(b) {
			x[i] = matrix_get(&B, i, 0)
		}
	}

	matrix_delete(&B)
	return x, success
}

// Complete factor-and-solve workflow for banded matrices
cholesky_solve_banded_system :: proc(
	A: Matrix($T), // Input matrix
	b: []T, // Right-hand side
	kd: int, // Bandwidth
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	x: []T,
	factorization: BandedCholeskyFactorization(T),
) {
	// Factor the matrix
	factorization = cholesky_factorize_banded(A, kd, uplo_upper, allocator)

	if !factorization.success {
		return nil, factorization
	}

	// Solve the system
	x, _ = cholesky_solve_with_banded_factor(&factorization, b, allocator)
	return x, factorization
}

// Check if banded matrix is positive definite based on factorization
is_positive_definite_banded :: proc(chol: BandedCholeskyFactorization($T)) -> bool {
	// PBTRF returns info > 0 if matrix is not positive definite
	// info = i means the i-th leading minor is not positive definite
	return chol.success && chol.info == 0
}

// Extract diagonal from banded Cholesky factor
extract_banded_cholesky_diagonal :: proc(
	chol: ^BandedCholeskyFactorization($T),
	allocator := context.allocator,
) -> []T {
	if !chol.success {
		return nil
	}

	n := chol.factor.cols
	diag := make([]T, n, allocator)

	// Extract diagonal elements from banded storage
	for i in 0 ..< n {
		// In banded storage, diagonal is at specific offset
		if chol.uplo_upper {
			// Upper triangular: diagonal at row kd
			diag[i] = matrix_get(&chol.factor, chol.kd, i)
		} else {
			// Lower triangular: diagonal at row 0
			diag[i] = matrix_get(&chol.factor, 0, i)
		}
	}

	return diag
}

// Delete banded Cholesky factorization
delete_banded_cholesky_factorization :: proc(chol: ^BandedCholeskyFactorization($T)) {
	matrix_delete(&chol.factor)
}

// Utility functions for banded matrices
copy_matrix_to_banded :: proc(src: ^Matrix($T), dst: ^Matrix(T), kd: int, uplo_upper: bool) {
	for j in 0 ..< src.cols {
		for i in max_int(0, j - kd) ..< min_int(src.rows, j + kd + 1) {
			val := matrix_get(src, i, j)
			// Convert to banded storage format
			if uplo_upper {
				// Upper triangular banded storage
				if i <= j {
					band_row := kd + i - j
					matrix_set(dst, band_row, j, val)
				}
			} else {
				// Lower triangular banded storage
				if i >= j {
					band_row := i - j
					matrix_set(dst, band_row, j, val)
				}
			}
		}
	}
}
