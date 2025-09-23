package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE CHOLESKY WITH PIVOTING AND TRIDIAGONAL CONDITION
// ===================================================================================

// Pivoted Cholesky factorization proc group
m_cholesky_pivoted :: proc {
	m_cholesky_pivoted_c64,
	m_cholesky_pivoted_f64,
	m_cholesky_pivoted_f32,
	m_cholesky_pivoted_c128,
}

// Tridiagonal condition number proc group
m_condition_tridiagonal_positive_definite :: proc {
	m_condition_tridiagonal_positive_definite_c64,
	m_condition_tridiagonal_positive_definite_f64,
	m_condition_tridiagonal_positive_definite_f32,
	m_condition_tridiagonal_positive_definite_c128,
}

// ===================================================================================
// PIVOTED CHOLESKY RESULT STRUCTURE
// ===================================================================================

// Result of pivoted Cholesky factorization
PivotedCholeskyResult :: struct {
	rank:                     int, // Computed rank of matrix
	pivot:                    []Blas_Int, // Pivot indices
	tolerance_used:           f64, // Tolerance used for rank detection
	is_full_rank:             bool, // True if rank == n
	is_positive_semidefinite: bool, // True if factorization succeeded
}

// ===================================================================================
// PIVOTED CHOLESKY FACTORIZATION IMPLEMENTATION
// ===================================================================================

// Cholesky factorization with pivoting (c64)
// Computes P^T * A * P = L * L^H with diagonal pivoting for rank revelation
m_cholesky_pivoted_c64 :: proc(
	A: ^Matrix(complex64), // Matrix to factor (input/output)
	tolerance := f32(-1.0), // Tolerance for rank detection (-1 for default)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PivotedCholeskyResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate pivot array
	result.pivot = make([]Blas_Int, A.rows, allocator)

	// Allocate workspace
	work := make([]f32, 2 * A.rows, allocator)
	defer delete(work)

	rank: Blas_Int
	tol := tolerance
	info_val: Info

	lapack.cpstrf_(
		uplo_c,
		&n,
		cast(^complex64)A.data,
		&lda,
		raw_data(result.pivot),
		&rank,
		&tol,
		raw_data(work),
		&info_val,
		len(uplo_c),
	)

	// Fill result
	result.rank = int(rank)
	result.tolerance_used = f64(tol)
	result.is_full_rank = result.rank == A.rows
	result.is_positive_semidefinite = info_val >= 0

	// Handle errors
	if info_val < 0 {
		delete(result.pivot)
		result.pivot = nil
	}

	return result, info_val
}

// Cholesky factorization with pivoting (f64)
// Computes P^T * A * P = L * L^T with diagonal pivoting for rank revelation
m_cholesky_pivoted_f64 :: proc(
	A: ^Matrix(f64), // Matrix to factor (input/output)
	tolerance := f64(-1.0), // Tolerance for rank detection (-1 for default)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PivotedCholeskyResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate pivot array
	result.pivot = make([]Blas_Int, A.rows, allocator)

	// Allocate workspace
	work := make([]f64, 2 * A.rows, allocator)
	defer delete(work)

	rank: Blas_Int
	tol := tolerance
	info_val: Info

	lapack.dpstrf_(
		uplo_c,
		&n,
		cast(^f64)A.data,
		&lda,
		raw_data(result.pivot),
		&rank,
		&tol,
		raw_data(work),
		&info_val,
		len(uplo_c),
	)

	// Fill result
	result.rank = int(rank)
	result.tolerance_used = tol
	result.is_full_rank = result.rank == A.rows
	result.is_positive_semidefinite = info_val >= 0

	// Handle errors
	if info_val < 0 {
		delete(result.pivot)
		result.pivot = nil
	}

	return result, info_val
}

// Cholesky factorization with pivoting (f32)
// Computes P^T * A * P = L * L^T with diagonal pivoting for rank revelation
m_cholesky_pivoted_f32 :: proc(
	A: ^Matrix(f32), // Matrix to factor (input/output)
	tolerance := f32(-1.0), // Tolerance for rank detection (-1 for default)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PivotedCholeskyResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate pivot array
	result.pivot = make([]Blas_Int, A.rows, allocator)

	// Allocate workspace
	work := make([]f32, 2 * A.rows, allocator)
	defer delete(work)

	rank: Blas_Int
	tol := tolerance
	info_val: Info

	lapack.spstrf_(
		uplo_c,
		&n,
		cast(^f32)A.data,
		&lda,
		raw_data(result.pivot),
		&rank,
		&tol,
		raw_data(work),
		&info_val,
		len(uplo_c),
	)

	// Fill result
	result.rank = int(rank)
	result.tolerance_used = f64(tol)
	result.is_full_rank = result.rank == A.rows
	result.is_positive_semidefinite = info_val >= 0

	// Handle errors
	if info_val < 0 {
		delete(result.pivot)
		result.pivot = nil
	}

	return result, info_val
}

// Cholesky factorization with pivoting (c128)
// Computes P^T * A * P = L * L^H with diagonal pivoting for rank revelation
m_cholesky_pivoted_c128 :: proc(
	A: ^Matrix(complex128), // Matrix to factor (input/output)
	tolerance := f64(-1.0), // Tolerance for rank detection (-1 for default)
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	result: PivotedCholeskyResult,
	info: Info,
) {
	// Validate inputs
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.stride)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate pivot array
	result.pivot = make([]Blas_Int, A.rows, allocator)

	// Allocate workspace
	work := make([]f64, 2 * A.rows, allocator)
	defer delete(work)

	rank: Blas_Int
	tol := tolerance
	info_val: Info

	lapack.zpstrf_(
		uplo_c,
		&n,
		cast(^complex128)A.data,
		&lda,
		raw_data(result.pivot),
		&rank,
		&tol,
		raw_data(work),
		&info_val,
		len(uplo_c),
	)

	// Fill result
	result.rank = int(rank)
	result.tolerance_used = tol
	result.is_full_rank = result.rank == A.rows
	result.is_positive_semidefinite = info_val >= 0

	// Handle errors
	if info_val < 0 {
		delete(result.pivot)
		result.pivot = nil
	}

	return result, info_val
}

// ===================================================================================
// TRIDIAGONAL POSITIVE DEFINITE CONDITION NUMBER
// ===================================================================================

// Estimate condition number of tridiagonal positive definite matrix (c64)
// Requires matrix to be already factored
m_condition_tridiagonal_positive_definite_c64 :: proc(
	D: []f32, // Diagonal elements
	E: []complex64, // Off-diagonal elements
	anorm: f32, // Norm of original matrix
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)

	// Allocate workspace
	rwork := make([]f32, n, allocator)
	defer delete(rwork)

	rcond_val: f32
	anorm_val := anorm
	info_val: Info

	lapack.cptcon_(
		&n_val,
		raw_data(D),
		raw_data(E),
		&anorm_val,
		&rcond_val,
		raw_data(rwork),
		&info_val,
	)

	return rcond_val, info_val
}

// Estimate condition number of tridiagonal positive definite matrix (f64)
// Requires matrix to be already factored
m_condition_tridiagonal_positive_definite_f64 :: proc(
	D: []f64, // Diagonal elements
	E: []f64, // Off-diagonal elements
	anorm: f64, // Norm of original matrix
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)

	// Allocate workspace
	work := make([]f64, n, allocator)
	defer delete(work)

	rcond_val: f64
	anorm_val := anorm
	info_val: Info

	lapack.dptcon_(
		&n_val,
		raw_data(D),
		raw_data(E),
		&anorm_val,
		&rcond_val,
		raw_data(work),
		&info_val,
	)

	return rcond_val, info_val
}

// Estimate condition number of tridiagonal positive definite matrix (f32)
// Requires matrix to be already factored
m_condition_tridiagonal_positive_definite_f32 :: proc(
	D: []f32, // Diagonal elements
	E: []f32, // Off-diagonal elements
	anorm: f32, // Norm of original matrix
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)

	// Allocate workspace
	work := make([]f32, n, allocator)
	defer delete(work)

	rcond_val: f32
	anorm_val := anorm
	info_val: Info

	lapack.sptcon_(
		&n_val,
		raw_data(D),
		raw_data(E),
		&anorm_val,
		&rcond_val,
		raw_data(work),
		&info_val,
	)

	return rcond_val, info_val
}

// Estimate condition number of tridiagonal positive definite matrix (c128)
// Requires matrix to be already factored
m_condition_tridiagonal_positive_definite_c128 :: proc(
	D: []f64, // Diagonal elements
	E: []complex128, // Off-diagonal elements
	anorm: f64, // Norm of original matrix
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
) {
	// Validate inputs
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	n_val := Blas_Int(n)

	// Allocate workspace
	rwork := make([]f64, n, allocator)
	defer delete(rwork)

	rcond_val: f64
	anorm_val := anorm
	info_val: Info

	lapack.zptcon_(
		&n_val,
		raw_data(D),
		raw_data(E),
		&anorm_val,
		&rcond_val,
		raw_data(rwork),
		&info_val,
	)

	return rcond_val, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Perform rank-revealing Cholesky factorization
rank_revealing_cholesky :: proc(
	A: ^Matrix($T),
	tolerance := -1.0, // -1 for automatic tolerance
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	factor: Matrix(T),
	result: PivotedCholeskyResult,
) {
	// Clone matrix for factorization
	factor = matrix_clone(A, allocator)

	// Perform pivoted factorization
	when T == complex64 {
		result, _ = m_cholesky_pivoted_c64(&factor, f32(tolerance), uplo_upper, allocator)
	} else when T == f64 {
		result, _ = m_cholesky_pivoted_f64(&factor, tolerance, uplo_upper, allocator)
	} else when T == f32 {
		result, _ = m_cholesky_pivoted_f32(&factor, f32(tolerance), uplo_upper, allocator)
	} else when T == complex128 {
		result, _ = m_cholesky_pivoted_c128(&factor, tolerance, uplo_upper, allocator)
	}

	// Zero out unused parts if rank-deficient
	if !result.is_full_rank {
		zero_rank_deficient_part(&factor, result.rank, uplo_upper)
	}

	return factor, result
}

// Apply pivot permutation to matrix
apply_pivot_permutation :: proc(
	A: ^Matrix($T),
	pivot: []Blas_Int,
	forward := true, // True for P*A*P^T, false for P^T*A*P
) {
	n := A.rows
	if len(pivot) != n {
		panic("Pivot array size mismatch")
	}

	// Create permutation matrix effect
	if forward {
		// Apply P from left and P^T from right
		for i in 0 ..< n {
			if pivot[i] != Blas_Int(i + 1) { 	// LAPACK uses 1-based indexing
				swap_rows(A, i, int(pivot[i] - 1))
				swap_cols(A, i, int(pivot[i] - 1))
			}
		}
	} else {
		// Apply P^T from left and P from right (inverse)
		for i := n - 1; i >= 0; i -= 1 {
			if pivot[i] != Blas_Int(i + 1) {
				swap_rows(A, i, int(pivot[i] - 1))
				swap_cols(A, i, int(pivot[i] - 1))
			}
		}
	}
}

// Extract low-rank approximation from pivoted Cholesky
extract_low_rank :: proc(
	factor: ^Matrix($T),
	rank: int,
	uplo_upper: bool,
	allocator := context.allocator,
) -> Matrix(T) {
	if rank > factor.rows {
		panic("Rank exceeds matrix dimension")
	}

	// Extract rank-r approximation
	L := create_matrix(T, factor.rows, rank, allocator)

	if uplo_upper {
		// Extract from upper triangular factor
		for j in 0 ..< rank {
			for i in 0 ..= j {
				if i < factor.rows && j < factor.cols {
					matrix_set(&L, i, j, matrix_get(factor, i, j))
				}
			}
		}
	} else {
		// Extract from lower triangular factor
		for j in 0 ..< rank {
			for i in j ..< factor.rows {
				if i < factor.rows && j < factor.cols {
					matrix_set(&L, i, j, matrix_get(factor, i, j))
				}
			}
		}
	}

	return L
}

// Analyze tridiagonal positive definite matrix condition
analyze_tridiagonal_condition :: proc(
	D: []$T, // Diagonal elements
	E: []$S, // Off-diagonal elements
	allocator := context.allocator,
) -> TridiagonalConditionAnalysis {
	analysis: TridiagonalConditionAnalysis

	// Compute norm of tridiagonal matrix
	anorm := compute_tridiagonal_norm(D, E)

	// Estimate condition number
	when T == f32 && S == complex64 {
		rcond, info := m_condition_tridiagonal_positive_definite_c64(D, E, f32(anorm), allocator)
		analysis.rcond = f64(rcond)
		analysis.success = info == 0
	} else when T == f64 && S == f64 {
		rcond, info := m_condition_tridiagonal_positive_definite_f64(D, E, anorm, allocator)
		analysis.rcond = rcond
		analysis.success = info == 0
	} else when T == f32 && S == f32 {
		rcond, info := m_condition_tridiagonal_positive_definite_f32(D, E, f32(anorm), allocator)
		analysis.rcond = f64(rcond)
		analysis.success = info == 0
	} else when T == f64 && S == complex128 {
		rcond, info := m_condition_tridiagonal_positive_definite_c128(D, E, anorm, allocator)
		analysis.rcond = rcond
		analysis.success = info == 0
	}

	// Compute condition number
	if analysis.rcond > 0 {
		analysis.condition_number = 1.0 / analysis.rcond
		analysis.is_well_conditioned = analysis.condition_number < 1e6
	} else {
		analysis.condition_number = math.INF_F64
		analysis.is_well_conditioned = false
	}

	// Estimate relative error bound
	analysis.relative_error_bound = analysis.condition_number * builtin.F64_EPSILON

	return analysis
}

// Tridiagonal condition analysis structure
TridiagonalConditionAnalysis :: struct {
	rcond:                f64,
	condition_number:     f64,
	is_well_conditioned:  bool,
	relative_error_bound: f64,
	success:              bool,
}

// Check numerical rank of matrix
check_numerical_rank :: proc(
	A: ^Matrix($T),
	tolerance := -1.0,
	allocator := context.allocator,
) -> (
	rank: int,
	is_full_rank: bool,
	effective_condition: f64,
) {
	// Perform pivoted Cholesky
	A_copy := matrix_clone(A, allocator)
	defer matrix_delete(&A_copy)

	var; result: PivotedCholeskyResult
	when T == complex64 {
		result, _ = m_cholesky_pivoted_c64(&A_copy, f32(tolerance), true, allocator)
	} else when T == f64 {
		result, _ = m_cholesky_pivoted_f64(&A_copy, tolerance, true, allocator)
	} else when T == f32 {
		result, _ = m_cholesky_pivoted_f32(&A_copy, f32(tolerance), true, allocator)
	} else when T == complex128 {
		result, _ = m_cholesky_pivoted_c128(&A_copy, tolerance, true, allocator)
	}
	defer if result.pivot != nil do delete(result.pivot)

	rank = result.rank
	is_full_rank = result.is_full_rank

	// Estimate condition number of rank-r approximation
	if rank > 0 {
		// Get diagonal elements to estimate condition
		min_diag := math.INF_F64
		max_diag := 0.0

		for i in 0 ..< rank {
			diag_elem := matrix_get(&A_copy, i, i)
			when T == complex64 || T == complex128 {
				abs_val := abs_complex(diag_elem)
			} else {
				abs_val := f64(abs(diag_elem))
			}
			min_diag = min(min_diag, abs_val)
			max_diag = max(max_diag, abs_val)
		}

		if min_diag > 0 {
			effective_condition = (max_diag / min_diag) * (max_diag / min_diag)
		} else {
			effective_condition = math.INF_F64
		}
	} else {
		effective_condition = math.INF_F64
	}

	return rank, is_full_rank, effective_condition
}

// Solve rank-deficient system using pivoted Cholesky
solve_rank_deficient :: proc(
	A: ^Matrix($T),
	B: ^Matrix(T),
	tolerance := -1.0,
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	rank: int,
	success: bool,
) {
	// Perform pivoted factorization
	factor, result := rank_revealing_cholesky(A, tolerance, true, allocator)
	defer matrix_delete(&factor)
	defer if result.pivot != nil do delete(result.pivot)

	rank = result.rank
	if rank == 0 {
		// Matrix is zero, return zero solution
		X = create_matrix(T, A.cols, B.cols, allocator)
		return X, rank, false
	}

	// Apply permutation to B
	B_perm := matrix_clone(B, allocator)
	defer matrix_delete(&B_perm)
	apply_pivot_to_rhs(&B_perm, result.pivot, true)

	// Solve using rank-r factor
	X = solve_with_rank_r_factor(&factor, &B_perm, rank, true, allocator)

	// Apply inverse permutation to solution
	apply_pivot_to_rhs(&X, result.pivot, false)

	return X, rank, true
}

// Create tridiagonal positive definite matrix
create_tridiagonal_matrix :: proc(
	D: []$T, // Diagonal elements
	E: []T, // Off-diagonal elements
	allocator := context.allocator,
) -> Matrix(T) {
	n := len(D)
	if len(E) != n - 1 && n > 1 {
		panic("Off-diagonal must have n-1 elements")
	}

	A := create_matrix(T, n, n, allocator)

	// Set diagonal
	for i in 0 ..< n {
		matrix_set(&A, i, i, D[i])
	}

	// Set off-diagonal
	for i in 0 ..< n - 1 {
		matrix_set(&A, i, i + 1, E[i])
		matrix_set(&A, i + 1, i, E[i]) // Symmetric
	}

	return A
}

// Helper functions

zero_rank_deficient_part :: proc(A: ^Matrix($T), rank: int, uplo_upper: bool) {
	n := A.rows
	if uplo_upper {
		// Zero out columns beyond rank
		for j in rank ..< n {
			for i in 0 ..= j {
				matrix_set(A, i, j, T(0))
			}
		}
	} else {
		// Zero out rows beyond rank
		for i in rank ..< n {
			for j in 0 ..= i {
				matrix_set(A, i, j, T(0))
			}
		}
	}
}

swap_rows :: proc(A: ^Matrix($T), i, j: int) {
	if i == j do return
	for k in 0 ..< A.cols {
		temp := matrix_get(A, i, k)
		matrix_set(A, i, k, matrix_get(A, j, k))
		matrix_set(A, j, k, temp)
	}
}

swap_cols :: proc(A: ^Matrix($T), i, j: int) {
	if i == j do return
	for k in 0 ..< A.rows {
		temp := matrix_get(A, k, i)
		matrix_set(A, k, i, matrix_get(A, k, j))
		matrix_set(A, k, j, temp)
	}
}

compute_tridiagonal_norm :: proc(D: []$T, E: []$S) -> f64 {
	// Compute 1-norm of tridiagonal matrix
	norm := 0.0

	// First column
	if len(D) > 0 {
		when T == complex64 || T == complex128 {
			col_sum := abs_complex(D[0])
		} else {
			col_sum := f64(abs(D[0]))
		}
		if len(E) > 0 {
			when S == complex64 || S == complex128 {
				col_sum += abs_complex(E[0])
			} else {
				col_sum += f64(abs(E[0]))
			}
		}
		norm = max(norm, col_sum)
	}

	// Middle columns
	for i in 1 ..< len(D) - 1 {
		when T == complex64 || T == complex128 {
			col_sum := abs_complex(D[i])
		} else {
			col_sum := f64(abs(D[i]))
		}
		if i - 1 < len(E) {
			when S == complex64 || S == complex128 {
				col_sum += abs_complex(E[i - 1])
			} else {
				col_sum += f64(abs(E[i - 1]))
			}
		}
		if i < len(E) {
			when S == complex64 || S == complex128 {
				col_sum += abs_complex(E[i])
			} else {
				col_sum += f64(abs(E[i]))
			}
		}
		norm = max(norm, col_sum)
	}

	// Last column
	if len(D) > 1 {
		i := len(D) - 1
		when T == complex64 || T == complex128 {
			col_sum := abs_complex(D[i])
		} else {
			col_sum := f64(abs(D[i]))
		}
		if i - 1 < len(E) {
			when S == complex64 || S == complex128 {
				col_sum += abs_complex(E[i - 1])
			} else {
				col_sum += f64(abs(E[i - 1]))
			}
		}
		norm = max(norm, col_sum)
	}

	return norm
}

apply_pivot_to_rhs :: proc(B: ^Matrix($T), pivot: []Blas_Int, forward: bool) {
	// Apply row permutation to RHS
	n := B.rows
	if forward {
		for i in 0 ..< n {
			if pivot[i] != Blas_Int(i + 1) {
				swap_rows(B, i, int(pivot[i] - 1))
			}
		}
	} else {
		for i := n - 1; i >= 0; i -= 1 {
			if pivot[i] != Blas_Int(i + 1) {
				swap_rows(B, i, int(pivot[i] - 1))
			}
		}
	}
}

solve_with_rank_r_factor :: proc(
	factor: ^Matrix($T),
	B: ^Matrix(T),
	rank: int,
	uplo_upper: bool,
	allocator: mem.Allocator,
) -> Matrix(T) {
	// Solve using only the rank-r part of the factorization
	X := matrix_clone(B, allocator)

	// This would use triangular solvers on the rank-r submatrix
	// For now, return the clone
	return X
}
