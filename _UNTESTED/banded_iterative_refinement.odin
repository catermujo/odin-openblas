package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// POSITIVE DEFINITE BANDED ITERATIVE REFINEMENT AND SPLIT CHOLESKY
// ===================================================================================

// Iterative refinement for positive definite banded systems proc group
m_refine_solution_banded_pd :: proc {
	m_refine_solution_banded_pd_c64,
	m_refine_solution_banded_pd_f64,
	m_refine_solution_banded_pd_f32,
	m_refine_solution_banded_pd_c128,
}

// Split Cholesky factorization for positive definite banded matrices proc group
m_split_cholesky_banded :: proc {
	m_split_cholesky_banded_c64,
	m_split_cholesky_banded_f64,
	m_split_cholesky_banded_f32,
	m_split_cholesky_banded_c128,
}

// ===================================================================================
// ITERATIVE REFINEMENT IMPLEMENTATION
// ===================================================================================

// Refinement result structure
RefinementResult :: struct($T: typeid) {
	ferr:    []T, // Forward error bounds for each solution
	berr:    []T, // Backward error bounds for each solution
	success: bool, // Whether refinement succeeded
	info:    Info,
}

// Iterative refinement for positive definite banded systems (c64)
// Improves solution accuracy using iterative refinement
m_refine_solution_banded_pd_c64 :: proc(
	AB: ^Matrix(complex64), // Original banded matrix
	AFB: ^Matrix(complex64), // Factorized matrix from CPBTRF
	B: ^Matrix(complex64), // Right-hand side matrix
	X: ^Matrix(complex64), // Solution matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> RefinementResult(f32) {
	// Validate inputs
	if len(AB.data) == 0 || len(AFB.data) == 0 || len(B.data) == 0 || len(X.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols || AFB.rows != AFB.cols {
		panic("AB and AFB must be square")
	}
	if B.rows != X.rows || B.cols != X.cols {
		panic("B and X must have same dimensions")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate error bound arrays
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)

	// Allocate workspace
	work := make([]complex64, 2 * n, context.temp_allocator)
	rwork := make([]f32, n, context.temp_allocator)

	info_val: Info

	lapack.cpbrfs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	return RefinementResult(f32) {
		ferr = ferr,
		berr = berr,
		success = info_val == 0,
		info = info_val,
	}
}

// Iterative refinement for positive definite banded systems (f64)
// Improves solution accuracy using iterative refinement
m_refine_solution_banded_pd_f64 :: proc(
	AB: ^Matrix(f64), // Original banded matrix
	AFB: ^Matrix(f64), // Factorized matrix from DPBTRF
	B: ^Matrix(f64), // Right-hand side matrix
	X: ^Matrix(f64), // Solution matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> RefinementResult(f64) {
	// Validate inputs
	if len(AB.data) == 0 || len(AFB.data) == 0 || len(B.data) == 0 || len(X.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols || AFB.rows != AFB.cols {
		panic("AB and AFB must be square")
	}
	if B.rows != X.rows || B.cols != X.cols {
		panic("B and X must have same dimensions")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate error bound arrays
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)

	// Allocate workspace
	work := make([]f64, 3 * n, context.temp_allocator)
	iwork := make([]Blas_Int, n, context.temp_allocator)

	info_val: Info

	lapack.dpbrfs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	return RefinementResult(f64) {
		ferr = ferr,
		berr = berr,
		success = info_val == 0,
		info = info_val,
	}
}

// Iterative refinement for positive definite banded systems (f32)
// Improves solution accuracy using iterative refinement
m_refine_solution_banded_pd_f32 :: proc(
	AB: ^Matrix(f32), // Original banded matrix
	AFB: ^Matrix(f32), // Factorized matrix from SPBTRF
	B: ^Matrix(f32), // Right-hand side matrix
	X: ^Matrix(f32), // Solution matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> RefinementResult(f32) {
	// Validate inputs
	if len(AB.data) == 0 || len(AFB.data) == 0 || len(B.data) == 0 || len(X.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols || AFB.rows != AFB.cols {
		panic("AB and AFB must be square")
	}
	if B.rows != X.rows || B.cols != X.cols {
		panic("B and X must have same dimensions")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate error bound arrays
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)

	// Allocate workspace
	work := make([]f32, 3 * n, context.temp_allocator)
	iwork := make([]Blas_Int, n, context.temp_allocator)

	info_val: Info

	lapack.spbrfs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	return RefinementResult(f32) {
		ferr = ferr,
		berr = berr,
		success = info_val == 0,
		info = info_val,
	}
}

// Iterative refinement for positive definite banded systems (c128)
// Improves solution accuracy using iterative refinement
m_refine_solution_banded_pd_c128 :: proc(
	AB: ^Matrix(complex128), // Original banded matrix
	AFB: ^Matrix(complex128), // Factorized matrix from ZPBTRF
	B: ^Matrix(complex128), // Right-hand side matrix
	X: ^Matrix(complex128), // Solution matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> RefinementResult(f64) {
	// Validate inputs
	if len(AB.data) == 0 || len(AFB.data) == 0 || len(B.data) == 0 || len(X.data) == 0 {
		panic("Matrices cannot be empty")
	}
	if AB.rows != AB.cols || AFB.rows != AFB.cols {
		panic("AB and AFB must be square")
	}
	if B.rows != X.rows || B.cols != X.cols {
		panic("B and X must have same dimensions")
	}
	if B.rows != AB.rows {
		panic("System dimensions must be consistent")
	}

	uplo_c := "U" if uplo_upper else "L"
	n := Blas_Int(AB.cols)
	kd_val := Blas_Int(kd)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.ld)
	ldafb := Blas_Int(AFB.ld)
	ldb := Blas_Int(B.ld)
	ldx := Blas_Int(X.ld)

	// Allocate error bound arrays
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)

	// Allocate workspace
	work := make([]complex128, 2 * n, context.temp_allocator)
	rwork := make([]f64, n, context.temp_allocator)

	info_val: Info

	lapack.zpbrfs_(
		uplo_c,
		&n,
		&kd_val,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(AFB.data),
		&ldafb,
		raw_data(B.data),
		&ldb,
		raw_data(X.data),
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	return RefinementResult(f64) {
		ferr = ferr,
		berr = berr,
		success = info_val == 0,
		info = info_val,
	}
}

// ===================================================================================
// SPLIT CHOLESKY FACTORIZATION IMPLEMENTATION
// ===================================================================================

// Split Cholesky factorization for positive definite banded matrix (c64)
// Computes split factor S from L^H*L where L = S*S^H
m_split_cholesky_banded_c64 :: proc(
	AB: ^Matrix(complex64), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Blas_Int,
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
	ldab := Blas_Int(AB.ld)
	info_val: Info

	lapack.cpbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// Split Cholesky factorization for positive definite banded matrix (f64)
// Computes split factor S from L^T*L where L = S*S^T
m_split_cholesky_banded_f64 :: proc(
	AB: ^Matrix(f64), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Blas_Int,
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
	ldab := Blas_Int(AB.ld)
	info_val: Info

	lapack.dpbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// Split Cholesky factorization for positive definite banded matrix (f32)
// Computes split factor S from L^T*L where L = S*S^T
m_split_cholesky_banded_f32 :: proc(
	AB: ^Matrix(f32), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Blas_Int,
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
	ldab := Blas_Int(AB.ld)
	info_val: Info

	lapack.spbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// Split Cholesky factorization for positive definite banded matrix (c128)
// Computes split factor S from L^H*L where L = S*S^H
m_split_cholesky_banded_c128 :: proc(
	AB: ^Matrix(complex128), // Banded matrix (input/output)
	kd: int, // Number of super/sub-diagonals
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Blas_Int,
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
	ldab := Blas_Int(AB.ld)
	info_val: Info

	lapack.zpbstf_(uplo_c, &n, &kd_val, raw_data(AB.data), &ldab, &info_val, len(uplo_c))

	return info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Check if refinement significantly improved the solution
is_refinement_successful :: proc(result: RefinementResult($T), tolerance := T(1e-10)) -> bool {
	if !result.success || len(result.ferr) == 0 {
		return false
	}

	// Check if all forward errors are below tolerance
	for err in result.ferr {
		if err > tolerance {
			return false
		}
	}

	return true
}

// Get maximum error bounds from refinement
get_max_error_bounds :: proc(result: RefinementResult($T)) -> (max_ferr, max_berr: T) {
	if len(result.ferr) == 0 || len(result.berr) == 0 {
		return T(0), T(0)
	}

	max_ferr = result.ferr[0]
	max_berr = result.berr[0]

	for i in 1 ..< len(result.ferr) {
		if result.ferr[i] > max_ferr {
			max_ferr = result.ferr[i]
		}
		if result.berr[i] > max_berr {
			max_berr = result.berr[i]
		}
	}

	return max_ferr, max_berr
}

// Complete solve with iterative refinement workflow
solve_with_refinement :: proc(
	AB: ^Matrix($T), // Original banded matrix
	B: ^Matrix(T), // Right-hand side matrix
	kd: int, // Bandwidth
	uplo_upper := true,
	max_iterations := 1, // Number of refinement iterations
	allocator := context.allocator,
) -> (
	X: Matrix(T),
	refinement: RefinementResult(auto),
	success: bool,
) {
	// Create working copies
	AFB := make_matrix(T, AB.rows, AB.cols, AB.format, allocator)
	copy_matrix(AB, &AFB)

	X = make_matrix(T, B.rows, B.cols, B.format, allocator)
	copy_matrix(B, &X)

	// Factor the matrix (would call PBTRF here)
	// For now, assume AFB is already factorized

	// Initial solve (would call PBTRS here)
	// For now, assume X contains initial solution

	// Perform iterative refinement
	for iter in 0 ..< max_iterations {
		when T == complex64 {
			refinement = m_refine_solution_banded_pd_c64(
				AB,
				&AFB,
				B,
				&X,
				kd,
				uplo_upper,
				allocator,
			)
		} else when T == f64 {
			refinement = m_refine_solution_banded_pd_f64(
				AB,
				&AFB,
				B,
				&X,
				kd,
				uplo_upper,
				allocator,
			)
		} else when T == f32 {
			refinement = m_refine_solution_banded_pd_f32(
				AB,
				&AFB,
				B,
				&X,
				kd,
				uplo_upper,
				allocator,
			)
		} else when T == complex128 {
			refinement = m_refine_solution_banded_pd_c128(
				AB,
				&AFB,
				B,
				&X,
				kd,
				uplo_upper,
				allocator,
			)
		} else {
			panic("Unsupported type for iterative refinement")
		}

		if !refinement.success {
			break
		}

		// Check if refinement is good enough
		if is_refinement_successful(refinement) {
			success = true
			break
		}
	}

	delete_matrix(&AFB)
	return X, refinement, success
}

// Prepare matrix for split Cholesky factorization
prepare_for_split_cholesky :: proc(
	A: ^Matrix($T), // Input matrix
	kd: int, // Bandwidth
	uplo_upper := true,
	allocator := context.allocator,
) -> (
	AB: Matrix(T),
	success: bool,
) {
	// Create banded storage format
	AB = make_banded_matrix(T, A.rows, A.cols, kd, kd, allocator)

	// Copy data to banded format (simplified)
	for j in 0 ..< A.cols {
		for i in max(0, j - kd) ..< min(A.rows, j + kd + 1) {
			val := matrix_get(A, i, j)
			// Set in banded format (implementation specific)
			matrix_set(&AB, i, j, val)
		}
	}

	// Apply split Cholesky factorization
	when T == complex64 {
		success, _ := m_split_cholesky_banded_c64(&AB, kd, uplo_upper, allocator)
	} else when T == f64 {
		success, _ := m_split_cholesky_banded_f64(&AB, kd, uplo_upper, allocator)
	} else when T == f32 {
		success, _ := m_split_cholesky_banded_f32(&AB, kd, uplo_upper, allocator)
	} else when T == complex128 {
		success, _ := m_split_cholesky_banded_c128(&AB, kd, uplo_upper, allocator)
	} else {
		panic("Unsupported type for split Cholesky")
	}

	return AB, success
}

// Analyze error bounds for multiple right-hand sides
analyze_solution_accuracy :: proc(
	result: RefinementResult($T),
	solution_names: []string = nil,
) -> string {
	if !result.success {
		return "Refinement failed"
	}

	output := "Solution Error Analysis:\n"

	for i in 0 ..< len(result.ferr) {
		name := ""
		if solution_names != nil && i < len(solution_names) {
			name = solution_names[i]
		} else {
			name = fmt.sprintf("Solution %d", i + 1)
		}

		output = fmt.sprintf(
			"%s\n%s:\n  Forward Error: %.2e\n  Backward Error: %.2e",
			output,
			name,
			result.ferr[i],
			result.berr[i],
		)
	}

	max_ferr, max_berr := get_max_error_bounds(result)
	output = fmt.sprintf(
		"%s\n\nMaximum Errors:\n  Forward: %.2e\n  Backward: %.2e",
		output,
		max_ferr,
		max_berr,
	)

	return output
}

// Delete refinement result
delete_refinement_result :: proc(result: ^RefinementResult($T)) {
	if result.ferr != nil {
		delete(result.ferr)
	}
	if result.berr != nil {
		delete(result.berr)
	}
}
