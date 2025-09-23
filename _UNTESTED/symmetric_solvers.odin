package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC SYSTEM SOLVERS
// ============================================================================
// Driver routines that combine factorization and solving for symmetric systems

// Standard solver result
SymmetricSolverResult :: struct($T: typeid) {
	factorization_successful: bool,
	solution_successful:      bool,
	pivot_indices:            []Blas_Int, // Pivot information from factorization
	determinant_sign:         int, // Sign of determinant (-1, 0, 1)
	is_singular:              bool, // True if matrix is singular
	condition_estimate:       f64, // Rough condition number estimate
}

// Aasen solver result
AasenSolverResult :: struct($T: typeid) {
	factorization_successful: bool,
	solution_successful:      bool,
	pivot_indices:            []Blas_Int,
	tridiagonal_factor:       Matrix(T), // T matrix from Aasen factorization
	determinant_sign:         int,
	is_singular:              bool,
	condition_estimate:       f64,
}

// 2-stage Aasen solver result
Aasen2StageSolverResult :: struct($T: typeid) {
	factorization_successful: bool,
	solution_successful:      bool,
	pivot_indices_1:          []Blas_Int, // First stage pivots
	pivot_indices_2:          []Blas_Int, // Second stage pivots
	band_matrix:              Matrix(T), // TB band matrix from 2-stage factorization
	determinant_sign:         int,
	is_singular:              bool,
	condition_estimate:       f64,
}

// RK (bounded Bunch-Kaufman) solver result
RKSolverResult :: struct($T: typeid) {
	factorization_successful: bool,
	solution_successful:      bool,
	pivot_indices:            []Blas_Int,
	e_factor:                 []T, // E vector from RK factorization
	determinant_sign:         int,
	is_singular:              bool,
	condition_estimate:       f64,
}

// Rook pivoting solver result
RookSolverResult :: struct($T: typeid) {
	factorization_successful: bool,
	solution_successful:      bool,
	pivot_indices:            []Blas_Int,
	determinant_sign:         int,
	is_singular:              bool,
	condition_estimate:       f64,
}

// Factorization flag for expert drivers
FactFlag :: enum {
	FACTORIZE, // 'N' - Factorize A first
	FACTORED, // 'F' - AF contains the factorization from a previous call
}

// Equilibration state
EquilibrationState :: enum {
	NONE, // 'N' - No equilibration
	APPLIED, // 'Y' - Equilibration was applied
}

// Solution quality assessment
SolutionQuality :: enum {
	EXCELLENT, // rcond > 0.1, berr < 1e-14
	GOOD, // rcond > 0.01, berr < 1e-10
	ACCEPTABLE, // rcond > machine_epsilon, berr < 1e-6
	POOR, // rcond <= machine_epsilon or berr >= 1e-6
}

// Expert solver result
ExpertSolverResult :: struct($T: typeid, $S: typeid) {
	factorization_successful: bool,
	solution_successful:      bool,
	pivot_indices:            []Blas_Int,
	rcond:                    f64, // Reciprocal condition number
	forward_errors:           []S, // Forward error bounds
	backward_errors:          []S, // Backward error bounds
	max_forward_error:        f64,
	max_backward_error:       f64,
	condition_number:         f64, // 1/rcond
	is_well_conditioned:      bool,
	solution_quality:         SolutionQuality,
}

// Extended expert solver result with equilibration and advanced error bounds
ExtendedExpertSolverResult :: struct($T: typeid, $S: typeid) {
	factorization_successful: bool,
	solution_successful:      bool,
	pivot_indices:            []Blas_Int,
	rcond:                    f64, // Reciprocal condition number
	rpvgrw:                   f64, // Reciprocal pivot growth factor
	backward_errors:          []S, // Backward error bounds
	err_bounds_norm:          Matrix(S), // Normwise error bounds
	err_bounds_comp:          Matrix(S), // Componentwise error bounds
	scale_factors:            []S, // Equilibration scale factors
	equilibration_used:       bool, // True if equilibration was applied
	max_backward_error:       f64,
	condition_number:         f64, // 1/rcond
	pivot_growth_acceptable:  bool, // True if rpvgrw is acceptable
	solution_quality:         SolutionQuality,
	stability_assessment:     StabilityAssessment,
}

// Stability assessment based on pivot growth
StabilityAssessment :: enum {
	EXCELLENT, // rpvgrw close to 1, very stable
	GOOD, // rpvgrw > 0.1, acceptable stability
	MARGINAL, // rpvgrw > machine_epsilon, marginal stability
	POOR, // rpvgrw <= machine_epsilon, poor stability
}

// ============================================================================
// STANDARD SYMMETRIC SOLVERS (BUNCH-KAUFMAN PIVOTING)
// ============================================================================

// Complex single precision symmetric solver
csysv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	b: Matrix(complex64), // RHS on input, solution on output
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: SymmetricSolverResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csysv_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.complex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.complex)b.data,
			&ldb,
			cast(^lapack.complex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex64, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csysv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	// Rough condition estimate (would need additional LAPACK calls for accurate estimate)
	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Double precision symmetric solver
dsysv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	b: Matrix(f64),
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: SymmetricSolverResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsysv_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsysv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Single precision symmetric solver
ssysv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	b: Matrix(f32),
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: SymmetricSolverResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssysv_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssysv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Complex double precision symmetric solver
zsysv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	b: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: SymmetricSolverResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsysv_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.doublecomplex)b.data,
			&ldb,
			cast(^lapack.doublecomplex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex128, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsysv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

sysv :: proc {
	csysv,
	dsysv,
	ssysv,
	zsysv,
}

// ============================================================================
// AASEN SYMMETRIC SOLVERS
// ============================================================================
// Uses Aasen's algorithm for symmetric indefinite matrices

// Complex single precision Aasen solver
csysv_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	b: Matrix(complex64), // RHS on input, solution on output
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: AasenSolverResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csysv_aa_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.complex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.complex)b.data,
			&ldb,
			cast(^lapack.complex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex64, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csysv_aa_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Aasen factorization produces L*T*L^T where T is tridiagonal
	// The T matrix is stored in the factorized A matrix
	result.tridiagonal_factor = a

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Double precision Aasen solver
dsysv_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	b: Matrix(f64),
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: AasenSolverResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsysv_aa_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsysv_aa_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0
	result.tridiagonal_factor = a

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Single precision Aasen solver
ssysv_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	b: Matrix(f32),
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: AasenSolverResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssysv_aa_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssysv_aa_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0
	result.tridiagonal_factor = a

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Complex double precision Aasen solver
zsysv_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	b: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: AasenSolverResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsysv_aa_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.doublecomplex)b.data,
			&ldb,
			cast(^lapack.doublecomplex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex128, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsysv_aa_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0
	result.tridiagonal_factor = a

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

sysv_aa :: proc {
	csysv_aa,
	dsysv_aa,
	ssysv_aa,
	zsysv_aa,
}

// ============================================================================
// 2-STAGE AASEN SYMMETRIC SOLVERS
// ============================================================================
// Two-stage Aasen algorithm for improved performance on large matrices

// Complex single precision 2-stage Aasen solver
csysv_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	tb: Matrix(complex64), // Band matrix storage (4*n, nb)
	b: Matrix(complex64), // RHS on input, solution on output
	ipiv: []Blas_Int = nil, // First stage pivot indices (size n)
	ipiv2: []Blas_Int = nil, // Second stage pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: Aasen2StageSolverResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot arrays if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	allocated_ipiv2 := ipiv2 == nil
	if allocated_ipiv2 {
		ipiv2 = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csysv_aa_2stage_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.complex)a.data,
			&lda,
			cast(^lapack.complex)tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
			cast(^lapack.complex)b.data,
			&ldb,
			cast(^lapack.complex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex64, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csysv_aa_2stage_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot arrays
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
			if ipiv2[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Double precision 2-stage Aasen solver
dsysv_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	tb: Matrix(f64),
	b: Matrix(f64),
	ipiv: []Blas_Int = nil,
	ipiv2: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: Aasen2StageSolverResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot arrays if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	allocated_ipiv2 := ipiv2 == nil
	if allocated_ipiv2 {
		ipiv2 = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsysv_aa_2stage_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsysv_aa_2stage_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot arrays
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
			if ipiv2[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Single precision 2-stage Aasen solver
ssysv_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	tb: Matrix(f32),
	b: Matrix(f32),
	ipiv: []Blas_Int = nil,
	ipiv2: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: Aasen2StageSolverResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot arrays if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	allocated_ipiv2 := ipiv2 == nil
	if allocated_ipiv2 {
		ipiv2 = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssysv_aa_2stage_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssysv_aa_2stage_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot arrays
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
			if ipiv2[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Complex double precision 2-stage Aasen solver
zsysv_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	tb: Matrix(complex128),
	b: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	ipiv2: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: Aasen2StageSolverResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot arrays if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	allocated_ipiv2 := ipiv2 == nil
	if allocated_ipiv2 {
		ipiv2 = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsysv_aa_2stage_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			cast(^lapack.doublecomplex)tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
			cast(^lapack.doublecomplex)b.data,
			&ldb,
			cast(^lapack.doublecomplex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex128, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsysv_aa_2stage_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot arrays
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
			if ipiv2[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

sysv_aa_2stage :: proc {
	csysv_aa_2stage,
	dsysv_aa_2stage,
	ssysv_aa_2stage,
	zsysv_aa_2stage,
}

// ============================================================================
// RK (BOUNDED BUNCH-KAUFMAN) SYMMETRIC SOLVERS
// ============================================================================
// Uses bounded Bunch-Kaufman pivoting with additional E factor

// Complex single precision RK solver
csysv_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	e: []complex64, // E factor from RK factorization (size n)
	b: Matrix(complex64), // RHS on input, solution on output
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: RKSolverResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csysv_rk_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.complex)a.data,
			&lda,
			cast(^lapack.complex)raw_data(e),
			raw_data(ipiv),
			cast(^lapack.complex)b.data,
			&ldb,
			cast(^lapack.complex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex64, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csysv_rk_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.e_factor = e
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Double precision RK solver
dsysv_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	e: []f64,
	b: Matrix(f64),
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: RKSolverResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsysv_rk_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(e),
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsysv_rk_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.e_factor = e
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Single precision RK solver
ssysv_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	e: []f32,
	b: Matrix(f32),
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: RKSolverResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssysv_rk_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(e),
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssysv_rk_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.e_factor = e
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Complex double precision RK solver
zsysv_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	e: []complex128,
	b: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: RKSolverResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsysv_rk_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			cast(^lapack.doublecomplex)raw_data(e),
			raw_data(ipiv),
			cast(^lapack.doublecomplex)b.data,
			&ldb,
			cast(^lapack.doublecomplex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex128, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsysv_rk_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.e_factor = e
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

sysv_rk :: proc {
	csysv_rk,
	dsysv_rk,
	ssysv_rk,
	zsysv_rk,
}

// ============================================================================
// ROOK PIVOTING SYMMETRIC SOLVERS
// ============================================================================
// Uses rook pivoting strategy for improved numerical stability

// Complex single precision rook solver
csysv_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	b: Matrix(complex64), // RHS on input, solution on output
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: RookSolverResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csysv_rook_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.complex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.complex)b.data,
			&ldb,
			cast(^lapack.complex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex64, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csysv_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Double precision rook solver
dsysv_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	b: Matrix(f64),
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: RookSolverResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsysv_rook_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsysv_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Single precision rook solver
ssysv_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	b: Matrix(f32),
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: RookSolverResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssysv_rook_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			raw_data(ipiv),
			b.data,
			&ldb,
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssysv_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

// Complex double precision rook solver
zsysv_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	b: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: RookSolverResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsysv_rook_(
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.doublecomplex)b.data,
			&ldb,
			cast(^lapack.doublecomplex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex128, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsysv_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.is_singular = info_int > 0

	// Estimate determinant sign from pivot array
	if !result.is_singular {
		result.determinant_sign = 1
		for i in 0 ..< n {
			if ipiv[i] != Blas_Int(i + 1) {
				result.determinant_sign *= -1
			}
		}
	}

	result.condition_estimate = result.is_singular ? math.INF_F64 : 1.0

	return
}

sysv_rook :: proc {
	csysv_rook,
	dsysv_rook,
	ssysv_rook,
	zsysv_rook,
}

// ============================================================================
// EXPERT SYMMETRIC SOLVERS
// ============================================================================
// Expert interface with condition estimation and error bounds

// Complex single precision expert solver
csysvx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Original matrix (if fact == FACTORIZE)
	af: Matrix(complex64), // Factorized matrix
	b: Matrix(complex64), // RHS matrix
	x: Matrix(complex64), // Solution matrix (output)
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	ferr: []f32 = nil, // Forward error bounds (size nrhs)
	berr: []f32 = nil, // Backward error bounds (size nrhs)
	work: []complex64 = nil, // Workspace (query if nil)
	rwork: []f32 = nil, // Real workspace (size n)
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(complex64, f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f32
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f32, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f32, nrhs, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csysvx_(
			fact_cstring,
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.complex)a.data,
			&lda,
			cast(^lapack.complex)af.data,
			&ldaf,
			raw_data(ipiv),
			cast(^lapack.complex)b.data,
			&ldb,
			cast(^lapack.complex)x.data,
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			cast(^lapack.complex)&work_query,
			&lwork_query,
			nil, // rwork (real workspace for complex functions)
			&info_int,
			1,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex64, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Allocate real workspace if not provided
	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f32, n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.csysvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)af.data,
		&ldaf,
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		raw_data(rwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = f64(rcond)
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_well_conditioned = rcond > machine_epsilon(f32)

	if nrhs > 0 {
		result.max_forward_error = f64(slice.max(ferr[:nrhs]))
		result.max_backward_error = f64(slice.max(berr[:nrhs]))

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f32) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}
	}

	return
}

// Double precision expert solver
dsysvx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	af: Matrix(f64),
	b: Matrix(f64),
	x: Matrix(f64),
	ipiv: []Blas_Int = nil,
	ferr: []f64 = nil,
	berr: []f64 = nil,
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(f64, f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f64
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f64, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f64, nrhs, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsysvx_(
			fact_cstring,
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			af.data,
			&ldaf,
			raw_data(ipiv),
			b.data,
			&ldb,
			x.data,
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			&work_query,
			&lwork_query,
			nil, // iwork
			&info_int,
			1,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Allocate integer workspace if not provided
	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.dsysvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		af.data,
		&ldaf,
		raw_data(ipiv),
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_well_conditioned = rcond > machine_epsilon(f64)

	if nrhs > 0 {
		result.max_forward_error = slice.max(ferr[:nrhs])
		result.max_backward_error = slice.max(berr[:nrhs])

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f64) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}
	}

	return
}

// Single precision expert solver
ssysvx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	af: Matrix(f32),
	b: Matrix(f32),
	x: Matrix(f32),
	ipiv: []Blas_Int = nil,
	ferr: []f32 = nil,
	berr: []f32 = nil,
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(f32, f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f32
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f32, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f32, nrhs, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssysvx_(
			fact_cstring,
			uplo_cstring,
			&n_int,
			&nrhs_int,
			a.data,
			&lda,
			af.data,
			&ldaf,
			raw_data(ipiv),
			b.data,
			&ldb,
			x.data,
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			&work_query,
			&lwork_query,
			nil, // iwork
			&info_int,
			1,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, 3 * n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Allocate integer workspace if not provided
	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.ssysvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		af.data,
		&ldaf,
		raw_data(ipiv),
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = f64(rcond)
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_well_conditioned = rcond > machine_epsilon(f32)

	if nrhs > 0 {
		result.max_forward_error = f64(slice.max(ferr[:nrhs]))
		result.max_backward_error = f64(slice.max(berr[:nrhs]))

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f32) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}
	}

	return
}

// Complex double precision expert solver
zsysvx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	af: Matrix(complex128),
	b: Matrix(complex128),
	x: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	ferr: []f64 = nil,
	berr: []f64 = nil,
	work: []complex128 = nil,
	rwork: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: ExpertSolverResult(complex128, f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f64
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f64, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f64, nrhs, allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsysvx_(
			fact_cstring,
			uplo_cstring,
			&n_int,
			&nrhs_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			cast(^lapack.doublecomplex)af.data,
			&ldaf,
			raw_data(ipiv),
			cast(^lapack.doublecomplex)b.data,
			&ldb,
			cast(^lapack.doublecomplex)x.data,
			&ldx,
			&rcond,
			raw_data(ferr),
			raw_data(berr),
			cast(^lapack.doublecomplex)&work_query,
			&lwork_query,
			nil, // rwork
			&info_int,
			1,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex128, lwork, allocator)
		} else {
			lwork = max(1, 2 * n)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Allocate real workspace if not provided
	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f64, n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.zsysvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)af.data,
		&ldaf,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		raw_data(rwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_well_conditioned = rcond > machine_epsilon(f64)

	if nrhs > 0 {
		result.max_forward_error = slice.max(ferr[:nrhs])
		result.max_backward_error = slice.max(berr[:nrhs])

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f64) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}
	}

	return
}

sysvx :: proc {
	csysvx,
	dsysvx,
	ssysvx,
	zsysvx,
}

// ============================================================================
// EXTENDED EXPERT SYMMETRIC SOLVERS
// ============================================================================
// Most advanced interface with equilibration and comprehensive error analysis

// Error bound types
ERROR_BOUND_TYPES :: 3 // Number of error bound types

// Complex single precision extended expert solver
csysvxx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Original matrix (if fact == FACTORIZE)
	af: Matrix(complex64), // Factorized matrix
	equed: EquilibrationState, // Equilibration state
	s: []f32, // Scale factors (if equed == APPLIED)
	b: Matrix(complex64), // RHS matrix
	x: Matrix(complex64), // Solution matrix (output)
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	berr: []f32 = nil, // Backward error bounds (size nrhs)
	n_err_bnds: int = ERROR_BOUND_TYPES,
	err_bounds_norm: []f32 = nil, // Normwise bounds (nrhs * n_err_bnds)
	err_bounds_comp: []f32 = nil, // Componentwise bounds (nrhs * n_err_bnds)
	nparams: int = 0, // Number of parameters
	params: []f32 = nil, // Parameters array
	work: []complex64 = nil, // Workspace
	rwork: []f32 = nil, // Real workspace
	allocator := context.allocator,
) -> (
	result: ExtendedExpertSolverResult(complex64, f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	equed_char: u8 = equed == .APPLIED ? 'Y' : 'N'
	equed_cstring := cstring(&equed_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f32
	rpvgrw: f32
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate scale factors if equilibration is used and not provided
	allocated_s := false
	if equed == .APPLIED && s == nil {
		s = make([]f32, n, allocator)
		allocated_s = true
	}

	// Allocate error arrays if not provided
	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f32, nrhs, allocator)
	}

	allocated_err_bounds_norm := err_bounds_norm == nil
	if allocated_err_bounds_norm {
		err_bounds_norm = make([]f32, nrhs * n_err_bnds, allocator)
	}

	allocated_err_bounds_comp := err_bounds_comp == nil
	if allocated_err_bounds_comp {
		err_bounds_comp = make([]f32, nrhs * n_err_bnds, allocator)
	}

	// Allocate workspace - these functions need large workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f32, 2 * n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.csysvxx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)af.data,
		&ldaf,
		raw_data(ipiv),
		equed_cstring,
		raw_data(s) if s != nil else nil,
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)x.data,
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_int,
		raw_data(err_bounds_norm),
		raw_data(err_bounds_comp),
		&nparams_int,
		raw_data(params) if params != nil else nil,
		cast(^lapack.complex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = f64(rcond)
	result.rpvgrw = f64(rpvgrw)
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64
	result.backward_errors = berr
	result.scale_factors = s
	result.equilibration_used = equed == .APPLIED
	result.pivot_growth_acceptable = rpvgrw > machine_epsilon(f32)

	// Convert error bounds to matrices
	if nrhs > 0 && n_err_bnds > 0 {
		result.err_bounds_norm = Matrix(f32) {
			data   = err_bounds_norm,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}
		result.err_bounds_comp = Matrix(f32) {
			data   = err_bounds_comp,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}

		result.max_backward_error = f64(slice.max(berr[:nrhs]))

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f32) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}

		// Assess stability based on pivot growth
		if rpvgrw > 0.5 {
			result.stability_assessment = .EXCELLENT
		} else if rpvgrw > 0.1 {
			result.stability_assessment = .GOOD
		} else if rpvgrw > machine_epsilon(f32) {
			result.stability_assessment = .MARGINAL
		} else {
			result.stability_assessment = .POOR
		}
	}

	if allocated_s do delete(s)

	return
}

// Double precision extended expert solver
dsysvxx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	af: Matrix(f64),
	equed: EquilibrationState,
	s: []f64,
	b: Matrix(f64),
	x: Matrix(f64),
	ipiv: []Blas_Int = nil,
	berr: []f64 = nil,
	n_err_bnds: int = ERROR_BOUND_TYPES,
	err_bounds_norm: []f64 = nil,
	err_bounds_comp: []f64 = nil,
	nparams: int = 0,
	params: []f64 = nil,
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ExtendedExpertSolverResult(f64, f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	equed_char: u8 = equed == .APPLIED ? 'Y' : 'N'
	equed_cstring := cstring(&equed_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f64
	rpvgrw: f64
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate scale factors if equilibration is used and not provided
	allocated_s := false
	if equed == .APPLIED && s == nil {
		s = make([]f64, n, allocator)
		allocated_s = true
	}

	// Allocate error arrays if not provided
	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f64, nrhs, allocator)
	}

	allocated_err_bounds_norm := err_bounds_norm == nil
	if allocated_err_bounds_norm {
		err_bounds_norm = make([]f64, nrhs * n_err_bnds, allocator)
	}

	allocated_err_bounds_comp := err_bounds_comp == nil
	if allocated_err_bounds_comp {
		err_bounds_comp = make([]f64, nrhs * n_err_bnds, allocator)
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 4 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.dsysvxx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		af.data,
		&ldaf,
		raw_data(ipiv),
		equed_cstring,
		raw_data(s) if s != nil else nil,
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_int,
		raw_data(err_bounds_norm),
		raw_data(err_bounds_comp),
		&nparams_int,
		raw_data(params) if params != nil else nil,
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = rcond
	result.rpvgrw = rpvgrw
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64
	result.backward_errors = berr
	result.scale_factors = s
	result.equilibration_used = equed == .APPLIED
	result.pivot_growth_acceptable = rpvgrw > machine_epsilon(f64)

	// Convert error bounds to matrices
	if nrhs > 0 && n_err_bnds > 0 {
		result.err_bounds_norm = Matrix(f64) {
			data   = err_bounds_norm,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}
		result.err_bounds_comp = Matrix(f64) {
			data   = err_bounds_comp,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}

		result.max_backward_error = slice.max(berr[:nrhs])

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f64) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}

		// Assess stability based on pivot growth
		if rpvgrw > 0.5 {
			result.stability_assessment = .EXCELLENT
		} else if rpvgrw > 0.1 {
			result.stability_assessment = .GOOD
		} else if rpvgrw > machine_epsilon(f64) {
			result.stability_assessment = .MARGINAL
		} else {
			result.stability_assessment = .POOR
		}
	}

	if allocated_s do delete(s)

	return
}

// Single precision extended expert solver
ssysvxx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	af: Matrix(f32),
	equed: EquilibrationState,
	s: []f32,
	b: Matrix(f32),
	x: Matrix(f32),
	ipiv: []Blas_Int = nil,
	berr: []f32 = nil,
	n_err_bnds: int = ERROR_BOUND_TYPES,
	err_bounds_norm: []f32 = nil,
	err_bounds_comp: []f32 = nil,
	nparams: int = 0,
	params: []f32 = nil,
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ExtendedExpertSolverResult(f32, f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	equed_char: u8 = equed == .APPLIED ? 'Y' : 'N'
	equed_cstring := cstring(&equed_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f32
	rpvgrw: f32
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate scale factors if equilibration is used and not provided
	allocated_s := false
	if equed == .APPLIED && s == nil {
		s = make([]f32, n, allocator)
		allocated_s = true
	}

	// Allocate error arrays if not provided
	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f32, nrhs, allocator)
	}

	allocated_err_bounds_norm := err_bounds_norm == nil
	if allocated_err_bounds_norm {
		err_bounds_norm = make([]f32, nrhs * n_err_bnds, allocator)
	}

	allocated_err_bounds_comp := err_bounds_comp == nil
	if allocated_err_bounds_comp {
		err_bounds_comp = make([]f32, nrhs * n_err_bnds, allocator)
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 4 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.ssysvxx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		af.data,
		&ldaf,
		raw_data(ipiv),
		equed_cstring,
		raw_data(s) if s != nil else nil,
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_int,
		raw_data(err_bounds_norm),
		raw_data(err_bounds_comp),
		&nparams_int,
		raw_data(params) if params != nil else nil,
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = f64(rcond)
	result.rpvgrw = f64(rpvgrw)
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64
	result.backward_errors = berr
	result.scale_factors = s
	result.equilibration_used = equed == .APPLIED
	result.pivot_growth_acceptable = rpvgrw > machine_epsilon(f32)

	// Convert error bounds to matrices
	if nrhs > 0 && n_err_bnds > 0 {
		result.err_bounds_norm = Matrix(f32) {
			data   = err_bounds_norm,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}
		result.err_bounds_comp = Matrix(f32) {
			data   = err_bounds_comp,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}

		result.max_backward_error = f64(slice.max(berr[:nrhs]))

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f32) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}

		// Assess stability based on pivot growth
		if rpvgrw > 0.5 {
			result.stability_assessment = .EXCELLENT
		} else if rpvgrw > 0.1 {
			result.stability_assessment = .GOOD
		} else if rpvgrw > machine_epsilon(f32) {
			result.stability_assessment = .MARGINAL
		} else {
			result.stability_assessment = .POOR
		}
	}

	if allocated_s do delete(s)

	return
}

// Complex double precision extended expert solver
zsysvxx :: proc(
	fact: FactFlag,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	af: Matrix(complex128),
	equed: EquilibrationState,
	s: []f64,
	b: Matrix(complex128),
	x: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	berr: []f64 = nil,
	n_err_bnds: int = ERROR_BOUND_TYPES,
	err_bounds_norm: []f64 = nil,
	err_bounds_comp: []f64 = nil,
	nparams: int = 0,
	params: []f64 = nil,
	work: []complex128 = nil,
	rwork: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: ExtendedExpertSolverResult(complex128, f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(af.rows >= n && af.cols >= n, "Matrix AF too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	fact_char: u8 = fact == .FACTORED ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	equed_char: u8 = equed == .APPLIED ? 'Y' : 'N'
	equed_cstring := cstring(&equed_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	rcond: f64
	rpvgrw: f64
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate pivot array if not provided
	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}

	// Allocate scale factors if equilibration is used and not provided
	allocated_s := false
	if equed == .APPLIED && s == nil {
		s = make([]f64, n, allocator)
		allocated_s = true
	}

	// Allocate error arrays if not provided
	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f64, nrhs, allocator)
	}

	allocated_err_bounds_norm := err_bounds_norm == nil
	if allocated_err_bounds_norm {
		err_bounds_norm = make([]f64, nrhs * n_err_bnds, allocator)
	}

	allocated_err_bounds_comp := err_bounds_comp == nil
	if allocated_err_bounds_comp {
		err_bounds_comp = make([]f64, nrhs * n_err_bnds, allocator)
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f64, 2 * n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.zsysvxx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)af.data,
		&ldaf,
		raw_data(ipiv),
		equed_cstring,
		raw_data(s) if s != nil else nil,
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)x.data,
		&ldx,
		&rcond,
		&rpvgrw,
		raw_data(berr),
		&n_err_bnds_int,
		raw_data(err_bounds_norm),
		raw_data(err_bounds_comp),
		&nparams_int,
		raw_data(params) if params != nil else nil,
		cast(^lapack.doublecomplex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.factorization_successful = info == .OK || info_int > n
	result.solution_successful = info == .OK
	result.pivot_indices = ipiv
	result.rcond = rcond
	result.rpvgrw = rpvgrw
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64
	result.backward_errors = berr
	result.scale_factors = s
	result.equilibration_used = equed == .APPLIED
	result.pivot_growth_acceptable = rpvgrw > machine_epsilon(f64)

	// Convert error bounds to matrices
	if nrhs > 0 && n_err_bnds > 0 {
		result.err_bounds_norm = Matrix(f64) {
			data   = err_bounds_norm,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}
		result.err_bounds_comp = Matrix(f64) {
			data   = err_bounds_comp,
			rows   = nrhs,
			cols   = n_err_bnds,
			stride = n_err_bnds,
		}

		result.max_backward_error = slice.max(berr[:nrhs])

		// Assess solution quality
		if rcond > 0.1 && result.max_backward_error < 1e-14 {
			result.solution_quality = .EXCELLENT
		} else if rcond > 0.01 && result.max_backward_error < 1e-10 {
			result.solution_quality = .GOOD
		} else if rcond > machine_epsilon(f64) && result.max_backward_error < 1e-6 {
			result.solution_quality = .ACCEPTABLE
		} else {
			result.solution_quality = .POOR
		}

		// Assess stability based on pivot growth
		if rpvgrw > 0.5 {
			result.stability_assessment = .EXCELLENT
		} else if rpvgrw > 0.1 {
			result.stability_assessment = .GOOD
		} else if rpvgrw > machine_epsilon(f64) {
			result.stability_assessment = .MARGINAL
		} else {
			result.stability_assessment = .POOR
		}
	}

	if allocated_s do delete(s)

	return
}

sysvxx :: proc {
	csysvxx,
	dsysvxx,
	ssysvxx,
	zsysvxx,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Solve symmetric system using default Bunch-Kaufman pivoting
solve_symmetric_system :: proc(
	a: Matrix($T),
	b: Matrix(T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	success: bool,
	pivot_indices: []Blas_Int,
	info: Info,
) {

	n := a.rows
	nrhs := b.cols

	when T == complex64 {
		result, info_val := csysv(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else when T == complex128 {
		result, info_val := zsysv(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else when T == f64 {
		result, info_val := dsysv(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else when T == f32 {
		result, info_val := ssysv(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else {
		#panic("Unsupported type for symmetric system solving")
	}
}

// Solve symmetric system using Aasen algorithm
solve_symmetric_system_aasen :: proc(
	a: Matrix($T),
	b: Matrix(T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	success: bool,
	pivot_indices: []Blas_Int,
	info: Info,
) {

	n := a.rows
	nrhs := b.cols

	when T == complex64 {
		result, info_val := csysv_aa(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else when T == complex128 {
		result, info_val := zsysv_aa(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else when T == f64 {
		result, info_val := dsysv_aa(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else when T == f32 {
		result, info_val := ssysv_aa(uplo, n, nrhs, a, b, allocator = allocator)
		return result.solution_successful, result.pivot_indices, info_val
	} else {
		#panic("Unsupported type for Aasen symmetric system solving")
	}
}

// Check if matrix factorization indicates positive definiteness
is_positive_definite :: proc(result: $R) -> bool {
	when R ==
		SymmetricSolverResult(
			f32,
		) || R == SymmetricSolverResult(f64) || R == SymmetricSolverResult(complex64) || R == SymmetricSolverResult(complex128) || R == AasenSolverResult(f32) || R == AasenSolverResult(f64) || R == AasenSolverResult(complex64) || R == AasenSolverResult(complex128) || R == Aasen2StageSolverResult(f32) || R == Aasen2StageSolverResult(f64) || R == Aasen2StageSolverResult(complex64) || R == Aasen2StageSolverResult(complex128) {
		return(
			result.factorization_successful &&
			!result.is_singular &&
			result.determinant_sign > 0 \
		)
	} else {
		#panic("Unsupported result type for positive definiteness check")
	}
}
