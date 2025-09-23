package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC ITERATIVE REFINEMENT
// ============================================================================
// Improves the computed solution to a symmetric system of linear equations

// Complex single precision symmetric refinement
csyrfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Original matrix A
	af: Matrix(complex64), // Factored matrix from csytrf
	ipiv: []Blas_Int, // Pivot indices from csytrf
	b: Matrix(complex64), // Right-hand side matrix
	x: Matrix(complex64), // Solution matrix (refined on output)
	ferr: []f32 = nil, // Forward error bounds (size nrhs)
	berr: []f32 = nil, // Backward error bounds (size nrhs)
	work: []complex64 = nil, // Workspace (size 2*n)
	rwork: []f32 = nil, // Real workspace (size n)
	allocator := context.allocator,
) -> (
	result: RefinementResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f32, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f32, nrhs, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f32, n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.csyrfs_(
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
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.complex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	if nrhs > 0 {
		result.max_forward_error = f64(slice.max(ferr[:nrhs]))
		result.max_backward_error = f64(slice.max(berr[:nrhs]))
		result.improved_accuracy = result.max_backward_error < 1.0
	}

	return
}

// Double precision symmetric refinement
dsyrfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	af: Matrix(f64),
	ipiv: []Blas_Int,
	b: Matrix(f64),
	x: Matrix(f64),
	ferr: []f64 = nil,
	berr: []f64 = nil,
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: RefinementResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f64, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f64, nrhs, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.dsyrfs_(
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
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	if nrhs > 0 {
		result.max_forward_error = slice.max(ferr[:nrhs])
		result.max_backward_error = slice.max(berr[:nrhs])
		result.improved_accuracy = result.max_backward_error < 1.0
	}

	return
}

// Single precision symmetric refinement
ssyrfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	af: Matrix(f32),
	ipiv: []Blas_Int,
	b: Matrix(f32),
	x: Matrix(f32),
	ferr: []f32 = nil,
	berr: []f32 = nil,
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: RefinementResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f32, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f32, nrhs, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.ssyrfs_(
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
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	if nrhs > 0 {
		result.max_forward_error = f64(slice.max(ferr[:nrhs]))
		result.max_backward_error = f64(slice.max(berr[:nrhs]))
		result.improved_accuracy = result.max_backward_error < 1.0
	}

	return
}

// Complex double precision symmetric refinement
zsyrfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	af: Matrix(complex128),
	ipiv: []Blas_Int,
	b: Matrix(complex128),
	x: Matrix(complex128),
	ferr: []f64 = nil,
	berr: []f64 = nil,
	work: []complex128 = nil,
	rwork: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: RefinementResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldaf := Blas_Int(af.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays if not provided
	allocated_ferr := ferr == nil
	if allocated_ferr {
		ferr = make([]f64, nrhs, allocator)
	}

	allocated_berr := berr == nil
	if allocated_berr {
		berr = make([]f64, nrhs, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f64, n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.zsyrfs_(
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
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.doublecomplex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	if nrhs > 0 {
		result.max_forward_error = slice.max(ferr[:nrhs])
		result.max_backward_error = slice.max(berr[:nrhs])
		result.improved_accuracy = result.max_backward_error < 1.0
	}

	return
}

syrfs :: proc {
	csyrfs,
	dsyrfs,
	ssyrfs,
	zsyrfs,
}

// ============================================================================
// EXTENDED SYMMETRIC ITERATIVE REFINEMENT
// ============================================================================
// Extended version with equilibration and advanced error bounds


// Error bound types
ERROR_BOUND_TYPES :: 3 // Number of error bound types

// Extended refinement result
ExtendedRefinementResult :: struct($T: typeid) {
	rcond:              f64, // Reciprocal condition number
	backward_errors:    []T, // Backward error bounds
	err_bounds_norm:    Matrix(T), // Normwise error bounds
	err_bounds_comp:    Matrix(T), // Componentwise error bounds
	max_backward_error: f64, // Maximum backward error
	condition_number:   f64, // Condition number (1/rcond)
	equilibration_used: bool, // True if equilibration was used
	solution_quality:   SolutionQuality, // Quality assessment
}

// Solution quality assessment
SolutionQuality :: enum {
	EXCELLENT, // rcond > 0.1, berr < 1e-14
	GOOD, // rcond > 0.01, berr < 1e-10
	ACCEPTABLE, // rcond > machine_epsilon, berr < 1e-6
	POOR, // rcond <= machine_epsilon or berr >= 1e-6
}

// Complex single precision extended refinement
csyrfsx :: proc(
	uplo: UpLoFlag,
	equed: EquilibrationState,
	n: int,
	nrhs: int,
	a: Matrix(complex64),
	af: Matrix(complex64),
	ipiv: []Blas_Int,
	s: []f32, // Scale factors (if equed == APPLIED)
	b: Matrix(complex64),
	x: Matrix(complex64),
	berr: []f32 = nil, // Backward errors (size nrhs)
	n_err_bnds: int = ERROR_BOUND_TYPES,
	err_bounds_norm: []f32 = nil, // Normwise bounds (nrhs * n_err_bnds)
	err_bounds_comp: []f32 = nil, // Componentwise bounds (nrhs * n_err_bnds)
	nparams: int = 0, // Number of parameters
	params: []f32 = nil, // Parameters array
	work: []complex64 = nil,
	rwork: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: ExtendedRefinementResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

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
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate scale factors if equilibration is used
	if equed == .APPLIED && s == nil {
		s = make([]f32, n, allocator)
		defer delete(s)
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

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f32, 3 * n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.csyrfsx_(
		uplo_cstring,
		equed_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)af.data,
		&ldaf,
		raw_data(ipiv),
		raw_data(s) if s != nil else nil,
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)x.data,
		&ldx,
		&rcond,
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
	)

	info = Info(info_int)

	// Fill result
	result.rcond = f64(rcond)
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64
	result.backward_errors = berr
	result.equilibration_used = equed == .APPLIED

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
	}

	return
}

// Double precision extended refinement
dsyrfsx :: proc(
	uplo: UpLoFlag,
	equed: EquilibrationState,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	af: Matrix(f64),
	ipiv: []Blas_Int,
	s: []f64,
	b: Matrix(f64),
	x: Matrix(f64),
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
	result: ExtendedRefinementResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

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
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate scale factors if equilibration is used
	if equed == .APPLIED && s == nil {
		s = make([]f64, n, allocator)
		defer delete(s)
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

	// Allocate workspace if not provided
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
	lapack.dsyrfsx_(
		uplo_cstring,
		equed_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		af.data,
		&ldaf,
		raw_data(ipiv),
		raw_data(s) if s != nil else nil,
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
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
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64
	result.backward_errors = berr
	result.equilibration_used = equed == .APPLIED

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
	}

	return
}

// Single precision extended refinement
ssyrfsx :: proc(
	uplo: UpLoFlag,
	equed: EquilibrationState,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	af: Matrix(f32),
	ipiv: []Blas_Int,
	s: []f32,
	b: Matrix(f32),
	x: Matrix(f32),
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
	result: ExtendedRefinementResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

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
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate scale factors if equilibration is used
	if equed == .APPLIED && s == nil {
		s = make([]f32, n, allocator)
		defer delete(s)
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

	// Allocate workspace if not provided
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
	lapack.ssyrfsx_(
		uplo_cstring,
		equed_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		af.data,
		&ldaf,
		raw_data(ipiv),
		raw_data(s) if s != nil else nil,
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
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
	)

	info = Info(info_int)

	// Fill result
	result.rcond = f64(rcond)
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64
	result.backward_errors = berr
	result.equilibration_used = equed == .APPLIED

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
	}

	return
}

// Complex double precision extended refinement
zsyrfsx :: proc(
	uplo: UpLoFlag,
	equed: EquilibrationState,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	af: Matrix(complex128),
	ipiv: []Blas_Int,
	s: []f64,
	b: Matrix(complex128),
	x: Matrix(complex128),
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
	result: ExtendedRefinementResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Original matrix too small")
	assert(af.rows >= n && af.cols >= n, "Factored matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

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
	n_err_bnds_int := Blas_Int(n_err_bnds)
	nparams_int := Blas_Int(nparams)
	info_int: Info

	// Allocate scale factors if equilibration is used
	if equed == .APPLIED && s == nil {
		s = make([]f64, n, allocator)
		defer delete(s)
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

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f64, 3 * n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.zsyrfsx_(
		uplo_cstring,
		equed_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)af.data,
		&ldaf,
		raw_data(ipiv),
		raw_data(s) if s != nil else nil,
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)x.data,
		&ldx,
		&rcond,
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
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64
	result.backward_errors = berr
	result.equilibration_used = equed == .APPLIED

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
	}

	return
}

syrfsx :: proc {
	csyrfsx,
	dsyrfsx,
	ssyrfsx,
	zsyrfsx,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Refine symmetric solution
refine_symmetric_solution :: proc(
	a: Matrix($T),
	af: Matrix(T),
	ipiv: []Blas_Int,
	b: Matrix(T),
	x: Matrix(T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	forward_errors: []$S,
	backward_errors: []S,
	improved: bool,
	info: Info,
) {

	n := a.rows
	nrhs := b.cols

	when T == complex64 {
		result, info_val := csyrfs(uplo, n, nrhs, a, af, ipiv, b, x, allocator = allocator)
		return result.forward_errors, result.backward_errors, result.improved_accuracy, info_val
	} else when T == complex128 {
		result, info_val := zsyrfs(uplo, n, nrhs, a, af, ipiv, b, x, allocator = allocator)
		return result.forward_errors, result.backward_errors, result.improved_accuracy, info_val
	} else when T == f64 {
		result, info_val := dsyrfs(uplo, n, nrhs, a, af, ipiv, b, x, allocator = allocator)
		return result.forward_errors, result.backward_errors, result.improved_accuracy, info_val
	} else when T == f32 {
		result, info_val := ssyrfs(uplo, n, nrhs, a, af, ipiv, b, x, allocator = allocator)
		return result.forward_errors, result.backward_errors, result.improved_accuracy, info_val
	} else {
		#panic("Unsupported type for symmetric refinement")
	}
}

// Extended refinement with error bounds
refine_symmetric_solution_extended :: proc(
	a: Matrix($T),
	af: Matrix(T),
	ipiv: []Blas_Int,
	s: []$S,
	b: Matrix(T),
	x: Matrix(T),
	equed := EquilibrationState.NONE,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	rcond: f64,
	backward_errors: []S,
	quality: SolutionQuality,
	info: Info,
) {

	n := a.rows
	nrhs := b.cols

	when T == complex64 {
		result, info_val := csyrfsx(
			uplo,
			equed,
			n,
			nrhs,
			a,
			af,
			ipiv,
			s,
			b,
			x,
			allocator = allocator,
		)
		return result.rcond, result.backward_errors, result.solution_quality, info_val
	} else when T == complex128 {
		result, info_val := zsyrfsx(
			uplo,
			equed,
			n,
			nrhs,
			a,
			af,
			ipiv,
			s,
			b,
			x,
			allocator = allocator,
		)
		return result.rcond, result.backward_errors, result.solution_quality, info_val
	} else when T == f64 {
		result, info_val := dsyrfsx(
			uplo,
			equed,
			n,
			nrhs,
			a,
			af,
			ipiv,
			s,
			b,
			x,
			allocator = allocator,
		)
		return result.rcond, result.backward_errors, result.solution_quality, info_val
	} else when T == f32 {
		result, info_val := ssyrfsx(
			uplo,
			equed,
			n,
			nrhs,
			a,
			af,
			ipiv,
			s,
			b,
			x,
			allocator = allocator,
		)
		return result.rcond, result.backward_errors, result.solution_quality, info_val
	} else {
		#panic("Unsupported type for extended symmetric refinement")
	}
}
