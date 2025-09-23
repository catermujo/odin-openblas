package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC LINEAR SYSTEM SOLVERS
// ============================================================================
// Solves linear systems A*X = B where A is a symmetric matrix stored in packed format

// ============================================================================
// SIMPLE PACKED SYMMETRIC SOLVER
// ============================================================================

// Packed solver result
PackedSolverResult :: struct {
	pivot_indices:  []Blas_Int, // Pivot indices from factorization
	is_singular:    bool, // True if matrix is singular
	singular_index: int, // Index where singularity detected (if singular)
}

// Complex single precision packed solver
cspsv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex64, // Packed matrix (modified to factorization)
	b: Matrix(complex64), // Right-hand side (modified to solution)
	allocator := context.allocator,
) -> (
	result: PackedSolverResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.cspsv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)raw_data(ap),
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1 // Convert to 0-based
	}

	return
}

// Double precision packed solver
dspsv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f64,
	b: Matrix(f64),
	allocator := context.allocator,
) -> (
	result: PackedSolverResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.dspsv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1
	}

	return
}

// Single precision packed solver
sspsv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f32,
	b: Matrix(f32),
	allocator := context.allocator,
) -> (
	result: PackedSolverResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.sspsv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1
	}

	return
}

// Complex double precision packed solver
zspsv :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex128,
	b: Matrix(complex128),
	allocator := context.allocator,
) -> (
	result: PackedSolverResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.zspsv_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)raw_data(ap),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1
	}

	return
}

spsv :: proc {
	cspsv,
	dspsv,
	sspsv,
	zspsv,
}

// ============================================================================
// EXPERT PACKED SYMMETRIC SOLVER
// ============================================================================

// Expert packed solver result
PackedExpertResult :: struct($T: typeid) {
	rcond:                f64, // Reciprocal condition number
	forward_errors:       []T, // Forward error bounds for each RHS
	backward_errors:      []T, // Backward error bounds for each RHS
	pivot_indices:        []Blas_Int, // Pivot indices from factorization
	is_singular:          bool, // True if matrix is singular
	factorization_reused: bool, // True if provided factorization was used
	max_forward_error:    f64, // Maximum forward error
	max_backward_error:   f64, // Maximum backward error
	condition_number:     f64, // Actual condition number (1/rcond)
}

// Complex single precision expert solver
cspsvx :: proc(
	fact: FactorizationOption,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex64, // Original packed matrix
	afp: []complex64 = nil, // Factored packed matrix (in/out)
	ipiv: []Blas_Int = nil, // Pivot indices (in/out)
	b: Matrix(complex64), // Right-hand side
	x: Matrix(complex64), // Solution matrix (output)
	work: []complex64 = nil, // Workspace (size 2*n)
	rwork: []f32 = nil, // Real workspace (size n)
	allocator := context.allocator,
) -> (
	result: PackedExpertResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	fact_char: u8 = fact == .UseProvided ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate factorization arrays if needed
	allocated_afp := afp == nil
	if allocated_afp {
		afp = make([]complex64, n * (n + 1) / 2, allocator)
		if fact == .Compute {
			copy(afp, ap)
		}
	}
	defer if allocated_afp do delete(afp)

	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}
	result.pivot_indices = ipiv

	// Allocate error arrays
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)

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

	rcond: f32

	// Call LAPACK
	lapack.cspsvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)raw_data(ap),
		cast(^lapack.complex)raw_data(afp),
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.complex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = f64(rcond)
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_singular = rcond < machine_epsilon(f32)
	result.factorization_reused = fact == .UseProvided
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64

	// Compute max errors
	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, f64(ferr[i]))
		result.max_backward_error = max(result.max_backward_error, f64(berr[i]))
	}

	return
}

// Double precision expert solver
dspsvx :: proc(
	fact: FactorizationOption,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f64,
	afp: []f64 = nil,
	ipiv: []Blas_Int = nil,
	b: Matrix(f64),
	x: Matrix(f64),
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: PackedExpertResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	fact_char: u8 = fact == .UseProvided ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate factorization arrays if needed
	allocated_afp := afp == nil
	if allocated_afp {
		afp = make([]f64, n * (n + 1) / 2, allocator)
		if fact == .Compute {
			copy(afp, ap)
		}
	}
	defer if allocated_afp do delete(afp)

	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}
	result.pivot_indices = ipiv

	// Allocate error arrays
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)

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

	rcond: f64

	// Call LAPACK
	lapack.dspsvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(afp),
		raw_data(ipiv),
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_singular = rcond < machine_epsilon(f64)
	result.factorization_reused = fact == .UseProvided
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64

	// Compute max errors
	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, ferr[i])
		result.max_backward_error = max(result.max_backward_error, berr[i])
	}

	return
}

// Single precision expert solver
sspsvx :: proc(
	fact: FactorizationOption,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f32,
	afp: []f32 = nil,
	ipiv: []Blas_Int = nil,
	b: Matrix(f32),
	x: Matrix(f32),
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: PackedExpertResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	fact_char: u8 = fact == .UseProvided ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate factorization arrays if needed
	allocated_afp := afp == nil
	if allocated_afp {
		afp = make([]f32, n * (n + 1) / 2, allocator)
		if fact == .Compute {
			copy(afp, ap)
		}
	}
	defer if allocated_afp do delete(afp)

	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}
	result.pivot_indices = ipiv

	// Allocate error arrays
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)

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

	rcond: f32

	// Call LAPACK
	lapack.sspsvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(afp),
		raw_data(ipiv),
		b.data,
		&ldb,
		x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = f64(rcond)
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_singular = rcond < machine_epsilon(f32)
	result.factorization_reused = fact == .UseProvided
	result.condition_number = rcond > 0 ? 1.0 / f64(rcond) : math.INF_F64

	// Compute max errors
	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, f64(ferr[i]))
		result.max_backward_error = max(result.max_backward_error, f64(berr[i]))
	}

	return
}

// Complex double precision expert solver
zspsvx :: proc(
	fact: FactorizationOption,
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex128,
	afp: []complex128 = nil,
	ipiv: []Blas_Int = nil,
	b: Matrix(complex128),
	x: Matrix(complex128),
	work: []complex128 = nil,
	rwork: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: PackedExpertResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	fact_char: u8 = fact == .UseProvided ? 'F' : 'N'
	fact_cstring := cstring(&fact_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate factorization arrays if needed
	allocated_afp := afp == nil
	if allocated_afp {
		afp = make([]complex128, n * (n + 1) / 2, allocator)
		if fact == .Compute {
			copy(afp, ap)
		}
	}
	defer if allocated_afp do delete(afp)

	allocated_ipiv := ipiv == nil
	if allocated_ipiv {
		ipiv = make([]Blas_Int, n, allocator)
	}
	result.pivot_indices = ipiv

	// Allocate error arrays
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)

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

	rcond: f64

	// Call LAPACK
	lapack.zspsvx_(
		fact_cstring,
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)raw_data(ap),
		cast(^lapack.doublecomplex)raw_data(afp),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)x.data,
		&ldx,
		&rcond,
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.doublecomplex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.forward_errors = ferr
	result.backward_errors = berr
	result.is_singular = rcond < machine_epsilon(f64)
	result.factorization_reused = fact == .UseProvided
	result.condition_number = rcond > 0 ? 1.0 / rcond : math.INF_F64

	// Compute max errors
	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, ferr[i])
		result.max_backward_error = max(result.max_backward_error, berr[i])
	}

	return
}

spsvx :: proc {
	cspsvx,
	dspsvx,
	sspsvx,
	zspsvx,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Simple solve for packed symmetric system
solve_packed_symmetric :: proc(
	ap: []$T,
	b: Matrix(T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	solution: Matrix(T),
	info: Info,
) {
	n := b.rows
	nrhs := b.cols

	// Make copies since they get modified
	ap_copy := make([]T, len(ap), allocator)
	copy(ap_copy, ap)
	defer delete(ap_copy)

	solution = matrix_clone(&b, allocator)

	when T == complex64 {
		result, info_val := cspsv(uplo, n, nrhs, ap_copy, solution, allocator)
		defer delete(result.pivot_indices)
		return solution, info_val
	} else when T == complex128 {
		result, info_val := zspsv(uplo, n, nrhs, ap_copy, solution, allocator)
		defer delete(result.pivot_indices)
		return solution, info_val
	} else when T == f64 {
		result, info_val := dspsv(uplo, n, nrhs, ap_copy, solution, allocator)
		defer delete(result.pivot_indices)
		return solution, info_val
	} else when T == f32 {
		result, info_val := sspsv(uplo, n, nrhs, ap_copy, solution, allocator)
		defer delete(result.pivot_indices)
		return solution, info_val
	} else {
		#panic("Unsupported type for packed symmetric solver")
	}
}

// Expert solve with condition estimation
solve_packed_with_conditioning :: proc(
	ap: []$T,
	b: Matrix(T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	solution: Matrix(T),
	condition: f64,
	success: bool,
) {
	n := b.rows
	nrhs := b.cols

	solution = create_matrix(T, n, nrhs, allocator)

	when T == complex64 {
		result, info := cspsvx(
			.Compute,
			uplo,
			n,
			nrhs,
			ap,
			b = b,
			x = solution,
			allocator = allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
			delete(result.pivot_indices)
		}
		condition = result.condition_number
		success = info == .OK && !result.is_singular
	} else when T == complex128 {
		result, info := zspsvx(
			.Compute,
			uplo,
			n,
			nrhs,
			ap,
			b = b,
			x = solution,
			allocator = allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
			delete(result.pivot_indices)
		}
		condition = result.condition_number
		success = info == .OK && !result.is_singular
	} else when T == f64 {
		result, info := dspsvx(
			.Compute,
			uplo,
			n,
			nrhs,
			ap,
			b = b,
			x = solution,
			allocator = allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
			delete(result.pivot_indices)
		}
		condition = result.condition_number
		success = info == .OK && !result.is_singular
	} else when T == f32 {
		result, info := sspsvx(
			.Compute,
			uplo,
			n,
			nrhs,
			ap,
			b = b,
			x = solution,
			allocator = allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
			delete(result.pivot_indices)
		}
		condition = result.condition_number
		success = info == .OK && !result.is_singular
	} else {
		#panic("Unsupported type")
	}

	if !success {
		matrix_delete(&solution)
		solution = Matrix(T){}
	}

	return
}

// Solve multiple systems with same packed coefficient matrix
solve_packed_multiple :: proc(
	ap: []$T,
	rhs_list: []Matrix(T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	solutions: []Matrix(T),
	all_success: bool,
) {
	if len(rhs_list) == 0 {
		return nil, false
	}

	n := rhs_list[0].rows
	solutions = make([]Matrix(T), len(rhs_list), allocator)
	all_success = true

	// Factor once
	afp := make([]T, len(ap), allocator)
	copy(afp, ap)
	defer delete(afp)

	ipiv := make([]Blas_Int, n, allocator)
	defer delete(ipiv)

	// Solve first system and get factorization
	solutions[0] = create_matrix(T, n, rhs_list[0].cols, allocator)

	when T == f64 {
		result, info := dspsvx(
			.Compute,
			uplo,
			n,
			rhs_list[0].cols,
			ap,
			afp = afp,
			ipiv = ipiv,
			b = rhs_list[0],
			x = solutions[0],
			allocator = allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
		}

		if info != .OK || result.is_singular {
			all_success = false
			return
		}

		// Reuse factorization for remaining systems
		for i in 1 ..< len(rhs_list) {
			solutions[i] = create_matrix(T, n, rhs_list[i].cols, allocator)
			result2, info2 := dspsvx(
				.UseProvided,
				uplo,
				n,
				rhs_list[i].cols,
				ap,
				afp = afp,
				ipiv = ipiv,
				b = rhs_list[i],
				x = solutions[i],
				allocator = allocator,
			)
			defer {
				delete(result2.forward_errors)
				delete(result2.backward_errors)
			}

			if info2 != .OK {
				all_success = false
			}
		}
	} else when T == f32 {
		result, info := sspsvx(
			.Compute,
			uplo,
			n,
			rhs_list[0].cols,
			ap,
			afp = afp,
			ipiv = ipiv,
			b = rhs_list[0],
			x = solutions[0],
			allocator = allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
		}

		if info != .OK || result.is_singular {
			all_success = false
			return
		}

		// Reuse factorization for remaining systems
		for i in 1 ..< len(rhs_list) {
			solutions[i] = create_matrix(T, n, rhs_list[i].cols, allocator)
			result2, info2 := sspsvx(
				.UseProvided,
				uplo,
				n,
				rhs_list[i].cols,
				ap,
				afp = afp,
				ipiv = ipiv,
				b = rhs_list[i],
				x = solutions[i],
				allocator = allocator,
			)
			defer {
				delete(result2.forward_errors)
				delete(result2.backward_errors)
			}

			if info2 != .OK {
				all_success = false
			}
		}
	}
	// Add complex cases as needed

	return
}

// Analyze solution quality for packed system
analyze_packed_solution :: proc(
	ap: []$T,
	b: Matrix(T),
	x: Matrix(T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	analysis: SolutionAnalysis,
) {
	n := b.rows
	nrhs := b.cols

	// Create temporary solution matrix
	x_temp := create_matrix(T, n, nrhs, allocator)
	defer matrix_delete(&x_temp)

	when T == f64 {
		result, info := dspsvx(
			.Compute,
			uplo,
			n,
			nrhs,
			ap,
			b = b,
			x = x_temp,
			allocator = allocator,
		)
		defer {
			delete(result.forward_errors)
			delete(result.backward_errors)
			delete(result.pivot_indices)
		}

		analysis.success = info == .OK
		analysis.is_singular = result.is_singular
		analysis.condition_number = result.condition_number
		analysis.rcond = result.rcond
		analysis.max_forward_error = result.max_forward_error
		analysis.max_backward_error = result.max_backward_error

		// Determine quality
		if result.is_singular {
			analysis.quality = .SINGULAR
		} else if result.condition_number > 1e15 {
			analysis.quality = .ILL_CONDITIONED
		} else if result.max_backward_error > 1e-6 {
			analysis.quality = .POOR
		} else if result.max_backward_error > 1e-10 {
			analysis.quality = .FAIR
		} else if result.max_backward_error > 1e-14 {
			analysis.quality = .GOOD
		} else {
			analysis.quality = .EXCELLENT
		}
	}
	// Add other type cases as needed

	return
}

// Solution analysis structure
SolutionAnalysis :: struct {
	success:            bool,
	is_singular:        bool,
	condition_number:   f64,
	rcond:              f64,
	max_forward_error:  f64,
	max_backward_error: f64,
	quality:            SolutionQuality,
}

SolutionQuality :: enum {
	EXCELLENT,
	GOOD,
	FAIR,
	POOR,
	ILL_CONDITIONED,
	SINGULAR,
}
