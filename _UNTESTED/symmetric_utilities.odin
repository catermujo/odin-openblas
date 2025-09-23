package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC MATRIX UTILITIES
// ============================================================================
// Utility functions for symmetric matrix operations

// ============================================================================
// SYMMETRIC MATRIX ROW/COLUMN SWAPPING
// ============================================================================
// Functions to swap rows and columns while maintaining symmetry

// Complex single precision symmetric swap
csyswapr :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64),
	i1: int, // First index (1-based)
	i2: int, // Second index (1-based)
) -> Info {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(i1 >= 1 && i1 <= n, "First index out of range")
	assert(i2 >= 1 && i2 <= n, "Second index out of range")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	i1_int := Blas_Int(i1)
	i2_int := Blas_Int(i2)

	// Call LAPACK
	lapack.csyswapr_(uplo_cstring, &n_int, cast(^lapack.complex)a.data, &lda, &i1_int, &i2_int, 1)

	return .OK
}

// Double precision symmetric swap
dsyswapr :: proc(uplo: UpLoFlag, n: int, a: Matrix(f64), i1: int, i2: int) -> Info {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(i1 >= 1 && i1 <= n, "First index out of range")
	assert(i2 >= 1 && i2 <= n, "Second index out of range")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	i1_int := Blas_Int(i1)
	i2_int := Blas_Int(i2)

	// Call LAPACK
	lapack.dsyswapr_(uplo_cstring, &n_int, a.data, &lda, &i1_int, &i2_int, 1)

	return .OK
}

// Single precision symmetric swap
ssyswapr :: proc(uplo: UpLoFlag, n: int, a: Matrix(f32), i1: int, i2: int) -> Info {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(i1 >= 1 && i1 <= n, "First index out of range")
	assert(i2 >= 1 && i2 <= n, "Second index out of range")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	i1_int := Blas_Int(i1)
	i2_int := Blas_Int(i2)

	// Call LAPACK
	lapack.ssyswapr_(uplo_cstring, &n_int, a.data, &lda, &i1_int, &i2_int, 1)

	return .OK
}

// Complex double precision symmetric swap
zsyswapr :: proc(uplo: UpLoFlag, n: int, a: Matrix(complex128), i1: int, i2: int) -> Info {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(i1 >= 1 && i1 <= n, "First index out of range")
	assert(i2 >= 1 && i2 <= n, "Second index out of range")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	i1_int := Blas_Int(i1)
	i2_int := Blas_Int(i2)

	// Call LAPACK
	lapack.zsyswapr_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		&i1_int,
		&i2_int,
		1,
	)

	return .OK
}

syswapr :: proc {
	csyswapr,
	dsyswapr,
	ssyswapr,
	zsyswapr,
}

// ============================================================================
// SYMMETRIC TRIDIAGONALIZATION
// ============================================================================
// Reduces symmetric matrices to tridiagonal form

// Tridiagonalization result
TridiagonalizationResult :: struct($T: typeid) {
	diagonal:          []T, // Diagonal elements
	off_diagonal:      []T, // Off-diagonal elements
	reflector_scalars: []T, // Householder reflector scalars
	success:           bool,
}

// 2-stage tridiagonalization result
Tridiagonalization2StageResult :: struct($T: typeid) {
	diagonal:            []T, // Diagonal elements
	off_diagonal:        []T, // Off-diagonal elements
	reflector_scalars:   []T, // Householder reflector scalars
	householder_vectors: []T, // Additional Householder vectors for 2-stage
	success:             bool,
}

// Transformation type for 2-stage algorithms
TransformationType :: enum {
	NO_VECTORS, // 'N' - Do not compute transformation matrix
	VECTORS, // 'V' - Compute transformation matrix
}

// Double precision symmetric tridiagonalization
dsytrd :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64), // Input symmetric matrix, tridiagonal on output
	d: []f64 = nil, // Diagonal elements (size n)
	e: []f64 = nil, // Off-diagonal elements (size n-1)
	tau: []f64 = nil, // Householder reflector scalars (size n-1)
	work: []f64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: TridiagonalizationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate output arrays if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f64, n, allocator)
	}

	allocated_e := e == nil
	if allocated_e {
		e = make([]f64, max(1, n - 1), allocator)
	}

	allocated_tau := tau == nil
	if allocated_tau {
		tau = make([]f64, max(1, n - 1), allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsytrd_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(d),
			raw_data(e),
			raw_data(tau),
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsytrd_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(d),
		raw_data(e),
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.diagonal = d
	result.off_diagonal = e
	result.reflector_scalars = tau
	result.success = info == .OK

	return
}

// Single precision symmetric tridiagonalization
ssytrd :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	d: []f32 = nil,
	e: []f32 = nil,
	tau: []f32 = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: TridiagonalizationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate output arrays if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f32, n, allocator)
	}

	allocated_e := e == nil
	if allocated_e {
		e = make([]f32, max(1, n - 1), allocator)
	}

	allocated_tau := tau == nil
	if allocated_tau {
		tau = make([]f32, max(1, n - 1), allocator)
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssytrd_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(d),
			raw_data(e),
			raw_data(tau),
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssytrd_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(d),
		raw_data(e),
		raw_data(tau),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.diagonal = d
	result.off_diagonal = e
	result.reflector_scalars = tau
	result.success = info == .OK

	return
}

// Double precision 2-stage symmetric tridiagonalization
dsytrd_2stage :: proc(
	vect: TransformationType,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64), // Input symmetric matrix, tridiagonal on output
	d: []f64 = nil, // Diagonal elements (size n)
	e: []f64 = nil, // Off-diagonal elements (size n-1)
	tau: []f64 = nil, // Householder reflector scalars (size n-1)
	hous2: []f64 = nil, // Additional Householder vectors (query if nil)
	work: []f64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: Tridiagonalization2StageResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	vect_char: u8 = vect == .VECTORS ? 'V' : 'N'
	vect_cstring := cstring(&vect_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate output arrays if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f64, n, allocator)
	}

	allocated_e := e == nil
	if allocated_e {
		e = make([]f64, max(1, n - 1), allocator)
	}

	allocated_tau := tau == nil
	if allocated_tau {
		tau = make([]f64, max(1, n - 1), allocator)
	}

	// Query HOUS2 size if not provided
	allocated_hous2 := hous2 == nil
	lhous2: Blas_Int
	if allocated_hous2 {
		// Query HOUS2 size first
		lhous2_query := Blas_Int(-1)
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsytrd_2stage_(
			vect_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(d),
			raw_data(e),
			raw_data(tau),
			&work_query, // Temporary for HOUS2 query
			&lhous2_query,
			&work_query,
			&lwork_query,
			&info_int,
			1,
			1,
		)

		if info_int == 0 {
			lhous2 = Blas_Int(work_query)
			hous2 = make([]f64, lhous2, allocator)
		} else {
			lhous2 = max(1, 2 * n)
			hous2 = make([]f64, lhous2, allocator)
		}
	} else {
		lhous2 = Blas_Int(len(hous2))
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsytrd_2stage_(
			vect_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(d),
			raw_data(e),
			raw_data(tau),
			raw_data(hous2),
			&lhous2,
			&work_query,
			&lwork_query,
			&info_int,
			1,
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
	lapack.dsytrd_2stage_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(d),
		raw_data(e),
		raw_data(tau),
		raw_data(hous2),
		&lhous2,
		raw_data(work),
		&lwork,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.diagonal = d
	result.off_diagonal = e
	result.reflector_scalars = tau
	result.householder_vectors = hous2
	result.success = info == .OK

	return
}

// Single precision 2-stage symmetric tridiagonalization
ssytrd_2stage :: proc(
	vect: TransformationType,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	d: []f32 = nil,
	e: []f32 = nil,
	tau: []f32 = nil,
	hous2: []f32 = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: Tridiagonalization2StageResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	vect_char: u8 = vect == .VECTORS ? 'V' : 'N'
	vect_cstring := cstring(&vect_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate output arrays if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f32, n, allocator)
	}

	allocated_e := e == nil
	if allocated_e {
		e = make([]f32, max(1, n - 1), allocator)
	}

	allocated_tau := tau == nil
	if allocated_tau {
		tau = make([]f32, max(1, n - 1), allocator)
	}

	// Query HOUS2 size if not provided
	allocated_hous2 := hous2 == nil
	lhous2: Blas_Int
	if allocated_hous2 {
		// Query HOUS2 size first
		lhous2_query := Blas_Int(-1)
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssytrd_2stage_(
			vect_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(d),
			raw_data(e),
			raw_data(tau),
			&work_query, // Temporary for HOUS2 query
			&lhous2_query,
			&work_query,
			&lwork_query,
			&info_int,
			1,
			1,
		)

		if info_int == 0 {
			lhous2 = Blas_Int(work_query)
			hous2 = make([]f32, lhous2, allocator)
		} else {
			lhous2 = max(1, 2 * n)
			hous2 = make([]f32, lhous2, allocator)
		}
	} else {
		lhous2 = Blas_Int(len(hous2))
	}

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssytrd_2stage_(
			vect_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(d),
			raw_data(e),
			raw_data(tau),
			raw_data(hous2),
			&lhous2,
			&work_query,
			&lwork_query,
			&info_int,
			1,
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
	lapack.ssytrd_2stage_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(d),
		raw_data(e),
		raw_data(tau),
		raw_data(hous2),
		&lhous2,
		raw_data(work),
		&lwork,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.diagonal = d
	result.off_diagonal = e
	result.reflector_scalars = tau
	result.householder_vectors = hous2
	result.success = info == .OK

	return
}

sytrd :: proc {
	dsytrd,
	ssytrd,
}
sytrd_2stage :: proc {
	dsytrd_2stage,
	ssytrd_2stage,
}

// ============================================================================
// SYMMETRIC MATRIX FACTORIZATION (BUNCH-KAUFMAN)
// ============================================================================
// Bunch-Kaufman factorization of symmetric indefinite matrices

// Factorization result
FactorizationResult :: struct($T: typeid) {
	pivot_indices:            []Blas_Int, // Pivot indices
	factorization_successful: bool,
	is_singular:              bool,
	determinant_sign:         int, // Sign of determinant
}

// Aasen factorization result
AasenFactorizationResult :: struct($T: typeid) {
	pivot_indices:            []Blas_Int, // Pivot indices
	tridiagonal_factor:       Matrix(T), // T matrix from Aasen factorization
	factorization_successful: bool,
	is_singular:              bool,
	determinant_sign:         int,
}

// 2-stage Aasen factorization result
Aasen2StageFactorizationResult :: struct($T: typeid) {
	pivot_indices_1:          []Blas_Int, // First stage pivots
	pivot_indices_2:          []Blas_Int, // Second stage pivots
	band_matrix:              Matrix(T), // TB band matrix
	factorization_successful: bool,
	is_singular:              bool,
	determinant_sign:         int,
}

// RK (bounded Bunch-Kaufman) factorization result
RKFactorizationResult :: struct($T: typeid) {
	pivot_indices:            []Blas_Int, // Pivot indices
	e_factor:                 []T, // E vector from RK factorization
	factorization_successful: bool,
	is_singular:              bool,
	determinant_sign:         int,
}

// Complex single precision symmetric factorization
csytrf :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: FactorizationResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.csytrf_(
			uplo_cstring,
			&n_int,
			cast(^lapack.complex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.complex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex64, lwork, allocator)
		} else {
			lwork = max(1, n)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csytrf_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

// Double precision symmetric factorization
dsytrf :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: FactorizationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsytrf_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f64, lwork, allocator)
		} else {
			lwork = max(1, n)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsytrf_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

// Single precision symmetric factorization
ssytrf :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: FactorizationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssytrf_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
			&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(work_query)
			work = make([]f32, lwork, allocator)
		} else {
			lwork = max(1, n)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssytrf_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

// Complex double precision symmetric factorization
zsytrf :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: FactorizationResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.zsytrf_(
			uplo_cstring,
			&n_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			raw_data(ipiv),
			cast(^lapack.doublecomplex)&work_query,
			&lwork_query,
			&info_int,
			1,
		)

		if info_int == 0 {
			lwork = Blas_Int(real(work_query))
			work = make([]complex128, lwork, allocator)
		} else {
			lwork = max(1, n)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsytrf_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

sytrf :: proc {
	csytrf,
	dsytrf,
	ssytrf,
	zsytrf,
}

// ============================================================================
// AASEN SYMMETRIC FACTORIZATION
// ============================================================================
// Aasen's algorithm for symmetric indefinite matrices

// Complex single precision Aasen factorization
csytrf_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: AasenFactorizationResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.csytrf_aa_(
			uplo_cstring,
			&n_int,
			cast(^lapack.complex)a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.csytrf_aa_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.tridiagonal_factor = a // Aasen factorization stores T in A
	result.factorization_successful = info == .OK
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

	return
}

// Double precision Aasen factorization
dsytrf_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: AasenFactorizationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsytrf_aa_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.dsytrf_aa_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.tridiagonal_factor = a
	result.factorization_successful = info == .OK
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

	return
}

// Single precision Aasen factorization
ssytrf_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: AasenFactorizationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssytrf_aa_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.ssytrf_aa_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.tridiagonal_factor = a
	result.factorization_successful = info == .OK
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

	return
}

// Complex double precision Aasen factorization
zsytrf_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: AasenFactorizationResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.zsytrf_aa_(
			uplo_cstring,
			&n_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.zsytrf_aa_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.tridiagonal_factor = a
	result.factorization_successful = info == .OK
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

	return
}

sytrf_aa :: proc {
	csytrf_aa,
	dsytrf_aa,
	ssytrf_aa,
	zsytrf_aa,
}

// ============================================================================
// 2-STAGE AASEN SYMMETRIC FACTORIZATION
// ============================================================================
// Two-stage Aasen algorithm for improved performance on large matrices

// Complex single precision 2-stage Aasen factorization
csytrf_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	tb: Matrix(complex64), // Band matrix storage (4*n, nb)
	ipiv: []Blas_Int = nil, // First stage pivot indices (size n)
	ipiv2: []Blas_Int = nil, // Second stage pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: Aasen2StageFactorizationResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
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

		lapack.csytrf_aa_2stage_(
			uplo_cstring,
			&n_int,
			cast(^lapack.complex)a.data,
			&lda,
			cast(^lapack.complex)tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
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
	lapack.csytrf_aa_2stage_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.factorization_successful = info == .OK
	result.is_singular = info_int > 0

	// Estimate determinant sign from both pivot arrays
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

	return
}

// Double precision 2-stage Aasen factorization
dsytrf_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	tb: Matrix(f64),
	ipiv: []Blas_Int = nil,
	ipiv2: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: Aasen2StageFactorizationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
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

		lapack.dsytrf_aa_2stage_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
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
	lapack.dsytrf_aa_2stage_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.factorization_successful = info == .OK
	result.is_singular = info_int > 0

	// Estimate determinant sign from both pivot arrays
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

	return
}

// Single precision 2-stage Aasen factorization
ssytrf_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	tb: Matrix(f32),
	ipiv: []Blas_Int = nil,
	ipiv2: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: Aasen2StageFactorizationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
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

		lapack.ssytrf_aa_2stage_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
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
	lapack.ssytrf_aa_2stage_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.factorization_successful = info == .OK
	result.is_singular = info_int > 0

	// Estimate determinant sign from both pivot arrays
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

	return
}

// Complex double precision 2-stage Aasen factorization
zsytrf_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	tb: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	ipiv2: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: Aasen2StageFactorizationResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
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

		lapack.zsytrf_aa_2stage_(
			uplo_cstring,
			&n_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			cast(^lapack.doublecomplex)tb.data,
			&ltb,
			raw_data(ipiv),
			raw_data(ipiv2),
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
	lapack.zsytrf_aa_2stage_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)tb.data,
		&ltb,
		raw_data(ipiv),
		raw_data(ipiv2),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices_1 = ipiv
	result.pivot_indices_2 = ipiv2
	result.band_matrix = tb
	result.factorization_successful = info == .OK
	result.is_singular = info_int > 0

	// Estimate determinant sign from both pivot arrays
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

	return
}

sytrf_aa_2stage :: proc {
	csytrf_aa_2stage,
	dsytrf_aa_2stage,
	ssytrf_aa_2stage,
	zsytrf_aa_2stage,
}

// ============================================================================
// RK (BOUNDED BUNCH-KAUFMAN) FACTORIZATION
// ============================================================================
// Bounded Bunch-Kaufman factorization with additional E factor

// Complex single precision RK factorization
csytrf_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	e: []complex64, // E factor from RK factorization (size n)
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: RKFactorizationResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.csytrf_rk_(
			uplo_cstring,
			&n_int,
			cast(^lapack.complex)a.data,
			&lda,
			cast(^lapack.complex)raw_data(e),
			raw_data(ipiv),
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
	lapack.csytrf_rk_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.e_factor = e
	result.factorization_successful = info == .OK
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

	return
}

// Double precision RK factorization
dsytrf_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	e: []f64,
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: RKFactorizationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsytrf_rk_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(e),
			raw_data(ipiv),
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
	lapack.dsytrf_rk_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.e_factor = e
	result.factorization_successful = info == .OK
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

	return
}

// Single precision RK factorization
ssytrf_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	e: []f32,
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: RKFactorizationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssytrf_rk_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(e),
			raw_data(ipiv),
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
	lapack.ssytrf_rk_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.e_factor = e
	result.factorization_successful = info == .OK
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

	return
}

// Complex double precision RK factorization
zsytrf_rk :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	e: []complex128,
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: RKFactorizationResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.zsytrf_rk_(
			uplo_cstring,
			&n_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			cast(^lapack.doublecomplex)raw_data(e),
			raw_data(ipiv),
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
	lapack.zsytrf_rk_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.e_factor = e
	result.factorization_successful = info == .OK
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

	return
}

sytrf_rk :: proc {
	csytrf_rk,
	dsytrf_rk,
	ssytrf_rk,
	zsytrf_rk,
}

// ============================================================================
// ROOK PIVOTING FACTORIZATION
// ============================================================================
// Rook pivoting factorization for enhanced numerical stability

// Rook pivoting factorization result structure
RookFactorizationResult :: struct($T: typeid) {
	pivot_indices:            []Blas_Int, // Pivot indices from rook pivoting
	factorization_successful: bool,
	is_singular:              bool,
	determinant_sign:         int,
}

// Complex single precision rook pivoting factorization
csytrf_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Input matrix, factorized on output
	ipiv: []Blas_Int = nil, // Pivot indices (size n)
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: RookFactorizationResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.csytrf_rook_(
			uplo_cstring,
			&n_int,
			cast(^lapack.complex)a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.csytrf_rook_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

// Double precision rook pivoting factorization
dsytrf_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: RookFactorizationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsytrf_rook_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.dsytrf_rook_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

// Single precision rook pivoting factorization
ssytrf_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: RookFactorizationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssytrf_rook_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.ssytrf_rook_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

// Complex double precision rook pivoting factorization
zsytrf_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: RookFactorizationResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.zsytrf_rook_(
			uplo_cstring,
			&n_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.zsytrf_rook_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.pivot_indices = ipiv
	result.factorization_successful = info == .OK
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

	return
}

sytrf_rook :: proc {
	csytrf_rook,
	dsytrf_rook,
	ssytrf_rook,
	zsytrf_rook,
}

// ============================================================================
// SYMMETRIC MATRIX INVERSION
// ============================================================================
// Inversion of symmetric indefinite matrices using factorizations

// Matrix inversion result structure
InversionResult :: struct($T: typeid) {
	inversion_successful: bool,
	is_singular:          bool,
}

// ============================================================================
// STANDARD SYMMETRIC MATRIX INVERSION
// ============================================================================
// Standard inversion using Bunch-Kaufman factorization

// Complex single precision symmetric matrix inversion
csytri :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Factorized matrix, inverse on output
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []complex64 = nil, // Workspace (size n)
	allocator := context.allocator,
) -> (
	result: InversionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csytri_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision symmetric matrix inversion
dsytri :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsytri_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision symmetric matrix inversion
ssytri :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssytri_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision symmetric matrix inversion
zsytri :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsytri_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytri :: proc {
	csytri,
	dsytri,
	ssytri,
	zsytri,
}

// ============================================================================
// IMPROVED SYMMETRIC MATRIX INVERSION (SYTRI2)
// ============================================================================
// Improved inversion algorithm with better cache efficiency

// Complex single precision improved symmetric matrix inversion
csytri2 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Factorized matrix, inverse on output
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: InversionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csytri2_(
			uplo_cstring,
			&n_int,
			cast(^lapack.complex)a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.csytri2_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision improved symmetric matrix inversion
dsytri2 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsytri2_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.dsytri2_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision improved symmetric matrix inversion
ssytri2 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssytri2_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.ssytri2_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision improved symmetric matrix inversion
zsytri2 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsytri2_(
			uplo_cstring,
			&n_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			raw_data(ipiv),
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
	lapack.zsytri2_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytri2 :: proc {
	csytri2,
	dsytri2,
	ssytri2,
	zsytri2,
}

// ============================================================================
// BLOCK-BASED SYMMETRIC MATRIX INVERSION (SYTRI2X)
// ============================================================================
// Block-based inversion algorithm for improved performance

// Complex single precision block-based symmetric matrix inversion
csytri2x :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Factorized matrix, inverse on output
	ipiv: []Blas_Int, // Pivot indices from factorization
	nb: int = 0, // Block size (0 = automatic)
	work: []complex64 = nil, // Workspace
	allocator := context.allocator,
) -> (
	result: InversionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	nb_int := Blas_Int(nb == 0 ? 64 : nb) // Default block size
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work_size := nb_int * (n_int + nb_int)
		work = make([]complex64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csytri2x_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&nb_int,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision block-based symmetric matrix inversion
dsytri2x :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	nb: int = 0,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	nb_int := Blas_Int(nb == 0 ? 64 : nb)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work_size := nb_int * (n_int + nb_int)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsytri2x_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&nb_int,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision block-based symmetric matrix inversion
ssytri2x :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	nb: int = 0,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	nb_int := Blas_Int(nb == 0 ? 64 : nb)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work_size := nb_int * (n_int + nb_int)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssytri2x_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(work),
		&nb_int,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision block-based symmetric matrix inversion
zsytri2x :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	nb: int = 0,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	nb_int := Blas_Int(nb == 0 ? 64 : nb)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work_size := nb_int * (n_int + nb_int)
		work = make([]complex128, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsytri2x_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&nb_int,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytri2x :: proc {
	csytri2x,
	dsytri2x,
	ssytri2x,
	zsytri2x,
}

// ============================================================================
// RK INVERSION WITH E FACTOR (SYTRI_3)
// ============================================================================
// Inversion using RK factorization with E factor

// Complex single precision RK inversion with E factor
csytri_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // RK factorized matrix, inverse on output
	e: []complex64, // E factor from RK factorization
	ipiv: []Blas_Int, // Pivot indices from RK factorization
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: InversionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csytri_3_(
			uplo_cstring,
			&n_int,
			cast(^lapack.complex)a.data,
			&lda,
			cast(^lapack.complex)raw_data(e),
			raw_data(ipiv),
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
	lapack.csytri_3_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision RK inversion with E factor
dsytri_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	e: []f64,
	ipiv: []Blas_Int,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsytri_3_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(e),
			raw_data(ipiv),
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
	lapack.dsytri_3_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision RK inversion with E factor
ssytri_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	e: []f32,
	ipiv: []Blas_Int,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssytri_3_(
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			raw_data(e),
			raw_data(ipiv),
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
	lapack.ssytri_3_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision RK inversion with E factor
zsytri_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	e: []complex128,
	ipiv: []Blas_Int,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: InversionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsytri_3_(
			uplo_cstring,
			&n_int,
			cast(^lapack.doublecomplex)a.data,
			&lda,
			cast(^lapack.doublecomplex)raw_data(e),
			raw_data(ipiv),
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
	lapack.zsytri_3_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.inversion_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytri_3 :: proc {
	csytri_3,
	dsytri_3,
	ssytri_3,
	zsytri_3,
}

// ============================================================================
// SYMMETRIC SYSTEM SOLUTION
// ============================================================================
// Solving symmetric systems using factorizations

// System solution result structure
SolutionResult :: struct($T: typeid) {
	solution_successful: bool,
	is_singular:         bool,
}

// ============================================================================
// STANDARD SYMMETRIC SYSTEM SOLUTION (SYTRS)
// ============================================================================
// Standard solution using Bunch-Kaufman factorization

// Complex single precision symmetric system solution
csytrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Factorized matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: Matrix(complex64), // RHS on input, solution on output
) -> (
	result: SolutionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.csytrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision symmetric system solution
dsytrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	b: Matrix(f64),
) -> (
	result: SolutionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.dsytrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision symmetric system solution
ssytrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	b: Matrix(f32),
) -> (
	result: SolutionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.ssytrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision symmetric system solution
zsytrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	b: Matrix(complex128),
) -> (
	result: SolutionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.zsytrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytrs :: proc {
	csytrs,
	dsytrs,
	ssytrs,
	zsytrs,
}

// ============================================================================
// IMPROVED SYMMETRIC SYSTEM SOLUTION (SYTRS2)
// ============================================================================
// Improved solution algorithm with workspace usage

// Complex single precision improved symmetric system solution
csytrs2 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Factorized matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: Matrix(complex64), // RHS on input, solution on output
	work: []complex64 = nil, // Workspace (size n)
	allocator := context.allocator,
) -> (
	result: SolutionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csytrs2_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision improved symmetric system solution
dsytrs2 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	b: Matrix(f64),
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: SolutionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsytrs2_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision improved symmetric system solution
ssytrs2 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	b: Matrix(f32),
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: SolutionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssytrs2_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision improved symmetric system solution
zsytrs2 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	b: Matrix(complex128),
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: SolutionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsytrs2_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytrs2 :: proc {
	csytrs2,
	dsytrs2,
	ssytrs2,
	zsytrs2,
}

// ============================================================================
// RK SYSTEM SOLUTION WITH E FACTOR (SYTRS_3)
// ============================================================================
// System solution using RK factorization with E factor

// Complex single precision RK system solution with E factor
csytrs_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // RK factorized matrix
	e: []complex64, // E factor from RK factorization
	ipiv: []Blas_Int, // Pivot indices from RK factorization
	b: Matrix(complex64), // RHS on input, solution on output
) -> (
	result: SolutionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.csytrs_3_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision RK system solution with E factor
dsytrs_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	e: []f64,
	ipiv: []Blas_Int,
	b: Matrix(f64),
) -> (
	result: SolutionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.dsytrs_3_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision RK system solution with E factor
ssytrs_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	e: []f32,
	ipiv: []Blas_Int,
	b: Matrix(f32),
) -> (
	result: SolutionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.ssytrs_3_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(e),
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision RK system solution with E factor
zsytrs_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	e: []complex128,
	ipiv: []Blas_Int,
	b: Matrix(complex128),
) -> (
	result: SolutionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.zsytrs_3_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)raw_data(e),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytrs_3 :: proc {
	csytrs_3,
	dsytrs_3,
	ssytrs_3,
	zsytrs_3,
}

// ============================================================================
// AASEN SYSTEM SOLUTION (SYTRS_AA)
// ============================================================================
// System solution using Aasen factorization

// Complex single precision Aasen system solution
csytrs_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Aasen factorized matrix
	ipiv: []Blas_Int, // Pivot indices from Aasen factorization
	b: Matrix(complex64), // RHS on input, solution on output
	work: []complex64 = nil, // Workspace (query if nil)
	allocator := context.allocator,
) -> (
	result: SolutionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex64
		lwork_query := Blas_Int(-1)

		lapack.csytrs_aa_(
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
			lwork = max(1, 3 * n - 2)
			work = make([]complex64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csytrs_aa_(
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
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision Aasen system solution
dsytrs_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	b: Matrix(f64),
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: SolutionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsytrs_aa_(
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
			lwork = max(1, 3 * n - 2)
			work = make([]f64, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsytrs_aa_(
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
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision Aasen system solution
ssytrs_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	b: Matrix(f32),
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: SolutionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssytrs_aa_(
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
			lwork = max(1, 3 * n - 2)
			work = make([]f32, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssytrs_aa_(
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
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision Aasen system solution
zsytrs_aa :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	b: Matrix(complex128),
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: SolutionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Query optimal workspace size if not provided
	allocated_work := work == nil
	lwork: Blas_Int
	if allocated_work {
		work_query: complex128
		lwork_query := Blas_Int(-1)

		lapack.zsytrs_aa_(
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
			lwork = max(1, 3 * n - 2)
			work = make([]complex128, lwork, allocator)
		}
	} else {
		lwork = Blas_Int(len(work))
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsytrs_aa_(
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
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytrs_aa :: proc {
	csytrs_aa,
	dsytrs_aa,
	ssytrs_aa,
	zsytrs_aa,
}

// ============================================================================
// 2-STAGE AASEN SYSTEM SOLUTION (SYTRS_AA_2STAGE)
// ============================================================================
// System solution using 2-stage Aasen factorization

// Complex single precision 2-stage Aasen system solution
csytrs_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // 2-stage Aasen factorized matrix
	tb: Matrix(complex64), // Band matrix from 2-stage factorization
	ipiv: []Blas_Int, // First stage pivot indices
	ipiv2: []Blas_Int, // Second stage pivot indices
	b: Matrix(complex64), // RHS on input, solution on output
) -> (
	result: SolutionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.csytrs_aa_2stage_(
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
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision 2-stage Aasen system solution
dsytrs_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	tb: Matrix(f64),
	ipiv: []Blas_Int,
	ipiv2: []Blas_Int,
	b: Matrix(f64),
) -> (
	result: SolutionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.dsytrs_aa_2stage_(
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
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision 2-stage Aasen system solution
ssytrs_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	tb: Matrix(f32),
	ipiv: []Blas_Int,
	ipiv2: []Blas_Int,
	b: Matrix(f32),
) -> (
	result: SolutionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.ssytrs_aa_2stage_(
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
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision 2-stage Aasen system solution
zsytrs_aa_2stage :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	tb: Matrix(complex128),
	ipiv: []Blas_Int,
	ipiv2: []Blas_Int,
	b: Matrix(complex128),
) -> (
	result: SolutionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ltb := Blas_Int(tb.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.zsytrs_aa_2stage_(
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
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytrs_aa_2stage :: proc {
	csytrs_aa_2stage,
	dsytrs_aa_2stage,
	ssytrs_aa_2stage,
	zsytrs_aa_2stage,
}

// ============================================================================
// ROOK PIVOTING SYSTEM SOLUTION (SYTRS_ROOK)
// ============================================================================
// System solution using rook pivoting factorization

// Complex single precision rook pivoting system solution
csytrs_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex64), // Rook factorized matrix
	ipiv: []Blas_Int, // Pivot indices from rook factorization
	b: Matrix(complex64), // RHS on input, solution on output
) -> (
	result: SolutionResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.csytrs_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Double precision rook pivoting system solution
dsytrs_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	b: Matrix(f64),
) -> (
	result: SolutionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.dsytrs_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Single precision rook pivoting system solution
ssytrs_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	b: Matrix(f32),
) -> (
	result: SolutionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.ssytrs_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		a.data,
		&lda,
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

// Complex double precision rook pivoting system solution
zsytrs_rook :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	b: Matrix(complex128),
) -> (
	result: SolutionResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.zsytrs_rook_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.solution_successful = info == .OK
	result.is_singular = info_int > 0

	return
}

sytrs_rook :: proc {
	csytrs_rook,
	dsytrs_rook,
	ssytrs_rook,
	zsytrs_rook,
}

// ============================================================================
// TRIANGULAR BAND MATRIX CONDITION NUMBER
// ============================================================================
// Condition number estimation for triangular band matrices

// Norm type for condition number estimation
NormType :: enum {
	ONE_NORM, // '1' - 1-norm
	INFINITY_NORM, // 'I' - infinity-norm
}

// Diagonal type for triangular matrices
DiagonalType :: enum {
	NON_UNIT, // 'N' - non-unit diagonal
	UNIT, // 'U' - unit diagonal
}

// Condition number result structure
ConditionNumberResult :: struct($T: typeid) {
	rcond:                 T, // Reciprocal condition number
	condition_number:      T, // 1/rcond
	estimation_successful: bool,
}

// Complex single precision triangular band condition number
ctbcon :: proc(
	norm: NormType,
	uplo: UpLoFlag,
	diag: DiagonalType,
	n: int,
	kd: int, // Number of super/sub-diagonals
	ab: Matrix(complex64), // Band matrix in packed storage
	work: []complex64 = nil, // Workspace (size 2*n)
	rwork: []f32 = nil, // Real workspace (size n)
	allocator := context.allocator,
) -> (
	result: ConditionNumberResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")

	norm_char: u8 = norm == .ONE_NORM ? '1' : 'I'
	norm_cstring := cstring(&norm_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	rcond: f32
	info_int: Info

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
	lapack.ctbcon_(
		norm_cstring,
		uplo_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		cast(^lapack.complex)ab.data,
		&ldab,
		&rcond,
		cast(^lapack.complex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f32(max(f32))
	result.estimation_successful = info == .OK

	return
}

// Double precision triangular band condition number
dtbcon :: proc(
	norm: NormType,
	uplo: UpLoFlag,
	diag: DiagonalType,
	n: int,
	kd: int,
	ab: Matrix(f64),
	work: []f64 = nil, // Workspace (size 3*n)
	iwork: []Blas_Int = nil, // Integer workspace (size n)
	allocator := context.allocator,
) -> (
	result: ConditionNumberResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")

	norm_char: u8 = norm == .ONE_NORM ? '1' : 'I'
	norm_cstring := cstring(&norm_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	rcond: f64
	info_int: Info

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
	lapack.dtbcon_(
		norm_cstring,
		uplo_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f64(max(f64))
	result.estimation_successful = info == .OK

	return
}

// Single precision triangular band condition number
stbcon :: proc(
	norm: NormType,
	uplo: UpLoFlag,
	diag: DiagonalType,
	n: int,
	kd: int,
	ab: Matrix(f32),
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ConditionNumberResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")

	norm_char: u8 = norm == .ONE_NORM ? '1' : 'I'
	norm_cstring := cstring(&norm_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	rcond: f32
	info_int: Info

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
	lapack.stbcon_(
		norm_cstring,
		uplo_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f32(max(f32))
	result.estimation_successful = info == .OK

	return
}

// Complex double precision triangular band condition number
ztbcon :: proc(
	norm: NormType,
	uplo: UpLoFlag,
	diag: DiagonalType,
	n: int,
	kd: int,
	ab: Matrix(complex128),
	work: []complex128 = nil,
	rwork: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: ConditionNumberResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")

	norm_char: u8 = norm == .ONE_NORM ? '1' : 'I'
	norm_cstring := cstring(&norm_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	rcond: f64
	info_int: Info

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
	lapack.ztbcon_(
		norm_cstring,
		uplo_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		cast(^lapack.doublecomplex)ab.data,
		&ldab,
		&rcond,
		cast(^lapack.doublecomplex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f64(max(f64))
	result.estimation_successful = info == .OK

	return
}

tbcon :: proc {
	ctbcon,
	dtbcon,
	stbcon,
	ztbcon,
}

// ============================================================================
// TRIANGULAR BAND ERROR BOUNDS AND REFINEMENT
// ============================================================================
// Error bounds and iterative refinement for triangular band systems

// Transpose type for triangular operations
TransposeType :: enum {
	NO_TRANSPOSE, // 'N' - no transpose
	TRANSPOSE, // 'T' - transpose
	CONJUGATE, // 'C' - conjugate transpose (Hermitian)
}

// Error bounds result structure
ErrorBoundsResult :: struct($T: typeid, $S: typeid) {
	forward_errors:        []S, // Forward error bounds for each RHS
	backward_errors:       []S, // Backward error bounds for each RHS
	refinement_successful: bool,
	max_forward_error:     S,
	max_backward_error:    S,
}

// Complex single precision triangular band error bounds
ctbrfs :: proc(
	uplo: UpLoFlag,
	trans: TransposeType,
	diag: DiagonalType,
	n: int,
	kd: int, // Number of super/sub-diagonals
	nrhs: int,
	ab: Matrix(complex64), // Band matrix
	b: Matrix(complex64), // Original RHS
	x: Matrix(complex64), // Current solution
	work: []complex64 = nil, // Workspace (size 2*n)
	rwork: []f32 = nil, // Real workspace (size n)
	allocator := context.allocator,
) -> (
	result: ErrorBoundsResult(complex64, f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	trans_char: u8
	switch trans {
	case .NO_TRANSPOSE:
		trans_char = 'N'
	case .TRANSPOSE:
		trans_char = 'T'
	case .CONJUGATE:
		trans_char = 'C'
	}
	trans_cstring := cstring(&trans_char)

	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error bound arrays
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

	// Call LAPACK
	lapack.ctbrfs_(
		uplo_cstring,
		trans_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		&nrhs_int,
		cast(^lapack.complex)ab.data,
		&ldab,
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
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr
	result.refinement_successful = info == .OK

	// Calculate max errors
	result.max_forward_error = 0
	result.max_backward_error = 0
	for i in 0 ..< nrhs {
		if ferr[i] > result.max_forward_error {
			result.max_forward_error = ferr[i]
		}
		if berr[i] > result.max_backward_error {
			result.max_backward_error = berr[i]
		}
	}

	return
}

// Double precision triangular band error bounds
dtbrfs :: proc(
	uplo: UpLoFlag,
	trans: TransposeType,
	diag: DiagonalType,
	n: int,
	kd: int,
	nrhs: int,
	ab: Matrix(f64),
	b: Matrix(f64),
	x: Matrix(f64),
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ErrorBoundsResult(f64, f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	trans_char: u8
	switch trans {
	case .NO_TRANSPOSE:
		trans_char = 'N'
	case .TRANSPOSE:
		trans_char = 'T'
	case .CONJUGATE:
		trans_char = 'T' // For real matrices, C == T
	}
	trans_cstring := cstring(&trans_char)

	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error bound arrays
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

	// Call LAPACK
	lapack.dtbrfs_(
		uplo_cstring,
		trans_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		&nrhs_int,
		ab.data,
		&ldab,
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
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr
	result.refinement_successful = info == .OK

	// Calculate max errors
	result.max_forward_error = 0
	result.max_backward_error = 0
	for i in 0 ..< nrhs {
		if ferr[i] > result.max_forward_error {
			result.max_forward_error = ferr[i]
		}
		if berr[i] > result.max_backward_error {
			result.max_backward_error = berr[i]
		}
	}

	return
}

// Single precision triangular band error bounds
stbrfs :: proc(
	uplo: UpLoFlag,
	trans: TransposeType,
	diag: DiagonalType,
	n: int,
	kd: int,
	nrhs: int,
	ab: Matrix(f32),
	b: Matrix(f32),
	x: Matrix(f32),
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ErrorBoundsResult(f32, f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	trans_char: u8
	switch trans {
	case .NO_TRANSPOSE:
		trans_char = 'N'
	case .TRANSPOSE:
		trans_char = 'T'
	case .CONJUGATE:
		trans_char = 'T' // For real matrices, C == T
	}
	trans_cstring := cstring(&trans_char)

	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error bound arrays
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

	// Call LAPACK
	lapack.stbrfs_(
		uplo_cstring,
		trans_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		&nrhs_int,
		ab.data,
		&ldab,
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
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr
	result.refinement_successful = info == .OK

	// Calculate max errors
	result.max_forward_error = 0
	result.max_backward_error = 0
	for i in 0 ..< nrhs {
		if ferr[i] > result.max_forward_error {
			result.max_forward_error = ferr[i]
		}
		if berr[i] > result.max_backward_error {
			result.max_backward_error = berr[i]
		}
	}

	return
}

// Complex double precision triangular band error bounds
ztbrfs :: proc(
	uplo: UpLoFlag,
	trans: TransposeType,
	diag: DiagonalType,
	n: int,
	kd: int,
	nrhs: int,
	ab: Matrix(complex128),
	b: Matrix(complex128),
	x: Matrix(complex128),
	work: []complex128 = nil,
	rwork: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: ErrorBoundsResult(complex128, f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(nrhs >= 0, "Number of RHS must be non-negative")
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n && b.cols >= nrhs, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	trans_char: u8
	switch trans {
	case .NO_TRANSPOSE:
		trans_char = 'N'
	case .TRANSPOSE:
		trans_char = 'T'
	case .CONJUGATE:
		trans_char = 'C'
	}
	trans_cstring := cstring(&trans_char)

	diag_char: u8 = diag == .UNIT ? 'U' : 'N'
	diag_cstring := cstring(&diag_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.stride)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error bound arrays
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

	// Call LAPACK
	lapack.ztbrfs_(
		uplo_cstring,
		trans_cstring,
		diag_cstring,
		&n_int,
		&kd_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)ab.data,
		&ldab,
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
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr
	result.refinement_successful = info == .OK

	// Calculate max errors
	result.max_forward_error = 0
	result.max_backward_error = 0
	for i in 0 ..< nrhs {
		if ferr[i] > result.max_forward_error {
			result.max_forward_error = ferr[i]
		}
		if berr[i] > result.max_backward_error {
			result.max_backward_error = berr[i]
		}
	}

	return
}

tbrfs :: proc {
	ctbrfs,
	dtbrfs,
	stbrfs,
	ztbrfs,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Swap rows and columns in symmetric matrix
swap_symmetric_rows_columns :: proc(
	a: Matrix($T),
	i1: int,
	i2: int,
	uplo := UpLoFlag.Lower,
) -> Info {

	n := a.rows

	when T == complex64 {
		return csyswapr(uplo, n, a, i1, i2)
	} else when T == complex128 {
		return zsyswapr(uplo, n, a, i1, i2)
	} else when T == f64 {
		return dsyswapr(uplo, n, a, i1, i2)
	} else when T == f32 {
		return ssyswapr(uplo, n, a, i1, i2)
	} else {
		#panic("Unsupported type for symmetric matrix swapping")
	}
}

// Reduce symmetric matrix to tridiagonal form
tridiagonalize_symmetric :: proc(
	a: Matrix($T),
	uplo := UpLoFlag.Lower,
	use_2stage := false,
	vect := TransformationType.NO_VECTORS,
	allocator := context.allocator,
) -> (
	diagonal: []$S,
	off_diagonal: []S,
	tau: []S,
	info: Info,
) {

	n := a.rows

	when T == f64 {
		if use_2stage {
			result, info_val := dsytrd_2stage(vect, uplo, n, a, allocator = allocator)
			return result.diagonal, result.off_diagonal, result.reflector_scalars, info_val
		} else {
			result, info_val := dsytrd(uplo, n, a, allocator = allocator)
			return result.diagonal, result.off_diagonal, result.reflector_scalars, info_val
		}
	} else when T == f32 {
		if use_2stage {
			result, info_val := ssytrd_2stage(vect, uplo, n, a, allocator = allocator)
			return result.diagonal, result.off_diagonal, result.reflector_scalars, info_val
		} else {
			result, info_val := ssytrd(uplo, n, a, allocator = allocator)
			return result.diagonal, result.off_diagonal, result.reflector_scalars, info_val
		}
	} else {
		#panic("Unsupported type for symmetric tridiagonalization")
	}
}

// Factorize symmetric matrix using Bunch-Kaufman pivoting
factorize_symmetric :: proc(
	a: Matrix($T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	pivots: []Blas_Int,
	success: bool,
	info: Info,
) {

	n := a.rows

	when T == complex64 {
		result, info_val := csytrf(uplo, n, a, allocator = allocator)
		return result.pivot_indices, result.factorization_successful, info_val
	} else when T == complex128 {
		result, info_val := zsytrf(uplo, n, a, allocator = allocator)
		return result.pivot_indices, result.factorization_successful, info_val
	} else when T == f64 {
		result, info_val := dsytrf(uplo, n, a, allocator = allocator)
		return result.pivot_indices, result.factorization_successful, info_val
	} else when T == f32 {
		result, info_val := ssytrf(uplo, n, a, allocator = allocator)
		return result.pivot_indices, result.factorization_successful, info_val
	} else {
		#panic("Unsupported type for symmetric matrix factorization")
	}
}

// ==============================================================================
// Triangular Band System Solution Functions
// ==============================================================================

// Triangular band system solution result structure
TriangularBandSolutionResult :: struct($T: typeid) {
	solution_successful: bool,
	solution_matrix:     Matrix(T), // Solution matrix X
}

// Low-level triangular band system solution functions (ctbtrs, dtbtrs, stbtrs, ztbtrs)
ctbtrs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	kd: ^Blas_Int,
	nrhs: ^Blas_Int,
	AB: ^complex64,
	ldab: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctbtrs_(
		uplo,
		trans,
		diag,
		n,
		kd,
		nrhs,
		AB,
		ldab,
		B,
		ldb,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

dtbtrs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	kd: ^Blas_Int,
	nrhs: ^Blas_Int,
	AB: ^f64,
	ldab: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtbtrs_(
		uplo,
		trans,
		diag,
		n,
		kd,
		nrhs,
		AB,
		ldab,
		B,
		ldb,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

stbtrs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	kd: ^Blas_Int,
	nrhs: ^Blas_Int,
	AB: ^f32,
	ldab: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	info: ^Blas_Int,
) {
	stbtrs_(
		uplo,
		trans,
		diag,
		n,
		kd,
		nrhs,
		AB,
		ldab,
		B,
		ldb,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

ztbtrs :: proc(
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	n: ^Blas_Int,
	kd: ^Blas_Int,
	nrhs: ^Blas_Int,
	AB: ^complex128,
	ldab: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztbtrs_(
		uplo,
		trans,
		diag,
		n,
		kd,
		nrhs,
		AB,
		ldab,
		B,
		ldb,
		info,
		len(uplo),
		len(trans),
		len(diag),
	)
}

// High-level triangular band system solution wrapper functions
solve_triangular_band_system_complex64 :: proc(
	AB: Matrix(complex64),
	B: Matrix(complex64),
	kd: int,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularBandSolutionResult(complex64),
	err: LapackError,
) {

	n := Blas_Int(AB.cols)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.rows)
	ldb := Blas_Int(B.rows)
	kd_int := Blas_Int(kd)

	// Copy B matrix for solution
	solution_data := make([]complex64, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(complex64) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	info: Blas_Int
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	ctbtrs(
		uplo_str,
		trans_str,
		diag_str,
		&n,
		&kd_int,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(solution_data),
		&ldb,
		&info,
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solution_successful = true
	result.solution_matrix = solution_matrix
	return
}

solve_triangular_band_system_float64 :: proc(
	AB: Matrix(f64),
	B: Matrix(f64),
	kd: int,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularBandSolutionResult(f64),
	err: LapackError,
) {

	n := Blas_Int(AB.cols)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.rows)
	ldb := Blas_Int(B.rows)
	kd_int := Blas_Int(kd)

	// Copy B matrix for solution
	solution_data := make([]f64, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(f64) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	info: Blas_Int
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	dtbtrs(
		uplo_str,
		trans_str,
		diag_str,
		&n,
		&kd_int,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(solution_data),
		&ldb,
		&info,
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solution_successful = true
	result.solution_matrix = solution_matrix
	return
}

solve_triangular_band_system_float32 :: proc(
	AB: Matrix(f32),
	B: Matrix(f32),
	kd: int,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularBandSolutionResult(f32),
	err: LapackError,
) {

	n := Blas_Int(AB.cols)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.rows)
	ldb := Blas_Int(B.rows)
	kd_int := Blas_Int(kd)

	// Copy B matrix for solution
	solution_data := make([]f32, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(f32) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	info: Blas_Int
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	stbtrs(
		uplo_str,
		trans_str,
		diag_str,
		&n,
		&kd_int,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(solution_data),
		&ldb,
		&info,
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solution_successful = true
	result.solution_matrix = solution_matrix
	return
}

solve_triangular_band_system_complex128 :: proc(
	AB: Matrix(complex128),
	B: Matrix(complex128),
	kd: int,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularBandSolutionResult(complex128),
	err: LapackError,
) {

	n := Blas_Int(AB.cols)
	nrhs := Blas_Int(B.cols)
	ldab := Blas_Int(AB.rows)
	ldb := Blas_Int(B.rows)
	kd_int := Blas_Int(kd)

	// Copy B matrix for solution
	solution_data := make([]complex128, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(complex128) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	info: Blas_Int
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)

	ztbtrs(
		uplo_str,
		trans_str,
		diag_str,
		&n,
		&kd_int,
		&nrhs,
		raw_data(AB.data),
		&ldab,
		raw_data(solution_data),
		&ldb,
		&info,
	)

	if info != 0 {
		delete(solution_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.solution_successful = true
	result.solution_matrix = solution_matrix
	return
}

// Generic triangular band system solution function
solve_triangular_band_system :: proc {
	solve_triangular_band_system_complex64,
	solve_triangular_band_system_float64,
	solve_triangular_band_system_float32,
	solve_triangular_band_system_complex128,
}

// ==============================================================================
// Triangular Solve with RFP Format Functions
// ==============================================================================

// RFP (Rectangular Full Packed) triangular solve result structure
RFPTriangularSolveResult :: struct($T: typeid) {
	solve_successful: bool,
	solution_matrix:  Matrix(T), // Solution matrix X
}

// Low-level triangular solve with RFP format functions (ctfsm, dtfsm, stfsm, ztfsm)
ctfsm :: proc(
	transr: cstring,
	side: cstring,
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	alpha: ^complex64,
	A: ^complex64,
	B: ^complex64,
	ldb: ^Blas_Int,
) {
	ctfsm_(
		transr,
		side,
		uplo,
		trans,
		diag,
		m,
		n,
		alpha,
		A,
		B,
		ldb,
		len(transr),
		len(side),
		len(uplo),
		len(trans),
		len(diag),
	)
}

dtfsm :: proc(
	transr: cstring,
	side: cstring,
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	alpha: ^f64,
	A: ^f64,
	B: ^f64,
	ldb: ^Blas_Int,
) {
	dtfsm_(
		transr,
		side,
		uplo,
		trans,
		diag,
		m,
		n,
		alpha,
		A,
		B,
		ldb,
		len(transr),
		len(side),
		len(uplo),
		len(trans),
		len(diag),
	)
}

stfsm :: proc(
	transr: cstring,
	side: cstring,
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	alpha: ^f32,
	A: ^f32,
	B: ^f32,
	ldb: ^Blas_Int,
) {
	stfsm_(
		transr,
		side,
		uplo,
		trans,
		diag,
		m,
		n,
		alpha,
		A,
		B,
		ldb,
		len(transr),
		len(side),
		len(uplo),
		len(trans),
		len(diag),
	)
}

ztfsm :: proc(
	transr: cstring,
	side: cstring,
	uplo: cstring,
	trans: cstring,
	diag: cstring,
	m: ^Blas_Int,
	n: ^Blas_Int,
	alpha: ^complex128,
	A: ^complex128,
	B: ^complex128,
	ldb: ^Blas_Int,
) {
	ztfsm_(
		transr,
		side,
		uplo,
		trans,
		diag,
		m,
		n,
		alpha,
		A,
		B,
		ldb,
		len(transr),
		len(side),
		len(uplo),
		len(trans),
		len(diag),
	)
}

// High-level RFP triangular solve wrapper functions
solve_rfp_triangular_complex64 :: proc(
	A_rfp: []complex64,
	B: Matrix(complex64),
	alpha: complex64 = 1.0,
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularSolveResult(complex64),
	err: LapackError,
) {

	m := Blas_Int(B.rows)
	n := Blas_Int(B.cols)
	ldb := Blas_Int(B.rows)

	// Copy B matrix for solution
	solution_data := make([]complex64, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(complex64) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	transr_str := "N"
	if transr do transr_str = "T"
	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)
	alpha_copy := alpha

	ctfsm(
		transr_str,
		side_str,
		uplo_str,
		trans_str,
		diag_str,
		&m,
		&n,
		&alpha_copy,
		raw_data(A_rfp),
		raw_data(solution_data),
		&ldb,
	)

	result.solve_successful = true
	result.solution_matrix = solution_matrix
	return
}

solve_rfp_triangular_float64 :: proc(
	A_rfp: []f64,
	B: Matrix(f64),
	alpha: f64 = 1.0,
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularSolveResult(f64),
	err: LapackError,
) {

	m := Blas_Int(B.rows)
	n := Blas_Int(B.cols)
	ldb := Blas_Int(B.rows)

	// Copy B matrix for solution
	solution_data := make([]f64, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(f64) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	transr_str := "N"
	if transr do transr_str = "T"
	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)
	alpha_copy := alpha

	dtfsm(
		transr_str,
		side_str,
		uplo_str,
		trans_str,
		diag_str,
		&m,
		&n,
		&alpha_copy,
		raw_data(A_rfp),
		raw_data(solution_data),
		&ldb,
	)

	result.solve_successful = true
	result.solution_matrix = solution_matrix
	return
}

solve_rfp_triangular_float32 :: proc(
	A_rfp: []f32,
	B: Matrix(f32),
	alpha: f32 = 1.0,
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularSolveResult(f32),
	err: LapackError,
) {

	m := Blas_Int(B.rows)
	n := Blas_Int(B.cols)
	ldb := Blas_Int(B.rows)

	// Copy B matrix for solution
	solution_data := make([]f32, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(f32) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	transr_str := "N"
	if transr do transr_str = "T"
	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)
	alpha_copy := alpha

	stfsm(
		transr_str,
		side_str,
		uplo_str,
		trans_str,
		diag_str,
		&m,
		&n,
		&alpha_copy,
		raw_data(A_rfp),
		raw_data(solution_data),
		&ldb,
	)

	result.solve_successful = true
	result.solution_matrix = solution_matrix
	return
}

solve_rfp_triangular_complex128 :: proc(
	A_rfp: []complex128,
	B: Matrix(complex128),
	alpha: complex128 = 1.0,
	side: MatrixSide = .Left,
	uplo: MatrixTriangle = .Upper,
	trans: MatrixTranspose = .None,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularSolveResult(complex128),
	err: LapackError,
) {

	m := Blas_Int(B.rows)
	n := Blas_Int(B.cols)
	ldb := Blas_Int(B.rows)

	// Copy B matrix for solution
	solution_data := make([]complex128, B.rows * B.cols, allocator) or_return
	copy(solution_data, B.data[:B.rows * B.cols])

	solution_matrix := Matrix(complex128) {
		data = solution_data,
		rows = B.rows,
		cols = B.cols,
	}

	transr_str := "N"
	if transr do transr_str = "T"
	side_str := matrix_side_to_cstring(side)
	uplo_str := matrix_triangle_to_cstring(uplo)
	trans_str := matrix_transpose_to_cstring(trans)
	diag_str := matrix_diagonal_to_cstring(diag)
	alpha_copy := alpha

	ztfsm(
		transr_str,
		side_str,
		uplo_str,
		trans_str,
		diag_str,
		&m,
		&n,
		&alpha_copy,
		raw_data(A_rfp),
		raw_data(solution_data),
		&ldb,
	)

	result.solve_successful = true
	result.solution_matrix = solution_matrix
	return
}

// Generic RFP triangular solve function
solve_rfp_triangular :: proc {
	solve_rfp_triangular_complex64,
	solve_rfp_triangular_float64,
	solve_rfp_triangular_float32,
	solve_rfp_triangular_complex128,
}

// ==============================================================================
// Triangular Inversion with RFP Format Functions
// ==============================================================================

// RFP triangular inversion result structure
RFPTriangularInversionResult :: struct($T: typeid) {
	inversion_successful: bool,
	inverted_matrix_rfp:  []T, // Inverted matrix in RFP format
}

// Low-level triangular inversion with RFP format functions (ctftri, dtftri, stftri, ztftri)
ctftri :: proc(
	transr: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	A: ^complex64,
	info: ^Blas_Int,
) {
	ctftri_(transr, uplo, diag, n, A, info, len(transr), len(uplo), len(diag))
}

dtftri :: proc(
	transr: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	A: ^f64,
	info: ^Blas_Int,
) {
	dtftri_(transr, uplo, diag, n, A, info, len(transr), len(uplo), len(diag))
}

stftri :: proc(
	transr: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	A: ^f32,
	info: ^Blas_Int,
) {
	stftri_(transr, uplo, diag, n, A, info, len(transr), len(uplo), len(diag))
}

ztftri :: proc(
	transr: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	A: ^complex128,
	info: ^Blas_Int,
) {
	ztftri_(transr, uplo, diag, n, A, info, len(transr), len(uplo), len(diag))
}

// High-level RFP triangular inversion wrapper functions
invert_rfp_triangular_complex64 :: proc(
	A_rfp: []complex64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularInversionResult(complex64),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate expected RFP size for triangular matrix
	rfp_size := (n * (n + 1)) / 2
	if len(A_rfp) < rfp_size {
		return {}, .InvalidParameter
	}

	// Copy RFP data for inversion
	inverted_data := make([]complex64, len(A_rfp), allocator) or_return
	copy(inverted_data, A_rfp)

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	ctftri(transr_str, uplo_str, diag_str, &n_int, raw_data(inverted_data), &info)

	if info != 0 {
		delete(inverted_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.inversion_successful = true
	result.inverted_matrix_rfp = inverted_data
	return
}

invert_rfp_triangular_float64 :: proc(
	A_rfp: []f64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularInversionResult(f64),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate expected RFP size for triangular matrix
	rfp_size := (n * (n + 1)) / 2
	if len(A_rfp) < rfp_size {
		return {}, .InvalidParameter
	}

	// Copy RFP data for inversion
	inverted_data := make([]f64, len(A_rfp), allocator) or_return
	copy(inverted_data, A_rfp)

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	dtftri(transr_str, uplo_str, diag_str, &n_int, raw_data(inverted_data), &info)

	if info != 0 {
		delete(inverted_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.inversion_successful = true
	result.inverted_matrix_rfp = inverted_data
	return
}

invert_rfp_triangular_float32 :: proc(
	A_rfp: []f32,
	n: int,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularInversionResult(f32),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate expected RFP size for triangular matrix
	rfp_size := (n * (n + 1)) / 2
	if len(A_rfp) < rfp_size {
		return {}, .InvalidParameter
	}

	// Copy RFP data for inversion
	inverted_data := make([]f32, len(A_rfp), allocator) or_return
	copy(inverted_data, A_rfp)

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	stftri(transr_str, uplo_str, diag_str, &n_int, raw_data(inverted_data), &info)

	if info != 0 {
		delete(inverted_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.inversion_successful = true
	result.inverted_matrix_rfp = inverted_data
	return
}

invert_rfp_triangular_complex128 :: proc(
	A_rfp: []complex128,
	n: int,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPTriangularInversionResult(complex128),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate expected RFP size for triangular matrix
	rfp_size := (n * (n + 1)) / 2
	if len(A_rfp) < rfp_size {
		return {}, .InvalidParameter
	}

	// Copy RFP data for inversion
	inverted_data := make([]complex128, len(A_rfp), allocator) or_return
	copy(inverted_data, A_rfp)

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	ztftri(transr_str, uplo_str, diag_str, &n_int, raw_data(inverted_data), &info)

	if info != 0 {
		delete(inverted_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .SingularMatrix
		}
	}

	result.inversion_successful = true
	result.inverted_matrix_rfp = inverted_data
	return
}

// Generic RFP triangular inversion function
invert_rfp_triangular :: proc {
	invert_rfp_triangular_complex64,
	invert_rfp_triangular_float64,
	invert_rfp_triangular_float32,
	invert_rfp_triangular_complex128,
}

// ==============================================================================
// Summary and Convenience Overloads
// ==============================================================================

// Triangular band operation overloads
tbtrs :: proc {
	ctbtrs,
	dtbtrs,
	stbtrs,
	ztbtrs,
}

// RFP triangular operation overloads
tfsm :: proc {
	ctfsm,
	dtfsm,
	stfsm,
	ztfsm,
}
tftri :: proc {
	ctftri,
	dtftri,
	stftri,
	ztftri,
}

// ==============================================================================
// RFP Format Conversion Functions
// ==============================================================================

// RFP to packed format conversion result structure
RFPToPackedResult :: struct($T: typeid) {
	conversion_successful: bool,
	packed_matrix:         []T, // Matrix in packed format (AP)
}

// RFP to full format conversion result structure
RFPToFullResult :: struct($T: typeid) {
	conversion_successful: bool,
	full_matrix:           Matrix(T), // Matrix in standard full format
}

// Low-level RFP to packed format conversion functions (ctfttp, dtfttp, stfttp, ztfttp)
ctfttp :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^complex64,
	AP: ^complex64,
	info: ^Blas_Int,
) {
	ctfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

dtfttp :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^f64,
	AP: ^f64,
	info: ^Blas_Int,
) {
	dtfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

stfttp :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^f32,
	AP: ^f32,
	info: ^Blas_Int,
) {
	stfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

ztfttp :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^complex128,
	AP: ^complex128,
	info: ^Blas_Int,
) {
	ztfttp_(transr, uplo, n, ARF, AP, info, len(transr), len(uplo))
}

// High-level RFP to packed format conversion wrapper functions
convert_rfp_to_packed_complex64 :: proc(
	ARF: []complex64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToPackedResult(complex64),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]complex64, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	ctfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

convert_rfp_to_packed_float64 :: proc(
	ARF: []f64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToPackedResult(f64),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]f64, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	dtfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

convert_rfp_to_packed_float32 :: proc(
	ARF: []f32,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToPackedResult(f32),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]f32, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	stfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

convert_rfp_to_packed_complex128 :: proc(
	ARF: []complex128,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToPackedResult(complex128),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate packed storage size for triangular matrix
	packed_size := (n * (n + 1)) / 2

	// Allocate packed format array
	packed_data := make([]complex128, packed_size, allocator) or_return

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	ztfttp(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(packed_data), &info)

	if info != 0 {
		delete(packed_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.packed_matrix = packed_data
	return
}

// Generic RFP to packed format conversion function
convert_rfp_to_packed :: proc {
	convert_rfp_to_packed_complex64,
	convert_rfp_to_packed_float64,
	convert_rfp_to_packed_float32,
	convert_rfp_to_packed_complex128,
}

// ==============================================================================
// RFP to Full Format Conversion Functions
// ==============================================================================

// Low-level RFP to full format conversion functions (ctfttr, dtfttr, stfttr, ztfttr)
ctfttr :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^complex64,
	A: ^complex64,
	lda: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctfttr_(transr, uplo, n, ARF, A, lda, info, len(transr), len(uplo))
}

dtfttr :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^f64,
	A: ^f64,
	lda: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtfttr_(transr, uplo, n, ARF, A, lda, info, len(transr), len(uplo))
}

stfttr :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^f32,
	A: ^f32,
	lda: ^Blas_Int,
	info: ^Blas_Int,
) {
	stfttr_(transr, uplo, n, ARF, A, lda, info, len(transr), len(uplo))
}

ztfttr :: proc(
	transr: cstring,
	uplo: cstring,
	n: ^Blas_Int,
	ARF: ^complex128,
	A: ^complex128,
	lda: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztfttr_(transr, uplo, n, ARF, A, lda, info, len(transr), len(uplo))
}

// High-level RFP to full format conversion wrapper functions
convert_rfp_to_full_complex64 :: proc(
	ARF: []complex64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToFullResult(complex64),
	err: LapackError,
) {

	n_int := Blas_Int(n)
	lda := n_int

	// Allocate full format matrix
	full_data := make([]complex64, n * n, allocator) or_return

	full_matrix := Matrix(complex64) {
		data = full_data,
		rows = n,
		cols = n,
	}

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	ctfttr(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(full_data), &lda, &info)

	if info != 0 {
		delete(full_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.full_matrix = full_matrix
	return
}

convert_rfp_to_full_float64 :: proc(
	ARF: []f64,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToFullResult(f64),
	err: LapackError,
) {

	n_int := Blas_Int(n)
	lda := n_int

	// Allocate full format matrix
	full_data := make([]f64, n * n, allocator) or_return

	full_matrix := Matrix(f64) {
		data = full_data,
		rows = n,
		cols = n,
	}

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	dtfttr(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(full_data), &lda, &info)

	if info != 0 {
		delete(full_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.full_matrix = full_matrix
	return
}

convert_rfp_to_full_float32 :: proc(
	ARF: []f32,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToFullResult(f32),
	err: LapackError,
) {

	n_int := Blas_Int(n)
	lda := n_int

	// Allocate full format matrix
	full_data := make([]f32, n * n, allocator) or_return

	full_matrix := Matrix(f32) {
		data = full_data,
		rows = n,
		cols = n,
	}

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	stfttr(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(full_data), &lda, &info)

	if info != 0 {
		delete(full_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.full_matrix = full_matrix
	return
}

convert_rfp_to_full_complex128 :: proc(
	ARF: []complex128,
	n: int,
	uplo: MatrixTriangle = .Upper,
	transr: bool = false,
	allocator := context.allocator,
) -> (
	result: RFPToFullResult(complex128),
	err: LapackError,
) {

	n_int := Blas_Int(n)
	lda := n_int

	// Allocate full format matrix
	full_data := make([]complex128, n * n, allocator) or_return

	full_matrix := Matrix(complex128) {
		data = full_data,
		rows = n,
		cols = n,
	}

	info: Blas_Int
	transr_str := "N"
	if transr do transr_str = "T"
	uplo_str := matrix_triangle_to_cstring(uplo)

	ztfttr(transr_str, uplo_str, &n_int, raw_data(ARF), raw_data(full_data), &lda, &info)

	if info != 0 {
		delete(full_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.conversion_successful = true
	result.full_matrix = full_matrix
	return
}

// Generic RFP to full format conversion function
convert_rfp_to_full :: proc {
	convert_rfp_to_full_complex64,
	convert_rfp_to_full_float64,
	convert_rfp_to_full_float32,
	convert_rfp_to_full_complex128,
}

// ==============================================================================
// Generalized Eigenvector Functions
// ==============================================================================

// Eigenvector computation side specification
EigenvectorSide :: enum {
	Right, // Compute right eigenvectors only
	Left, // Compute left eigenvectors only
	Both, // Compute both left and right eigenvectors
}

// Eigenvector selection specification
EigenvectorSelection :: enum {
	All, // Compute all eigenvectors
	Backtransform, // Backtransform using DTGEVC
	Selected, // Compute selected eigenvectors
}

// Generalized eigenvector result structure
GeneralizedEigenvectorResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	left_eigenvectors:      Matrix(T), // Left eigenvectors VL
	right_eigenvectors:     Matrix(T), // Right eigenvectors VR
	num_computed:           int, // Number of eigenvectors computed
	selection_mask:         []bool, // Selection mask for computed eigenvectors
}

// Helper function to convert eigenvector side to string
eigenvector_side_to_cstring :: proc(side: EigenvectorSide) -> cstring {
	switch side {
	case .Right:
		return "R"
	case .Left:
		return "L"
	case .Both:
		return "B"
	}
	return "R"
}

// Helper function to convert eigenvector selection to string
eigenvector_selection_to_cstring :: proc(selection: EigenvectorSelection) -> cstring {
	switch selection {
	case .All:
		return "A"
	case .Backtransform:
		return "B"
	case .Selected:
		return "S"
	}
	return "A"
}

// Low-level generalized eigenvector functions (ctgevc, dtgevc, stgevc, ztgevc)
ctgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^complex64,
	lds: ^Blas_Int,
	P: ^complex64,
	ldp: ^Blas_Int,
	VL: ^complex64,
	ldvl: ^Blas_Int,
	VR: ^complex64,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex64,
	rwork: ^f32,
	info: ^Blas_Int,
) {
	ctgevc_(
		side,
		howmny,
		select,
		n,
		S,
		lds,
		P,
		ldp,
		VL,
		ldvl,
		VR,
		ldvr,
		mm,
		m,
		work,
		rwork,
		info,
		len(side),
		len(howmny),
	)
}

dtgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^f64,
	lds: ^Blas_Int,
	P: ^f64,
	ldp: ^Blas_Int,
	VL: ^f64,
	ldvl: ^Blas_Int,
	VR: ^f64,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f64,
	info: ^Blas_Int,
) {
	dtgevc_(
		side,
		howmny,
		select,
		n,
		S,
		lds,
		P,
		ldp,
		VL,
		ldvl,
		VR,
		ldvr,
		mm,
		m,
		work,
		info,
		len(side),
		len(howmny),
	)
}

stgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^f32,
	lds: ^Blas_Int,
	P: ^f32,
	ldp: ^Blas_Int,
	VL: ^f32,
	ldvl: ^Blas_Int,
	VR: ^f32,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f32,
	info: ^Blas_Int,
) {
	stgevc_(
		side,
		howmny,
		select,
		n,
		S,
		lds,
		P,
		ldp,
		VL,
		ldvl,
		VR,
		ldvr,
		mm,
		m,
		work,
		info,
		len(side),
		len(howmny),
	)
}

ztgevc :: proc(
	side: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	S: ^complex128,
	lds: ^Blas_Int,
	P: ^complex128,
	ldp: ^Blas_Int,
	VL: ^complex128,
	ldvl: ^Blas_Int,
	VR: ^complex128,
	ldvr: ^Blas_Int,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex128,
	rwork: ^f64,
	info: ^Blas_Int,
) {
	ztgevc_(
		side,
		howmny,
		select,
		n,
		S,
		lds,
		P,
		ldp,
		VL,
		ldvl,
		VR,
		ldvr,
		mm,
		m,
		work,
		rwork,
		info,
		len(side),
		len(howmny),
	)
}

// High-level generalized eigenvector wrapper functions
compute_generalized_eigenvectors_complex64 :: proc(
	S: Matrix(complex64),
	P: Matrix(complex64),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(complex64, f32),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []complex64 = nil
	right_data: []complex64 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]complex64, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]complex64, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]complex64, 2 * int(n), allocator) or_return
	rwork := make([]f32, 2 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	ctgevc(
		side_str,
		howmny_str,
		select_ptr,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(P.data),
		&ldp,
		left_ptr,
		&ldvl,
		right_ptr,
		&ldvr,
		&mm,
		&m,
		raw_data(work),
		raw_data(rwork),
		&info,
	)

	// Clean up workspace
	delete(work, allocator)
	delete(rwork, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(complex64)
	right_matrix: Matrix(complex64)

	if left_data != nil {
		left_matrix = Matrix(complex64) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(complex64) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

compute_generalized_eigenvectors_float64 :: proc(
	S: Matrix(f64),
	P: Matrix(f64),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(f64, f64),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []f64 = nil
	right_data: []f64 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]f64, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]f64, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]f64, 6 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	dtgevc(
		side_str,
		howmny_str,
		select_ptr,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(P.data),
		&ldp,
		left_ptr,
		&ldvl,
		right_ptr,
		&ldvr,
		&mm,
		&m,
		raw_data(work),
		&info,
	)

	// Clean up workspace
	delete(work, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(f64)
	right_matrix: Matrix(f64)

	if left_data != nil {
		left_matrix = Matrix(f64) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(f64) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

compute_generalized_eigenvectors_float32 :: proc(
	S: Matrix(f32),
	P: Matrix(f32),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(f32, f32),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []f32 = nil
	right_data: []f32 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]f32, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]f32, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]f32, 6 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	stgevc(
		side_str,
		howmny_str,
		select_ptr,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(P.data),
		&ldp,
		left_ptr,
		&ldvl,
		right_ptr,
		&ldvr,
		&mm,
		&m,
		raw_data(work),
		&info,
	)

	// Clean up workspace
	delete(work, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(f32)
	right_matrix: Matrix(f32)

	if left_data != nil {
		left_matrix = Matrix(f32) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(f32) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

compute_generalized_eigenvectors_complex128 :: proc(
	S: Matrix(complex128),
	P: Matrix(complex128),
	side: EigenvectorSide = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenvectorResult(complex128, f64),
	err: LapackError,
) {

	n := Blas_Int(S.rows)
	lds := Blas_Int(S.rows)
	ldp := Blas_Int(P.rows)

	// Allocate eigenvector matrices based on side selection
	left_data: []complex128 = nil
	right_data: []complex128 = nil
	ldvl: Blas_Int = 1
	ldvr: Blas_Int = 1

	if side == .Left || side == .Both {
		left_data = make([]complex128, int(n * n), allocator) or_return
		ldvl = n
	}

	if side == .Right || side == .Both {
		right_data = make([]complex128, int(n * n), allocator) or_return
		ldvr = n
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate workspace
	work := make([]complex128, 2 * int(n), allocator) or_return
	rwork := make([]f64, 2 * int(n), allocator) or_return

	mm := n // Maximum number of eigenvectors to compute
	m: Blas_Int // Actual number computed (output)
	info: Blas_Int

	side_str := eigenvector_side_to_cstring(side)
	howmny_str := eigenvector_selection_to_cstring(selection)

	// Call LAPACK function
	select_ptr := raw_data(select_array) if select_array != nil else nil
	left_ptr := raw_data(left_data) if left_data != nil else nil
	right_ptr := raw_data(right_data) if right_data != nil else nil

	ztgevc(
		side_str,
		howmny_str,
		select_ptr,
		&n,
		raw_data(S.data),
		&lds,
		raw_data(P.data),
		&ldp,
		left_ptr,
		&ldvl,
		right_ptr,
		&ldvr,
		&mm,
		&m,
		raw_data(work),
		raw_data(rwork),
		&info,
	)

	// Clean up workspace
	delete(work, allocator)
	delete(rwork, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if left_data != nil do delete(left_data, allocator)
		if right_data != nil do delete(right_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result matrices
	left_matrix: Matrix(complex128)
	right_matrix: Matrix(complex128)

	if left_data != nil {
		left_matrix = Matrix(complex128) {
			data = left_data,
			rows = int(n),
			cols = int(n),
		}
	}

	if right_data != nil {
		right_matrix = Matrix(complex128) {
			data = right_data,
			rows = int(n),
			cols = int(n),
		}
	}

	// Create selection mask for output
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.left_eigenvectors = left_matrix
	result.right_eigenvectors = right_matrix
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

// Generic generalized eigenvector computation function
compute_generalized_eigenvectors :: proc {
	compute_generalized_eigenvectors_complex64,
	compute_generalized_eigenvectors_float64,
	compute_generalized_eigenvectors_float32,
	compute_generalized_eigenvectors_complex128,
}

// ==============================================================================
// Final Convenience Overloads and Summary
// ==============================================================================

// RFP format conversion overloads
tfttp :: proc {
	ctfttp,
	dtfttp,
	stfttp,
	ztfttp,
}
tfttr :: proc {
	ctfttr,
	dtfttr,
	stfttr,
	ztfttr,
}

// Generalized eigenvector overloads
tgevc :: proc {
	ctgevc,
	dtgevc,
	stgevc,
	ztgevc,
}

// ==============================================================================
// Generalized Schur Form Reordering Functions
// ==============================================================================

// Generalized Schur form reordering result structure
GeneralizedSchurReorderResult :: struct($T: typeid) {
	reordering_successful: bool,
	reordered_A:           Matrix(T), // Reordered matrix A
	reordered_B:           Matrix(T), // Reordered matrix B
	updated_Q:             Matrix(T), // Updated orthogonal matrix Q (if requested)
	updated_Z:             Matrix(T), // Updated orthogonal matrix Z (if requested)
	final_ifst:            int, // Final position after reordering
	final_ilst:            int, // Final position after reordering
}

// Low-level generalized Schur form reordering functions (ctgexc, dtgexc, stgexc, ztgexc)
ctgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	Q: ^complex64,
	ldq: ^Blas_Int,
	Z: ^complex64,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, info)
}

dtgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	Q: ^f64,
	ldq: ^Blas_Int,
	Z: ^f64,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	work: ^f64,
	lwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, work, lwork, info)
}

stgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	Q: ^f32,
	ldq: ^Blas_Int,
	Z: ^f32,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	work: ^f32,
	lwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, work, lwork, info)
}

ztgexc :: proc(
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	Q: ^complex128,
	ldq: ^Blas_Int,
	Z: ^complex128,
	ldz: ^Blas_Int,
	ifst: ^Blas_Int,
	ilst: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgexc_(wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst, info)
}

// High-level generalized Schur form reordering wrapper functions
reorder_generalized_schur_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64),
	Q: Matrix(complex64) = {},
	Z: Matrix(complex64) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(complex64),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]complex64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(complex64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]complex64, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(complex64) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []complex64 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(complex64)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]complex64, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(complex64) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]complex64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(complex64) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []complex64 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(complex64)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]complex64, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(complex64) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]complex64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(complex64) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	ctgexc(
		&wantq,
		&wantz,
		&n,
		raw_data(a_data),
		&lda,
		raw_data(b_data),
		&ldb,
		q_ptr,
		&ldq,
		z_ptr,
		&ldz,
		&ifst_copy,
		&ilst_copy,
		&info,
	)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

reorder_generalized_schur_float64 :: proc(
	A: Matrix(f64),
	B: Matrix(f64),
	Q: Matrix(f64) = {},
	Z: Matrix(f64) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(f64),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]f64, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(f64) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]f64, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(f64) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []f64 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(f64)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]f64, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(f64) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]f64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(f64) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []f64 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(f64)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]f64, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(f64) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]f64, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(f64) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	// Query optimal workspace size
	work_query: f64
	lwork_query: Blas_Int = -1
	info_query: Blas_Int
	ifst_query := Blas_Int(ifst + 1)
	ilst_query := Blas_Int(ilst + 1)

	dtgexc(
		&wantq,
		&wantz,
		&n,
		raw_data(a_data),
		&lda,
		raw_data(b_data),
		&ldb,
		nil,
		&ldq,
		nil,
		&ldz,
		&ifst_query,
		&ilst_query,
		&work_query,
		&lwork_query,
		&info_query,
	)

	if info_query != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(work_query)
	work := make([]f64, int(lwork), allocator) or_return

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	dtgexc(
		&wantq,
		&wantz,
		&n,
		raw_data(a_data),
		&lda,
		raw_data(b_data),
		&ldb,
		q_ptr,
		&ldq,
		z_ptr,
		&ldz,
		&ifst_copy,
		&ilst_copy,
		raw_data(work),
		&lwork,
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

reorder_generalized_schur_float32 :: proc(
	A: Matrix(f32),
	B: Matrix(f32),
	Q: Matrix(f32) = {},
	Z: Matrix(f32) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(f32),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]f32, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(f32) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]f32, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(f32) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []f32 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(f32)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]f32, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(f32) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]f32, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(f32) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []f32 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(f32)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]f32, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(f32) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]f32, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(f32) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	// Query optimal workspace size
	work_query: f32
	lwork_query: Blas_Int = -1
	info_query: Blas_Int
	ifst_query := Blas_Int(ifst + 1)
	ilst_query := Blas_Int(ilst + 1)

	stgexc(
		&wantq,
		&wantz,
		&n,
		raw_data(a_data),
		&lda,
		raw_data(b_data),
		&ldb,
		nil,
		&ldq,
		nil,
		&ldz,
		&ifst_query,
		&ilst_query,
		&work_query,
		&lwork_query,
		&info_query,
	)

	if info_query != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(work_query)
	work := make([]f32, int(lwork), allocator) or_return

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	stgexc(
		&wantq,
		&wantz,
		&n,
		raw_data(a_data),
		&lda,
		raw_data(b_data),
		&ldb,
		q_ptr,
		&ldq,
		z_ptr,
		&ldz,
		&ifst_copy,
		&ilst_copy,
		raw_data(work),
		&lwork,
		&info,
	)

	delete(work, allocator)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

reorder_generalized_schur_complex128 :: proc(
	A: Matrix(complex128),
	B: Matrix(complex128),
	Q: Matrix(complex128) = {},
	Z: Matrix(complex128) = {},
	ifst: int,
	ilst: int,
	update_Q: bool = false,
	update_Z: bool = false,
	allocator := context.allocator,
) -> (
	result: GeneralizedSchurReorderResult(complex128),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)

	// Copy matrices for reordering
	a_data := make([]complex128, A.rows * A.cols, allocator) or_return
	copy(a_data, A.data[:A.rows * A.cols])
	reordered_A := Matrix(complex128) {
		data = a_data,
		rows = A.rows,
		cols = A.cols,
	}

	b_data := make([]complex128, B.rows * B.cols, allocator) or_return
	copy(b_data, B.data[:B.rows * B.cols])
	reordered_B := Matrix(complex128) {
		data = b_data,
		rows = B.rows,
		cols = B.cols,
	}

	// Handle Q matrix
	q_data: []complex128 = nil
	ldq: Blas_Int = 1
	updated_Q: Matrix(complex128)
	wantq: Blas_Int = update_Q ? 1 : 0

	if update_Q {
		if Q.data != nil {
			q_data = make([]complex128, Q.rows * Q.cols, allocator) or_return
			copy(q_data, Q.data[:Q.rows * Q.cols])
			updated_Q = Matrix(complex128) {
				data = q_data,
				rows = Q.rows,
				cols = Q.cols,
			}
			ldq = Blas_Int(Q.rows)
		} else {
			// Create identity matrix if Q not provided
			q_data = make([]complex128, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				q_data[i * int(n) + i] = 1.0
			}
			updated_Q = Matrix(complex128) {
				data = q_data,
				rows = int(n),
				cols = int(n),
			}
			ldq = n
		}
	}

	// Handle Z matrix
	z_data: []complex128 = nil
	ldz: Blas_Int = 1
	updated_Z: Matrix(complex128)
	wantz: Blas_Int = update_Z ? 1 : 0

	if update_Z {
		if Z.data != nil {
			z_data = make([]complex128, Z.rows * Z.cols, allocator) or_return
			copy(z_data, Z.data[:Z.rows * Z.cols])
			updated_Z = Matrix(complex128) {
				data = z_data,
				rows = Z.rows,
				cols = Z.cols,
			}
			ldz = Blas_Int(Z.rows)
		} else {
			// Create identity matrix if Z not provided
			z_data = make([]complex128, int(n * n), allocator) or_return
			for i in 0 ..< int(n) {
				z_data[i * int(n) + i] = 1.0
			}
			updated_Z = Matrix(complex128) {
				data = z_data,
				rows = int(n),
				cols = int(n),
			}
			ldz = n
		}
	}

	info: Blas_Int
	ifst_copy := Blas_Int(ifst + 1) // Convert to 1-based indexing
	ilst_copy := Blas_Int(ilst + 1) // Convert to 1-based indexing

	q_ptr := raw_data(q_data) if q_data != nil else nil
	z_ptr := raw_data(z_data) if z_data != nil else nil

	ztgexc(
		&wantq,
		&wantz,
		&n,
		raw_data(a_data),
		&lda,
		raw_data(b_data),
		&ldb,
		q_ptr,
		&ldq,
		z_ptr,
		&ldz,
		&ifst_copy,
		&ilst_copy,
		&info,
	)

	if info != 0 {
		delete(a_data, allocator)
		delete(b_data, allocator)
		if q_data != nil do delete(q_data, allocator)
		if z_data != nil do delete(z_data, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.reordering_successful = true
	result.reordered_A = reordered_A
	result.reordered_B = reordered_B
	result.updated_Q = updated_Q
	result.updated_Z = updated_Z
	result.final_ifst = int(ifst_copy - 1) // Convert back to 0-based indexing
	result.final_ilst = int(ilst_copy - 1) // Convert back to 0-based indexing
	return
}

// Generic generalized Schur form reordering function
reorder_generalized_schur :: proc {
	reorder_generalized_schur_complex64,
	reorder_generalized_schur_float64,
	reorder_generalized_schur_float32,
	reorder_generalized_schur_complex128,
}

// ==============================================================================
// Generalized Schur Form Condition Estimation and SVD Functions
// ==============================================================================

// Condition estimation job specification
ConditionJob :: enum {
	EigenvaluesOnly, // Compute eigenvalues only
	Reciprocal, // Compute reciprocal condition numbers
	EstimateBounds, // Estimate error bounds
	Full, // Full condition estimation
}

// SVD job specification
SVDJob :: enum {
	None, // Do not compute matrix
	Compute, // Compute matrix
}

// Generalized Schur condition result
GeneralizedSchurConditionResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	eigenvalues_alpha:      []T, // Alpha values (eigenvalues for complex)
	eigenvalues_beta:       []T, // Beta values
	selected_count:         int, // Number of selected eigenvalues
	reciprocal_left:        S, // Left reciprocal condition number
	reciprocal_right:       S, // Right reciprocal condition number
	error_bounds:           []S, // Error bound estimates
}

// Generalized SVD result
GeneralizedSVDResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	alpha_values:           []S, // Alpha values from GSVD
	beta_values:            []S, // Beta values from GSVD
	U_matrix:               Matrix(T), // Left orthogonal matrix U
	V_matrix:               Matrix(T), // Right orthogonal matrix V
	Q_matrix:               Matrix(T), // Orthogonal matrix Q
	cycles_performed:       int, // Number of Jacobi cycles
}

// Low-level condition estimation functions (ctgsen, dtgsen, stgsen, ztgsen)
ctgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	alpha: ^complex64,
	beta: ^complex64,
	Q: ^complex64,
	ldq: ^Blas_Int,
	Z: ^complex64,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f32,
	pr: ^f32,
	DIF: ^f32,
	work: ^complex64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsen_(
		ijob,
		wantq,
		wantz,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		alpha,
		beta,
		Q,
		ldq,
		Z,
		ldz,
		m,
		pl,
		pr,
		DIF,
		work,
		lwork,
		iwork,
		liwork,
		info,
	)
}

dtgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	alphar: ^f64,
	alphai: ^f64,
	beta: ^f64,
	Q: ^f64,
	ldq: ^Blas_Int,
	Z: ^f64,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f64,
	pr: ^f64,
	DIF: ^f64,
	work: ^f64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsen_(
		ijob,
		wantq,
		wantz,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		alphar,
		alphai,
		beta,
		Q,
		ldq,
		Z,
		ldz,
		m,
		pl,
		pr,
		DIF,
		work,
		lwork,
		iwork,
		liwork,
		info,
	)
}

stgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	alphar: ^f32,
	alphai: ^f32,
	beta: ^f32,
	Q: ^f32,
	ldq: ^Blas_Int,
	Z: ^f32,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f32,
	pr: ^f32,
	DIF: ^f32,
	work: ^f32,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsen_(
		ijob,
		wantq,
		wantz,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		alphar,
		alphai,
		beta,
		Q,
		ldq,
		Z,
		ldz,
		m,
		pl,
		pr,
		DIF,
		work,
		lwork,
		iwork,
		liwork,
		info,
	)
}

ztgsen :: proc(
	ijob: ^Blas_Int,
	wantq: ^Blas_Int,
	wantz: ^Blas_Int,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	alpha: ^complex128,
	beta: ^complex128,
	Q: ^complex128,
	ldq: ^Blas_Int,
	Z: ^complex128,
	ldz: ^Blas_Int,
	m: ^Blas_Int,
	pl: ^f64,
	pr: ^f64,
	DIF: ^f64,
	work: ^complex128,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	liwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsen_(
		ijob,
		wantq,
		wantz,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		alpha,
		beta,
		Q,
		ldq,
		Z,
		ldz,
		m,
		pl,
		pr,
		DIF,
		work,
		lwork,
		iwork,
		liwork,
		info,
	)
}

// Low-level generalized SVD functions (ctgsja, dtgsja, stgsja, ztgsja)
ctgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	tola: ^f32,
	tolb: ^f32,
	alpha: ^f32,
	beta: ^f32,
	U: ^complex64,
	ldu: ^Blas_Int,
	V: ^complex64,
	ldv: ^Blas_Int,
	Q: ^complex64,
	ldq: ^Blas_Int,
	work: ^complex64,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsja_(
		jobu,
		jobv,
		jobq,
		m,
		p,
		n,
		k,
		l,
		A,
		lda,
		B,
		ldb,
		tola,
		tolb,
		alpha,
		beta,
		U,
		ldu,
		V,
		ldv,
		Q,
		ldq,
		work,
		ncycle,
		info,
		len(jobu),
		len(jobv),
		len(jobq),
	)
}

dtgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	tola: ^f64,
	tolb: ^f64,
	alpha: ^f64,
	beta: ^f64,
	U: ^f64,
	ldu: ^Blas_Int,
	V: ^f64,
	ldv: ^Blas_Int,
	Q: ^f64,
	ldq: ^Blas_Int,
	work: ^f64,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsja_(
		jobu,
		jobv,
		jobq,
		m,
		p,
		n,
		k,
		l,
		A,
		lda,
		B,
		ldb,
		tola,
		tolb,
		alpha,
		beta,
		U,
		ldu,
		V,
		ldv,
		Q,
		ldq,
		work,
		ncycle,
		info,
		len(jobu),
		len(jobv),
		len(jobq),
	)
}

stgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	tola: ^f32,
	tolb: ^f32,
	alpha: ^f32,
	beta: ^f32,
	U: ^f32,
	ldu: ^Blas_Int,
	V: ^f32,
	ldv: ^Blas_Int,
	Q: ^f32,
	ldq: ^Blas_Int,
	work: ^f32,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsja_(
		jobu,
		jobv,
		jobq,
		m,
		p,
		n,
		k,
		l,
		A,
		lda,
		B,
		ldb,
		tola,
		tolb,
		alpha,
		beta,
		U,
		ldu,
		V,
		ldv,
		Q,
		ldq,
		work,
		ncycle,
		info,
		len(jobu),
		len(jobv),
		len(jobq),
	)
}

ztgsja :: proc(
	jobu: cstring,
	jobv: cstring,
	jobq: cstring,
	m: ^Blas_Int,
	p: ^Blas_Int,
	n: ^Blas_Int,
	k: ^Blas_Int,
	l: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	tola: ^f64,
	tolb: ^f64,
	alpha: ^f64,
	beta: ^f64,
	U: ^complex128,
	ldu: ^Blas_Int,
	V: ^complex128,
	ldv: ^Blas_Int,
	Q: ^complex128,
	ldq: ^Blas_Int,
	work: ^complex128,
	ncycle: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsja_(
		jobu,
		jobv,
		jobq,
		m,
		p,
		n,
		k,
		l,
		A,
		lda,
		B,
		ldb,
		tola,
		tolb,
		alpha,
		beta,
		U,
		ldu,
		V,
		ldv,
		Q,
		ldq,
		work,
		ncycle,
		info,
		len(jobu),
		len(jobv),
		len(jobq),
	)
}

// ==============================================================================
// Final Convenience Overloads
// ==============================================================================

// Generalized Schur reordering overloads
tgexc :: proc {
	ctgexc,
	dtgexc,
	stgexc,
	ztgexc,
}

// Generalized Schur condition estimation overloads
tgsen :: proc {
	ctgsen,
	dtgsen,
	stgsen,
	ztgsen,
}

// Generalized SVD overloads
tgsja :: proc {
	ctgsja,
	dtgsja,
	stgsja,
	ztgsja,
}

// ==============================================================================
// Generalized Eigenvalue Sensitivity Analysis Functions
// ==============================================================================

// Sensitivity analysis job specification
SensitivityJob :: enum {
	EigenvaluesOnly, // Compute eigenvalue condition numbers only
	SubspacesOnly, // Compute invariant subspace condition numbers only
	Both, // Compute both eigenvalue and subspace condition numbers
}

// Generalized eigenvalue sensitivity result
GeneralizedSensitivityResult :: struct($T: typeid, $S: typeid) {
	computation_successful: bool,
	condition_numbers_S:    []S, // Eigenvalue condition numbers
	condition_numbers_DIF:  []S, // Invariant subspace condition numbers
	num_computed:           int, // Number of condition numbers computed
	selection_mask:         []bool, // Selection mask for computed values
}

// Low-level generalized eigenvalue sensitivity functions (ctgsna, dtgsna, stgsna, ztgsna)
ctgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	VL: ^complex64,
	ldvl: ^Blas_Int,
	VR: ^complex64,
	ldvr: ^Blas_Int,
	S: ^f32,
	DIF: ^f32,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsna_(
		job,
		howmny,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		VL,
		ldvl,
		VR,
		ldvr,
		S,
		DIF,
		mm,
		m,
		work,
		lwork,
		iwork,
		info,
		len(job),
		len(howmny),
	)
}

dtgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	VL: ^f64,
	ldvl: ^Blas_Int,
	VR: ^f64,
	ldvr: ^Blas_Int,
	S: ^f64,
	DIF: ^f64,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsna_(
		job,
		howmny,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		VL,
		ldvl,
		VR,
		ldvr,
		S,
		DIF,
		mm,
		m,
		work,
		lwork,
		iwork,
		info,
		len(job),
		len(howmny),
	)
}

stgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	VL: ^f32,
	ldvl: ^Blas_Int,
	VR: ^f32,
	ldvr: ^Blas_Int,
	S: ^f32,
	DIF: ^f32,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^f32,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsna_(
		job,
		howmny,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		VL,
		ldvl,
		VR,
		ldvr,
		S,
		DIF,
		mm,
		m,
		work,
		lwork,
		iwork,
		info,
		len(job),
		len(howmny),
	)
}

ztgsna :: proc(
	job: cstring,
	howmny: cstring,
	select: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	VL: ^complex128,
	ldvl: ^Blas_Int,
	VR: ^complex128,
	ldvr: ^Blas_Int,
	S: ^f64,
	DIF: ^f64,
	mm: ^Blas_Int,
	m: ^Blas_Int,
	work: ^complex128,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsna_(
		job,
		howmny,
		select,
		n,
		A,
		lda,
		B,
		ldb,
		VL,
		ldvl,
		VR,
		ldvr,
		S,
		DIF,
		mm,
		m,
		work,
		lwork,
		iwork,
		info,
		len(job),
		len(howmny),
	)
}

// Helper function to convert sensitivity job to string
sensitivity_job_to_cstring :: proc(job: SensitivityJob) -> cstring {
	switch job {
	case .EigenvaluesOnly:
		return "E"
	case .SubspacesOnly:
		return "V"
	case .Both:
		return "B"
	}
	return "B"
}

// High-level generalized eigenvalue sensitivity wrapper function
compute_generalized_sensitivity_complex64 :: proc(
	A: Matrix(complex64),
	B: Matrix(complex64),
	VL: Matrix(complex64) = {},
	VR: Matrix(complex64) = {},
	job: SensitivityJob = .Both,
	selection: EigenvectorSelection = .All,
	select_mask: []bool = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedSensitivityResult(complex64, f32),
	err: LapackError,
) {

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.rows)
	ldb := Blas_Int(B.rows)
	ldvl := Blas_Int(1)
	ldvr := Blas_Int(1)

	// Handle left eigenvectors
	vl_ptr: ^complex64 = nil
	if VL.data != nil {
		ldvl = Blas_Int(VL.rows)
		vl_ptr = raw_data(VL.data)
	}

	// Handle right eigenvectors
	vr_ptr: ^complex64 = nil
	if VR.data != nil {
		ldvr = Blas_Int(VR.rows)
		vr_ptr = raw_data(VR.data)
	}

	// Setup selection array
	select_array: []Blas_Int = nil
	if selection == .Selected && select_mask != nil {
		select_array = make([]Blas_Int, len(select_mask), allocator) or_return
		for i, selected in select_mask {
			select_array[i] = selected ? 1 : 0
		}
	}

	// Allocate output arrays
	s_values: []f32 = nil
	dif_values: []f32 = nil

	max_compute := int(n)
	if select_mask != nil {
		max_compute = len(select_mask)
	}

	if job == .EigenvaluesOnly || job == .Both {
		s_values = make([]f32, max_compute, allocator) or_return
	}

	if job == .SubspacesOnly || job == .Both {
		dif_values = make([]f32, max_compute, allocator) or_return
	}

	// Query workspace size
	work_query: complex64
	lwork_query: Blas_Int = -1
	info_query: Blas_Int
	mm := Blas_Int(max_compute)
	m: Blas_Int

	job_str := sensitivity_job_to_cstring(job)
	howmny_str := eigenvector_selection_to_cstring(selection)
	select_ptr := raw_data(select_array) if select_array != nil else nil
	s_ptr := raw_data(s_values) if s_values != nil else nil
	dif_ptr := raw_data(dif_values) if dif_values != nil else nil

	ctgsna(
		job_str,
		howmny_str,
		select_ptr,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		s_ptr,
		dif_ptr,
		&mm,
		&m,
		&work_query,
		&lwork_query,
		nil,
		&info_query,
	)

	if info_query != 0 && info_query != -17 {
		if s_values != nil do delete(s_values, allocator)
		if dif_values != nil do delete(dif_values, allocator)
		if select_array != nil do delete(select_array, allocator)
		return {}, .InvalidParameter
	}

	// Allocate workspace
	lwork := Blas_Int(real(work_query))
	work := make([]complex64, int(lwork), allocator) or_return
	iwork := make([]Blas_Int, int(n + 2), allocator) or_return

	info: Blas_Int
	ctgsna(
		job_str,
		howmny_str,
		select_ptr,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		vl_ptr,
		&ldvl,
		vr_ptr,
		&ldvr,
		s_ptr,
		dif_ptr,
		&mm,
		&m,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&info,
	)

	delete(work, allocator)
	delete(iwork, allocator)
	if select_array != nil do delete(select_array, allocator)

	if info != 0 {
		if s_values != nil do delete(s_values, allocator)
		if dif_values != nil do delete(dif_values, allocator)
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	// Create result mask
	result_mask: []bool = nil
	if select_mask != nil {
		result_mask = make([]bool, len(select_mask), allocator) or_return
		copy(result_mask, select_mask)
	}

	result.computation_successful = true
	result.condition_numbers_S = s_values
	result.condition_numbers_DIF = dif_values
	result.num_computed = int(m)
	result.selection_mask = result_mask
	return
}

// ==============================================================================
// Generalized Sylvester Equation Functions
// ==============================================================================

// Generalized Sylvester equation result
GeneralizedSylvesterResult :: struct($T: typeid, $S: typeid) {
	solution_successful: bool,
	solution_C:          Matrix(T), // Solution matrix C
	solution_F:          Matrix(T), // Solution matrix F
	dif_estimate:        S, // DIF estimate (if requested)
	scale_factor:        S, // Scale factor applied to solution
}

// Low-level generalized Sylvester equation functions (ctgsyl, dtgsyl, stgsyl, ztgsyl)
ctgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex64,
	lda: ^Blas_Int,
	B: ^complex64,
	ldb: ^Blas_Int,
	C: ^complex64,
	ldc: ^Blas_Int,
	D: ^complex64,
	ldd: ^Blas_Int,
	E: ^complex64,
	lde: ^Blas_Int,
	F: ^complex64,
	ldf: ^Blas_Int,
	dif: ^f32,
	scale: ^f32,
	work: ^complex64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ctgsyl_(
		trans,
		ijob,
		m,
		n,
		A,
		lda,
		B,
		ldb,
		C,
		ldc,
		D,
		ldd,
		E,
		lde,
		F,
		ldf,
		dif,
		scale,
		work,
		lwork,
		iwork,
		info,
		len(trans),
	)
}

dtgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f64,
	lda: ^Blas_Int,
	B: ^f64,
	ldb: ^Blas_Int,
	C: ^f64,
	ldc: ^Blas_Int,
	D: ^f64,
	ldd: ^Blas_Int,
	E: ^f64,
	lde: ^Blas_Int,
	F: ^f64,
	ldf: ^Blas_Int,
	dif: ^f64,
	scale: ^f64,
	work: ^f64,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtgsyl_(
		trans,
		ijob,
		m,
		n,
		A,
		lda,
		B,
		ldb,
		C,
		ldc,
		D,
		ldd,
		E,
		lde,
		F,
		ldf,
		dif,
		scale,
		work,
		lwork,
		iwork,
		info,
		len(trans),
	)
}

stgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^f32,
	lda: ^Blas_Int,
	B: ^f32,
	ldb: ^Blas_Int,
	C: ^f32,
	ldc: ^Blas_Int,
	D: ^f32,
	ldd: ^Blas_Int,
	E: ^f32,
	lde: ^Blas_Int,
	F: ^f32,
	ldf: ^Blas_Int,
	dif: ^f32,
	scale: ^f32,
	work: ^f32,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stgsyl_(
		trans,
		ijob,
		m,
		n,
		A,
		lda,
		B,
		ldb,
		C,
		ldc,
		D,
		ldd,
		E,
		lde,
		F,
		ldf,
		dif,
		scale,
		work,
		lwork,
		iwork,
		info,
		len(trans),
	)
}

ztgsyl :: proc(
	trans: cstring,
	ijob: ^Blas_Int,
	m: ^Blas_Int,
	n: ^Blas_Int,
	A: ^complex128,
	lda: ^Blas_Int,
	B: ^complex128,
	ldb: ^Blas_Int,
	C: ^complex128,
	ldc: ^Blas_Int,
	D: ^complex128,
	ldd: ^Blas_Int,
	E: ^complex128,
	lde: ^Blas_Int,
	F: ^complex128,
	ldf: ^Blas_Int,
	dif: ^f64,
	scale: ^f64,
	work: ^complex128,
	lwork: ^Blas_Int,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	ztgsyl_(
		trans,
		ijob,
		m,
		n,
		A,
		lda,
		B,
		ldb,
		C,
		ldc,
		D,
		ldd,
		E,
		lde,
		F,
		ldf,
		dif,
		scale,
		work,
		lwork,
		iwork,
		info,
		len(trans),
	)
}

// ==============================================================================
// Triangular Packed Condition Number Functions
// ==============================================================================

// Triangular packed condition number result
TriangularPackedConditionResult :: struct($T: typeid) {
	computation_successful: bool,
	reciprocal_condition:   T, // Reciprocal condition number
	condition_number:       T, // 1/rcond
}

// Low-level triangular packed condition number functions (ctpcon, dtpcon, stpcon, ztpcon)
ctpcon :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	AP: ^complex64,
	rcond: ^f32,
	work: ^complex64,
	rwork: ^f32,
	info: ^Blas_Int,
) {
	ctpcon_(norm, uplo, diag, n, AP, rcond, work, rwork, info, len(norm), len(uplo), len(diag))
}

dtpcon :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	AP: ^f64,
	rcond: ^f64,
	work: ^f64,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	dtpcon_(norm, uplo, diag, n, AP, rcond, work, iwork, info, len(norm), len(uplo), len(diag))
}

stpcon :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	AP: ^f32,
	rcond: ^f32,
	work: ^f32,
	iwork: ^Blas_Int,
	info: ^Blas_Int,
) {
	stpcon_(norm, uplo, diag, n, AP, rcond, work, iwork, info, len(norm), len(uplo), len(diag))
}

ztpcon :: proc(
	norm: cstring,
	uplo: cstring,
	diag: cstring,
	n: ^Blas_Int,
	AP: ^complex128,
	rcond: ^f64,
	work: ^complex128,
	rwork: ^f64,
	info: ^Blas_Int,
) {
	ztpcon_(norm, uplo, diag, n, AP, rcond, work, rwork, info, len(norm), len(uplo), len(diag))
}

// High-level triangular packed condition number wrapper function
estimate_triangular_packed_condition_complex64 :: proc(
	AP: []complex64,
	n: int,
	norm: MatrixNorm = .OneNorm,
	uplo: MatrixTriangle = .Upper,
	diag: MatrixDiagonal = .NonUnit,
	allocator := context.allocator,
) -> (
	result: TriangularPackedConditionResult(f32),
	err: LapackError,
) {

	n_int := Blas_Int(n)

	// Calculate expected packed size
	packed_size := (n * (n + 1)) / 2
	if len(AP) < packed_size {
		return {}, .InvalidParameter
	}

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator) or_return
	rwork := make([]f32, n, allocator) or_return

	rcond: f32
	info: Blas_Int
	norm_str := matrix_norm_to_cstring(norm)
	uplo_str := matrix_triangle_to_cstring(uplo)
	diag_str := matrix_diagonal_to_cstring(diag)

	ctpcon(
		norm_str,
		uplo_str,
		diag_str,
		&n_int,
		raw_data(AP),
		&rcond,
		raw_data(work),
		raw_data(rwork),
		&info,
	)

	delete(work, allocator)
	delete(rwork, allocator)

	if info != 0 {
		if info < 0 {
			return {}, .InvalidParameter
		} else {
			return {}, .ComputationFailed
		}
	}

	result.computation_successful = true
	result.reciprocal_condition = rcond
	result.condition_number = rcond > 0 ? 1.0 / rcond : f32(max(f32))
	return
}

// ==============================================================================
// Final Advanced Function Overloads
// ==============================================================================

// Generalized eigenvalue sensitivity overloads
tgsna :: proc {
	ctgsna,
	dtgsna,
	stgsna,
	ztgsna,
}

// Generalized Sylvester equation overloads
tgsyl :: proc {
	ctgsyl,
	dtgsyl,
	stgsyl,
	ztgsyl,
}

// Triangular packed condition number overloads
tpcon :: proc {
	ctpcon,
	dtpcon,
	stpcon,
	ztpcon,
}
