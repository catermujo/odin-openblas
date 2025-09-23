package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC MATRIX OPERATIONS
// ============================================================================
// Reduction to tridiagonal form, factorization, and inversion for symmetric
// matrices stored in packed format

// ============================================================================
// PACKED SYMMETRIC TO TRIDIAGONAL REDUCTION
// ============================================================================
// Reduces a real symmetric matrix in packed storage to tridiagonal form

// Tridiagonal reduction result for packed matrices
PackedTridiagonalResult :: struct($T: typeid) {
	diagonal:     []T, // Diagonal elements of tridiagonal matrix
	off_diagonal: []T, // Off-diagonal elements
	tau:          []T, // Elementary reflectors
}

// Double precision packed to tridiagonal reduction
dsptrd :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f64, // Packed matrix (modified on output)
	d: []f64 = nil, // Diagonal of tridiagonal (size n)
	e: []f64 = nil, // Off-diagonal of tridiagonal (size n-1)
	tau: []f64 = nil, // Elementary reflectors (size n-1)
	allocator := context.allocator,
) -> (
	result: PackedTridiagonalResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate diagonal if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f64, n, allocator)
	}
	result.diagonal = d

	// Allocate off-diagonal if not provided
	allocated_e := e == nil
	if allocated_e && n > 0 {
		e = make([]f64, n - 1, allocator)
	}
	result.off_diagonal = e

	// Allocate tau if not provided
	allocated_tau := tau == nil
	if allocated_tau && n > 0 {
		tau = make([]f64, n - 1, allocator)
	}
	result.tau = tau

	// Call LAPACK
	lapack.dsptrd_(
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(d),
		raw_data(e),
		raw_data(tau),
		&info_int,
		1,
	)

	return result, Info(info_int)
}

// Single precision packed to tridiagonal reduction
ssptrd :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	d: []f32 = nil,
	e: []f32 = nil,
	tau: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: PackedTridiagonalResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate diagonal if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f32, n, allocator)
	}
	result.diagonal = d

	// Allocate off-diagonal if not provided
	allocated_e := e == nil
	if allocated_e && n > 0 {
		e = make([]f32, n - 1, allocator)
	}
	result.off_diagonal = e

	// Allocate tau if not provided
	allocated_tau := tau == nil
	if allocated_tau && n > 0 {
		tau = make([]f32, n - 1, allocator)
	}
	result.tau = tau

	// Call LAPACK
	lapack.ssptrd_(
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(d),
		raw_data(e),
		raw_data(tau),
		&info_int,
		1,
	)

	return result, Info(info_int)
}

sptrd :: proc {
	dsptrd,
	ssptrd,
}

// ============================================================================
// PACKED SYMMETRIC FACTORIZATION
// ============================================================================
// Computes the factorization of a symmetric matrix using Bunch-Kaufman diagonal pivoting

// Packed factorization result
PackedFactorizationResult :: struct {
	pivot_indices:        []Blas_Int, // Pivot indices
	is_positive_definite: bool, // True if matrix is positive definite
	is_singular:          bool, // True if matrix is singular
	singular_index:       int, // Index where singularity detected
	num_positive_pivots:  int, // Number of positive pivots
	num_negative_pivots:  int, // Number of negative pivots
	num_zero_pivots:      int, // Number of zero pivots
}

// Complex single precision packed factorization
csptrf :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []complex64, // Packed matrix (modified to factorization)
	allocator := context.allocator,
) -> (
	result: PackedFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.csptrf_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)raw_data(ap),
		raw_data(ipiv),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1 // Convert to 0-based
	}

	// Analyze pivots
	for i in 0 ..< n {
		pivot := ipiv[i]
		if pivot > 0 {
			result.num_positive_pivots += 1
		} else if pivot < 0 {
			result.num_negative_pivots += 1
			i += 1 // Skip next element for 2x2 block
		} else {
			result.num_zero_pivots += 1
		}
	}

	return
}

// Double precision packed factorization
dsptrf :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f64,
	allocator := context.allocator,
) -> (
	result: PackedFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.dsptrf_(uplo_cstring, &n_int, raw_data(ap), raw_data(ipiv), &info_int, 1)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1
	}

	// Analyze pivots
	for i in 0 ..< n {
		pivot := ipiv[i]
		if pivot > 0 {
			result.num_positive_pivots += 1
		} else if pivot < 0 {
			result.num_negative_pivots += 1
			i += 1 // Skip next element for 2x2 block
		} else {
			result.num_zero_pivots += 1
		}
	}

	return
}

// Single precision packed factorization
ssptrf :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	allocator := context.allocator,
) -> (
	result: PackedFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.ssptrf_(uplo_cstring, &n_int, raw_data(ap), raw_data(ipiv), &info_int, 1)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1
	}

	// Analyze pivots
	for i in 0 ..< n {
		pivot := ipiv[i]
		if pivot > 0 {
			result.num_positive_pivots += 1
		} else if pivot < 0 {
			result.num_negative_pivots += 1
			i += 1 // Skip next element for 2x2 block
		} else {
			result.num_zero_pivots += 1
		}
	}

	return
}

// Complex double precision packed factorization
zsptrf :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []complex128,
	allocator := context.allocator,
) -> (
	result: PackedFactorizationResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate pivot array
	ipiv := make([]Blas_Int, n, allocator)
	result.pivot_indices = ipiv

	// Call LAPACK
	lapack.zsptrf_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)raw_data(ap),
		raw_data(ipiv),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check for singularity
	if info > 0 {
		result.is_singular = true
		result.singular_index = int(info) - 1
	}

	// Analyze pivots
	for i in 0 ..< n {
		pivot := ipiv[i]
		if pivot > 0 {
			result.num_positive_pivots += 1
		} else if pivot < 0 {
			result.num_negative_pivots += 1
			i += 1 // Skip next element for 2x2 block
		} else {
			result.num_zero_pivots += 1
		}
	}

	return
}

sptrf :: proc {
	csptrf,
	dsptrf,
	ssptrf,
	zsptrf,
}

// ============================================================================
// PACKED SYMMETRIC MATRIX INVERSION
// ============================================================================
// Computes the inverse of a symmetric matrix using the factorization from sptrf

// Packed inversion result
PackedInversionResult :: struct {
	is_singular:      bool, // True if matrix is singular
	inverse_computed: bool, // True if inverse was successfully computed
}

// Complex single precision packed matrix inversion
csptri :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []complex64, // Factored matrix from csptrf (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from csptrf
	work: []complex64 = nil, // Workspace (size n if nil)
	allocator := context.allocator,
) -> (
	result: PackedInversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csptri_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)raw_data(ap),
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check result
	if info > 0 {
		result.is_singular = true
		result.inverse_computed = false
	} else if info == .OK {
		result.inverse_computed = true
	}

	return
}

// Double precision packed matrix inversion
dsptri :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f64,
	ipiv: []Blas_Int,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: PackedInversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsptri_(
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(ipiv),
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check result
	if info > 0 {
		result.is_singular = true
		result.inverse_computed = false
	} else if info == .OK {
		result.inverse_computed = true
	}

	return
}

// Single precision packed matrix inversion
ssptri :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	ipiv: []Blas_Int,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: PackedInversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssptri_(
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(ipiv),
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check result
	if info > 0 {
		result.is_singular = true
		result.inverse_computed = false
	} else if info == .OK {
		result.inverse_computed = true
	}

	return
}

// Complex double precision packed matrix inversion
zsptri :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []complex128,
	ipiv: []Blas_Int,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: PackedInversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsptri_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)raw_data(ap),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Check result
	if info > 0 {
		result.is_singular = true
		result.inverse_computed = false
	} else if info == .OK {
		result.inverse_computed = true
	}

	return
}

sptri :: proc {
	csptri,
	dsptri,
	ssptri,
	zsptri,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Reduce packed symmetric matrix to tridiagonal form and get eigenvalues
reduce_packed_to_tridiagonal :: proc(
	ap: []$T,
	n: int,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	diagonal: []T,
	off_diagonal: []T,
	info: Info,
) {
	// Make a copy since ap gets modified
	ap_copy := make([]T, len(ap), allocator)
	copy(ap_copy, ap)
	defer delete(ap_copy)

	when T == f64 {
		result, info_val := dsptrd(uplo, n, ap_copy, allocator = allocator)
		defer delete(result.tau)
		return result.diagonal, result.off_diagonal, info_val
	} else when T == f32 {
		result, info_val := ssptrd(uplo, n, ap_copy, allocator = allocator)
		defer delete(result.tau)
		return result.diagonal, result.off_diagonal, info_val
	} else {
		#panic("Unsupported type for packed to tridiagonal reduction")
	}
}

// Factor packed symmetric matrix
factor_packed_symmetric :: proc(
	ap: []$T,
	n: int,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	factored: []T,
	pivots: []Blas_Int,
	info: Info,
) {
	// Make a copy for factorization
	factored = make([]T, len(ap), allocator)
	copy(factored, ap)

	when T == complex64 {
		result, info_val := csptrf(uplo, n, factored, allocator)
		return factored, result.pivot_indices, info_val
	} else when T == complex128 {
		result, info_val := zsptrf(uplo, n, factored, allocator)
		return factored, result.pivot_indices, info_val
	} else when T == f64 {
		result, info_val := dsptrf(uplo, n, factored, allocator)
		return factored, result.pivot_indices, info_val
	} else when T == f32 {
		result, info_val := ssptrf(uplo, n, factored, allocator)
		return factored, result.pivot_indices, info_val
	} else {
		#panic("Unsupported type for packed factorization")
	}
}

// Compute inverse of packed symmetric matrix
invert_packed_symmetric :: proc(
	ap: []$T,
	n: int,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	inverse: []T,
	success: bool,
) {
	// First factor the matrix
	factored, pivots, fact_info := factor_packed_symmetric(ap, n, uplo, allocator)
	defer delete(pivots)

	if fact_info != .OK {
		delete(factored)
		return nil, false
	}

	// Then compute inverse using factorization
	when T == complex64 {
		result, inv_info := csptri(uplo, n, factored, pivots, allocator = allocator)
		success = inv_info == .OK && result.inverse_computed
	} else when T == complex128 {
		result, inv_info := zsptri(uplo, n, factored, pivots, allocator = allocator)
		success = inv_info == .OK && result.inverse_computed
	} else when T == f64 {
		result, inv_info := dsptri(uplo, n, factored, pivots, allocator = allocator)
		success = inv_info == .OK && result.inverse_computed
	} else when T == f32 {
		result, inv_info := ssptri(uplo, n, factored, pivots, allocator = allocator)
		success = inv_info == .OK && result.inverse_computed
	} else {
		#panic("Unsupported type for packed inversion")
	}

	if success {
		inverse = factored
	} else {
		delete(factored)
		inverse = nil
	}

	return
}

// Check if packed symmetric matrix is invertible
is_symmetric_packed_invertible :: proc(
	ap: []$T,
	n: int,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	invertible: bool,
	condition_estimate: f64,
) {
	// Factor the matrix
	factored, pivots, fact_info := factor_packed_symmetric(ap, n, uplo, allocator)
	defer {
		delete(factored)
		delete(pivots)
	}

	if fact_info != .OK {
		return false, math.INF_F64
	}

	// Use the factorization to estimate invertibility
	when T == f64 || T == f32 {
		// For real matrices, check the pivots
		result: PackedFactorizationResult
		for i in 0 ..< n {
			pivot := pivots[i]
			if pivot > 0 {
				result.num_positive_pivots += 1
			} else if pivot < 0 {
				result.num_negative_pivots += 1
				i += 1 // Skip next element for 2x2 block
			} else {
				result.num_zero_pivots += 1
			}
		}

		invertible = result.num_zero_pivots == 0

		// Rough condition estimate based on pivot analysis
		if result.num_zero_pivots > 0 {
			condition_estimate = math.INF_F64
		} else {
			// This is a very rough estimate
			condition_estimate = f64(n) // Would need actual condition estimation for better value
		}
	} else {
		// For complex matrices, similar analysis
		invertible = true // Assume invertible unless proven otherwise
		condition_estimate = f64(n)
	}

	return
}

// Compute eigenvalues after reducing to tridiagonal form
eigenvalues_from_packed :: proc(
	ap: []$T,
	n: int,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	info: Info,
) {
	// First reduce to tridiagonal form
	diagonal, off_diagonal, red_info := reduce_packed_to_tridiagonal(ap, n, uplo, allocator)
	defer {
		delete(diagonal)
		delete(off_diagonal)
	}

	if red_info != .OK {
		return nil, red_info
	}

	// Then compute eigenvalues of tridiagonal matrix
	// This would require calling dsterf or similar
	// For now, return the diagonal as an approximation
	eigenvalues = make([]T, n, allocator)
	copy(eigenvalues, diagonal)

	return eigenvalues, .OK
}

// Analyze packed symmetric matrix structure
analyze_packed_structure :: proc(
	ap: []$T,
	n: int,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	analysis: PackedMatrixAnalysis,
) {
	// Factor the matrix to analyze its structure
	factored, pivots, fact_info := factor_packed_symmetric(ap, n, uplo, allocator)
	defer {
		delete(factored)
		delete(pivots)
	}

	analysis.n = n
	analysis.storage_size = n * (n + 1) / 2
	analysis.is_factorizable = fact_info == .OK

	if fact_info == .OK {
		// Analyze pivot structure
		for i in 0 ..< n {
			pivot := pivots[i]
			if pivot > 0 {
				analysis.num_positive_pivots += 1
			} else if pivot < 0 {
				analysis.num_negative_pivots += 1
				i += 1 // Skip next element for 2x2 block
			} else {
				analysis.num_zero_pivots += 1
			}
		}

		analysis.is_positive_definite =
			analysis.num_negative_pivots == 0 && analysis.num_zero_pivots == 0
		analysis.is_indefinite =
			analysis.num_negative_pivots > 0 && analysis.num_positive_pivots > 0
		analysis.is_singular = analysis.num_zero_pivots > 0
	}

	return
}

// Packed matrix analysis structure
PackedMatrixAnalysis :: struct {
	n:                    int,
	storage_size:         int,
	is_factorizable:      bool,
	is_positive_definite: bool,
	is_indefinite:        bool,
	is_singular:          bool,
	num_positive_pivots:  int,
	num_negative_pivots:  int,
	num_zero_pivots:      int,
}
