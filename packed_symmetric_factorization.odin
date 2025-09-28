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

// Reduce packed symmetric to tridiagonal for f32/f64
m_reduce_packed_symmetric_tridiagonal_f32_f64 :: proc(
	ap: []$T, // Packed matrix (modified on output)
	d: []T, // Pre-allocated diagonal of tridiagonal (size n)
	e: []T, // Pre-allocated off-diagonal of tridiagonal (size n-1)
	tau: []T, // Pre-allocated elementary reflectors (size n-1)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(d) >= n, "Diagonal array too small")
	if n > 0 {
		assert(len(e) >= n - 1, "Off-diagonal array too small")
		assert(len(tau) >= n - 1, "Tau array too small")
	}

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.ssptrd_(&uplo_c, &n_int, raw_data(ap), raw_data(d), raw_data(e), raw_data(tau), &info, 1)
	} else when T == f64 {
		lapack.dsptrd_(&uplo_c, &n_int, raw_data(ap), raw_data(d), raw_data(e), raw_data(tau), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Reduce packed Hermitian to tridiagonal for complex64/complex128
m_reduce_packed_hermitian_tridiagonal_c64_c128 :: proc(
	ap: []$T, // Packed Hermitian matrix (modified on output)
	d: []$R, // Pre-allocated diagonal of tridiagonal (size n)
	e: []R, // Pre-allocated off-diagonal of tridiagonal (size n-1)
	tau: []T, // Pre-allocated elementary reflectors (size n-1)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(d) >= n, "Diagonal array too small")
	if n > 0 {
		assert(len(e) >= n - 1, "Off-diagonal array too small")
		assert(len(tau) >= n - 1, "Tau array too small")
	}

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == complex64 {
		lapack.chptrd_(&uplo_c, &n_int, cast(^lapack.complex)raw_data(ap), raw_data(d), raw_data(e), cast(^lapack.complex)raw_data(tau), &info, 1)
	} else when T == complex128 {
		lapack.zhptrd_(&uplo_c, &n_int, cast(^lapack.doublecomplex)raw_data(ap), raw_data(d), raw_data(e), cast(^lapack.doublecomplex)raw_data(tau), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed to tridiagonal reduction
m_reduce_packed_tridiagonal :: proc {
	m_reduce_packed_symmetric_tridiagonal_f32_f64,
	m_reduce_packed_hermitian_tridiagonal_c64_c128,
}

// ============================================================================
// PACKED SYMMETRIC FACTORIZATION
// ============================================================================
// Computes the factorization of a symmetric matrix using Bunch-Kaufman diagonal pivoting

// Factor packed symmetric matrix for f32/f64
m_factor_packed_symmetric_f32_f64 :: proc(
	ap: []$T, // Packed matrix (modified to factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.ssptrf_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), &info, 1)
	} else when T == f64 {
		lapack.dsptrf_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Factor packed symmetric matrix for complex64/complex128
m_factor_packed_symmetric_c64_c128 :: proc(
	ap: []$T, // Packed matrix (modified to factorization)
	ipiv: []Blas_Int, // Pre-allocated pivot indices (size n)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == complex64 {
		lapack.csptrf_(&uplo_c, &n_int, cast(^lapack.complex)raw_data(ap), raw_data(ipiv), &info, 1)
	} else when T == complex128 {
		lapack.zsptrf_(&uplo_c, &n_int, cast(^lapack.doublecomplex)raw_data(ap), raw_data(ipiv), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed symmetric factorization
m_factor_packed_symmetric :: proc {
	m_factor_packed_symmetric_f32_f64,
	m_factor_packed_symmetric_c64_c128,
}

// ============================================================================
// PACKED SYMMETRIC MATRIX INVERSION
// ============================================================================
// Computes the inverse of a symmetric matrix using the factorization from sptrf

// Invert packed symmetric matrix for f32/f64
m_invert_packed_symmetric_f32_f64 :: proc(
	ap: []$T, // Factored matrix from sptrf (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from sptrf
	work: []T, // Pre-allocated workspace (size n)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.ssptri_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), raw_data(work), &info, 1)
	} else when T == f64 {
		lapack.dsptri_(&uplo_c, &n_int, raw_data(ap), raw_data(ipiv), raw_data(work), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Invert packed symmetric matrix for complex64/complex128
m_invert_packed_symmetric_c64_c128 :: proc(
	ap: []$T, // Factored matrix from sptrf (modified to inverse)
	ipiv: []Blas_Int, // Pivot indices from sptrf
	work: []T, // Pre-allocated workspace (size n)
	uplo := MatrixRegion.Upper,
	n: int, // Matrix dimension
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)

	when T == complex64 {
		lapack.csptri_(&uplo_c, &n_int, cast(^lapack.complex)raw_data(ap), raw_data(ipiv), cast(^lapack.complex)raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zsptri_(&uplo_c, &n_int, cast(^lapack.doublecomplex)raw_data(ap), raw_data(ipiv), cast(^lapack.doublecomplex)raw_data(work), &info, 1)
	}

	ok = info == 0
	return info, ok
}

// Procedure group for packed symmetric matrix inversion
m_invert_packed_symmetric :: proc {
	m_invert_packed_symmetric_f32_f64,
	m_invert_packed_symmetric_c64_c128,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Reduce packed symmetric matrix to tridiagonal form and get eigenvalues (allocating version for convenience)
reduce_packed_to_tridiagonal :: proc(ap: []$T, n: int, uplo := MatrixRegion.Upper, allocator := context.allocator) -> (diagonal: []T, off_diagonal: []T, tau: []T, info: Info) {
	// Make a copy since ap gets modified
	ap_copy := make([]T, len(ap), allocator)
	copy(ap_copy, ap)

	// Allocate output arrays
	diagonal = make([]T, n, allocator)
	off_diagonal = make([]T, max(0, n - 1), allocator)
	tau = make([]T, max(0, n - 1), allocator)

	when T == f32 || T == f64 {
		info, _ = m_reduce_packed_symmetric_tridiagonal_f32_f64(ap_copy, diagonal, off_diagonal, tau, uplo, n)
	} else {
		#panic("Unsupported type for packed to tridiagonal reduction - use complex Hermitian version")
	}

	delete(ap_copy)
	return diagonal, off_diagonal, tau, info
}

// Factor packed symmetric matrix (allocating version for convenience)
factor_packed_symmetric :: proc(ap: []$T, n: int, uplo := MatrixRegion.Upper, allocator := context.allocator) -> (factored: []T, pivots: []Blas_Int, info: Info) {
	// Make a copy for factorization
	factored = make([]T, len(ap), allocator)
	copy(factored, ap)

	// Allocate pivot array
	pivots = make([]Blas_Int, n, allocator)

	when T == f32 || T == f64 {
		info, _ = m_factor_packed_symmetric_f32_f64(factored, pivots, uplo, n)
	} else when T == complex64 || T == complex128 {
		info, _ = m_factor_packed_symmetric_c64_c128(factored, pivots, uplo, n)
	} else {
		#panic("Unsupported type for packed factorization")
	}

	return factored, pivots, info
}

// Compute inverse of packed symmetric matrix (allocating version for convenience)
invert_packed_symmetric :: proc(ap: []$T, n: int, uplo := MatrixRegion.Upper, allocator := context.allocator) -> (inverse: []T, success: bool) {
	// First factor the matrix
	factored, pivots, fact_info := factor_packed_symmetric(ap, n, uplo, allocator)
	defer delete(pivots)

	if fact_info != 0 {
		delete(factored)
		return nil, false
	}

	// Allocate workspace for inversion
	work := make([]T, n, allocator)
	defer delete(work)

	// Compute inverse using factorization
	var; inv_info: Info
	when T == f32 || T == f64 {
		inv_info, success = m_invert_packed_symmetric_f32_f64(factored, pivots, work, uplo, n)
	} else when T == complex64 || T == complex128 {
		inv_info, success = m_invert_packed_symmetric_c64_c128(factored, pivots, work, uplo, n)
	} else {
		#panic("Unsupported type for packed inversion")
	}

	if success && inv_info == 0 {
		inverse = factored
	} else {
		delete(factored)
		inverse = nil
		success = false
	}

	return
}
