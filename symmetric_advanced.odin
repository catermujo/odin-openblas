package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC MATRIX CONDITION ESTIMATION WITH E FACTOR
// ============================================================================
// Uses the factorization with block diagonal matrix E from sytrf_rk

// Query workspace for symmetric condition estimation with E factor
query_workspace_condition_symmetric_e :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	when T == complex64 || T == complex128 {
		// Complex types only need work array (2*n)
		work_size = 2 * n
		iwork_size = 0
	} else {
		// Real types need both work (2*n) and iwork (n)
		work_size = 2 * n
		iwork_size = n
	}
	return work_size, iwork_size
}

// Estimate condition number with E factor for f32/f64
m_condition_symmetric_e_f32_f64 :: proc(
	a: ^Matrix($T), // Factored matrix from sytrf_rk
	e: []T, // Block diagonal matrix E from sytrf_rk
	ipiv: []Blas_Int, // Pivot indices from sytrf_rk
	anorm: T, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size 2*n)
	iwork: []Blas_Int, // Pre-allocated integer workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(e) >= n, "E array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	anorm_val := anorm

	when T == f32 {
		lapack.ssycon_3_(&uplo_c, &n_int, raw_data(a.data), &lda, raw_data(e), raw_data(ipiv), &anorm_val, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	} else when T == f64 {
		lapack.dsycon_3_(&uplo_c, &n_int, raw_data(a.data), &lda, raw_data(e), raw_data(ipiv), &anorm_val, &rcond, raw_data(work), raw_data(iwork), &info, 1)
	}

	ok = info == 0
	return rcond, info, ok
}

// Estimate condition number with E factor for complex64/complex128
m_condition_symmetric_e_c64_c128 :: proc(
	a: ^Matrix($T), // Factored matrix from csytrf_rk
	e: []T, // Block diagonal matrix E from csytrf_rk
	ipiv: []Blas_Int, // Pivot indices from csytrf_rk
	anorm: $R, // 1-norm of original matrix
	work: []T, // Pre-allocated workspace (size 2*n)
	uplo := MatrixRegion.Upper,
) -> (
	rcond: R,
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(e) >= n, "E array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= 2 * n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	anorm_val := anorm

	when T == complex64 {
		lapack.csycon_3_(&uplo_c, &n_int, cast(^lapack.complex)raw_data(a.data), &lda, cast(^lapack.complex)raw_data(e), raw_data(ipiv), &anorm_val, &rcond, cast(^lapack.complex)raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zsycon_3_(&uplo_c, &n_int, cast(^lapack.complex16)raw_data(a.data), &lda, cast(^lapack.complex16)raw_data(e), raw_data(ipiv), &anorm_val, &rcond, cast(^lapack.complex16)raw_data(work), &info, 1)
	}

	ok = info == 0
	return rcond, info, ok
}

// Procedure group for symmetric condition estimation with E factor
m_condition_symmetric_e :: proc {
	m_condition_symmetric_e_f32_f64,
	m_condition_symmetric_e_c64_c128,
}


// ============================================================================
// SYMMETRIC MATRIX CONVERSION
// ============================================================================
// Convert between different storage formats for symmetric matrices

// Conversion direction
ConversionWay :: enum {
	CONVERT, // 'C' - Convert from standard to split format
	REVERT, // 'R' - Revert from split format to standard
}

// Helper to convert ConversionWay to char
conversion_way_to_char :: proc(way: ConversionWay) -> u8 {
	switch way {
	case .CONVERT:
		return 'C'
	case .REVERT:
		return 'R'
	}
	return 'C'
}

// Convert symmetric matrix storage format for all types
m_convert_symmetric :: proc(
	a: ^Matrix($T), // Matrix to convert (modified in place)
	ipiv: []Blas_Int, // Pivot indices
	e: []T, // Block diagonal matrix E (size n, output)
	way := ConversionWay.CONVERT,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(e) >= n, "E array too small")

	uplo_c := matrix_region_to_char(uplo)
	way_c := conversion_way_to_char(way)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.ssyconv_(&uplo_c, &way_c, &n_int, raw_data(a.data), &lda, raw_data(ipiv), raw_data(e), &info, 1, 1)
	} else when T == f64 {
		lapack.dsyconv_(&uplo_c, &way_c, &n_int, raw_data(a.data), &lda, raw_data(ipiv), raw_data(e), &info, 1, 1)
	} else when T == complex64 {
		lapack.csyconv_(&uplo_c, &way_c, &n_int, cast(^lapack.complex)raw_data(a.data), &lda, raw_data(ipiv), cast(^lapack.complex)raw_data(e), &info, 1, 1)
	} else when T == complex128 {
		lapack.zsyconv_(&uplo_c, &way_c, &n_int, cast(^lapack.complex16)raw_data(a.data), &lda, raw_data(ipiv), cast(^lapack.complex16)raw_data(e), &info, 1, 1)
	}

	ok = info == 0
	return info, ok
}

// Conversion result
ConversionResult :: struct {
	conversion_complete: bool, // True if conversion was successful
	format_changed:      bool, // True if format actually changed
}


// ============================================================================
// SYMMETRIC MATRIX EQUILIBRATION
// ============================================================================
// Compute scaling factors to equilibrate a symmetric matrix

// Compute scaling factors to equilibrate symmetric matrix for f32/f64
m_equilibrate_symmetric_f32_f64 :: proc(
	a: ^Matrix($T), // Matrix to equilibrate
	s: []T, // Pre-allocated scale factors (size n)
	work: []T, // Pre-allocated workspace (size 3*n)
	uplo := MatrixRegion.Upper,
) -> (
	scond: T,
	amax: T,
	info: Info,
	ok: bool, // Ratio of smallest to largest scale factor// Absolute maximum element
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(s) >= n, "Scale factors array too small")
	assert(len(work) >= 3 * n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.ssyequb_(&uplo_c, &n_int, a.data, &lda, raw_data(s), &scond, &amax, raw_data(work), &info, 1)
	} else when T == f64 {
		lapack.dsyequb_(&uplo_c, &n_int, a.data, &lda, raw_data(s), &scond, &amax, raw_data(work), &info, 1)
	}

	ok = info == 0
	return scond, amax, info, ok
}

// Compute scaling factors to equilibrate symmetric matrix for complex64/complex128
m_equilibrate_symmetric_c64_c128 :: proc(
	a: ^Matrix($T), // Matrix to equilibrate
	s: []$R, // Pre-allocated scale factors (size n)
	work: []T, // Pre-allocated workspace (size 3*n)
	uplo := MatrixRegion.Upper,
) -> (
	scond: R,
	amax: R,
	info: Info,
	ok: bool, // Ratio of smallest to largest scale factor// Absolute maximum element
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(s) >= n, "Scale factors array too small")
	assert(len(work) >= 3 * n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == complex64 {
		lapack.csyequb_(&uplo_c, &n_int, cast(^lapack.complex)a.data, &lda, raw_data(s), &scond, &amax, cast(^lapack.complex)raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zsyequb_(&uplo_c, &n_int, cast(^lapack.doublecomplex)a.data, &lda, raw_data(s), &scond, &amax, cast(^lapack.doublecomplex)raw_data(work), &info, 1)
	}

	ok = info == 0
	return scond, amax, info, ok
}

// Procedure group for symmetric equilibration
m_equilibrate_symmetric :: proc {
	m_equilibrate_symmetric_f32_f64,
	m_equilibrate_symmetric_c64_c128,
}
