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

// Swap rows and corresponding columns in symmetric matrix
m_swap_symmetric_rows :: proc(
	a: ^Matrix($T),
	i1: int, // First index (0-based)
	i2: int, // Second index (0-based)
	uplo := MatrixRegion.Upper,
) -> (
	ok: bool,
) where is_float(T) || is_complex(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(i1 >= 0 && i1 < n, "First index out of range")
	assert(i2 >= 0 && i2 < n, "Second index out of range")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	i1_int := Blas_Int(i1 + 1)
	i2_int := Blas_Int(i2 + 1)

	when T == f32 {
		lapack.ssyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	} else when T == f64 {
		lapack.dsyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	} else when T == complex64 {
		lapack.csyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	} else when T == complex128 {
		lapack.zsyswapr_(&uplo_c, &n_int, a.data, &lda, &i1_int, &i2_int, 1)
	}

	return true
}

// ============================================================================
// SYMMETRIC TRIDIAGONALIZATION
// ============================================================================
// Reduces symmetric matrices to tridiagonal form

// Transformation type for 2-stage algorithms
TransformationType :: enum {
	NO_VECTORS, // 'N' - Do not compute transformation matrix
	VECTORS, // 'V' - Compute transformation matrix
}

// Query workspace for symmetric tridiagonalization
query_workspace_tridiagonalize_symmetric :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where is_float(T) {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrd_(&uplo_c, &n_int, nil, &lda, nil, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	}

	return work_size
}

// Reduce symmetric matrix to tridiagonal form
m_tridiagonalize_symmetric :: proc(
	a: ^Matrix($T), // Symmetric matrix (modified in place)
	d: []T, // Diagonal elements (size n)
	e: []T, // Off-diagonal elements (size n-1)
	tau: []T, // Householder reflector scalars (size n-1)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where is_float(T) {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1, "Tau array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrd_(&uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrd_(&uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// Query workspace for 2-stage symmetric tridiagonalization
query_workspace_tridiagonalize_symmetric_2stage :: proc($T: typeid, n: int, vect := TransformationType.NO_VECTORS, uplo := MatrixRegion.Upper) -> (work_size: int, hous2_size: int) where T == f32 || T == f64 {
	n_int := Blas_Int(n)
	vect_c := transformation_type_to_char(vect)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	lhous2 := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		hous2_query: f32
		lapack.ssytrd_2stage_(&vect_c, &uplo_c, &n_int, nil, &lda, nil, nil, nil, &hous2_query, &lhous2, &work_query, &lwork, &info, 1, 1)
		work_size = int(work_query)
		hous2_size = int(hous2_query)
	} else when T == f64 {
		work_query: f64
		hous2_query: f64
		lapack.dsytrd_2stage_(&vect_c, &uplo_c, &n_int, nil, &lda, nil, nil, nil, &hous2_query, &lhous2, &work_query, &lwork, &info, 1, 1)
		work_size = int(work_query)
		hous2_size = int(hous2_query)
	}

	return work_size, hous2_size
}

// Reduce symmetric matrix to tridiagonal form using 2-stage algorithm
m_tridiagonalize_symmetric_2stage :: proc(
	a: ^Matrix($T), // Symmetric matrix (modified in place)
	d: []T, // Diagonal elements (size n)
	e: []T, // Off-diagonal elements (size n-1)
	tau: []T, // Householder reflector scalars (size n-1)
	hous2: []T, // Additional Householder vectors
	work: []T, // Workspace
	vect := TransformationType.NO_VECTORS,
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1, "Off-diagonal array too small")
	assert(len(tau) >= n - 1, "Tau array too small")
	assert(len(hous2) > 0, "HOUS2 array required")
	assert(len(work) > 0, "Workspace required")

	vect_c := transformation_type_to_char(vect)
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lhous2 := Blas_Int(len(hous2))
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrd_2stage_(&vect_c, &uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(hous2), &lhous2, raw_data(work), &lwork, &info, 1, 1)
	} else when T == f64 {
		lapack.dsytrd_2stage_(&vect_c, &uplo_c, &n_int, a.data, &lda, raw_data(d), raw_data(e), raw_data(tau), raw_data(hous2), &lhous2, raw_data(work), &lwork, &info, 1, 1)
	}

	return info, info == 0
}

// ============================================================================
// SYMMETRIC MATRIX FACTORIZATION (BUNCH-KAUFMAN)
// ============================================================================
// Bunch-Kaufman factorization of symmetric indefinite matrices

// Query workspace for symmetric factorization (Bunch-Kaufman)
query_workspace_factorize_symmetric :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using Bunch-Kaufman pivoting
m_factorize_symmetric :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// AASEN SYMMETRIC FACTORIZATION
// ============================================================================
// Aasen's algorithm for symmetric indefinite matrices

// Query workspace for Aasen symmetric factorization
query_workspace_factorize_symmetric_aasen :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_aa_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using Aasen's algorithm
m_factorize_symmetric_aasen :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_aa_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// 2-STAGE AASEN SYMMETRIC FACTORIZATION
// ============================================================================
// Two-stage Aasen algorithm for improved performance on large matrices

// Query workspace for 2-stage Aasen symmetric factorization
query_workspace_factorize_symmetric_aasen_2stage :: proc(
	$T: typeid,
	n: int,
	nb: int, // Block size for band matrix
	uplo := MatrixRegion.Upper,
) -> (
	work_size: int,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	ltb := Blas_Int(max(1, 4 * n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_aa_2stage_(&uplo_c, &n_int, nil, &lda, nil, &ltb, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using 2-stage Aasen's algorithm
m_factorize_symmetric_aasen_2stage :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	tb: ^Matrix(T), // Band matrix storage (4*n x nb)
	ipiv: []Blas_Int, // First stage pivot indices (size n)
	ipiv2: []Blas_Int, // Second stage pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(tb.rows >= 4 * n, "Band matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	ltb := Blas_Int(tb.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_aa_2stage_(&uplo_c, &n_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// RK (BOUNDED BUNCH-KAUFMAN) FACTORIZATION
// ============================================================================
// Bounded Bunch-Kaufman factorization with additional E factor

// Query workspace for RK (bounded Bunch-Kaufman) symmetric factorization
query_workspace_factorize_symmetric_rk :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n_int := Blas_Int(n)
	uplo_c := matrix_region_to_char(uplo)
	lda := Blas_Int(max(1, n))
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_rk_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using RK (bounded Bunch-Kaufman) algorithm
m_factorize_symmetric_rk :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	e: []T, // E factor from RK factorization (size n)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_rk_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// ROOK PIVOTING FACTORIZATION
// ============================================================================
// Rook pivoting factorization for enhanced numerical stability

// Query workspace for symmetric matrix factorization using rook pivoting
query_workspace_factorize_symmetric_rook :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrf_rook_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Symmetric matrix factorization using rook pivoting algorithm
m_factorize_symmetric_rook :: proc(
	a: ^Matrix($T), // Symmetric matrix (factorized in place)
	ipiv: []Blas_Int, // Pivot indices (size n)
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrf_rook_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// SYMMETRIC MATRIX INVERSION
// ============================================================================
// Inversion of symmetric indefinite matrices using factorizations

// ============================================================================
// STANDARD SYMMETRIC MATRIX INVERSION
// ============================================================================
// Standard inversion using Bunch-Kaufman factorization

// Query workspace for symmetric matrix inversion
query_workspace_invert_symmetric :: proc($T: typeid, n: int) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// SYTRI requires workspace of size n
	work_size = n
	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Symmetric matrix inversion using factorization
m_invert_symmetric :: proc(
	a: ^Matrix($T), // Factorized matrix (inverted in place)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.ssytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	} else when T == f64 {
		lapack.dsytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	} else when T == complex64 {
		lapack.csytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zsytri_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// IMPROVED SYMMETRIC MATRIX INVERSION (SYTRI2)
// ============================================================================
// Improved inversion algorithm with better cache efficiency

// Query workspace for improved symmetric matrix inversion
query_workspace_invert_symmetric_improved :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytri2_(&uplo_c, &n_int, nil, &lda, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Improved symmetric matrix inversion using factorization
m_invert_symmetric_improved :: proc(
	a: ^Matrix($T), // Factorized matrix (inverted in place)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytri2_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// BLOCK-BASED SYMMETRIC MATRIX INVERSION (SYTRI2X)
// ============================================================================
// Block-based inversion algorithm for improved performance

// Query workspace for block-based symmetric matrix inversion
query_workspace_invert_symmetric_blocked :: proc($T: typeid, n: int, nb: int = 64) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// SYTRI2X requires workspace of size nb*(n+nb)
	effective_nb := nb
	if effective_nb <= 0 {
		effective_nb = 64 // Default block size
	}
	work_size = effective_nb * (n + effective_nb)
	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Block-based symmetric matrix inversion using factorization
m_invert_symmetric_blocked :: proc(
	a: ^Matrix($T), // Factorized matrix (inverted in place)
	ipiv: []Blas_Int, // Pivot indices from factorization
	work: []T, // Workspace (size nb*(n+nb))
	uplo := MatrixRegion.Upper,
	nb: int = 64, // Block size
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= nb * (n + nb), "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	nb_int := Blas_Int(nb)

	when T == f32 {
		lapack.ssytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	} else when T == f64 {
		lapack.dsytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	} else when T == complex64 {
		lapack.csytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	} else when T == complex128 {
		lapack.zsytri2x_(&uplo_c, &n_int, a.data, &lda, raw_data(ipiv), raw_data(work), &nb_int, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// RK INVERSION WITH E FACTOR (SYTRI_3)
// ============================================================================
// Inversion using RK factorization with E factor

// Query workspace for RK inversion with E factor
query_workspace_invert_symmetric_rk :: proc($T: typeid, n: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytri_3_(&uplo_c, &n_int, nil, &lda, nil, nil, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// RK inversion with E factor using factorization
m_invert_symmetric_rk :: proc(
	a: ^Matrix($T), // RK factorized matrix (inverted in place)
	e: []T, // E factor from RK factorization
	ipiv: []Blas_Int, // Pivot indices from RK factorization
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Matrix too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytri_3_(&uplo_c, &n_int, a.data, &lda, raw_data(e), raw_data(ipiv), raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
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

// Symmetric system solution using factorization
m_solve_symmetric_sytrs :: proc(
	a: ^Matrix($T), // Factorized matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// IMPROVED SYMMETRIC SYSTEM SOLUTION (SYTRS2)
// ============================================================================
// Improved solution algorithm with workspace usage

// Query workspace for improved symmetric system solution
query_workspace_solve_symmetric_improved :: proc($T: typeid, n: int) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// SYTRS2 requires workspace of size n
	work_size = n
	if work_size < 1 {
		work_size = 1
	}
	return work_size
}

// Improved symmetric system solution using factorization
m_solve_symmetric_improved :: proc(
	a: ^Matrix($T), // Factorized matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: ^Matrix(T), // RHS on input, solution on output
	work: []T, // Workspace (size n)
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(len(work) >= n, "Workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	} else when T == f64 {
		lapack.dsytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	} else when T == complex64 {
		lapack.csytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	} else when T == complex128 {
		lapack.zsytrs2_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// RK SYSTEM SOLUTION WITH E FACTOR (SYTRS_3)
// ============================================================================
// System solution using RK factorization with E factor

// RK system solution with E factor using factorization
m_solve_symmetric_rk_sytrs3 :: proc(
	a: ^Matrix($T), // RK factorized matrix
	e: []T, // E factor from RK factorization
	ipiv: []Blas_Int, // Pivot indices from RK factorization
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(len(e) >= n, "E vector too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n, "Matrix B too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_3_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(e), raw_data(ipiv), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// AASEN SYSTEM SOLUTION (SYTRS_AA)
// ============================================================================
// System solution using Aasen factorization

// Query workspace for Aasen system solution
query_workspace_solve_symmetric_aasen_sytrs_aa :: proc($T: typeid, n: int, nrhs: int, uplo := MatrixRegion.Upper) -> (work_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(n)
	ldb := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.ssytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dsytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.csytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.zsytrs_aa_(&uplo_c, &n_int, &nrhs_int, nil, &lda, nil, nil, &ldb, &work_query, &lwork, &info, 1)
		work_size = int(real(work_query))
	}

	return work_size
}

// Aasen system solution using factorization
m_solve_symmetric_aasen_sytrs_aa :: proc(
	a: ^Matrix($T), // Aasen factorized matrix
	ipiv: []Blas_Int, // Pivot indices from Aasen factorization
	b: ^Matrix(T), // RHS on input, solution on output
	work: []T, // Workspace
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(len(work) > 0, "Workspace required")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))

	when T == f32 {
		lapack.ssytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_aa_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, raw_data(work), &lwork, &info, 1)
	}

	return info, info == 0
}

// ============================================================================
// 2-STAGE AASEN SYSTEM SOLUTION (SYTRS_AA_2STAGE)
// ============================================================================
// System solution using 2-stage Aasen factorization

// 2-stage Aasen system solution using factorization
m_solve_symmetric_aasen_2stage_sytrs_aa_2stage :: proc(
	a: ^Matrix($T), // 2-stage Aasen factorized matrix
	tb: ^Matrix(T), // Band matrix from 2-stage factorization
	ipiv: []Blas_Int, // First stage pivot indices
	ipiv2: []Blas_Int, // Second stage pivot indices
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(tb.rows >= 4 * n, "TB matrix too small")
	assert(len(ipiv) >= n, "First pivot array too small")
	assert(len(ipiv2) >= n, "Second pivot array too small")
	assert(b.rows >= n, "Matrix B too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ltb := Blas_Int(tb.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_aa_2stage_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, tb.data, &ltb, raw_data(ipiv), raw_data(ipiv2), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// ROOK PIVOTING SYSTEM SOLUTION (SYTRS_ROOK)
// ============================================================================
// System solution using rook pivoting factorization

// Rook pivoting system solution using factorization
m_solve_symmetric_rook_sytrs :: proc(
	a: ^Matrix($T), // Rook factorized matrix
	ipiv: []Blas_Int, // Pivot indices from rook factorization
	b: ^Matrix(T), // RHS on input, solution on output
	uplo := MatrixRegion.Upper,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	nrhs := b.cols
	assert(a.rows >= n, "Matrix A too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n, "Matrix B too small")

	uplo_c := matrix_region_to_char(uplo)
	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.ssytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == f64 {
		lapack.dsytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex64 {
		lapack.csytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	} else when T == complex128 {
		lapack.zsytrs_rook_(&uplo_c, &n_int, &nrhs_int, a.data, &lda, raw_data(ipiv), b.data, &ldb, &info, 1)
	}

	return info, info == 0
}


// ============================================================================
// TRIANGULAR BAND MATRIX CONDITION NUMBER
// ============================================================================
// Condition number estimation for triangular band matrices

// Diagonal type for triangular matrices
DiagonalType :: enum {
	NON_UNIT, // 'N' - non-unit diagonal
	UNIT, // 'U' - unit diagonal
}

// Convert diagonal type to character
diagonal_type_to_char :: proc(diag: DiagonalType) -> u8 {
	switch diag {
	case .NON_UNIT:
		return 'N'
	case .UNIT:
		return 'U'
	}
	return 'N'
}

// Helper function to convert TransposeMode to char
transpose_mode_to_char :: proc(t: TransposeMode) -> u8 {
	switch t {
	case .None:
		return 'N'
	case .Transpose:
		return 'T'
	case .ConjugateTranspose:
		return 'C'
	}
	return 'N'
}

// Helper function to convert RFPTranspose to char
rfp_transpose_to_char :: proc(trans: RFPTranspose) -> u8 {
	switch trans {
	case .NORMAL:
		return 'N'
	case .TRANSPOSE:
		return 'T'
	case .CONJUGATE:
		return 'C'
	}
	return 'N'
}

// Query workspace for triangular band condition number - real types
query_workspace_condition_triangular_band_real :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Real types need: work = 3*n, iwork = n
	work_size = 3 * n
	iwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if iwork_size < 1 {
		iwork_size = 1
	}
	return
}

// Query workspace for triangular band condition number - complex types
query_workspace_condition_triangular_band_complex :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) where T == complex64 || T == complex128 {
	// Complex types need: work = 2*n, rwork = n
	work_size = 2 * n
	rwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if rwork_size < 1 {
		rwork_size = 1
	}
	return
}

// Triangular band condition number for real types
m_condition_triangular_band_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix in packed storage
	work: []T, // Workspace (size 3*n)
	iwork: []Blas_Int, // Integer workspace (size n)
	norm := MatrixNorm.OneNorm,
	uplo := MatrixRegion.Upper,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	rcond: T,
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	norm_c: c.char = norm == .ONE_NORM ? '1' : 'I'
	uplo_c := matrix_region_to_char(uplo)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)

	when T == f32 {
		lapack.stbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	}

	return rcond, info, info == 0
}

// Triangular band condition number for complex types
m_condition_triangular_band_c64_c128 :: proc(
	ab: ^Matrix($T), // Band matrix in packed storage
	work: []T, // Complex workspace (size 2*n)
	rwork: []$R, // Real workspace (size n)
	norm := MatrixNorm.OneNorm,
	uplo := MatrixRegion.Upper,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	rcond: R,
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := ab.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	norm_c: c.char = norm == .ONE_NORM ? '1' : 'I'
	uplo_c := matrix_region_to_char(uplo)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.ld)

	when T == complex64 {
		lapack.ctbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztbcon_(&norm_c, &uplo_c, &diag_c, &n_int, &kd_int, ab.data, &ldab, &rcond, raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	}

	return rcond, info, info == 0
}

// Procedure group for triangular band condition number
m_condition_triangular_band :: proc {
	m_condition_triangular_band_f32_f64,
	m_condition_triangular_band_c64_c128,
}

// ============================================================================
// TRIANGULAR BAND ERROR BOUNDS AND REFINEMENT
// ============================================================================
// Error bounds and iterative refinement for triangular band systems

// Query workspace for triangular band error bounds - real types
query_workspace_refine_triangular_band_real :: proc($T: typeid, n: int) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 {
	// Real types need: work = 3*n, iwork = n
	work_size = 3 * n
	iwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if iwork_size < 1 {
		iwork_size = 1
	}
	return
}

// Query workspace for triangular band error bounds - complex types
query_workspace_refine_triangular_band_complex :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) where T == complex64 || T == complex128 {
	// Complex types need: work = 2*n, rwork = n
	work_size = 2 * n
	rwork_size = n
	if work_size < 1 {
		work_size = 1
	}
	if rwork_size < 1 {
		rwork_size = 1
	}
	return
}

// Query result sizes for triangular band error bounds
query_result_sizes_refine_triangular_band :: proc(nrhs: int) -> (ferr_size: int, berr_size: int) {
	return nrhs, nrhs
}

// Triangular band error bounds for real types
m_refine_triangular_band_f32_f64 :: proc(
	ab: ^Matrix($T), // Band matrix
	b: ^Matrix(T), // Original RHS
	x: ^Matrix(T), // Current solution
	ferr: []T, // Forward error bounds (size nrhs)
	berr: []T, // Backward error bounds (size nrhs)
	work: []T, // Workspace (size 3*n)
	iwork: []Blas_Int, // Integer workspace (size n)
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := ab.cols
	nrhs := b.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 3 * n, "Workspace too small")
	assert(len(iwork) >= n, "Integer workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.ld)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == f32 {
		lapack.stbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(iwork), &info, 1, 1, 1)
	}

	return info, info == 0
}

// Triangular band error bounds for complex types
m_refine_triangular_band_c64_c128 :: proc(
	ab: ^Matrix($T), // Band matrix
	b: ^Matrix(T), // Original RHS
	x: ^Matrix(T), // Current solution
	ferr: []$R, // Forward error bounds (size nrhs)
	berr: []R, // Backward error bounds (size nrhs)
	work: []T, // Complex workspace (size 2*n)
	rwork: []R, // Real workspace (size n)
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
	kd: int = 0, // Number of super/sub-diagonals
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := ab.cols
	nrhs := b.cols
	assert(ab.rows >= kd + 1, "Band matrix too small")
	assert(b.rows >= n, "Matrix B too small")
	assert(x.rows >= n && x.cols >= nrhs, "Matrix X too small")
	assert(len(ferr) >= nrhs, "Forward error array too small")
	assert(len(berr) >= nrhs, "Backward error array too small")
	assert(len(work) >= 2 * n, "Workspace too small")
	assert(len(rwork) >= n, "Real workspace too small")

	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c: c.char = diag == .UNIT ? 'U' : 'N'
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.ld)
	ldb := Blas_Int(b.ld)
	ldx := Blas_Int(x.ld)

	when T == complex64 {
		lapack.ctbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztbrfs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, x.data, &ldx, raw_data(ferr), raw_data(berr), raw_data(work), raw_data(rwork), &info, 1, 1, 1)
	}

	return info, info == 0
}

// Procedure group for triangular band error bounds
m_refine_triangular_band :: proc {
	m_refine_triangular_band_f32_f64,
	m_refine_triangular_band_c64_c128,
}


// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Swap rows and columns in symmetric matrix
swap_symmetric_rows_columns :: proc(a: ^Matrix($T), i1: int, i2: int, uplo := MatrixRegion.Upper) -> bool {
	return m_swap_symmetric_rows(a, i1, i2, uplo)
}


// ==============================================================================
// Triangular Band System Solution Functions
// ==============================================================================

// ============================================================================
// TRIANGULAR BAND SYSTEM SOLUTION
// ============================================================================
// Solves triangular band systems of equations

// Solve triangular band system
// Solves A * X = B or A^T * X = B where A is triangular band matrix
m_solve_triangular_band :: proc(
	ab: ^Matrix($T), // Triangular band matrix
	b: ^Matrix(T), // Right-hand side matrix (solution on output)
	kd: int, // Number of super/sub-diagonals
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := ab.cols
	nrhs := b.cols
	assert(ab.rows >= kd + 1, "Band matrix storage too small")
	assert(b.rows >= n, "B matrix too small")

	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c := diagonal_type_to_char(diag)
	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	nrhs_int := Blas_Int(nrhs)
	ldab := Blas_Int(ab.ld)
	ldb := Blas_Int(b.ld)

	when T == f32 {
		lapack.stbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	} else when T == complex64 {
		lapack.ctbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztbtrs_(&uplo_c, &trans_c, &diag_c, &n_int, &kd_int, &nrhs_int, ab.data, &ldab, b.data, &ldb, &info, 1, 1, 1)
	}

	return info, info == 0
}


// ==============================================================================
// Triangular Solve with RFP Format Functions
// ==============================================================================

// ============================================================================
// RFP (RECTANGULAR FULL PACKED) TRIANGULAR SOLVE
// ============================================================================
// Solves triangular systems with matrices stored in RFP format
// RFP format stores triangular/symmetric matrices in a compact rectangular array

// Solve triangular system with RFP format matrix
// Solves: op(A) * X = alpha * B  (side = Left)
//     or: X * op(A) = alpha * B  (side = Right)
// where A is stored in RFP format
m_solve_triangular_rfp :: proc(
	a_rfp: []$T, // Triangular matrix in RFP format
	b: ^Matrix(T), // Right-hand side matrix (solution on output)
	alpha: T, // Scalar multiplier
	side := OrthogonalSide.Left,
	uplo := MatrixRegion.Upper,
	trans := TransposeMode.None,
	diag := DiagonalType.NON_UNIT,
	transr := RFPTranspose.NORMAL,
) -> (
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	m := b.rows
	n := b.cols

	// Determine the size of A based on side
	a_size := side == .Left ? m : n
	assert(len(a_rfp) >= a_size * (a_size + 1) / 2, "RFP array too small")
	assert(b.rows >= m && b.cols >= n, "B matrix too small")

	side_c := side == .Left ? u8('L') : u8('R')
	uplo_c := matrix_region_to_char(uplo)
	trans_c := transpose_mode_to_char(trans)
	diag_c := diagonal_type_to_char(diag)
	transr_c := rfp_transpose_to_char(transr)

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	ldb := Blas_Int(b.ld)
	alpha_copy := alpha

	when T == f32 {
		lapack.stfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	} else when T == f64 {
		lapack.dtfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	} else when T == complex64 {
		lapack.ctfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztfsm_(&transr_c, &side_c, &uplo_c, &trans_c, &diag_c, &m_int, &n_int, &alpha_copy, raw_data(a_rfp), b.data, &ldb, 1, 1, 1, 1, 1)
	}

	return true
}


// ============================================================================
// RFP (RECTANGULAR FULL PACKED) TRIANGULAR INVERSION
// ============================================================================
// Inverts triangular matrices stored in RFP format

// Invert triangular matrix in RFP format
// Computes the inverse of a triangular matrix A stored in RFP format
m_invert_triangular_rfp :: proc(
	a_rfp: []$T, // Triangular matrix in RFP format (inverted in place)
	n: int, // Order of the matrix
	uplo := MatrixRegion.Upper,
	diag := DiagonalType.NON_UNIT,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Check that RFP array has correct size
	rfp_size := n * (n + 1) / 2
	assert(len(a_rfp) >= rfp_size, "RFP array too small")

	uplo_c := matrix_region_to_char(uplo)
	diag_c := diagonal_type_to_char(diag)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.stftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	} else when T == f64 {
		lapack.dtftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	} else when T == complex64 {
		lapack.ctftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	} else when T == complex128 {
		lapack.ztftri_(&transr_c, &uplo_c, &diag_c, &n_int, raw_data(a_rfp), &info, 1, 1, 1)
	}

	return info, info == 0
}

// ============================================================================
// RFP TO PACKED FORMAT CONVERSION
// ============================================================================
// Converts matrices between RFP and packed storage formats

// Convert RFP format to packed format
// Converts a triangular/symmetric matrix from RFP format to packed storage
m_convert_rfp_to_packed :: proc(
	a_rfp: []$T, // Matrix in RFP format
	a_packed: []T, // Output matrix in packed format
	n: int, // Order of the matrix
	uplo := MatrixRegion.Upper,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Check that arrays have correct sizes
	rfp_size := n * (n + 1) / 2
	packed_size := n * (n + 1) / 2
	assert(len(a_rfp) >= rfp_size, "RFP array too small")
	assert(len(a_packed) >= packed_size, "Packed array too small")

	uplo_c := matrix_region_to_char(uplo)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.stfttp_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), raw_data(a_packed), &info, 1, 1)
	} else when T == f64 {
		lapack.dtfttp_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), raw_data(a_packed), &info, 1, 1)
	} else when T == complex64 {
		lapack.ctfttp_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), raw_data(a_packed), &info, 1, 1)
	} else when T == complex128 {
		lapack.ztfttp_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), raw_data(a_packed), &info, 1, 1)
	}

	return info, info == 0
}

// Convert packed format to RFP format
// Converts a triangular/symmetric matrix from packed storage to RFP format
m_convert_packed_to_rfp :: proc(
	a_packed: []$T, // Matrix in packed format
	a_rfp: []T, // Output matrix in RFP format
	n: int, // Order of the matrix
	uplo := MatrixRegion.Upper,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Check that arrays have correct sizes
	rfp_size := n * (n + 1) / 2
	packed_size := n * (n + 1) / 2
	assert(len(a_packed) >= packed_size, "Packed array too small")
	assert(len(a_rfp) >= rfp_size, "RFP array too small")

	uplo_c := matrix_region_to_char(uplo)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)

	when T == f32 {
		lapack.stpttf_(&transr_c, &uplo_c, &n_int, raw_data(a_packed), raw_data(a_rfp), &info, 1, 1)
	} else when T == f64 {
		lapack.dtpttf_(&transr_c, &uplo_c, &n_int, raw_data(a_packed), raw_data(a_rfp), &info, 1, 1)
	} else when T == complex64 {
		lapack.ctpttf_(&transr_c, &uplo_c, &n_int, raw_data(a_packed), raw_data(a_rfp), &info, 1, 1)
	} else when T == complex128 {
		lapack.ztpttf_(&transr_c, &uplo_c, &n_int, raw_data(a_packed), raw_data(a_rfp), &info, 1, 1)
	}

	return info, info == 0
}


// ============================================================================
// RFP (RECTANGULAR FULL PACKED) FORMAT CONVERSION
// ============================================================================
// Converts between RFP and standard full matrix formats

// Convert RFP format to full matrix format
// Converts a triangular/symmetric matrix from RFP format to standard full storage
m_convert_rfp_to_full :: proc(
	a_rfp: []$T, // Matrix in RFP format
	a: ^Matrix(T), // Output full matrix
	uplo := MatrixRegion.Upper,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Output matrix too small")

	// Check that RFP array has correct size
	rfp_size := n * (n + 1) / 2
	assert(len(a_rfp) >= rfp_size, "RFP array too small")

	uplo_c := matrix_region_to_char(uplo)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.stfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	} else when T == f64 {
		lapack.dtfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	} else when T == complex64 {
		lapack.ctfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	} else when T == complex128 {
		lapack.ztfttr_(&transr_c, &uplo_c, &n_int, raw_data(a_rfp), a.data, &lda, &info, 1, 1)
	}

	return info, info == 0
}

// Convert full matrix format to RFP format
// Converts a triangular/symmetric matrix from standard full storage to RFP format
m_convert_full_to_rfp :: proc(
	a: ^Matrix($T), // Full matrix
	a_rfp: []T, // Output matrix in RFP format
	uplo := MatrixRegion.Upper,
	transr := RFPTranspose.NORMAL,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n, "Input matrix too small")

	// Check that RFP array has correct size
	rfp_size := n * (n + 1) / 2
	assert(len(a_rfp) >= rfp_size, "RFP array too small")

	uplo_c := matrix_region_to_char(uplo)
	transr_c := rfp_transpose_to_char(transr)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)

	when T == f32 {
		lapack.strttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	} else when T == f64 {
		lapack.dtrttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	} else when T == complex64 {
		lapack.ctrttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	} else when T == complex128 {
		lapack.ztrttf_(&transr_c, &uplo_c, &n_int, a.data, &lda, raw_data(a_rfp), &info, 1, 1)
	}

	return info, info == 0
}


// ============================================================================
// GENERALIZED EIGENVECTOR COMPUTATION
// ============================================================================
// Computes eigenvectors of a generalized eigenvalue problem


// Query workspace for generalized eigenvector computation - complex types
query_workspace_compute_generalized_eigenvectors_complex :: proc($T: typeid, n: int) -> (work_size: int, rwork_size: int) where T == complex64 || T == complex128 {
	// Complex types need: work = 2*n, rwork = 2*n
	work_size = 2 * n
	rwork_size = 2 * n
	return
}

// Query workspace for generalized eigenvector computation - real types
query_workspace_compute_generalized_eigenvectors_real :: proc($T: typeid, n: int) -> (work_size: int) where T == f32 || T == f64 {
	// Real types need: work = 6*n
	work_size = 6 * n
	return
}

// Compute generalized eigenvectors for real types (f32/f64)
m_compute_generalized_eigenvectors_real :: proc(
	s: ^Matrix($T), // Real Schur form of matrix A from QZ decomposition
	p: ^Matrix(T), // Real Schur form of matrix B from QZ decomposition
	vl: ^Matrix(T) = nil, // Left eigenvectors (optional)
	vr: ^Matrix(T) = nil, // Right eigenvectors (optional)
	select_mask: []Blas_Int = nil, // Selection array (for selected eigenvectors)
	work: []T, // Workspace
	side := EigenvectorSide.Both,
	selection := EigenvectorSelection.All,
) -> (
	m_selected: Blas_Int,
	info: Info,
	ok: bool, // Number of eigenvectors computed
) where T == f32 || T == f64 {
	n := s.cols
	assert(s.rows >= n && p.rows >= n && p.cols >= n, "Matrices too small")
	assert(len(work) >= 6 * n, "Insufficient workspace")

	side_c := eigenvector_side_to_char(side)
	howmny_c := eigenvector_selection_to_char(selection)
	n_int := Blas_Int(n)
	lds := Blas_Int(s.ld)
	ldp := Blas_Int(p.ld)

	// Handle eigenvector matrices
	ldvl: Blas_Int = 1
	vl_ptr: rawptr = nil
	if (side == .Left || side == .Both) && vl != nil {
		assert(vl.rows >= n && vl.cols >= n, "Left eigenvector matrix too small")
		ldvl = Blas_Int(vl.ld)
		vl_ptr = raw_data(vl.data)
	}

	ldvr: Blas_Int = 1
	vr_ptr: rawptr = nil
	if (side == .Right || side == .Both) && vr != nil {
		assert(vr.rows >= n && vr.cols >= n, "Right eigenvector matrix too small")
		ldvr = Blas_Int(vr.ld)
		vr_ptr = raw_data(vr.data)
	}

	// Maximum number of eigenvectors
	mm := Blas_Int(n)

	when T == f32 {
		lapack.stgevc_(&side_c, &howmny_c, raw_data(select_mask) if select_mask != nil else nil, &n_int, s.data, &lds, p.data, &ldp, vl_ptr, &ldvl, vr_ptr, &ldvr, &mm, &m_selected, raw_data(work), &info, 1, 1)
	} else when T == f64 {
		lapack.dtgevc_(&side_c, &howmny_c, raw_data(select_mask) if select_mask != nil else nil, &n_int, s.data, &lds, p.data, &ldp, vl_ptr, &ldvl, vr_ptr, &ldvr, &mm, &m_selected, raw_data(work), &info, 1, 1)
	}

	return m_selected, info, info == 0
}

// Compute generalized eigenvectors for complex types (complex64/complex128)
m_compute_generalized_eigenvectors_complex :: proc(
	s: ^Matrix($T), // Upper triangular matrix from QZ decomposition
	p: ^Matrix(T), // Upper triangular matrix from QZ decomposition
	vl: ^Matrix(T) = nil, // Left eigenvectors (optional)
	vr: ^Matrix(T) = nil, // Right eigenvectors (optional)
	select_mask: []Blas_Int = nil, // Selection array (for selected eigenvectors)
	work: []T, // Complex workspace
	rwork: []$R, // Real workspace
	side := EigenvectorSide.Both,
	selection := EigenvectorSelection.All,
) -> (
	m_selected: Blas_Int,
	info: Info,
	ok: bool, // Number of eigenvectors computed
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := s.cols
	assert(s.rows >= n && p.rows >= n && p.cols >= n, "Matrices too small")
	assert(len(work) >= 2 * n, "Insufficient complex workspace")
	assert(len(rwork) >= 2 * n, "Insufficient real workspace")

	side_c := eigenvector_side_to_char(side)
	howmny_c := eigenvector_selection_to_char(selection)
	n_int := Blas_Int(n)
	lds := Blas_Int(s.ld)
	ldp := Blas_Int(p.ld)

	// Handle eigenvector matrices
	ldvl: Blas_Int = 1
	vl_ptr: rawptr = nil
	if (side == .Left || side == .Both) && vl != nil {
		assert(vl.rows >= n && vl.cols >= n, "Left eigenvector matrix too small")
		ldvl = Blas_Int(vl.ld)
		vl_ptr = raw_data(vl.data)
	}

	ldvr: Blas_Int = 1
	vr_ptr: rawptr = nil
	if (side == .Right || side == .Both) && vr != nil {
		assert(vr.rows >= n && vr.cols >= n, "Right eigenvector matrix too small")
		ldvr = Blas_Int(vr.ld)
		vr_ptr = raw_data(vr.data)
	}

	// Maximum number of eigenvectors
	mm := Blas_Int(n)

	when T == complex64 {
		lapack.ctgevc_(
			&side_c,
			&howmny_c,
			raw_data(select_mask) if select_mask != nil else nil,
			&n_int,
			s.data,
			&lds,
			p.data,
			&ldp,
			vl_ptr,
			&ldvl,
			vr_ptr,
			&ldvr,
			&mm,
			&m_selected,
			raw_data(work),
			raw_data(rwork),
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.ztgevc_(
			&side_c,
			&howmny_c,
			raw_data(select_mask) if select_mask != nil else nil,
			&n_int,
			s.data,
			&lds,
			p.data,
			&ldp,
			vl_ptr,
			&ldvl,
			vr_ptr,
			&ldvr,
			&mm,
			&m_selected,
			raw_data(work),
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	return m_selected, info, info == 0
}

// Procedure group for generalized eigenvector computation
m_compute_generalized_eigenvectors :: proc {
	m_compute_generalized_eigenvectors_real,
	m_compute_generalized_eigenvectors_complex,
}


// ============================================================================
// GENERALIZED SCHUR FORM REORDERING
// ============================================================================
// Reorders eigenvalues in the generalized Schur form

// Query workspace for generalized Schur reordering - real types
query_workspace_reorder_generalized_schur_real :: proc($T: typeid, n: int) -> (work_size: int) where T == f32 || T == f64 {
	// Query LAPACK for optimal workspace size
	wantq := Blas_Int(1) // Update Q
	wantz := Blas_Int(1) // Update Z
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	ldq := Blas_Int(max(1, n))
	ldz := Blas_Int(max(1, n))
	ifst := Blas_Int(1)
	ilst := Blas_Int(n)
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.stgexc_(&wantq, &wantz, &n_int, nil, &lda, nil, &ldb, nil, &ldq, nil, &ldz, &ifst, &ilst, &work_query, &lwork, &info)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dtgexc_(&wantq, &wantz, &n_int, nil, &lda, nil, &ldb, nil, &ldq, nil, &ldz, &ifst, &ilst, &work_query, &lwork, &info)
		work_size = int(work_query)
	}

	if work_size < 1 {
		work_size = 4 * n + 16 // Minimum workspace
	}
	return work_size
}

// Reorder generalized Schur form for complex types (complex64/complex128)
m_reorder_generalized_schur_complex :: proc(
	a: ^Matrix($T), // Generalized Schur matrix A (modified)
	b: ^Matrix(T), // Generalized Schur matrix B (modified)
	q: ^Matrix(T) = nil, // Orthogonal/unitary matrix Q (optional, updated)
	z: ^Matrix(T) = nil, // Orthogonal/unitary matrix Z (optional, updated)
	ifst: int, // Initial position of eigenvalue to move (1-based)
	ilst: int, // Target position (1-based)
	update_q := true,
	update_z := true,
) -> (
	info: Info,
	ok: bool,
) where T == complex64 || T == complex128 {
	n := a.cols
	assert(a.rows >= n && b.rows >= n && b.cols >= n, "Matrices too small")
	assert(ifst >= 1 && ifst <= n, "ifst out of range")
	assert(ilst >= 1 && ilst <= n, "ilst out of range")

	wantq := Blas_Int(update_q ? 1 : 0)
	wantz := Blas_Int(update_z ? 1 : 0)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)

	// Handle Q and Z matrices
	ldq: Blas_Int = 1
	q_ptr: rawptr = nil
	if update_q && q != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.ld)
		q_ptr = raw_data(q.data)
	}

	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if update_z && z != nil {
		assert(z.rows >= n && z.cols >= n, "Z matrix too small")
		ldz = Blas_Int(z.ld)
		z_ptr = raw_data(z.data)
	}

	ifst_int := Blas_Int(ifst)
	ilst_int := Blas_Int(ilst)

	when T == complex64 {
		lapack.ctgexc_(&wantq, &wantz, &n_int, a.data, &lda, b.data, &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_int, &ilst_int, &info)
	} else when T == complex128 {
		lapack.ztgexc_(&wantq, &wantz, &n_int, a.data, &lda, b.data, &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_int, &ilst_int, &info)
	}

	return info, info == 0
}

// Reorder generalized Schur form for real types (f32/f64)
m_reorder_generalized_schur_real :: proc(
	a: ^Matrix($T), // Generalized Schur matrix A (modified)
	b: ^Matrix(T), // Generalized Schur matrix B (modified)
	q: ^Matrix(T) = nil, // Orthogonal matrix Q (optional, updated)
	z: ^Matrix(T) = nil, // Orthogonal matrix Z (optional, updated)
	work: []T, // Workspace
	ifst: int, // Initial position of eigenvalue to move (1-based)
	ilst: int, // Target position (1-based)
	update_q := true,
	update_z := true,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n && b.rows >= n && b.cols >= n, "Matrices too small")
	assert(ifst >= 1 && ifst <= n, "ifst out of range")
	assert(ilst >= 1 && ilst <= n, "ilst out of range")
	assert(len(work) > 0, "Workspace required")

	wantq := Blas_Int(update_q ? 1 : 0)
	wantz := Blas_Int(update_z ? 1 : 0)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))

	// Handle Q and Z matrices
	ldq: Blas_Int = 1
	q_ptr: rawptr = nil
	if update_q && q != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.ld)
		q_ptr = raw_data(q.data)
	}

	ldz: Blas_Int = 1
	z_ptr: rawptr = nil
	if update_z && z != nil {
		assert(z.rows >= n && z.cols >= n, "Z matrix too small")
		ldz = Blas_Int(z.ld)
		z_ptr = raw_data(z.data)
	}

	ifst_int := Blas_Int(ifst)
	ilst_int := Blas_Int(ilst)

	when T == f32 {
		lapack.stgexc_(&wantq, &wantz, &n_int, a.data, &lda, b.data, &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_int, &ilst_int, raw_data(work), &lwork, &info)
	} else when T == f64 {
		lapack.dtgexc_(&wantq, &wantz, &n_int, a.data, &lda, b.data, &ldb, q_ptr, &ldq, z_ptr, &ldz, &ifst_int, &ilst_int, raw_data(work), &lwork, &info)
	}

	return info, info == 0
}

// Procedure group for generalized Schur reordering
m_reorder_generalized_schur :: proc {
	m_reorder_generalized_schur_complex,
	m_reorder_generalized_schur_real,
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

// ============================================================================
// GENERALIZED EIGENVALUE SENSITIVITY ANALYSIS
// ============================================================================
// Estimates condition numbers for eigenvalues and eigenvectors of generalized eigenvalue problems

// Helper function to convert sensitivity job to char
sensitivity_job_to_char :: proc(job: SensitivityJob) -> u8 {
	switch job {
	case .EigenvaluesOnly:
		return 'E'
	case .SubspacesOnly:
		return 'V'
	case .Both:
		return 'B'
	}
	return 'B'
}

// Query workspace for generalized eigenvalue sensitivity - all types
query_workspace_compute_generalized_sensitivity :: proc($T: typeid, n: int, job: SensitivityJob) -> (work_size: int, iwork_size: int) where T == f32 || T == f64 || T == complex64 || T == complex128 {
	// Query LAPACK for optimal workspace size
	job_c := sensitivity_job_to_char(job)
	howmny_c := u8('A') // All eigenvalues
	n_int := Blas_Int(n)
	lda := Blas_Int(max(1, n))
	ldb := Blas_Int(max(1, n))
	ldvl := Blas_Int(max(1, n))
	ldvr := Blas_Int(max(1, n))
	mm := Blas_Int(n)
	m: Blas_Int
	lwork := Blas_Int(QUERY_WORKSPACE)
	info: Info

	when T == f32 {
		work_query: f32
		lapack.stgsna_(&job_c, &howmny_c, nil, &n_int, nil, &lda, nil, &ldb, nil, &ldvl, nil, &ldvr, nil, nil, &mm, &m, &work_query, &lwork, nil, &info, 1, 1)
		work_size = int(work_query)
	} else when T == f64 {
		work_query: f64
		lapack.dtgsna_(&job_c, &howmny_c, nil, &n_int, nil, &lda, nil, &ldb, nil, &ldvl, nil, &ldvr, nil, nil, &mm, &m, &work_query, &lwork, nil, &info, 1, 1)
		work_size = int(work_query)
	} else when T == complex64 {
		work_query: complex64
		lapack.ctgsna_(&job_c, &howmny_c, nil, &n_int, nil, &lda, nil, &ldb, nil, &ldvl, nil, &ldvr, nil, nil, &mm, &m, &work_query, &lwork, nil, &info, 1, 1)
		work_size = int(real(work_query))
	} else when T == complex128 {
		work_query: complex128
		lapack.ztgsna_(&job_c, &howmny_c, nil, &n_int, nil, &lda, nil, &ldb, nil, &ldvl, nil, &ldvr, nil, nil, &mm, &m, &work_query, &lwork, nil, &info, 1, 1)
		work_size = int(real(work_query))
	}

	// Integer workspace
	if job == .Both || job == .SubspacesOnly {
		iwork_size = n + 2
	} else {
		iwork_size = 1
	}

	return work_size, iwork_size
}

// Compute generalized eigenvalue sensitivity for real types (f32/f64)
m_compute_generalized_sensitivity_real :: proc(
	a: ^Matrix($T), // Upper quasi-triangular matrix from generalized Schur decomposition
	b: ^Matrix(T), // Upper triangular matrix from generalized Schur decomposition
	vl: ^Matrix(T) = nil, // Left eigenvectors (optional)
	vr: ^Matrix(T) = nil, // Right eigenvectors (optional)
	s: []T = nil, // Reciprocal condition numbers for eigenvalues (optional)
	dif: []T = nil, // Condition numbers for deflating subspaces (optional)
	select_mask: []Blas_Int = nil, // Selection array (for selected eigenvalues)
	work: []T, // Workspace
	iwork: []Blas_Int = nil, // Integer workspace (needed if job != EigenvaluesOnly)
	job := SensitivityJob.Both,
	selection := EigenvectorSelection.All,
) -> (
	m_computed: Blas_Int,
	info: Info,
	ok: bool, // Number of eigenvalues/vectors for which condition numbers computed
) where T == f32 || T == f64 {
	n := a.cols
	assert(a.rows >= n && b.rows >= n && b.cols >= n, "Matrices too small")
	assert(len(work) > 0, "Workspace required")
	if job != .EigenvaluesOnly {
		assert(len(iwork) >= n + 2, "Integer workspace required for subspace condition numbers")
	}

	job_c := sensitivity_job_to_char(job)
	howmny_c := eigenvector_selection_to_char(selection)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))

	// Handle eigenvector matrices
	ldvl: Blas_Int = 1
	vl_ptr: rawptr = nil
	if vl != nil {
		assert(vl.rows >= n && vl.cols >= n, "Left eigenvector matrix too small")
		ldvl = Blas_Int(vl.ld)
		vl_ptr = raw_data(vl.data)
	}

	ldvr: Blas_Int = 1
	vr_ptr: rawptr = nil
	if vr != nil {
		assert(vr.rows >= n && vr.cols >= n, "Right eigenvector matrix too small")
		ldvr = Blas_Int(vr.ld)
		vr_ptr = raw_data(vr.data)
	}

	// Maximum number to compute
	mm := Blas_Int(n)

	when T == f32 {
		lapack.stgsna_(
			&job_c,
			&howmny_c,
			raw_data(select_mask) if select_mask != nil else nil,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			vl_ptr,
			&ldvl,
			vr_ptr,
			&ldvr,
			raw_data(s) if s != nil else nil,
			raw_data(dif) if dif != nil else nil,
			&mm,
			&m_computed,
			raw_data(work),
			&lwork,
			raw_data(iwork) if iwork != nil else nil,
			&info,
			1,
			1,
		)
	} else when T == f64 {
		lapack.dtgsna_(
			&job_c,
			&howmny_c,
			raw_data(select_mask) if select_mask != nil else nil,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			vl_ptr,
			&ldvl,
			vr_ptr,
			&ldvr,
			raw_data(s) if s != nil else nil,
			raw_data(dif) if dif != nil else nil,
			&mm,
			&m_computed,
			raw_data(work),
			&lwork,
			raw_data(iwork) if iwork != nil else nil,
			&info,
			1,
			1,
		)
	}

	return m_computed, info, info == 0
}

// Compute generalized eigenvalue sensitivity for complex types (complex64/complex128)
m_compute_generalized_sensitivity_complex :: proc(
	a: ^Matrix($T), // Upper triangular matrix from generalized Schur decomposition
	b: ^Matrix(T), // Upper triangular matrix from generalized Schur decomposition
	vl: ^Matrix(T) = nil, // Left eigenvectors (optional)
	vr: ^Matrix(T) = nil, // Right eigenvectors (optional)
	s: []$R = nil, // Reciprocal condition numbers for eigenvalues (optional, real)
	dif: []R = nil, // Condition numbers for deflating subspaces (optional, real)
	select_mask: []Blas_Int = nil, // Selection array (for selected eigenvalues)
	work: []T, // Complex workspace
	iwork: []Blas_Int = nil, // Integer workspace (needed if job != EigenvaluesOnly)
	job := SensitivityJob.Both,
	selection := EigenvectorSelection.All,
) -> (
	m_computed: Blas_Int,
	info: Info,
	ok: bool, // Number of eigenvalues/vectors for which condition numbers computed
) where T == complex64 || T == complex128,
	R == real_type_of(T) {
	n := a.cols
	assert(a.rows >= n && b.rows >= n && b.cols >= n, "Matrices too small")
	assert(len(work) > 0, "Workspace required")
	if job != .EigenvaluesOnly {
		assert(len(iwork) >= n + 2, "Integer workspace required for subspace condition numbers")
	}

	job_c := sensitivity_job_to_char(job)
	howmny_c := eigenvector_selection_to_char(selection)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.ld)
	ldb := Blas_Int(b.ld)
	lwork := Blas_Int(len(work))

	// Handle eigenvector matrices
	ldvl: Blas_Int = 1
	vl_ptr: rawptr = nil
	if vl != nil {
		assert(vl.rows >= n && vl.cols >= n, "Left eigenvector matrix too small")
		ldvl = Blas_Int(vl.ld)
		vl_ptr = raw_data(vl.data)
	}

	ldvr: Blas_Int = 1
	vr_ptr: rawptr = nil
	if vr != nil {
		assert(vr.rows >= n && vr.cols >= n, "Right eigenvector matrix too small")
		ldvr = Blas_Int(vr.ld)
		vr_ptr = raw_data(vr.data)
	}

	// Maximum number to compute
	mm := Blas_Int(n)

	when T == complex64 {
		lapack.ctgsna_(
			&job_c,
			&howmny_c,
			raw_data(select_mask) if select_mask != nil else nil,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			vl_ptr,
			&ldvl,
			vr_ptr,
			&ldvr,
			raw_data(s) if s != nil else nil,
			raw_data(dif) if dif != nil else nil,
			&mm,
			&m_computed,
			raw_data(work),
			&lwork,
			raw_data(iwork) if iwork != nil else nil,
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.ztgsna_(
			&job_c,
			&howmny_c,
			raw_data(select_mask) if select_mask != nil else nil,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			vl_ptr,
			&ldvl,
			vr_ptr,
			&ldvr,
			raw_data(s) if s != nil else nil,
			raw_data(dif) if dif != nil else nil,
			&mm,
			&m_computed,
			raw_data(work),
			&lwork,
			raw_data(iwork) if iwork != nil else nil,
			&info,
			1,
			1,
		)
	}

	return m_computed, info, info == 0
}

// Procedure group for generalized eigenvalue sensitivity
m_compute_generalized_sensitivity :: proc {
	m_compute_generalized_sensitivity_real,
	m_compute_generalized_sensitivity_complex,
}
