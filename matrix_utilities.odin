package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"
import "core:slice"

// ===================================================================================
// MATRIX INITIALIZATION, SORTING, AND UTILITY OPERATIONS
// ===================================================================================

// m_initialize_matrix

// v_sort

v_sum_of_squares :: proc {
	v_sum_of_squares_f32_c64,
	v_sum_of_squares_f64_c128,
}

// ===================================================================================
// MATRIX INITIALIZATION
// ===================================================================================
// Sets diagonal elements to beta, off-diagonal elements to alpha
m_initialize_matrix :: proc(
	A: ^Matrix($T),
	alpha: T, // Off-diagonal value
	beta: T, // Diagonal value
	region := MatrixRegion.Full,
) where is_float(T) || is_complex(T) {
	assert(A.data != nil, "Matrix data cannot be empty")

	uplo_c := matrix_region_to_cstring(region)
	m := A.rows
	n := A.cols
	lda := A.ld
	alpha_val := alpha
	beta_val := beta

	when T == f32 {
		lapack.slaset_(uplo_c, &m, &n, &alpha_val, &beta_val, raw_data(A.data), &lda, len(uplo_c))
	} else when T == f64 {
		lapack.dlaset_(uplo_c, &m, &n, &alpha_val, &beta_val, raw_data(A.data), &lda, len(uplo_c))
	} else when T == complex64 {
		lapack.claset_(uplo_c, &m, &n, &alpha_val, &beta_val, raw_data(A.data), &lda, len(uplo_c))
	} else when T == complex128 {
		lapack.zlaset_(uplo_c, &m, &n, &alpha_val, &beta_val, raw_data(A.data), &lda, len(uplo_c))
	}
}

// ===================================================================================
// VECTOR SORTING
// ===================================================================================
v_sort :: proc(
	D: Vector($T),
	direction := SortDirection.Increasing,
) -> (
	success: bool,
	info: Info,
) where is_float(T) {
	// Validate input
	if len(D.data) == 0 {return true, 0}

	id_c := sort_direction_to_cstring(direction)
	n := D.size
	info_val: Info
	when T == f32 {
		lapack.slasrt_(id_c, &n, raw_data(D.data), &info_val, len(id_c))
	} else when T == f64 {
		lapack.dlasrt_(id_c, &n, raw_data(D.data), &info_val, len(id_c))
	}

	return info_val == 0, info_val
}


// ===================================================================================
// SUM OF SQUARES COMPUTATION
// ===================================================================================

// Sum of squares computation (c64)
// Updates scale and sumsq such that scale^2 * sumsq = sum of squares of X
v_sum_of_squares_f32_c64 :: proc(
	X: ^Vector($T),
	scale: ^f32, // In/out: scaling factor
	sumsq: ^f32, // In/out: sum of squares
) where T == f32 || T == complex64 {
	if len(X.data) == 0 {return}

	n := Blas_Int(len(X.data))
	incx := X.incr

	when T == f32 {
		lapack.slassq_(&n, &X.data[X.offset], &incx, scale, sumsq)
	} else when T == complex64 {
		lapack.classq_(&n, &X.data[X.offset], &incx, scale, sumsq)
	}
}

// Sum of squares computation (f64)
// Updates scale and sumsq such that scale^2 * sumsq = sum of squares of X
v_sum_of_squares_f64_c128 :: proc(
	X: ^Vector($T),
	scale: ^f64, // In/out: scaling factor
	sumsq: ^f64, // In/out: sum of squares
) where T == f64 || T == complex128 {
	// Validate input
	if len(X.data) == 0 {return}

	n := Blas_Int(len(X.data))
	incx := X.incr

	when T == f64 {
		lapack.dlassq_(&n, &X.data[X.offset], &incx, scale, sumsq)
	} else when T == complex128 {
		lapack.zlassq_(&n, &X.data[X.offset], &incx, scale, sumsq)

	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Initialize matrix to identity
m_identity :: proc(A: ^Matrix($T)) -> bool where is_float(T) || is_complex(T) {
	// Only works for square matrices
	assert(A.rows == A.cols, "Identity matrix must be square")

	when T == f32 {
		return m_initialize_matrix_f32(A, 0, 1, .Full)
	} else when T == f64 {
		return m_initialize_matrix_f64(A, 0, 1, .Full)
	} else when T == complex64 {
		return m_initialize_matrix_c64(A, 0, 1, .Full)
	} else when T == complex128 {
		return m_initialize_matrix_c128(A, 0, 1, .Full)
	}
}

// Compute vector norm using sum of squares
v_norm :: proc(X: ^Vector($T)) -> f64 where is_float(T) || is_complex(T) {
	when T == f32 || T == complex64 {
		scale := f32(1.0)
		sumsq := f32(0.0)
		v_sum_of_squares_f32_c64(X, &scale, &sumsq)
		return f64(scale) * sqrt(f64(sumsq))
	} else when T == f64 || T == complex128 {
		scale := f64(1.0)
		sumsq := f64(0.0)
		m_sum_of_squares_f64_c128(X, &scale, &sumsq)
		return scale * sqrt(sumsq)
	}
}

copy_matrix :: proc(src: ^Matrix($T), dst: ^Matrix(T)) {
	min_rows := min(src.rows, dst.rows)
	min_cols := min(src.cols, dst.cols)
	for j in 0 ..< min_cols {
		for i in 0 ..< min_rows {
			matrix_set(dst, i, j, matrix_get(src, i, j))
		}
	}
}

max_array :: proc(arr: []$T) -> f64 {
	if len(arr) == 0 {
		return 0.0
	}
	max_val := f64(arr[0])
	for val in arr[1:] {
		max_val = max(max_val, f64(val))
	}
	return max_val
}

abs :: proc(x: $T) -> f64 {
	when T == complex64 {
		r, i := real(x), imag(x)
		return f64(math.sqrt(r * r + i * i))
	} else when T == complex128 {
		r, i := real(x), imag(x)
		return math.sqrt(r * r + i * i)
	} else {
		val := f64(x)
		return val >= 0 ? val : -val
	}
}
