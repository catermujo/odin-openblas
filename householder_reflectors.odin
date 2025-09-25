package openblas

import lapack "./f77"
import "base:builtin"
import "core:c"
import "core:math"
import "core:mem"


// ===================================================================================
// SINGLE HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================

// Query workspace for applying Householder reflector
query_workspace_householder_reflector :: proc(
	$T: typeid,
	m: int,
	n: int,
	side: ReflectorSide,
) -> (
	work: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	return side == .Left ? Blas_Int(n) : Blas_Int(m)
}

// Apply single Householder reflector (generic)
solve_apply_householder_reflector :: proc(
	side: ReflectorSide,
	m: int,
	n: int,
	V: ^Vector($T),
	tau: T,
	C: ^Matrix(T),
	work: []T,
) where is_float(T) ||
	is_complex(T) {
	// Validate workspace
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	incv := V.incr
	ldc := C.ld
	side_c := side_to_cstring(side)
	tau_val := tau

	when T == f32 {
		lapack.slarf_(
			side_c,
			&m_int,
			&n_int,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	} else when T == f64 {
		lapack.dlarf_(
			side_c,
			&m_int,
			&n_int,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	} else when T == complex64 {
		lapack.clarf_(
			side_c,
			&m_int,
			&n_int,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	} else when T == complex128 {
		lapack.zlarf_(
			side_c,
			&m_int,
			&n_int,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	}
}

// ===================================================================================
// BLOCK HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================

// Query workspace for applying block Householder reflector
query_workspace_block_householder_reflector :: proc(
	$T: typeid,
	m: int,
	n: int,
	k: int,
	side: ReflectorSide,
) -> (
	work: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	ldwork := side == .Left ? n : m
	return Blas_Int(ldwork * k)
}

// Apply block Householder reflector (generic)
solve_apply_block_householder_reflector :: proc(
	side: ReflectorSide,
	trans: ReflectorTranspose,
	direct: ReflectorDirection,
	storev: ReflectorStorage,
	m: int,
	n: int,
	k: int,
	V: ^Matrix($T),
	T_mat: ^Matrix(T),
	C: ^Matrix(T),
	work: []T,
) where is_float(T) ||
	is_complex(T) {
	// Validate workspace
	ldwork := side == .Left ? n : m
	assert(len(work) >= ldwork * k, "Work array too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := V.ld
	ldt := T_mat.ld
	ldc := C.ld
	ldwork_val := Blas_Int(ldwork)

	side_c := side_to_cstring(side)
	trans_c := trans_to_cstring(trans)
	direct_c := direct_to_cstring(direct)
	storev_c := storev_to_cstring(storev)

	when T == f32 {
		lapack.slarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m_int,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(T_mat.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			len(side_c),
			len(trans_c),
			len(direct_c),
			len(storev_c),
		)
	} else when T == f64 {
		lapack.dlarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m_int,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(T_mat.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			len(side_c),
			len(trans_c),
			len(direct_c),
			len(storev_c),
		)
	} else when T == complex64 {
		lapack.clarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m_int,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(T_mat.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			len(side_c),
			len(trans_c),
			len(direct_c),
			len(storev_c),
		)
	} else when T == complex128 {
		lapack.zlarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m_int,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(T_mat.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			len(side_c),
			len(trans_c),
			len(direct_c),
			len(storev_c),
		)
	}
}

// ===================================================================================
// HOUSEHOLDER REFLECTOR GENERATION
// ===================================================================================

// Query result sizes for Householder reflector generation
// tau is a single scalar value
query_result_sizes_householder_reflector :: proc(n: int) -> (tau_size: int) {
	return 1 // tau is a single scalar
}

// Generate Householder reflector (generic)
solve_generate_householder_reflector :: proc(
	n: int,
	alpha: ^$T,
	X: ^Vector(T),
	tau: ^T,
) where is_float(T) ||
	is_complex(T) {
	n_int := Blas_Int(n)
	incx := X.incr

	when T == f32 {
		lapack.slarfg_(&n_int, alpha, data_ptr(X), &incx, tau)
	} else when T == f64 {
		lapack.dlarfg_(&n_int, alpha, data_ptr(X), &incx, tau)
	} else when T == complex64 {
		lapack.clarfg_(&n_int, alpha, data_ptr(X), &incx, tau)
	} else when T == complex128 {
		lapack.zlarfg_(&n_int, alpha, data_ptr(X), &incx, tau)
	}
}

// ===================================================================================
// TRIANGULAR T MATRIX FORMATION
// ===================================================================================

// Query result sizes for triangular T matrix formation
query_result_sizes_triangular_t_matrix :: proc(
	k: int,
) -> (
	tau_size: int,
	T_mat_rows: int,
	T_mat_cols: int,
) {
	return k, k, k // tau array of size k, T matrix is k×k
}

// Form triangular T matrix for block Householder reflector (generic)
solve_form_triangular_t_matrix :: proc(
	direct: ReflectorDirection,
	storev: ReflectorStorage,
	n: int,
	k: int,
	V: ^Matrix($T),
	tau: []T,
	T_mat: ^Matrix(T),
) where is_float(T) ||
	is_complex(T) {
	assert(len(tau) >= k, "Tau array too small")

	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	ldv := V.ld
	ldt := T_mat.ld

	direct_c := direct_to_cstring(direct)
	storev_c := storev_to_cstring(storev)

	when T == f32 {
		lapack.slarft_(
			direct_c,
			storev_c,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T_mat.data),
			&ldt,
			len(direct_c),
			len(storev_c),
		)
	} else when T == f64 {
		lapack.dlarft_(
			direct_c,
			storev_c,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T_mat.data),
			&ldt,
			len(direct_c),
			len(storev_c),
		)
	} else when T == complex64 {
		lapack.clarft_(
			direct_c,
			storev_c,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T_mat.data),
			&ldt,
			len(direct_c),
			len(storev_c),
		)
	} else when T == complex128 {
		lapack.zlarft_(
			direct_c,
			storev_c,
			&n_int,
			&k_int,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T_mat.data),
			&ldt,
			len(direct_c),
			len(storev_c),
		)
	}
}

// ===================================================================================
// SMALL HOUSEHOLDER REFLECTOR APPLICATION (OPTIMIZED)
// ===================================================================================

// Query workspace for small Householder reflector
query_workspace_small_householder_reflector :: proc(
	$T: typeid,
	m: int,
	n: int,
	side: ReflectorSide,
) -> (
	work: Blas_Int,
) where is_float(T) ||
	is_complex(T) {
	return side == .Left ? Blas_Int(n) : Blas_Int(m)
}

// Apply small Householder reflector optimized for small matrices (generic)
solve_apply_small_householder_reflector :: proc(
	side: ReflectorSide,
	m: int,
	n: int,
	V: []$T,
	tau: T,
	C: ^Matrix(T),
	work: []T,
) where is_float(T) ||
	is_complex(T) {
	// Validate workspace
	work_size := side == .Left ? n : m
	assert(len(work) >= work_size, "Work array too small")
	assert(len(V) >= (side == .Left ? m : n), "Vector too small")

	m_int := Blas_Int(m)
	n_int := Blas_Int(n)
	ldc := C.ld
	side_c := side_to_cstring(side)
	tau_val := tau

	when T == f32 {
		lapack.slarfx_(
			side_c,
			&m_int,
			&n_int,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	} else when T == f64 {
		lapack.dlarfx_(
			side_c,
			&m_int,
			&n_int,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	} else when T == complex64 {
		lapack.clarfx_(
			side_c,
			&m_int,
			&n_int,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	} else when T == complex128 {
		lapack.zlarfx_(
			side_c,
			&m_int,
			&n_int,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			len(side_c),
		)
	}
}


// ===================================================================================
// RANDOM VECTOR GENERATION
// ===================================================================================

// Generate random vector (generic)
solve_generate_random_vector :: proc(
	idist: RandomDistribution,
	n: int,
	X: ^Vector($T),
	iseed: ^[4]Blas_Int, // LAPACK seed (must be provided, will be updated)
) where is_float(T) || is_complex(T) {
	idist_int := Blas_Int(idist)
	n_int := X.size

	when T == f32 {
		lapack.slarnv_(&idist_int, &iseed[0], &n_int, data_ptr(X))
	} else when T == f64 {
		lapack.dlarnv_(&idist_int, &iseed[0], &n_int, data_ptr(X))
	} else when T == complex64 {
		lapack.clarnv_(&idist_int, &iseed[0], &n_int, data_ptr(X))
	} else when T == complex128 {
		lapack.zlarnv_(&idist_int, &iseed[0], &n_int, data_ptr(X))
	}
}
