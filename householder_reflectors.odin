package openblas

import lapack "./f77"
import "base:builtin"
import "core:c"
import "core:math"
import "core:mem"

// m_apply_householder_reflector

// m_apply_block_householder_reflector

// v_generate_householder_reflector

// m_form_triangular_t_matrix

// m_apply_small_householder_reflector

// v_generate_random


// ===================================================================================
// SINGLE HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================
m_apply_householder_reflector :: proc(
	C: ^Matrix($T),
	V: ^Vector(T),
	tau: T,
	side: ReflectorSide = .Left,
	allocator := context.allocator,
) where is_float(T) ||
	is_complex(T) {

	// Validate input
	switch side {
	case .Left:
		assert(V.size == C.rows, "Vector length must match matrix rows for left application")
	case .Right:
		assert(V.size == C.cols, "Vector length must match matrix columns for right application")
	}

	m := C.rows
	n := C.cols
	incv := V.incr
	ldc := C.ld
	side_c := side_to_char(side)
	tau_val := tau

	// Allocate workspace
	work_size := side == .Left ? n : m
	work := make([]T, work_size)
	defer delete(work)

	when T == f32 {
		lapack.slarf_(
			side_c,
			&m,
			&n,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	} else when T == f64 {
		lapack.dlarf_(
			side_c,
			&m,
			&n,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	} else when T == complex64 {
		lapack.clarf_(
			side_c,
			&m,
			&n,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	} else when T == complex128 {
		lapack.zlarf_(
			side_c,
			&m,
			&n,
			data_ptr(V),
			&incv,
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	}
}

// ===================================================================================
// BLOCK HOUSEHOLDER REFLECTOR APPLICATION
// ===================================================================================

// Apply block Householder reflector (c64)
m_apply_block_householder_reflector :: proc(
	C: ^Matrix($Type),
	V: ^Matrix(Type),
	T: ^Matrix(Type),
	side: ReflectorSide = .Left,
	trans: ReflectorTranspose = .None,
	direct: ReflectorDirection = .Forward,
	storev: ReflectorStorage = .ColumnWise,
	allocator := context.allocator,
) where is_float(Type) ||
	is_complex(Type) {
	// Validate dimensions
	k := V.cols if storev == .ColumnWise else V.rows
	assert(
		T.rows == k && T.cols == k,
		"T matrix must be k x k where k is the number of reflectors",
	)

	m := C.rows
	n := C.cols
	k_val := k
	ldv := V.ld
	ldt := T.ld
	ldc := C.ld

	side_c := side_to_char(side)
	trans_c := trans_to_char(trans)
	direct_c := direct_to_char(direct)
	storev_c := storev_to_char(storev)

	// Allocate workspace
	ldwork := side == .Left ? n : m
	work := make([]Type, ldwork * k)
	defer delete(work)
	ldwork_val := ldwork

	when Type == f32 {
		lapack.slarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(T.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			c.size_t(len(side_c)),
			c.size_t(len(trans_c)),
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	} else when Type == f64 {
		lapack.dlarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(T.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			c.size_t(len(side_c)),
			c.size_t(len(trans_c)),
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	} else when Type == complex64 {
		lapack.clarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(T.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			c.size_t(len(side_c)),
			c.size_t(len(trans_c)),
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	} else when Type == complex128 {
		lapack.zlarfb_(
			side_c,
			trans_c,
			direct_c,
			storev_c,
			&m,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(T.data),
			&ldt,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			&ldwork_val,
			c.size_t(len(side_c)),
			c.size_t(len(trans_c)),
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	}
}

// ===================================================================================
// HOUSEHOLDER REFLECTOR GENERATION
// ===================================================================================

// Generate Householder reflector (c64)
v_generate_householder_reflector :: proc(
	X: ^Vector($T),
	alpha: ^T,
	allocator := context.allocator,
) -> (
	tau: T,
	success: bool,
) where is_float(Type) ||
	is_complex(Type) {
	// Validate input
	assert(X.size > 0, "Vector must have positive length")

	n := X.size
	incx := X.incr
	tau_val: T

	when T == f32 {
		lapack.slarfg_(&n, alpha, data_ptr(X), &incx, &tau_val)
	} else when T == f64 {
		lapack.dlarfg_(&n, alpha, data_ptr(X), &incx, &tau_val)
	} else when T == complex64 {
		lapack.clarfg_(&n, alpha, data_ptr(X), &incx, &tau_val)
	} else when T == complex128 {
		lapack.zlarfg_(&n, alpha, data_ptr(X), &incx, &tau_val)
	}

	return tau_val, true
}

// ===================================================================================
// TRIANGULAR T MATRIX FORMATION
// ===================================================================================

// Form triangular T matrix for block Householder reflector (c64)
m_form_triangular_t_matrix :: proc(
	V: ^Matrix($Type),
	tau: []Type,
	T: ^Matrix(Type),
	direct: ReflectorDirection = .Forward,
	storev: ReflectorStorage = .ColumnWise,
	allocator := context.allocator,
) where is_float(Type) ||
	is_complex(Type) {
	// Validate input
	k := Blas_Int(len(tau))
	assert(
		T.rows == k && T.cols == k,
		"T matrix must be k x k where k is the number of reflectors",
	)

	n := storev == .ColumnWise ? V.rows : V.cols
	k_val := k
	ldv := V.ld
	ldt := T.ld

	direct_c := direct_to_char(direct)
	storev_c := storev_to_char(storev)

	when Type == f32 {
		lapack.slarft_(
			direct_c,
			storev_c,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T.data),
			&ldt,
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	} else when Type == f64 {
		lapack.dlarft_(
			direct_c,
			storev_c,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T.data),
			&ldt,
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	} else when Type == complex64 {
		lapack.clarft_(
			direct_c,
			storev_c,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T.data),
			&ldt,
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	} else when Type == complex128 {
		lapack.zlarft_(
			direct_c,
			storev_c,
			&n,
			&k_val,
			raw_data(V.data),
			&ldv,
			raw_data(tau),
			raw_data(T.data),
			&ldt,
			c.size_t(len(direct_c)),
			c.size_t(len(storev_c)),
		)
	}

	return true
}

// ===================================================================================
// SMALL HOUSEHOLDER REFLECTOR APPLICATION (OPTIMIZED)
// ===================================================================================

// Apply small Householder reflector optimized for small matrices (c64)
m_apply_small_householder_reflector :: proc(
	C: ^Matrix($T),
	V: []T,
	tau: T,
	side: ReflectorSide = .Left,
	allocator := context.allocator,
) where is_float(Type) ||
	is_complex(Type) {
	// Validate input
	assert(side != .Left || len(V) == int(C.rows), "Vector length must match matrix rows for left application")
	assert(side != .Right || len(V) == int(C.cols), "Vector length must match matrix columns for right application")

	m := C.rows
	n := C.cols
	ldc := C.ld
	side_c := side_to_char(side)
	tau_val := tau

	// Allocate workspace (smaller than regular larfx)
	work_size := side == .Left ? n : m
	work := make([]T, work_size)
	defer delete(work)

	when T == f32 {
		lapack.slarfx_(
			side_c,
			&m,
			&n,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	} else when T == f64 {
		lapack.dlarfx_(
			side_c,
			&m,
			&n,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	} else when T == complex64 {
		lapack.clarfx_(
			side_c,
			&m,
			&n,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	} else when T == complex128 {
		lapack.zlarfx_(
			side_c,
			&m,
			&n,
			raw_data(V),
			&tau_val,
			raw_data(C.data),
			&ldc,
			raw_data(work),
			c.size_t(len(side_c)),
		)
	}
}


// ===================================================================================
// RANDOM VECTOR GENERATION
// ===================================================================================
// Generate random vector (c64) - Direct LAPACK wrapper
v_generate_random :: proc(
	X: ^Vector($T),
	idist: RandomDistribution, // 1=real uniform, 2=imaginary uniform, 3=normal, 5=uniform on unit circle
	iseed: ^[4]i32, // LAPACK seed (must be provided, will be updated)
) -> bool where is_float(T) || is_complex(T) {
	// Convert i32 seed to Blas_Int for LAPACK call
	seed_blas := [4]Blas_Int {
		Blas_Int(iseed[0]),
		Blas_Int(iseed[1]),
		Blas_Int(iseed[2]),
		Blas_Int(iseed[3]),
	}

	idist := Blas_Int(idist)
	n := X.size
	when T == f32 {
		lapack.slarnv_(&idist, &seed_blas[0], &n, data_ptr(X))
	} else when T == f64 {
		lapack.dlarnv_(&idist, &seed_blas[0], &n, data_ptr(X))
	} else when T == complex64 {
		lapack.clarnv_(&idist, &seed_blas[0], &n, data_ptr(X))
	} else when T == complex128 {
		lapack.zlarnv_(&idist, &seed_blas[0], &n, data_ptr(X))
	}

	// Update the original seed with modified values
	iseed[0] = i32(seed_blas[0])
	iseed[1] = i32(seed_blas[1])
	iseed[2] = i32(seed_blas[2])
	iseed[3] = i32(seed_blas[3])

	return true
}
