package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// GIVENS ROTATIONS AND MATRIX SCALING
// ===================================================================================

// ===================================================================================
// GIVENS ROTATION GENERATION
// ===================================================================================

// Generate robust Givens rotation
// Generates plane rotation with robustness against overflow
m_generate_robust_givens :: proc(f, g: $T) -> (cs, sn, r: T) where is_float(T) {
	cs_val, sn_val, r_val: T
	f_val, g_val := f, g
	when T == f32 {
		lapack.slartgp_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	} else when T == f64 {
		lapack.dlartgp_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	}
	return cs_val, sn_val, r_val
}

// Generate standard Givens rotation
// Generates plane rotation, choosing sign to avoid cancellation
m_generate_givens :: proc(f, g: $T) -> (cs, sn, r: T) where is_float(T) {
	cs_val, sn_val, r_val: T
	f_val, g_val := f, g
	when T == f32 {
		lapack.slartgs_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	} else when T == f64 {
		lapack.dlartgs_(&f_val, &g_val, &cs_val, &sn_val, &r_val)
	}
	return cs_val, sn_val, r_val
}

// ===================================================================================
// MATRIX SCALING OPERATIONS
// ===================================================================================

// Scale matrix - polymorphic for all numeric types
m_scale :: proc(
	A: ^Matrix($T),
	cfrom, cto: $ScaleType,
	scaling_type: MatrixScalingType = .General,
	kl: int = 0, // Lower bandwidth (for banded matrices)
	ku: int = 0, // Upper bandwidth (for banded matrices)
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) where (T == f32 && ScaleType == f32) || (T == f64 && ScaleType == f64) || (T == complex64 && ScaleType == f32) || (T == complex128 && ScaleType == f64) {
	// Validate matrix
	assert(len(A.data) > 0, "Matrix data cannot be empty")

	type_c := scaling_type_to_cstring(scaling_type)
	kl_val := Blas_Int(kl)
	ku_val := Blas_Int(ku)
	m := Blas_Int(A.rows)
	n := Blas_Int(A.cols)
	lda := Blas_Int(A.ld)
	cfrom_val, cto_val := cfrom, cto

	when T == f32 {
		lapack.slascl_(
			type_c,
			&kl_val,
			&ku_val,
			&cfrom_val,
			&cto_val,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&info,
			len(type_c),
		)
	} else when T == f64 {
		lapack.dlascl_(
			type_c,
			&kl_val,
			&ku_val,
			&cfrom_val,
			&cto_val,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&info,
			len(type_c),
		)
	} else when T == complex64 {
		lapack.clascl_(
			type_c,
			&kl_val,
			&ku_val,
			&cfrom_val,
			&cto_val,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&info,
			len(type_c),
		)
	} else when T == complex128 {
		lapack.zlascl_(
			type_c,
			&kl_val,
			&ku_val,
			&cfrom_val,
			&cto_val,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&info,
			len(type_c),
		)
	}

	return info == 0, info
}

// ===================================================================================
// CONVENIENCE FUNCTIONS FOR GIVENS ROTATIONS
// ===================================================================================

// Givens rotation parameters
GivensRotation :: struct($T: typeid) where is_float(T) {
	cs, sn: T, // Cosine and sine of rotation angle
	i, j:   int, // Indices of rows/columns to rotate
}

// Apply Givens rotation to vector pair
// Applies rotation [ cs  sn] [x] = [x']
//                  [-sn  cs] [y]   [y']
apply_givens_rotation :: proc(cs, sn: $T, x, y: ^T) {
	temp := cs * x^ + sn * y^
	y^ = -sn * x^ + cs * y^
	x^ = temp
}

// Create Givens rotation to zero out second component
// Returns rotation that transforms [a, b] -> [r, 0]
create_givens_to_zero :: proc(a, b: $T) -> (cs, sn, r: T) where is_float(T) {
	return m_generate_robust_givens(a, b)
}

// Pivoting type for plane rotations
GivensPivot :: enum {
	Variable, // "V" - Variable pivot (rotations between i and i+1)
	Bottom,   // "B" - Bottom pivot (rotations with last row/column)
	Top,      // "T" - Top pivot (rotations with first row/column)
}

pivot_to_cstring :: proc(pivot: GivensPivot) -> cstring {
	switch pivot {
	case .Variable:
		return "V"
	case .Bottom:
		return "B"
	case .Top:
		return "T"
	}
	unreachable()
}

// Direction for applying rotations
GivensDirection :: enum {
	Forward,  // "F" - Forward (from first to last)
	Backward, // "B" - Backward (from last to first)
}

direction_to_cstring :: proc(dir: GivensDirection) -> cstring {
	switch dir {
	case .Forward:
		return "F"
	case .Backward:
		return "B"
	}
	unreachable()
}

// Apply sequence of Givens rotations to matrix using LAPACK DLASR
// This is more efficient than manual application for large sequences
m_apply_givens_sequence :: proc(
	A: ^Matrix($T),
	c, s: []$RotType,  // Cosine and sine arrays
	side: ReflectorSide = .Left,
	pivot: GivensPivot = .Variable,
	direction: GivensDirection = .Forward,
) where (T == f32 && RotType == f32) ||
        (T == f64 && RotType == f64) ||
        (T == complex64 && RotType == f32) ||
        (T == complex128 && RotType == f64) {
	assert(len(c) == len(s), "Cosine and sine arrays must have same length")

	m := A.rows
	n := A.cols
	lda := A.ld

	side_c := side_to_char(side)
	pivot_c := pivot_to_cstring(pivot)
	direct_c := direction_to_cstring(direction)

	when T == f32 {
		lapack.slasr_(side_c, pivot_c, direct_c, &m, &n,
		              raw_data(c), raw_data(s), raw_data(A.data), &lda,
		              len(side_c), len(pivot_c), len(direct_c))
	} else when T == f64 {
		lapack.dlasr_(side_c, pivot_c, direct_c, &m, &n,
		              raw_data(c), raw_data(s), raw_data(A.data), &lda,
		              len(side_c), len(pivot_c), len(direct_c))
	} else when T == complex64 {
		lapack.clasr_(side_c, pivot_c, direct_c, &m, &n,
		              raw_data(c), raw_data(s), raw_data(A.data), &lda,
		              len(side_c), len(pivot_c), len(direct_c))
	} else when T == complex128 {
		lapack.zlasr_(side_c, pivot_c, direct_c, &m, &n,
		              raw_data(c), raw_data(s), raw_data(A.data), &lda,
		              len(side_c), len(pivot_c), len(direct_c))
	}
}

// Apply sequence of Givens rotations to matrix (convenience wrapper)
// Applies rotations to consecutive rows/columns
apply_givens_sequence :: proc(
	A: ^Matrix($T),
	rotations: []GivensRotation(T),
	apply_to_rows: bool = true,
	allocator := context.allocator,
) where is_float(T) {
	if len(rotations) == 0 do return

	// Extract c and s arrays from rotations
	c := make([]T, len(rotations), context.temp_allocator)
	s := make([]T, len(rotations), context.temp_allocator)
	defer delete(c)
	defer delete(s)

	for i, rot in rotations {
		c[i] = rot.cs
		s[i] = rot.sn
	}

	// Use LAPACK for efficient application
	side := apply_to_rows ? ReflectorSide.Left : ReflectorSide.Right
	m_apply_givens_sequence(A, c, s, side, .Variable, .Forward)
}

// ===================================================================================
// CONVENIENCE FUNCTIONS FOR MATRIX SCALING
// ===================================================================================

// Scale matrix to avoid over/underflow
// Automatically determines appropriate scaling factors
m_scale_safe :: proc(
	A: ^Matrix($T),
	scaling_type := MatrixScalingType.General,
	kl := 0,
	ku := 0,
	allocator := context.allocator,
) -> (
	success: bool,
	scale_factor: T,
) where is_float(T) ||
	is_complex(T) {
	// Find maximum absolute value in matrix
	max_val := T(0)
	when T == complex64 || T == complex128 {
		for val in A.data {
			abs_val := abs(val)
			if abs_val > max_val {
				max_val = abs_val
			}
		}
	} else {
		for val in A.data {
			abs_val := abs(val)
			if abs_val > max_val {
				max_val = abs_val
			}
		}
	}

	if max_val == 0 {
		return true, T(1) // Matrix is zero
	}

	// Determine safe scaling factor
	when T == f64 || T == complex128 {
		safe_max := 1e150 // Safe maximum for f64
		safe_min := 1e-150 // Safe minimum for f64
		scale_to := 1.0
	} else {
		safe_max := 1e30 // Safe maximum for f32
		safe_min := 1e-30 // Safe minimum for f32
		scale_to := 1.0
	}

	scale_factor = T(1)
	cfrom := max_val
	cto := T(scale_to)

	if max_val > safe_max {
		scale_factor = T(scale_to) / max_val
	} else if max_val < safe_min && max_val > 0 {
		scale_factor = T(scale_to) / max_val
	} else {
		return true, T(1) // No scaling needed
	}

	// Apply scaling
	when T == complex64 {
		success, _ := m_scale(A, f32(cfrom), f32(cto), scaling_type, kl, ku, allocator)
		return success, scale_factor
	} else when T == f64 {
		success, _ := m_scale(A, cfrom, cto, scaling_type, kl, ku, allocator)
		return success, scale_factor
	} else when T == f32 {
		success, _ := m_scale(A, cfrom, cto, scaling_type, kl, ku, allocator)
		return success, scale_factor
	} else when T == complex128 {
		success, _ := m_scale(A, f64(cfrom), f64(cto), scaling_type, kl, ku, allocator)
		return success, scale_factor
	}
}

// Scale matrix by constant factor
m_scale_by_factor :: proc(
	A: ^Matrix($T),
	factor: T,
	scaling_type := MatrixScalingType.General,
	kl := 0,
	ku := 0,
	allocator := context.allocator,
) -> (
	success: bool,
) where is_float(T) ||
	is_complex(T) {
	cfrom := T(1)
	cto := factor

	when T == complex64 {
		success, _ := m_scale(A, f32(cfrom), f32(cto), scaling_type, kl, ku, allocator)
		return success
	} else when T == f64 {
		success, _ := m_scale(A, cfrom, cto, scaling_type, kl, ku, allocator)
		return success
	} else when T == f32 {
		success, _ := m_scale(A, cfrom, cto, scaling_type, kl, ku, allocator)
		return success
	} else when T == complex128 {
		success, _ := m_scale(A, f64(cfrom), f64(cto), scaling_type, kl, ku, allocator)
		return success
	}
}

// Normalize matrix to unit scale
m_normalize :: proc(
	A: ^Matrix($T),
	scaling_type := MatrixScalingType.General,
	kl := 0,
	ku := 0,
	allocator := context.allocator,
) -> (
	success: bool,
	norm: T,
) where is_float(T) ||
	is_complex(T) {
	// Calculate Frobenius norm using LAPACK
	when T == f32 || T == complex64 {
		norm = T(m_norm_general(A, .FrobeniusNorm, allocator))
	} else when T == f64 || T == complex128 {
		norm = T(m_norm_general(A, .FrobeniusNorm, allocator))
	}

	if norm == 0 {
		return true, T(0) // Matrix is zero
	}

	// Scale by inverse of norm
	cfrom := norm
	cto := T(1)

	when T == f32 {
		success, _ := m_scale(A, cfrom, cto, scaling_type, kl, ku, allocator)
		return success, norm
	} else when T == f64 {
		success, _ := m_scale(A, cfrom, cto, scaling_type, kl, ku, allocator)
		return success, norm
	} else when T == complex64 {
		success, _ := m_scale(A, f32(cfrom), f32(cto), scaling_type, kl, ku, allocator)
		return success, norm
	} else when T == complex128 {
		success, _ := m_scale(A, f64(cfrom), f64(cto), scaling_type, kl, ku, allocator)
		return success, norm
	}
}
