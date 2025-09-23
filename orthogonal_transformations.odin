package openblas

import lapack "./f77"
import "base:builtin"
import "core:c"
import "core:math"
import "core:mem"

// ===================================================================================
// ORTHOGONAL TRANSFORMATION FUNCTIONS
// ===================================================================================

// m_apply_random_orthogonal - Apply random orthogonal transformation to matrix
m_apply_random_orthogonal :: proc {
	m_apply_random_orthogonal_f32_c64,
	m_apply_random_orthogonal_f64_c128,
}

// m_apply_givens_rotation - Apply Givens rotation to adjacent rows/columns
m_apply_givens_rotation :: proc {
	m_apply_givens_rotation_f32_c64,
	m_apply_givens_rotation_f64_c128,
}


// ===================================================================================
// RANDOM ORTHOGONAL TRANSFORMATION (DLAROR/SLAROR/CLAROR/ZLAROR)
// ===================================================================================

// Apply random orthogonal transformation for f32/complex64
m_apply_random_orthogonal_f32_c64 :: proc(
	A: ^Matrix($T),
	side: OrthogonalSide = .Left,
	init: OrthogonalInit = .None,
	seed: ^[4]i32,
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 ||
	T == complex64 {
	// Validate input
	assert(A != nil && A.data != nil, "Matrix A cannot be nil")
	assert(seed != nil, "Seed cannot be nil")
	assert(seed[3] % 2 == 1, "seed[3] must be odd")

	m := A.rows
	n := A.cols
	lda := A.ld

	// Convert enums to cstrings
	side_c := orthogonal_side_to_char(side)
	init_c := orthogonal_init_to_char(init)

	// Allocate workspace
	work_size: Blas_Int
	switch side {
	case .Left:
		work_size = m
	case .Right:
		work_size = n
	case .Both:
		work_size = max(m, n)
	}

	work := make([]T, work_size, allocator)
	defer delete(work)

	// Convert i32 seed to Blas_Int
	seed_blas := [4]Blas_Int {
		Blas_Int(seed[0]),
		Blas_Int(seed[1]),
		Blas_Int(seed[2]),
		Blas_Int(seed[3]),
	}

	info_val: Info

	when T == f32 {
		lapack.slaror_(
			side_c,
			init_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&seed_blas[0],
			raw_data(work),
			&info_val,
			c.size_t(len(side_c)),
			c.size_t(len(init_c)),
		)
	} else when T == complex64 {
		lapack.claror_(
			side_c,
			init_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&seed_blas[0],
			raw_data(work),
			&info_val,
			c.size_t(len(side_c)),
			c.size_t(len(init_c)),
		)
	}

	// Update the original seed
	seed[0] = i32(seed_blas[0])
	seed[1] = i32(seed_blas[1])
	seed[2] = i32(seed_blas[2])
	seed[3] = i32(seed_blas[3])

	return info_val
}

// Apply random orthogonal transformation for f64/complex128
m_apply_random_orthogonal_f64_c128 :: proc(
	A: ^Matrix($T),
	side: OrthogonalSide = .Left,
	init: OrthogonalInit = .None,
	seed: ^[4]i32,
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f64 ||
	T == complex128 {
	// Validate input
	assert(A != nil && A.data != nil, "Matrix A cannot be nil")
	assert(seed != nil, "Seed cannot be nil")
	assert(seed[3] % 2 == 1, "seed[3] must be odd")

	m := A.rows
	n := A.cols
	lda := A.ld

	// Convert enums to cstrings
	side_c := orthogonal_side_to_char(side)
	init_c := orthogonal_init_to_char(init)

	// Allocate workspace
	work_size: Blas_Int
	switch side {
	case .Left:
		work_size = m
	case .Right:
		work_size = n
	case .Both:
		work_size = max(m, n)
	}

	work := make([]T, work_size, allocator)
	defer delete(work)

	// Convert i32 seed to Blas_Int
	seed_blas := [4]Blas_Int {
		Blas_Int(seed[0]),
		Blas_Int(seed[1]),
		Blas_Int(seed[2]),
		Blas_Int(seed[3]),
	}

	info_val: Info

	when T == f64 {
		lapack.dlaror_(
			side_c,
			init_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&seed_blas[0],
			raw_data(work),
			&info_val,
			c.size_t(len(side_c)),
			c.size_t(len(init_c)),
		)
	} else when T == complex128 {
		lapack.zlaror_(
			side_c,
			init_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&seed_blas[0],
			raw_data(work),
			&info_val,
			c.size_t(len(side_c)),
			c.size_t(len(init_c)),
		)
	}

	// Update the original seed
	seed[0] = i32(seed_blas[0])
	seed[1] = i32(seed_blas[1])
	seed[2] = i32(seed_blas[2])
	seed[3] = i32(seed_blas[3])

	return info_val
}

// ===================================================================================
// GIVENS ROTATION APPLICATION (DLAROT/SLAROT/CLAROT/ZLAROT)
// ===================================================================================

// Apply Givens rotation for f32/complex64
m_apply_givens_rotation_f32_c64 :: proc(
	A: ^Matrix($T),
	c: f32, // Cosine of rotation angle (real for all types)
	s: T, // Sine of rotation angle (same type as matrix)
	row_idx: int, // First row/column index (0-based)
	apply_to_rows: bool, // true = apply to rows, false = apply to columns
	left_extend: bool, // true if rotation extends to the left
	right_extend: bool, // true if rotation extends to the right
	nl: int, // Number of elements to rotate
	xleft: T, // Value for left extension
	xright: T, // Value for right extension
	allocator := context.allocator,
) where T == f32 || T == complex64 {
	// Validate input
	assert(A != nil && A.data != nil, "Matrix A cannot be nil")
	assert(row_idx >= 0, "Row/column index must be non-negative")
	assert(nl > 0, "Number of elements must be positive")

	lda := A.ld

	// Convert boolean flags to FORTRAN logical (0 = false, 1 = true)
	lrows := Blas_Int(apply_to_rows ? 1 : 0)
	lleft := Blas_Int(left_extend ? 1 : 0)
	lright := Blas_Int(right_extend ? 1 : 0)
	nl_val := Blas_Int(nl)

	// Calculate starting position in matrix
	// For row operations: A[row_idx, :]
	// For column operations: A[:, row_idx]
	start_ptr: ^T
	if apply_to_rows {
		// Start at A[row_idx, 0] for row operations
		start_ptr = &A.data[row_idx]
	} else {
		// Start at A[0, row_idx] for column operations
		start_ptr = &A.data[row_idx * lda]
	}

	c_val := c
	s_val := s
	xleft_val := xleft
	xright_val := xright

	when T == f32 {
		lapack.slarot_(
			&lrows,
			&lleft,
			&lright,
			&nl_val,
			&c_val,
			&s_val,
			start_ptr,
			&lda,
			&xleft_val,
			&xright_val,
		)
	} else when T == complex64 {
		lapack.clarot_(
			&lrows,
			&lleft,
			&lright,
			&nl_val,
			&c_val,
			&s_val,
			start_ptr,
			&lda,
			&xleft_val,
			&xright_val,
		)
	}
}

// Apply Givens rotation for f64/complex128
m_apply_givens_rotation_f64_c128 :: proc(
	A: ^Matrix($T),
	c: f64, // Cosine of rotation angle (real for all types)
	s: T, // Sine of rotation angle (same type as matrix)
	row_idx: int, // First row/column index (0-based)
	apply_to_rows: bool, // true = apply to rows, false = apply to columns
	left_extend: bool, // true if rotation extends to the left
	right_extend: bool, // true if rotation extends to the right
	nl: int, // Number of elements to rotate
	xleft: T, // Value for left extension
	xright: T, // Value for right extension
	allocator := context.allocator,
) where T == f64 || T == complex128 {
	// Validate input
	assert(A != nil && A.data != nil, "Matrix A cannot be nil")
	assert(row_idx >= 0, "Row/column index must be non-negative")
	assert(nl > 0, "Number of elements must be positive")

	lda := A.ld

	// Convert boolean flags to FORTRAN logical (0 = false, 1 = true)
	lrows := Blas_Int(apply_to_rows ? 1 : 0)
	lleft := Blas_Int(left_extend ? 1 : 0)
	lright := Blas_Int(right_extend ? 1 : 0)
	nl_val := Blas_Int(nl)

	// Calculate starting position in matrix
	// For row operations: A[row_idx, :]
	// For column operations: A[:, row_idx]
	start_ptr: ^T
	if apply_to_rows {
		// Start at A[row_idx, 0] for row operations
		start_ptr = &A.data[row_idx]
	} else {
		// Start at A[0, row_idx] for column operations
		start_ptr = &A.data[row_idx * lda]
	}

	c_val := c
	s_val := s
	xleft_val := xleft
	xright_val := xright

	when T == f64 {
		lapack.dlarot_(
			&lrows,
			&lleft,
			&lright,
			&nl_val,
			&c_val,
			&s_val,
			start_ptr,
			&lda,
			&xleft_val,
			&xright_val,
		)
	} else when T == complex128 {
		lapack.zlarot_(
			&lrows,
			&lleft,
			&lright,
			&nl_val,
			&c_val,
			&s_val,
			start_ptr,
			&lda,
			&xleft_val,
			&xright_val,
		)
	}
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Generate random orthogonal matrix
m_generate_random_orthogonal :: proc(
	A: ^Matrix($T),
	seed: ^[4]i32,
	allocator := context.allocator,
) -> Info where is_float(T) ||
	is_complex(T) {
	// Initialize to identity and apply random orthogonal transformation
	when T == f32 || T == complex64 {
		return m_apply_random_orthogonal_f32_c64(A, .Left, .Identity, seed, allocator)
	} else when T == f64 || T == complex128 {
		return m_apply_random_orthogonal_f64_c128(A, .Left, .Identity, seed, allocator)
	}
}

// Apply Givens rotation to eliminate an element
m_apply_givens_elimination :: proc(
	A: ^Matrix($T),
	row1: int,
	row2: int,
	col: int,
	allocator := context.allocator,
) where is_float(T) ||
	is_complex(T) {
	// Compute rotation parameters to eliminate A[row2, col]
	a11 := matrix_get(A, row1, col)
	a21 := matrix_get(A, row2, col)

	// Compute Givens rotation parameters
	c: f64
	s: T

	when T == f32 || T == f64 {
		// Real case
		r := math.hypot(f64(a11), f64(a21))
		if r != 0 {
			c = f64(a11) / r
			s = T(-f64(a21) / r)
		} else {
			c = 1.0
			s = 0.0
		}
	} else when T == complex64 || T == complex128 {
		// Complex case - need to handle complex arithmetic
		// For complex Givens rotations, c is real and s is complex
		abs_a11 := abs(a11)
		abs_a21 := abs(a21)
		r := math.hypot(abs_a11, abs_a21)
		if r != 0 {
			c = abs_a11 / r
			// s needs to be computed to eliminate a21
			// s = -conj(a21) * a11 / (|a11| * r)
			when T == complex64 {
				s = complex64(-conj(a21) * a11 / complex64(abs_a11 * r, 0))
			} else {
				s = complex128(-conj(a21) * a11 / complex128(abs_a11 * r, 0))
			}
		} else {
			c = 1.0
			s = 0
		}
	}

	// Apply the rotation to the entire row
	n_cols := A.cols - col

	when T == f32 || T == complex64 {
		m_apply_givens_rotation_f32_c64(
			A,
			f32(c),
			s,
			row1,
			true, // apply to rows
			false, // no left extension
			false, // no right extension
			n_cols,
			0,
			0,
			allocator,
		)
	} else when T == f64 || T == complex128 {
		m_apply_givens_rotation_f64_c128(
			A,
			c,
			s,
			row1,
			true, // apply to rows
			false, // no left extension
			false, // no right extension
			n_cols,
			0,
			0,
			allocator,
		)
	}
}
