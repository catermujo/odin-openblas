package openblas

import lapack "./f77"
import "base:builtin"
import "core:c"
import "core:math"
import "core:mem"

// ===================================================================================
// MATRIX NORM COMPUTATION
// ===================================================================================

// Compute banded matrix norm proc group
m_norm_banded :: proc {
	m_norm_banded_f32_c64,
	m_norm_banded_f64_c128,
}

// Compute general matrix norm proc group
m_norm_general :: proc {
	m_norm_general_f32_c64,
	m_norm_general_f64_c128,
}

// Compute tridiagonal matrix norm proc group
m_norm_tridiagonal :: proc {
	m_norm_tridiagonal_f32_c64,
	m_norm_tridiagonal_f64_c128,
}

// Compute Hermitian banded matrix norm proc group
m_norm_hermitian_banded :: proc {
	m_norm_hermitian_banded_c64,
	m_norm_hermitian_banded_c128,
}

// Compute Hermitian matrix norm proc group
m_norm_hermitian :: proc {
	m_norm_hermitian_c64,
	m_norm_hermitian_c128,
}

// Compute Hermitian packed matrix norm proc group
m_norm_hermitian_packed :: proc {
	m_norm_hermitian_packed_c64,
	m_norm_hermitian_packed_c128,
}

// Compute Hessenberg matrix norm proc group
m_norm_hessenberg :: proc {
	m_norm_hessenberg_f32_c64,
	m_norm_hessenberg_f64_c128,
}

// Compute symmetric tridiagonal matrix norm proc group
// m_norm_symmetric_tridiagonal

// Compute symmetric matrix norm proc group
m_norm_symmetric :: proc {
	m_norm_symmetric_f32_c64,
	m_norm_symmetric_f64_c128,
}

// Compute triangular banded matrix norm proc group
m_norm_triangular_banded :: proc {
	m_norm_triangular_banded_f64_c128,
	m_norm_triangular_banded_f32_c64,
}

// Compute triangular packed matrix norm proc group
m_norm_triangular_packed :: proc {
	m_norm_triangular_packed_f64_c128,
	m_norm_triangular_packed_f32_c64,
}

// Compute triangular matrix norm proc group (general storage)
m_norm_triangular_general :: proc {
	m_norm_triangular_general_f64_c128,
	m_norm_triangular_general_f32_c64,
}

// Matrix row permutation proc group
// m_permute_rows

// Matrix column permutation proc group
// m_permute_columns

// Euclidean norm proc group
euclidean_norm :: proc {
	euclidean_norm_2d,
	euclidean_norm_3d,
}

// Real-complex matrix multiplication proc group
m_multiply_real_complex :: proc {
	m_multiply_real32_complex64,
	m_multiply_real64_complex128,
}

// ===================================================================================
// BANDED MATRIX NORMS
// ===================================================================================

// Compute norm of banded matrix (c64)
m_norm_banded_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .Banded, "Matrix must be in banded format")

	n := A.rows
	kl := A.storage.banded.kl
	ku := A.storage.banded.ku
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f32 {
		result = lapack.slangb_(
			norm_c,
			&n,
			&kl,
			&ku,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(work_len),
		)
	} else when T == complex64 {
		result = lapack.clangb_(
			norm_c,
			&n,
			&kl,
			&ku,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(work_len),
		)
	}

	return result, true
}

// Compute norm of banded matrix (f64)
m_norm_banded_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	if A.format != .Banded {
		panic("Matrix must be in banded format")
	}

	n := A.rows
	kl := A.storage.banded.kl
	ku := A.storage.banded.ku
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f64 {
		result = lapack.dlangb_(
			norm_c,
			&n,
			&kl,
			&ku,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(work_len),
		)
	} else when T == complex128 {
		result = lapack.zlangb_(
			norm_c,
			&n,
			&kl,
			&ku,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(work_len),
		)
	}

	return result, true
}

// ===================================================================================
// GENERAL MATRIX NORMS
// ===================================================================================

// Compute norm of general matrix (c64)
m_norm_general_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .General, "Matrix must be in general format")

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm {
		work = make([]f32, m)
	} else if norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f32 {
		result = lapack.slange_(
			norm_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(work_len),
		)
	} else when T == complex64 {
		result_val = lapack.clange_(
			norm_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(work_len),
		)
	}

	return result, true
}

// Compute norm of general matrix (f64)
m_norm_general_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	if A.format != .General {
		panic("Matrix must be in general format")
	}

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm {
		work = make([]f64, m)
	} else if norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f64 {
		result = lapack.dlange_(
			norm_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(work_len),
		)
	} else when T == complex128 {
		result = lapack.zlange_(
			norm_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(work_len),
		)
	}

	return result, true
}

// ===================================================================================
// TRIDIAGONAL MATRIX NORMS
// ===================================================================================

// Compute norm of tridiagonal matrix (c64)
m_norm_tridiagonal_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) {
	// Validate matrix
	assert(A.format == .Tridiagonal, "Matrix must be in tridiagonal format")

	n := A.rows
	norm_c := norm_to_cstring(norm)

	// Extract tridiagonal diagonals
	dl_offset := A.storage.tridiagonal.dl_offset
	d_offset := A.storage.tridiagonal.d_offset
	du_offset := A.storage.tridiagonal.du_offset

	dl_ptr := &A.data[dl_offset] if dl_offset >= 0 && dl_offset < len(A.data) else nil
	d_ptr := &A.data[d_offset] if d_offset >= 0 && d_offset < len(A.data) else nil
	du_ptr := &A.data[du_offset] if du_offset >= 0 && du_offset < len(A.data) else nil

	when T == f32 {
		result = lapack.slangt_(
			norm_c,
			&n,
			dl_ptr,
			d_ptr,
			du_ptr,
			c.size_t(0), // norm string length (not used in practice)
		)
	} else when T == complex64 {
		result = lapack.clangt_(
			norm_c,
			&n,
			dl_ptr,
			d_ptr,
			du_ptr,
			c.size_t(0), // norm string length (not used in practice)
		)
	}

	return result, true
}

// Compute norm of tridiagonal matrix (f64)
m_norm_tridiagonal_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .Tridiagonal, "Matrix must be in tridiagonal format")

	n := A.rows
	norm_c := norm_to_cstring(norm)

	// Extract tridiagonal diagonals
	dl_offset := A.storage.tridiagonal.dl_offset
	d_offset := A.storage.tridiagonal.d_offset
	du_offset := A.storage.tridiagonal.du_offset

	dl_ptr := &A.data[dl_offset] if dl_offset >= 0 && dl_offset < len(A.data) else nil
	d_ptr := &A.data[d_offset] if d_offset >= 0 && d_offset < len(A.data) else nil
	du_ptr := &A.data[du_offset] if du_offset >= 0 && du_offset < len(A.data) else nil

	when T == f64 {
		result = lapack.dlangt_(
			norm_c,
			&n,
			dl_ptr,
			d_ptr,
			du_ptr,
			c.size_t(0), // norm string length (not used in practice)
		)
	} else when T == complex128 {
		result = lapack.zlangt_(
			norm_c,
			&n,
			dl_ptr,
			d_ptr,
			du_ptr,
			c.size_t(0), // norm string length (not used in practice)
		)

	}

	return result, true
}

// ===================================================================================
// HERMITIAN BANDED MATRIX NORMS
// ===================================================================================

// Compute norm of Hermitian banded matrix (c64)
m_norm_hermitian_banded_c64 :: proc(
	A: ^Matrix(complex64),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) {
	// Validate matrix
	assert(A.format == .Hermitian, "Matrix must be in Hermitian format")

	n := A.rows
	k := A.storage.banded.ku // For Hermitian, kl = ku
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.hermitian.uplo != nil {
		uplo_c = A.storage.hermitian.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0


	result = lapack.clanhb_(
		norm_c,
		uplo_c,
		&n,
		&k,
		raw_data(A.data),
		&ldab,
		work_ptr,
		c.size_t(len(norm_c)), // norm string length
		c.size_t(len(uplo_c)), // uplo string length
	)

	return f32(result), true
}

// Compute norm of Hermitian banded matrix (c128)
m_norm_hermitian_banded_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) {
	// Validate matrix
	assert(A.format == .Hermitian, "Matrix must be in Hermitian format")

	n := A.rows
	k := A.storage.banded.ku // For Hermitian, kl = ku
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.hermitian.uplo != nil {
		uplo_c = A.storage.hermitian.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	result = lapack.zlanhb_(
		norm_c,
		uplo_c,
		&n,
		&k,
		raw_data(A.data),
		&ldab,
		work_ptr,
		c.size_t(len(norm_c)), // norm string length
		c.size_t(len(uplo_c)), // uplo string length
	)

	return result, true
}

// ===================================================================================
// HERMITIAN MATRIX NORMS
// ===================================================================================

// Compute norm of Hermitian matrix (c64)
m_norm_hermitian_c64 :: proc(
	A: ^Matrix(complex64),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) {
	// Validate matrix
	assert(A.format == .Hermitian, "Matrix must be in Hermitian format")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.hermitian.uplo != nil {
		uplo_c = A.storage.hermitian.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	result = lapack.clanhe_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		work_ptr,
		c.size_t(len(norm_c)), // norm string length
		c.size_t(len(uplo_c)), // uplo string length
	)

	return f32(result), true
}

// Compute norm of Hermitian matrix (c128)
m_norm_hermitian_c128 :: proc(
	A: ^Matrix(complex128),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) {
	// Validate matrix
	assert(A.format == .Hermitian, "Matrix must be in Hermitian format")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.hermitian.uplo != nil {
		uplo_c = A.storage.hermitian.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	result = lapack.zlanhe_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		work_ptr,
		c.size_t(len(norm_c)), // norm string length
		c.size_t(len(uplo_c)), // uplo string length
	)

	return result, true
}

// ===================================================================================
// HERMITIAN PACKED MATRIX NORMS
// ===================================================================================

// Compute norm of Hermitian packed matrix (c64)
m_norm_hermitian_packed_c64 :: proc(
	A: ^Matrix(complex64),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	result = lapack.clanhp_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		work_ptr,
		c.size_t(len(norm_c)), // norm string length
		c.size_t(len(uplo_c)), // uplo string length
	)

	return result, true
}

// Compute norm of Hermitian packed matrix (c128)
m_norm_hermitian_packed_c128 :: proc(
	A: ^Matrix(complex128),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	result = lapack.zlanhp_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		work_ptr,
		c.size_t(len(norm_c)), // norm string length
		c.size_t(len(uplo_c)), // uplo string length
	)

	return result, true
}

// ===================================================================================
// HESSENBERG MATRIX NORMS
// ===================================================================================

// Compute norm of Hessenberg matrix (c64)
m_norm_hessenberg_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) where T == f32 ||
	T == complex64 {
	// Validate matrix (Hessenberg matrices are stored as general matrices)
	assert(A.format == .General, "Hessenberg matrix must be stored in general format")
	assert(A.rows == A.cols, "Hessenberg matrix must be square")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f32 {
		result = lapack.slanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
		)
	} else when T == complex64 {
		result = lapack.clanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
		)
	}

	return result, true
}

// Compute norm of Hessenberg matrix (f64)
m_norm_hessenberg_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix (Hessenberg matrices are stored as general matrices)
	assert(A.format == .General, "Hessenberg matrix must be stored in general format")
	assert(A.rows == A.cols, "Hessenberg matrix must be square")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f64 {
		result = lapack.dlanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
		)
	} else when T == complex128 {
		result = lapack.zlanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
		)
	}

	return result, true
}

// ===================================================================================
// SYMMETRIC TRIDIAGONAL MATRIX NORMS
// ===================================================================================

// Compute norm of symmetric tridiagonal matrix (f64)
m_norm_symmetric_tridiagonal :: proc(
	D: []$T, // Main diagonal
	E: []T, // Off-diagonal elements
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: T,
	success: bool,
) where is_float(T) {
	// Validate input
	assert(len(D) != 0, "Main diagonal array cannot be empty")
	assert(
		len(E) == len(D) - 1 || len(E) == len(D),
		"Off-diagonal array must have length n-1 or n",
	)

	n := Blas_Int(len(D))
	norm_c := norm_to_cstring(norm)

	when T == f32 {
		result = lapack.slanst_(
			norm_c,
			&n,
			raw_data(D),
			raw_data(E),
			c.size_t(len(norm_c)), // norm string length
		)
	} else when T == f64 {
		result = lapack.dlanst_(
			norm_c,
			&n,
			raw_data(D),
			raw_data(E),
			c.size_t(len(norm_c)), // norm string length
		)
	}

	return result, true
}

// ===================================================================================
// SYMMETRIC MATRIX NORMS
// ===================================================================================

// Compute norm of symmetric matrix (c64)
m_norm_symmetric_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .Symmetric, "Matrix must be in symmetric format")
	assert(A.rows == A.cols, "Symmetric matrix must be square")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.symmetric.uplo != nil {
		uplo_c = A.storage.symmetric.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f32 {
		result = lapack.slansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
		)
	} else when T == complex64 {
		result = lapack.clansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
		)
	}

	return result, true
}

// Compute norm of symmetric matrix (f64)
m_norm_symmetric_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .Symmetric, "Matrix must be in symmetric format")
	assert(A.rows == A.cols, "Symmetric matrix must be square")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.symmetric.uplo != nil {
		uplo_c = A.storage.symmetric.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	work_len := len(work) if len(work) > 0 else 0

	when T == f64 {
		result = lapack.dlansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
		)
	} else when T == complex128 {
		result = lapack.zlansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
		)
	}

	return result, true
}

// ===================================================================================
// TRIANGULAR BANDED MATRIX NORMS
// ===================================================================================

// Compute norm of triangular banded matrix (c64)
m_norm_triangular_banded_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	upper: bool = true,
	unit_diagonal: bool = false,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .Triangular, "Matrix must be in triangular format")

	n := A.rows
	k := A.storage.banded.ku // bandwidth
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U" if upper else "L"
	diag_c := "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.triangular.uplo != nil {
		uplo_c = A.storage.triangular.uplo
	}
	if A.storage.triangular.diag != nil {
		diag_c = A.storage.triangular.diag
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f32 {

	} else when T == complex64 {
		result = lapack.clantb_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			&k,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	}

	return result, true
}

// Compute norm of triangular banded matrix (f64)
m_norm_triangular_banded_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	upper: bool = true,
	unit_diagonal: bool = false,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .Triangular, "Matrix must be in triangular format")

	n := A.rows
	k := A.storage.banded.ku // bandwidth
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U" if upper else "L"
	diag_c := "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.triangular.uplo != nil {
		uplo_c = A.storage.triangular.uplo
	}
	if A.storage.triangular.diag != nil {
		diag_c = A.storage.triangular.diag
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f64 {
		result = lapack.dlantb_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			&k,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	} else when T == complex128 {
		result = lapack.zlantb_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			&k,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	}

	return result, true
}

// ===================================================================================
// TRIANGULAR PACKED MATRIX NORMS
// ===================================================================================

// Compute norm of triangular packed matrix (c64)
m_norm_triangular_packed_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	upper: bool = true,
	unit_diagonal: bool = false,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U" if upper else "L"
	diag_c := "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f32 {
		result = lapack.slantp_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			raw_data(A.data),
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	} else when T == complex64 {
		result = lapack.clantp_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			raw_data(A.data),
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	}

	return result, true
}

// Compute norm of triangular packed matrix (f64)
m_norm_triangular_packed_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	upper: bool = true,
	unit_diagonal: bool = false,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U" if upper else "L"
	diag_c := "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, n)
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f64 {
		result = lapack.dlantp_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			raw_data(A.data),
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	} else when T == complex128 {
		result = lapack.zlantp_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			raw_data(A.data),
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	}

	return result, true
}

// ===================================================================================
// TRIANGULAR MATRIX NORMS (GENERAL STORAGE)
// ===================================================================================

// Compute norm of triangular matrix in general storage (c64)
m_norm_triangular_general_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	upper: bool = true,
	unit_diagonal: bool = false,
	allocator := context.allocator,
) -> (
	result: f32,
	success: bool,
) where T == f32 ||
	T == complex64 {
	// Validate matrix (can be stored as general or triangular format)
	if A.format != .General && A.format != .Triangular {
		panic("Matrix must be in general or triangular format")
	}

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U" if upper else "L"
	diag_c := "U" if unit_diagonal else "N"

	// Override with stored values if available and format is triangular
	if A.format == .Triangular {
		if A.storage.triangular.uplo != nil {
			uplo_c = A.storage.triangular.uplo
		}
		if A.storage.triangular.diag != nil {
			diag_c = A.storage.triangular.diag
		}
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f32
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f32, max(m, n))
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil
	when T == f32 {
		result = lapack.slantr_(
			norm_c,
			uplo_c,
			diag_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	} else when T == complex64 {
		result = lapack.clantr_(
			norm_c,
			uplo_c,
			diag_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	}


	return result, true
}

// Compute norm of triangular matrix in general storage (f64)
m_norm_triangular_general_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm = .FrobeniusNorm,
	upper: bool = true,
	unit_diagonal: bool = false,
	allocator := context.allocator,
) -> (
	result: f64,
	success: bool,
) where T == f64 ||
	T == complex128 {
	// Validate matrix (can be stored as general or triangular format)
	if A.format != .General && A.format != .Triangular {
		panic("Matrix must be in general or triangular format")
	}

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U" if upper else "L"
	diag_c := "U" if unit_diagonal else "N"

	// Override with stored values if available and format is triangular
	if A.format == .Triangular {
		if A.storage.triangular.uplo != nil {
			uplo_c = A.storage.triangular.uplo
		}
		if A.storage.triangular.diag != nil {
			diag_c = A.storage.triangular.diag
		}
	}

	// Allocate workspace if needed (for 1-norm or infinity-norm)
	work: []f64
	defer if work != nil {delete(work)}
	if norm == .OneNorm || norm == .InfinityNorm {
		work = make([]f64, max(m, n))
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f64 {
		result = lapack.dlantr_(
			norm_c,
			uplo_c,
			diag_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	} else when T == complex128 {
		result = lapack.zlantr_(
			norm_c,
			uplo_c,
			diag_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)), // norm string length
			c.size_t(len(uplo_c)), // uplo string length
			c.size_t(len(diag_c)), // diag string length
		)
	}

	return result, true
}

// ===================================================================================
// MATRIX ROW PERMUTATION
// ===================================================================================

// Permute rows of matrix (c64)
m_permute_row :: proc(
	A: ^Matrix($T),
	permutation: []int,
	forward: bool = true,
	allocator := context.allocator,
) -> (
	success: bool,
) where is_float(T) ||
	is_complex(T) {
	// Validate input
	assert(len(permutation) == int(A.rows), "Permutation array must match number of rows")

	m := A.rows
	n := A.cols
	ldx := A.ld
	forwrd: Blas_Int = 1 if forward else 0

	// Convert permutation to LAPACK format (1-based indexing)
	K := make([]Blas_Int, len(permutation))
	for p, i in permutation {
		K[i] = Blas_Int(p + 1) // Convert to 1-based
	}

	when T == f32 {
		lapack.slapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == f64 {
		lapack.dlapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex64 {
		lapack.clapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex128 {
		lapack.zlapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	}

	return true
}

// ===================================================================================
// MATRIX COLUMN PERMUTATION
// ===================================================================================

// Permute columns of matrix (c64)
m_permute_columns :: proc(
	A: ^Matrix($T),
	permutation: []int,
	forward: bool = true,
	allocator := context.allocator,
) where is_float(T) ||
	is_complex(T) {
	// Validate input
	assert(len(permutation) == A.cols, "Permutation array must match number of columns")

	m := A.rows
	n := A.cols
	ldx := A.ld
	forwrd: Blas_Int = 1 if forward else 0

	// Convert permutation to LAPACK format (1-based indexing)
	K := make([]Blas_Int, len(permutation))
	for p, i in permutation {
		K[i] = Blas_Int(p + 1) // Convert to 1-based
	}

	when T == f32 {
		lapack.slapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == f64 {
		lapack.dlapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex64 {
		lapack.clapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex128 {
		lapack.zlapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))

	}
	return true
}

// ===================================================================================
// EUCLIDEAN NORM UTILITIES
// ===================================================================================

// Compute Euclidean norm of 2D vector (f64)
euclidean_norm_2d :: proc(x, y: $T) -> T where is_float(T) {
	x_val := x
	y_val := y
	when T == f32 {
		return f32(lapack.slapy2_(&x_val, &y_val))
	} else when T == f64 {
		return lapack.dlapy2_(&x_val, &y_val)
	}
	unreachable()
}

// Compute Euclidean norm of 3D vector (f64)
euclidean_norm_3d :: proc(x, y, z: $T) -> T where is_float(T) {
	x_val := x
	y_val := y
	z_val := z
	when T == f32 {
		return f32(lapack.slapy3_(&x_val, &y_val, &z_val))
	} else when T == f64 {
		return lapack.dlapy3_(&x_val, &y_val, &z_val)
	}
	unreachable()
}

// ===================================================================================
// REAL-COMPLEX MATRIX MULTIPLICATION
// ===================================================================================

// Multiply real matrix by complex matrix (c64)
m_multiply_real32_complex64 :: proc(
	A: ^Matrix(f32), // Real matrix (m x k)
	B: ^Matrix(complex64), // Complex matrix (k x n)
	C: ^Matrix(complex64), // Result matrix (m x n)
	allocator := context.allocator,
) {
	assert(A.cols == B.rows, "Matrix dimensions incompatible for multiplication")
	assert(C.rows == A.rows && C.cols == B.cols, "Result matrix has incorrect dimensions")

	m := A.rows
	n := B.cols
	lda := A.ld
	ldb := B.ld
	ldc := C.ld

	// Allocate workspace for real part operations
	rwork := make([]f32, m * n)
	defer delete(rwork)

	lapack.clarcm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}

// Multiply real matrix by complex matrix (c128)
m_multiply_real64_complex128 :: proc(
	A: ^Matrix(f64), // Real matrix (m x k)
	B: ^Matrix(complex128), // Complex matrix (k x n)
	C: ^Matrix(complex128), // Result matrix (m x n)
	allocator := context.allocator,
) {
	assert(A.cols == B.rows, "Matrix dimensions incompatible for multiplication")
	assert(C.rows == A.rows && C.cols == B.cols, "Result matrix has incorrect dimensions")

	m := A.rows
	n := B.cols
	lda := A.ld
	ldb := B.ld
	ldc := C.ld

	// Allocate workspace for real part operations
	rwork := make([]f64, m * n)
	defer delete(rwork)

	lapack.zlarcm_(
		&m,
		&n,
		raw_data(A.data),
		&lda,
		raw_data(B.data),
		&ldb,
		raw_data(C.data),
		&ldc,
		raw_data(rwork),
	)
}
