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
norm_banded :: proc {
	norm_banded_f32_c64,
	norm_banded_f64_c128,
}

// Compute general matrix norm proc group
norm_general :: proc {
	norm_general_f32_c64,
	norm_general_f64_c128,
}

// Compute tridiagonal matrix norm proc group
norm_tridiagonal :: proc {
	norm_tridiagonal_f32_c64,
	norm_tridiagonal_f64_c128,
}

// Compute Hermitian banded matrix norm proc group
norm_banded_hermitian :: proc {
	norm_banded_hermitian_c64,
	norm_banded_hermitian_c128,
}

// Compute Hermitian matrix norm proc group
norm_hermitian :: proc {
	norm_hermitian_c64,
	norm_hermitian_c128,
}

// Compute Hermitian packed matrix norm proc group
norm_hermitian_packed :: proc {
	norm_hermitian_packed_c64,
	norm_hermitian_packed_c128,
}

// Compute Hessenberg matrix norm proc group
norm_hessenberg :: proc {
	norm_hessenberg_f32_c64,
	norm_hessenberg_f64_c128,
}

// Compute symmetric tridiagonal matrix norm proc group
// norm_symmetric_tridiagonal

// Compute symmetric matrix norm proc group
norm_symmetric :: proc {
	norm_symmetric_f32_c64,
	norm_symmetric_f64_c128,
}

// Compute triangular banded matrix norm proc group
norm_triangular_banded :: proc {
	norm_triangular_banded_f64_c128,
	norm_triangular_banded_f32_c64,
}

// Compute triangular packed matrix norm proc group
norm_triangular_packed :: proc {
	norm_triangular_packed_f64_c128,
	norm_triangular_packed_f32_c64,
}

// Compute triangular matrix norm proc group (general storage)
norm_triangular_general :: proc {
	norm_triangular_general_f64_c128,
	norm_triangular_general_f32_c64,
}

// Matrix row permutation
// permute_rows

// Matrix column permutation
// m_permute_columns

// Euclidean norm proc group
euclidean_norm :: proc {
	euclidean_norm_2d,
	euclidean_norm_3d,
}

// Real-complex matrix multiplication proc group
multiply_real_complex :: proc {
	multiply_real32_complex64,
	multiply_real64_complex128,
}

// ===================================================================================
// BANDED MATRIX NORMS
// ===================================================================================

// Query workspace for banded matrix norm
query_workspace_norm_banded :: proc(
	$T: typeid,
	n: int,
	norm: MatrixNorm,
) -> (
	work: int,
) where is_float(T) ||
	is_complex(T) {
	// For complex types, the work array is real-valued
	// For f32/complex64, work is []f32
	// For f64/complex128, work is []f64
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of banded matrix (f32/c64)
norm_banded_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f32 = nil,
) -> (
	result: f32,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .Banded, "Matrix must be in banded format")

	n := A.rows
	kl := A.storage.banded.kl
	ku := A.storage.banded.ku
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
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

	return result
}

// Compute norm of banded matrix (f64/c128)
norm_banded_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f64 = nil,
) -> (
	result: f64,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .Banded, "Matrix must be in banded format")

	n := A.rows
	kl := A.storage.banded.kl
	ku := A.storage.banded.ku
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
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

	return result
}

// ===================================================================================
// GENERAL MATRIX NORMS
// ===================================================================================

// Query workspace for general matrix norm
query_workspace_norm_general :: proc(
	$T: typeid,
	m: int,
	n: int,
	norm: MatrixNorm,
) -> (
	work: int,
) where is_float(T) ||
	is_complex(T) {
	// For complex types, the work array is real-valued
	if norm == .OneNorm {
		return m
	} else if norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of general matrix (f32/c64)
norm_general_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f32 = nil,
) -> (
	result: f32,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .General, "Matrix must be in general format")

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Validate workspace
	if norm == .OneNorm {
		assert(len(work) >= int(m), "Workspace too small for 1-norm")
	} else if norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small for infinity-norm")
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
		result = lapack.clange_(
			norm_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(work_len),
		)
	}

	return result
}

// Compute norm of general matrix (f64/c128)
norm_general_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f64 = nil,
) -> (
	result: f64,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .General, "Matrix must be in general format")

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Validate workspace
	if norm == .OneNorm {
		assert(len(work) >= int(m), "Workspace too small for 1-norm")
	} else if norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small for infinity-norm")
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

	return result
}

// ===================================================================================
// TRIDIAGONAL MATRIX NORMS
// ===================================================================================

// Compute norm of tridiagonal matrix (f32/c64)
norm_tridiagonal_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
) -> (
	result: f32,
) where T == f32 ||
	T == complex64 {
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
		result = lapack.slangt_(norm_c, &n, dl_ptr, d_ptr, du_ptr, c.size_t(len(norm_c)))
	} else when T == complex64 {
		result = lapack.clangt_(norm_c, &n, dl_ptr, d_ptr, du_ptr, c.size_t(len(norm_c)))
	}

	return result
}

// Compute norm of tridiagonal matrix (f64/c128)
norm_tridiagonal_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
) -> (
	result: f64,
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
		result = lapack.dlangt_(norm_c, &n, dl_ptr, d_ptr, du_ptr, c.size_t(len(norm_c)))
	} else when T == complex128 {
		result = lapack.zlangt_(norm_c, &n, dl_ptr, d_ptr, du_ptr, c.size_t(len(norm_c)))
	}

	return result
}

// ===================================================================================
// HERMITIAN BANDED MATRIX NORMS
// ===================================================================================

// Query workspace for Hermitian banded matrix norm
query_workspace_norm_banded_hermitian :: proc(n: int, norm: MatrixNorm) -> (work: int) {
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of Hermitian banded matrix (c64)
norm_banded_hermitian_c64 :: proc(
	A: ^Matrix(complex64),
	norm: MatrixNorm,
	work: []f32 = nil,
) -> (
	result: f32,
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

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	result = lapack.clanhb_(
		norm_c,
		uplo_c,
		&n,
		&k,
		raw_data(A.data),
		&ldab,
		work_ptr,
		c.size_t(len(norm_c)),
		c.size_t(len(uplo_c)),
	)

	return result
}

// Compute norm of Hermitian banded matrix (c128)
norm_banded_hermitian_c128 :: proc(
	A: ^Matrix(complex128),
	norm: MatrixNorm,
	work: []f64 = nil,
) -> (
	result: f64,
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

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	result = lapack.zlanhb_(
		norm_c,
		uplo_c,
		&n,
		&k,
		raw_data(A.data),
		&ldab,
		work_ptr,
		c.size_t(len(norm_c)),
		c.size_t(len(uplo_c)),
	)

	return result
}

// ===================================================================================
// HERMITIAN MATRIX NORMS
// ===================================================================================

// Query workspace for Hermitian matrix norm
query_workspace_norm_hermitian :: proc(n: int, norm: MatrixNorm) -> (work: int) {
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of Hermitian matrix (c64)
norm_hermitian_c64 :: proc(
	A: ^Matrix(complex64),
	norm: MatrixNorm,
	work: []f32 = nil,
) -> (
	result: f32,
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

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	result = lapack.clanhe_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		work_ptr,
		c.size_t(len(norm_c)),
		c.size_t(len(uplo_c)),
	)

	return result
}

// Compute norm of Hermitian matrix (c128)
norm_hermitian_c128 :: proc(
	A: ^Matrix(complex128),
	norm: MatrixNorm,
	work: []f64 = nil,
) -> (
	result: f64,
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

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	result = lapack.zlanhe_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		&lda,
		work_ptr,
		c.size_t(len(norm_c)),
		c.size_t(len(uplo_c)),
	)

	return result
}

// ===================================================================================
// HERMITIAN PACKED MATRIX NORMS
// ===================================================================================

// Query workspace for Hermitian packed matrix norm
query_workspace_norm_hermitian_packed :: proc(n: int, norm: MatrixNorm) -> (work: int) {
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of Hermitian packed matrix (c64)
norm_hermitian_packed_c64 :: proc(
	A: ^Matrix(complex64),
	norm: MatrixNorm,
	work: []f32 = nil,
) -> (
	result: f32,
) {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	result = lapack.clanhp_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		work_ptr,
		c.size_t(len(norm_c)),
		c.size_t(len(uplo_c)),
	)

	return result
}

// Compute norm of Hermitian packed matrix (c128)
norm_hermitian_packed_c128 :: proc(
	A: ^Matrix(complex128),
	norm: MatrixNorm,
	work: []f64 = nil,
) -> (
	result: f64,
) {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c: cstring = "U"
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	result = lapack.zlanhp_(
		norm_c,
		uplo_c,
		&n,
		raw_data(A.data),
		work_ptr,
		c.size_t(len(norm_c)),
		c.size_t(len(uplo_c)),
	)

	return result
}

// ===================================================================================
// HESSENBERG MATRIX NORMS
// ===================================================================================

// Query workspace for Hessenberg matrix norm
query_workspace_norm_hessenberg :: proc(
	$T: typeid,
	n: int,
	norm: MatrixNorm,
) -> (
	work: int,
) where is_float(T) ||
	is_complex(T) {
	// For complex types, the work array is real-valued
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of Hessenberg matrix (f32/c64)
norm_hessenberg_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f32 = nil,
) -> (
	result: f32,
) where T == f32 ||
	T == complex64 {
	// Validate matrix (Hessenberg matrices are stored as general matrices)
	assert(A.format == .General, "Hessenberg matrix must be stored in general format")
	assert(A.rows == A.cols, "Hessenberg matrix must be square")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f32 {
		result = lapack.slanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
		)
	} else when T == complex64 {
		result = lapack.clanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
		)
	}

	return result
}

// Compute norm of Hessenberg matrix (f64/c128)
norm_hessenberg_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f64 = nil,
) -> (
	result: f64,
) where T == f64 ||
	T == complex128 {
	// Validate matrix (Hessenberg matrices are stored as general matrices)
	assert(A.format == .General, "Hessenberg matrix must be stored in general format")
	assert(A.rows == A.cols, "Hessenberg matrix must be square")

	n := A.rows
	lda := A.ld
	norm_c := norm_to_cstring(norm)

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f64 {
		result = lapack.dlanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
		)
	} else when T == complex128 {
		result = lapack.zlanhs_(
			norm_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
		)
	}

	return result
}

// ===================================================================================
// SYMMETRIC TRIDIAGONAL MATRIX NORMS
// ===================================================================================

// Compute norm of symmetric tridiagonal matrix
norm_symmetric_tridiagonal :: proc(
	D: []$T, // Main diagonal
	E: []T, // Off-diagonal elements
	norm: MatrixNorm,
) -> (
	result: T,
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
		result = lapack.slanst_(norm_c, &n, raw_data(D), raw_data(E), c.size_t(len(norm_c)))
	} else when T == f64 {
		result = lapack.dlanst_(norm_c, &n, raw_data(D), raw_data(E), c.size_t(len(norm_c)))
	}

	return result
}

// ===================================================================================
// SYMMETRIC MATRIX NORMS
// ===================================================================================

// Query workspace for symmetric matrix norm
query_workspace_norm_symmetric :: proc(
	$T: typeid,
	n: int,
	norm: MatrixNorm,
) -> (
	work: int,
) where is_float(T) ||
	is_complex(T) {
	// For complex types, the work array is real-valued
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of symmetric matrix (f32/c64)
norm_symmetric_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f32 = nil,
) -> (
	result: f32,
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

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f32 {
		result = lapack.slansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
		)
	} else when T == complex64 {
		result = lapack.clansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
		)
	}

	return result
}

// Compute norm of symmetric matrix (f64/c128)
norm_symmetric_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f64 = nil,
) -> (
	result: f64,
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

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f64 {
		result = lapack.dlansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
		)
	} else when T == complex128 {
		result = lapack.zlansy_(
			norm_c,
			uplo_c,
			&n,
			raw_data(A.data),
			&lda,
			work_ptr,
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
		)
	}

	return result
}

// ===================================================================================
// TRIANGULAR BANDED MATRIX NORMS
// ===================================================================================

// Query workspace for triangular banded matrix norm
query_workspace_norm_triangular_banded :: proc(
	$T: typeid,
	n: int,
	norm: MatrixNorm,
) -> (
	work: int,
) where is_float(T) ||
	is_complex(T) {
	// For complex types, the work array is real-valued
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of triangular banded matrix (f32/c64)
norm_triangular_banded_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f32 = nil,
	region: MatrixRegion = .Upper,
	unit_diagonal: bool = false,
) -> (
	result: f32,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .Triangular, "Matrix must be in triangular format")
	assert(region != .Full, "Triangular matrix cannot use Full region")

	n := A.rows
	k := A.storage.banded.ku // bandwidth
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)
	uplo_c := matrix_region_to_cstring(region)
	diag_c: cstring = "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.triangular.uplo != nil {
		uplo_c = A.storage.triangular.uplo
	}
	if A.storage.triangular.diag != nil {
		diag_c = A.storage.triangular.diag
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
	}

	work_ptr := raw_data(work) if len(work) > 0 else nil

	when T == f32 {
		result = lapack.slantb_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			&k,
			raw_data(A.data),
			&ldab,
			work_ptr,
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	}

	return result
}

// Compute norm of triangular banded matrix (f64/c128)
norm_triangular_banded_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f64 = nil,
	region: MatrixRegion = .Upper,
	unit_diagonal: bool = false,
) -> (
	result: f64,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .Triangular, "Matrix must be in triangular format")
	assert(region != .Full, "Triangular matrix cannot use Full region")

	n := A.rows
	k := A.storage.banded.ku // bandwidth
	ldab := A.storage.banded.ldab
	norm_c := norm_to_cstring(norm)
	uplo_c := matrix_region_to_cstring(region)
	diag_c: cstring = "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.triangular.uplo != nil {
		uplo_c = A.storage.triangular.uplo
	}
	if A.storage.triangular.diag != nil {
		diag_c = A.storage.triangular.diag
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	}

	return result
}

// ===================================================================================
// TRIANGULAR PACKED MATRIX NORMS
// ===================================================================================

// Query workspace for triangular packed matrix norm
query_workspace_norm_triangular_packed :: proc(
	$T: typeid,
	n: int,
	norm: MatrixNorm,
) -> (
	work: int,
) where is_float(T) ||
	is_complex(T) {
	// For complex types, the work array is real-valued
	if norm == .OneNorm || norm == .InfinityNorm {
		return n
	}
	return 0
}

// Compute norm of triangular packed matrix (f32/c64)
norm_triangular_packed_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f32 = nil,
	region: MatrixRegion = .Upper,
	unit_diagonal: bool = false,
) -> (
	result: f32,
) where T == f32 ||
	T == complex64 {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")
	assert(region != .Full, "Triangular matrix cannot use Full region")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c := matrix_region_to_cstring(region)
	diag_c: cstring = "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	} else when T == complex64 {
		result = lapack.clantp_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			raw_data(A.data),
			work_ptr,
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	}

	return result
}

// Compute norm of triangular packed matrix (f64/c128)
norm_triangular_packed_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f64 = nil,
	region: MatrixRegion = .Upper,
	unit_diagonal: bool = false,
) -> (
	result: f64,
) where T == f64 ||
	T == complex128 {
	// Validate matrix
	assert(A.format == .Packed, "Matrix must be in packed format")
	assert(region != .Full, "Triangular matrix cannot use Full region")

	n := A.rows
	norm_c := norm_to_cstring(norm)
	uplo_c := matrix_region_to_cstring(region)
	diag_c: cstring = "U" if unit_diagonal else "N"

	// Override with stored values if available
	if A.storage.packed.uplo != nil {
		uplo_c = A.storage.packed.uplo
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= int(n), "Workspace too small")
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	} else when T == complex128 {
		result = lapack.zlantp_(
			norm_c,
			uplo_c,
			diag_c,
			&n,
			raw_data(A.data),
			work_ptr,
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	}

	return result
}

// ===================================================================================
// TRIANGULAR MATRIX NORMS (GENERAL STORAGE)
// ===================================================================================

// Query workspace for triangular matrix norm (general storage)
query_workspace_norm_triangular_general :: proc(
	$T: typeid,
	m: int,
	n: int,
	norm: MatrixNorm,
) -> (
	work: int,
) where is_float(T) ||
	is_complex(T) {
	// For complex types, the work array is real-valued
	if norm == .OneNorm || norm == .InfinityNorm {
		return max(m, n)
	}
	return 0
}

// Compute norm of triangular matrix in general storage (f32/c64)
norm_triangular_general_f32_c64 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f32 = nil,
	region: MatrixRegion = .Upper,
	unit_diagonal: bool = false,
) -> (
	result: f32,
) where T == f32 ||
	T == complex64 {
	// Validate matrix (can be stored as general or triangular format)
	assert(
		A.format == .General || A.format == .Triangular,
		"Matrix must be in general or triangular format",
	)
	assert(region != .Full, "Triangular matrix cannot use Full region")

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c := matrix_region_to_cstring(region)
	diag_c: cstring = "U" if unit_diagonal else "N"

	// Override with stored values if available and format is triangular
	if A.format == .Triangular {
		if A.storage.triangular.uplo != nil {
			uplo_c = A.storage.triangular.uplo
		}
		if A.storage.triangular.diag != nil {
			diag_c = A.storage.triangular.diag
		}
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= max(int(m), int(n)), "Workspace too small")
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	}

	return result
}

// Compute norm of triangular matrix in general storage (f64/c128)
norm_triangular_general_f64_c128 :: proc(
	A: ^Matrix($T),
	norm: MatrixNorm,
	work: []f64 = nil,
	region: MatrixRegion = .Upper,
	unit_diagonal: bool = false,
) -> (
	result: f64,
) where T == f64 ||
	T == complex128 {
	// Validate matrix (can be stored as general or triangular format)
	assert(
		A.format == .General || A.format == .Triangular,
		"Matrix must be in general or triangular format",
	)
	assert(region != .Full, "Triangular matrix cannot use Full region")

	m := A.rows
	n := A.cols
	lda := A.ld
	norm_c := norm_to_cstring(norm)
	uplo_c := matrix_region_to_cstring(region)
	diag_c: cstring = "U" if unit_diagonal else "N"

	// Override with stored values if available and format is triangular
	if A.format == .Triangular {
		if A.storage.triangular.uplo != nil {
			uplo_c = A.storage.triangular.uplo
		}
		if A.storage.triangular.diag != nil {
			diag_c = A.storage.triangular.diag
		}
	}

	// Validate workspace
	if norm == .OneNorm || norm == .InfinityNorm {
		assert(len(work) >= max(int(m), int(n)), "Workspace too small")
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
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
			c.size_t(len(norm_c)),
			c.size_t(len(uplo_c)),
			c.size_t(len(diag_c)),
		)
	}

	return result
}

// ===================================================================================
// MATRIX ROW PERMUTATION
// ===================================================================================

// Query result sizes for row permutation
query_result_sizes_permute_rows :: proc(n_rows: int) -> (k_size: int) {
	return n_rows
}

// Permute rows of matrix
permute_rows :: proc(
	A: ^Matrix($T),
	K: []Blas_Int, // Pre-allocated permutation array (1-based indexing)
	forward: bool = true,
) where is_float(T) || is_complex(T) {
	// Validate input
	assert(len(K) >= int(A.rows), "Permutation array too small")

	m := A.rows
	n := A.cols
	ldx := A.ld
	forwrd: Blas_Int = 1 if forward else 0

	when T == f32 {
		lapack.slapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == f64 {
		lapack.dlapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex64 {
		lapack.clapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex128 {
		lapack.zlapmr_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	}
}

// ===================================================================================
// MATRIX COLUMN PERMUTATION
// ===================================================================================

// Query result sizes for column permutation
query_result_sizes_permute_columns :: proc(n_cols: int) -> (k_size: int) {
	return n_cols
}

// Permute columns of matrix
permute_columns :: proc(
	A: ^Matrix($T),
	K: []Blas_Int, // Pre-allocated permutation array (1-based indexing)
	forward: bool = true,
) where is_float(T) || is_complex(T) {
	// Validate input
	assert(len(K) >= int(A.cols), "Permutation array too small")

	m := A.rows
	n := A.cols
	ldx := A.ld
	forwrd: Blas_Int = 1 if forward else 0

	when T == f32 {
		lapack.slapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == f64 {
		lapack.dlapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex64 {
		lapack.clapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	} else when T == complex128 {
		lapack.zlapmt_(&forwrd, &m, &n, raw_data(A.data), &ldx, raw_data(K))
	}
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

// Query workspace for real-complex matrix multiplication
query_workspace_multiply_real_complex :: proc(m: int, n: int) -> (rwork: int) {
	return m * n
}

// Multiply real matrix by complex matrix (f32/complex64)
multiply_real32_complex64 :: proc(
	A: ^Matrix(f32), // Real matrix (m x k)
	B: ^Matrix(complex64), // Complex matrix (k x n)
	C: ^Matrix(complex64), // Result matrix (m x n)
	rwork: []f32, // Pre-allocated workspace
) {
	assert(A.cols == B.rows, "Matrix dimensions incompatible for multiplication")
	assert(C.rows == A.rows && C.cols == B.cols, "Result matrix has incorrect dimensions")
	assert(len(rwork) >= int(A.rows * B.cols), "Workspace too small")

	m := A.rows
	n := B.cols
	lda := A.ld
	ldb := B.ld
	ldc := C.ld

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

// Multiply real matrix by complex matrix (f64/complex128)
multiply_real64_complex128 :: proc(
	A: ^Matrix(f64), // Real matrix (m x k)
	B: ^Matrix(complex128), // Complex matrix (k x n)
	C: ^Matrix(complex128), // Result matrix (m x n)
	rwork: []f64, // Pre-allocated workspace
) {
	assert(A.cols == B.rows, "Matrix dimensions incompatible for multiplication")
	assert(C.rows == A.rows && C.cols == B.cols, "Result matrix has incorrect dimensions")
	assert(len(rwork) >= int(A.rows * B.cols), "Workspace too small")

	m := A.rows
	n := B.cols
	lda := A.ld
	ldb := B.ld
	ldc := C.ld

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
