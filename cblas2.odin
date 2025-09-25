package openblas

import blas "./c"
import "base:intrinsics"

// ==== BLAS Level 2: Matrix-vector operations ====

// ===================================================================================
// MATRIX-VECTOR MULTIPLICATION
// ===================================================================================

// Matrix-vector multiplication: y = alpha*A*x + beta*y
// Performs general matrix-vector multiplication with scaling.
// A can optionally be transposed or conjugate-transposed.
// Result is stored in y, which is scaled by beta before adding the product.
// Supported types: f32, f64, complex64, complex128
mv_mul :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	y: ^Vector(T),
	alpha: T,
	beta: T,
	trans: blas.CBLAS_TRANSPOSE,
) where is_float(T) ||
	is_complex(T) {
	m, n := i64(A.rows), i64(A.cols)

	// Adjust dimensions based on transpose
	x_len, y_len := n, m
	if trans != .NoTrans {
		x_len, y_len = m, n
	}

	assert(x.size == x_len, "Input vector dimension must match matrix")
	assert(y.size == y_len, "Output vector dimension must match matrix")

	lda := i64(A.ld)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_sgemv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == f64 {
		blas.cblas_dgemv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == complex64 {
		alpha := alpha
		beta := beta
		blas.cblas_cgemv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == complex128 {
		alpha := alpha
		beta := beta
		blas.cblas_zgemv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	}
}

// ===================================================================================
// RANK-1 UPDATES (OUTER PRODUCT)
// ===================================================================================

// Rank-1 update: A = alpha*x*y^T + A
// Performs the outer product of vectors x and y, scaled by alpha, and adds to matrix A.
// For real types, computes standard outer product.
// For complex types, this is the unconjugated version (use mv_gerc for conjugated).
// The matrix A is modified in-place.
// Supported types: f32, f64, complex64, complex128
mv_ger :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	A: ^Matrix(T),
	alpha: T,
) where is_float(T) ||
	is_complex(T) {
	assert(x.size == A.rows, "x length must match matrix rows")
	assert(y.size == A.cols, "y length must match matrix columns")

	m := i64(A.rows)
	n := i64(A.cols)
	incx := i64(x.incr)
	incy := i64(y.incr)
	lda := i64(A.ld)
	alpha := alpha

	when T == f32 {
		blas.cblas_sger(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == f64 {
		blas.cblas_dger(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == complex64 {
		alpha := alpha
		blas.cblas_cgeru(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == complex128 {
		alpha := alpha
		blas.cblas_zgeru(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	}
}

// Alias for complex unconjugated rank-1 update (same as mv_ger for complex types)
mv_geru :: mv_ger

// Complex rank-1 update (conjugated): A = alpha*x*conj(y)^T + A
// Performs conjugated outer product for complex vectors.
// The y vector is conjugated before the outer product.
// For real types, behaves identically to mv_ger (conjugation is identity).
// The matrix A is modified in-place.
// Supported types: f32, f64, complex64, complex128
mv_ger_conj :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	A: ^Matrix(T),
	alpha: T,
) where is_float(T) ||
	is_complex(T) {
	assert(x.size == A.rows, "x length must match matrix rows")
	assert(y.size == A.cols, "y length must match matrix columns")

	m := i64(A.rows)
	n := i64(A.cols)
	incx := i64(x.incr)
	incy := i64(y.incr)
	lda := i64(A.ld)
	alpha := alpha

	when T == f32 {
		// For real types, conjugate is identity, so same as ger
		blas.cblas_sger(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == f64 {
		// For real types, conjugate is identity, so same as ger
		blas.cblas_dger(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == complex64 {
		alpha := alpha
		blas.cblas_cgerc(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == complex128 {
		alpha := alpha
		blas.cblas_zgerc(
			blas.CBLAS_ORDER.ColMajor,
			m,
			n,
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	}
}

// ===================================================================================
// TRIANGULAR MATRIX-VECTOR OPERATIONS
// ===================================================================================

// Triangular matrix-vector solve: solves A*x = b or A^T*x = b
// Solves a system of linear equations where A is triangular.
// The vector x is modified in-place with the solution.
// Supported types: f32, f64, complex64, complex128
mv_trsv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where is_float(T) ||
	is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector length must match matrix dimension")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_strsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == f64 {
		blas.cblas_dtrsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex64 {
		blas.cblas_ctrsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex128 {
		blas.cblas_ztrsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	}
}

// Triangular matrix-vector multiplication: x = A*x or x = A^T*x
// Multiplies a triangular matrix by a vector.
// The vector x is modified in-place with the result.
// Supported types: f32, f64, complex64, complex128
mv_trmv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where is_float(T) ||
	is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector length must match matrix dimension")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_strmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == f64 {
		blas.cblas_dtrmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex64 {
		blas.cblas_ctrmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex128 {
		blas.cblas_ztrmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	}
}

// ===================================================================================
// SYMMETRIC/HERMITIAN RANK UPDATES
// ===================================================================================

// Symmetric rank-1 update: A = alpha*x*x^T + A
// Updates a symmetric matrix with the outer product of a vector with itself.
// Only the upper or lower triangle of A is referenced and modified.
// For real types only (use mv_her for complex Hermitian update).
// Supported types: f32, f64
mv_syr :: proc(
	x: ^Vector($T),
	A: ^Matrix(T),
	alpha: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector length must match matrix dimension")

	n := i64(A.rows)
	incx := i64(x.incr)
	lda := i64(A.ld)

	when T == f32 {
		blas.cblas_ssyr(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(A.data),
			lda,
		)
	} else when T == f64 {
		blas.cblas_dsyr(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(A.data),
			lda,
		)
	}
}

// Hermitian rank-1 update: A = alpha*x*conj(x)^T + A
// Updates a Hermitian matrix with the outer product of a complex vector with its conjugate.
// Only the upper or lower triangle of A is referenced and modified.
// Alpha must be real for the result to remain Hermitian.
// For complex types only (use mv_syr for real symmetric update).
// Supported types: complex64 (alpha is f32), complex128 (alpha is f64)
mv_her :: proc(
	x: ^Vector($T),
	A: ^Matrix(T),
	alpha: $R,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector length must match matrix dimension")

	n := i64(A.rows)
	incx := i64(x.incr)
	lda := i64(A.ld)

	when T == complex64 {
		static_assert(R == f32, "Alpha must be f32 for complex64")
		blas.cblas_cher(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(A.data),
			lda,
		)
	} else when T == complex128 {
		static_assert(R == f64, "Alpha must be f64 for complex128")
		blas.cblas_zher(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(A.data),
			lda,
		)
	}
}

// Symmetric rank-2 update: A = alpha*(x*y^T + y*x^T) + A
// Updates a symmetric matrix with the sum of two outer products.
// Only the upper or lower triangle of A is referenced and modified.
// For real types only (use mv_her2 for complex Hermitian update).
// Supported types: f32, f64
mv_syr2 :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	A: ^Matrix(T),
	alpha: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "x length must match matrix dimension")
	assert(y.size == A.rows, "y length must match matrix dimension")

	n := i64(A.rows)
	incx := i64(x.incr)
	incy := i64(y.incr)
	lda := i64(A.ld)

	when T == f32 {
		blas.cblas_ssyr2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == f64 {
		blas.cblas_dsyr2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	}
}

// Hermitian rank-2 update: A = alpha*x*conj(y)^T + conj(alpha)*y*conj(x)^T + A
// Updates a Hermitian matrix with the sum of two conjugate outer products.
// Only the upper or lower triangle of A is referenced and modified.
// For complex types only (use mv_syr2 for real symmetric update).
// Supported types: complex64, complex128
mv_her2 :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	A: ^Matrix(T),
	alpha: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "x length must match matrix dimension")
	assert(y.size == A.rows, "y length must match matrix dimension")

	n := i64(A.rows)
	incx := i64(x.incr)
	incy := i64(y.incr)
	lda := i64(A.ld)
	alpha := alpha

	when T == complex64 {
		blas.cblas_cher2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	} else when T == complex128 {
		blas.cblas_zher2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(A.data),
			lda,
		)
	}
}

// ===================================================================================
// BANDED MATRIX-VECTOR OPERATIONS
// ===================================================================================

// General banded matrix-vector multiplication: y = alpha*A*x + beta*y
// A is a banded matrix with KL subdiagonals and KU superdiagonals.
// Efficient storage format for sparse matrices with non-zero elements near the diagonal.
// Supported types: f32, f64, complex64, complex128
mv_gbmv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	y: ^Vector(T),
	kl: int, // Number of subdiagonals
	ku: int, // Number of superdiagonals
	alpha: T,
	beta: T,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
) where is_float(T) || is_complex(T) {
	m, n := i64(A.rows), i64(A.cols)

	// Adjust dimensions based on transpose
	x_len, y_len := n, m
	if trans != .NoTrans {
		x_len, y_len = m, n
	}

	assert(x.size == x_len, "Input vector dimension must match matrix")
	assert(y.size == y_len, "Output vector dimension must match matrix")

	lda := i64(A.ld)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_sgbmv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			i64(kl),
			i64(ku),
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == f64 {
		blas.cblas_dgbmv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			i64(kl),
			i64(ku),
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		blas.cblas_cgbmv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			i64(kl),
			i64(ku),
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		blas.cblas_zgbmv(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			m,
			n,
			i64(kl),
			i64(ku),
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	}
}

// Symmetric banded matrix-vector multiplication: y = alpha*A*x + beta*y
// A is a symmetric banded matrix with K superdiagonals (or subdiagonals).
// Only the upper or lower triangle is stored in banded format.
// For real types only (use mv_hbmv for complex Hermitian).
// Supported types: f32, f64
mv_sbmv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	y: ^Vector(T),
	k: int, // Number of superdiagonals
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector dimensions must match matrix")
	assert(y.size == A.rows, "Output vector dimension must match matrix")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_ssbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			i64(k),
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == f64 {
		blas.cblas_dsbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			i64(k),
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	}
}

// Triangular banded matrix-vector multiplication: x = A*x or x = A^T*x
// A is a triangular banded matrix with K superdiagonals or subdiagonals.
// The vector x is modified in-place with the result.
// Supported types: f32, f64, complex64, complex128
mv_tbmv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	k: int, // Number of superdiagonals (if upper) or subdiagonals (if lower)
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where is_float(T) || is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector length must match matrix dimension")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_stbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == f64 {
		blas.cblas_dtbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex64 {
		blas.cblas_ctbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex128 {
		blas.cblas_ztbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	}
}

// Triangular banded matrix solve: solves A*x = b or A^T*x = b
// A is a triangular banded matrix with K superdiagonals or subdiagonals.
// The vector x is modified in-place with the solution.
// Supported types: f32, f64, complex64, complex128
mv_tbsv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	k: int, // Number of superdiagonals (if upper) or subdiagonals (if lower)
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where is_float(T) || is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector length must match matrix dimension")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_stbsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == f64 {
		blas.cblas_dtbsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex64 {
		blas.cblas_ctbsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex128 {
		blas.cblas_ztbsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			n,
			i64(k),
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
		)
	}
}

// ===================================================================================
// PACKED MATRIX-VECTOR OPERATIONS
// ===================================================================================

// Triangular packed matrix-vector multiplication: x = A*x or x = A^T*x
// A is stored in packed format (upper or lower triangle only, no gaps).
// The vector x is modified in-place with the result.
// Supported types: f32, f64, complex64, complex128
mv_tpmv :: proc(
	Ap: []$T, // Packed triangular matrix
	x: ^Vector(T),
	n: int, // Matrix dimension
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where is_float(T) || is_complex(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "Vector length must match matrix dimension")

	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_stpmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	} else when T == f64 {
		blas.cblas_dtpmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex64 {
		blas.cblas_ctpmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex128 {
		blas.cblas_ztpmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	}
}

// Triangular packed matrix solve: solves A*x = b or A^T*x = b
// A is stored in packed format (upper or lower triangle only, no gaps).
// The vector x is modified in-place with the solution.
// Supported types: f32, f64, complex64, complex128
mv_tpsv :: proc(
	Ap: []$T, // Packed triangular matrix
	x: ^Vector(T),
	n: int, // Matrix dimension
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where is_float(T) || is_complex(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "Vector length must match matrix dimension")

	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_stpsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	} else when T == f64 {
		blas.cblas_dtpsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex64 {
		blas.cblas_ctpsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	} else when T == complex128 {
		blas.cblas_ztpsv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			diag,
			i64(n),
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
		)
	}
}

// Symmetric matrix-vector multiplication: y = alpha*A*x + beta*y
// A is a symmetric matrix, only upper or lower triangle is referenced.
// For real types only (use mv_hemv for complex Hermitian).
// Supported types: f32, f64
mv_symv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	y: ^Vector(T),
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Input vector dimension must match matrix")
	assert(y.size == A.rows, "Output vector dimension must match matrix")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_ssymv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == f64 {
		blas.cblas_dsymv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	}
}

// Hermitian matrix-vector multiplication: y = alpha*A*x + beta*y
// A is a Hermitian matrix, only upper or lower triangle is referenced.
// For complex types only (use mv_symv for real symmetric).
// Supported types: complex64, complex128
mv_hemv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	y: ^Vector(T),
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Input vector dimension must match matrix")
	assert(y.size == A.rows, "Output vector dimension must match matrix")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)
	incy := i64(y.incr)
	alpha, beta := alpha, beta

	when T == complex64 {
		blas.cblas_chemv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == complex128 {
		blas.cblas_zhemv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	}
}

// Symmetric packed matrix-vector multiplication: y = alpha*A*x + beta*y
// A is stored in packed format (upper or lower triangle only, no gaps).
// For real types only.
// Supported types: f32, f64
mv_spmv :: proc(
	Ap: []$T, // Packed symmetric matrix
	x: ^Vector(T),
	y: ^Vector(T),
	n: int, // Matrix dimension
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "Input vector dimension must match matrix")
	assert(y.size == n, "Output vector dimension must match matrix")

	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_sspmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == f64 {
		blas.cblas_dspmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
			beta,
			vector_data_ptr(y),
			incy,
		)
	}
}

// ===================================================================================
// PACKED SYMMETRIC/HERMITIAN RANK UPDATES
// ===================================================================================

// Symmetric packed rank-1 update: A = alpha*x*x^T + A
// A is stored in packed format (upper or lower triangle only).
// Updates a packed symmetric matrix with the outer product of a vector with itself.
// For real types only.
// Supported types: f32, f64
mv_spr :: proc(
	x: ^Vector($T),
	Ap: []T, // Packed symmetric matrix
	n: int, // Matrix dimension
	alpha: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "Vector length must match matrix dimension")

	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_sspr(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(Ap),
		)
	} else when T == f64 {
		blas.cblas_dspr(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(Ap),
		)
	}
}

// Hermitian packed rank-1 update: A = alpha*x*conj(x)^T + A
// A is stored in packed format (upper or lower triangle only).
// Updates a packed Hermitian matrix with the outer product of a complex vector with its conjugate.
// Alpha must be real for the result to remain Hermitian.
// For complex types only.
// Supported types: complex64 (alpha is f32), complex128 (alpha is f64)
mv_hpr :: proc(
	x: ^Vector($T),
	Ap: []T, // Packed Hermitian matrix
	n: int, // Matrix dimension
	alpha: $R,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "Vector length must match matrix dimension")

	incx := i64(x.incr)

	when T == complex64 {
		static_assert(R == f32, "Alpha must be f32 for complex64")
		blas.cblas_chpr(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(Ap),
		)
	} else when T == complex128 {
		static_assert(R == f64, "Alpha must be f64 for complex128")
		blas.cblas_zhpr(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			vector_data_ptr(x),
			incx,
			raw_data(Ap),
		)
	}
}

// Symmetric packed rank-2 update: A = alpha*(x*y^T + y*x^T) + A
// A is stored in packed format (upper or lower triangle only).
// Updates a packed symmetric matrix with the sum of two outer products.
// For real types only.
// Supported types: f32, f64
mv_spr2 :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	Ap: []T, // Packed symmetric matrix
	n: int, // Matrix dimension
	alpha: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "x length must match matrix dimension")
	assert(y.size == n, "y length must match matrix dimension")

	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_sspr2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(Ap),
		)
	} else when T == f64 {
		blas.cblas_dspr2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(Ap),
		)
	}
}

// Hermitian packed rank-2 update: A = alpha*x*conj(y)^T + conj(alpha)*y*conj(x)^T + A
// A is stored in packed format (upper or lower triangle only).
// Updates a packed Hermitian matrix with the sum of two conjugate outer products.
// For complex types only.
// Supported types: complex64, complex128
mv_hpr2 :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	Ap: []T, // Packed Hermitian matrix
	n: int, // Matrix dimension
	alpha: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "x length must match matrix dimension")
	assert(y.size == n, "y length must match matrix dimension")

	incx := i64(x.incr)
	incy := i64(y.incr)
	alpha := alpha

	when T == complex64 {
		blas.cblas_chpr2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(Ap),
		)
	} else when T == complex128 {
		blas.cblas_zhpr2(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			&alpha,
			vector_data_ptr(x),
			incx,
			vector_data_ptr(y),
			incy,
			raw_data(Ap),
		)
	}
}

// ===================================================================================
// HERMITIAN BANDED AND PACKED MATRIX-VECTOR OPERATIONS
// ===================================================================================

// Hermitian banded matrix-vector multiplication: y = alpha*A*x + beta*y
// A is a Hermitian banded matrix with K superdiagonals (or subdiagonals).
// Only the upper or lower triangle is stored in banded format.
// For complex types only (use mv_sbmv for real symmetric).
// Supported types: complex64, complex128
mv_hbmv :: proc(
	A: ^Matrix($T),
	x: ^Vector(T),
	y: ^Vector(T),
	k: int, // Number of superdiagonals
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	assert(A.rows == A.cols, "Matrix must be square")
	assert(x.size == A.rows, "Vector dimensions must match matrix")
	assert(y.size == A.rows, "Output vector dimension must match matrix")

	n := i64(A.rows)
	lda := i64(A.ld)
	incx := i64(x.incr)
	incy := i64(y.incr)
	alpha, beta := alpha, beta

	when T == complex64 {
		blas.cblas_chbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			i64(k),
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == complex128 {
		blas.cblas_zhbmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			n,
			i64(k),
			&alpha,
			raw_data(A.data),
			lda,
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	}
}

// Hermitian packed matrix-vector multiplication: y = alpha*A*x + beta*y
// A is stored in packed format (upper or lower triangle only, no gaps).
// For complex types only (use mv_spmv for real symmetric).
// Supported types: complex64, complex128
mv_hpmv :: proc(
	Ap: []$T, // Packed Hermitian matrix
	x: ^Vector(T),
	y: ^Vector(T),
	n: int, // Matrix dimension
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	assert(len(Ap) >= n * (n + 1) / 2, "Packed array must have at least n*(n+1)/2 elements")
	assert(x.size == n, "Input vector dimension must match matrix")
	assert(y.size == n, "Output vector dimension must match matrix")

	incx := i64(x.incr)
	incy := i64(y.incr)
	alpha, beta := alpha, beta

	when T == complex64 {
		blas.cblas_chpmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			&alpha,
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	} else when T == complex128 {
		blas.cblas_zhpmv(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			i64(n),
			&alpha,
			raw_data(Ap),
			vector_data_ptr(x),
			incx,
			&beta,
			vector_data_ptr(y),
			incy,
		)
	}
}
