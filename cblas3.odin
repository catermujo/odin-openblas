package openblas

import blas "./c"
import "base:intrinsics"

// ==== BLAS Level 3: Matrix-matrix operations ====

// ===================================================================================
// MATRIX COPY AND TRANSPOSE
// ===================================================================================

// Out-of-place matrix copy with optional transpose and scaling: B = alpha * op(A)
// Copies matrix A to B with optional transpose/conjugate-transpose and scaling.
// This is useful for matrix transposition, scaling, and format conversion.
// Supported types: f32, f64, complex64, complex128
m_copy :: proc(
	A: ^Matrix( $T),
	B: ^Matrix( T),
	alpha: T,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
) where is_float(T) ||
	is_complex(T) {
	// Determine expected dimensions of B based on transpose
	if trans == .NoTrans {
		assert(B.rows == A.rows, "B rows must match A rows")
		assert(B.cols == A.cols, "B columns must match A columns")
	} else {
		assert(B.rows == A.cols, "B rows must match A columns when transposed")
		assert(B.cols == A.rows, "B columns must match A rows when transposed")
	}

	rows := i64(A.rows)
	cols := i64(A.cols)
	lda := i64(A.ld)
	ldb := i64(B.ld)

	when T == f32 {
		blas.cblas_somatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == f64 {
		blas.cblas_domatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == complex64 {
		alpha := alpha
		alpha_ptr := cast(^f32)&alpha
		blas.cblas_comatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha_ptr,
			cast(^f32)raw_data(A.data),
			lda,
			cast(^f32)raw_data(B.data),
			ldb,
		)
	} else when T == complex128 {
		alpha := alpha
		alpha_ptr := cast(^f64)&alpha
		blas.cblas_zomatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha_ptr,
			cast(^f64)raw_data(A.data),
			lda,
			cast(^f64)raw_data(B.data),
			ldb,
		)
	} else {
		#panic("Unsupported type for omatcopy")
	}
}

// In-place matrix transpose and scaling: A = alpha * op(A)
// Transposes and scales matrix A in-place.
// Note: For transpose operations, the matrix must be square or have sufficient storage.
// For non-square matrices being transposed, ldb specifies the leading dimension after transpose.
// Supported types: f32, f64, complex64, complex128
m_copy_inplace :: proc(
	A: ^Matrix( $T),
	alpha: T,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
	ldb: int = 0, // Leading dimension after operation (0 = use default)
) where is_float(T) || is_complex(T) {
	rows := i64(A.rows)
	cols := i64(A.cols)
	lda := i64(A.ld)

	// For transpose operations on non-square matrices, need to specify new leading dimension
	ldb_actual: i64
	if ldb > 0 {
		ldb_actual = i64(ldb)
	} else if trans == .NoTrans {
		ldb_actual = lda // No change in leading dimension
	} else {
		// For transpose, the new leading dimension should be cols (original)
		ldb_actual = cols
		// After transpose, update the dimensions
		if trans != .NoTrans {
			A.rows = int(cols)
			A.cols = int(rows)
			A.ld = int(cols)  // New leading dimension after transpose
		}
	}

	when T == f32 {
		blas.cblas_simatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha,
			raw_data(A.data),
			lda,
			ldb_actual,
		)
	} else when T == f64 {
		blas.cblas_dimatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha,
			raw_data(A.data),
			lda,
			ldb_actual,
		)
	} else when T == complex64 {
		alpha := alpha
		alpha_ptr := cast(^f32)&alpha
		blas.cblas_cimatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha_ptr,
			cast(^f32)raw_data(A.data),
			lda,
			ldb_actual,
		)
	} else when T == complex128 {
		alpha := alpha
		alpha_ptr := cast(^f64)&alpha
		blas.cblas_zimatcopy(
			blas.CBLAS_ORDER.ColMajor,
			trans,
			rows,
			cols,
			alpha_ptr,
			cast(^f64)raw_data(A.data),
			lda,
			ldb_actual,
		)
	} else {
		#panic("Unsupported type for imatcopy")
	}
}

// Matrix addition with scaling: C = alpha*A + beta*C
// Adds matrix A to C with optional scaling of both matrices.
// This is useful for matrix accumulation and linear combinations.
// Supported types: f32, f64, complex64, complex128
m_add :: proc(
	A: ^Matrix( $T),
	C: ^Matrix( T),
	alpha: T,
	beta: T,
) where is_float(T) ||
	is_complex(T) {
	assert(A.rows == C.rows, "Matrix rows must match")
	assert(A.cols == C.cols, "Matrix columns must match")

	rows := i64(A.rows)
	cols := i64(A.cols)
	lda := i64(A.ld)
	ldc := i64(C.ld)

	when T == f32 {
		blas.cblas_sgeadd(
			blas.CBLAS_ORDER.ColMajor,
			rows,
			cols,
			alpha,
			raw_data(A.data),
			lda,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == f64 {
		blas.cblas_dgeadd(
			blas.CBLAS_ORDER.ColMajor,
			rows,
			cols,
			alpha,
			raw_data(A.data),
			lda,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		alpha_ptr := cast(^f32)&alpha
		beta_ptr := cast(^f32)&beta
		blas.cblas_cgeadd(
			blas.CBLAS_ORDER.ColMajor,
			rows,
			cols,
			alpha_ptr,
			cast(^f32)raw_data(A.data),
			lda,
			beta_ptr,
			cast(^f32)raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		alpha_ptr := cast(^f64)&alpha
		beta_ptr := cast(^f64)&beta
		blas.cblas_zgeadd(
			blas.CBLAS_ORDER.ColMajor,
			rows,
			cols,
			alpha_ptr,
			cast(^f64)raw_data(A.data),
			lda,
			beta_ptr,
			cast(^f64)raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for geadd")
	}
}

// ===================================================================================
// GENERAL MATRIX-MATRIX MULTIPLICATION
// ===================================================================================

// Matrix multiplication: C = alpha*A*B + beta*C
// Performs general matrix multiplication with optional transpose operations.
// A, B, and C can be transposed or conjugate-transposed as specified.
// Result is stored in C, which is scaled by beta before adding the product.
// Supported types: f32, f64, complex64, complex128
m_mul :: proc(
	A: ^Matrix( $T),
	B: ^Matrix( T),
	C: ^Matrix( T),
	alpha: T,
	beta: T,
	transA: blas.CBLAS_TRANSPOSE = .NoTrans,
	transB: blas.CBLAS_TRANSPOSE = .NoTrans,
) where is_float(T) ||
	is_complex(T) {
	// Get dimensions based on transpose operations
	m := i64(C.rows) // Rows of C
	n := i64(C.cols) // Columns of C

	// K is the shared dimension between A and B
	k: i64
	if transA == .NoTrans {
		k = i64(A.cols)
		assert(A.rows == int(m), "A rows must match C rows")
	} else {
		k = i64(A.rows)
		assert(A.cols == int(m), "A columns must match C rows when transposed")
	}

	if transB == .NoTrans {
		assert(B.rows == int(k), "B rows must match A columns")
		assert(B.cols == int(n), "B columns must match C columns")
	} else {
		assert(B.cols == int(k), "B columns must match A columns when transposed")
		assert(B.rows == int(n), "B rows must match C columns when transposed")
	}

	lda := i64(A.ld) // Leading dimension of A
	ldb := i64(B.ld) // Leading dimension of B
	ldc := i64(C.ld) // Leading dimension of C

	when T == f32 {
		blas.cblas_sgemm(
			blas.CBLAS_ORDER.ColMajor,
			transA,
			transB,
			m,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == f64 {
		blas.cblas_dgemm(
			blas.CBLAS_ORDER.ColMajor,
			transA,
			transB,
			m,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		blas.cblas_cgemm(
			blas.CBLAS_ORDER.ColMajor,
			transA,
			transB,
			m,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		blas.cblas_zgemm(
			blas.CBLAS_ORDER.ColMajor,
			transA,
			transB,
			m,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for gemm")
	}
}

// Matrix multiplication with 3M algorithm: C = alpha*A*B + beta*C
// Uses the 3M algorithm for complex matrices which reduces the number of real multiplications.
// This can be faster for complex matrices on some architectures.
// Only available for complex types.
// Supported types: complex64, complex128
m_mul_3m :: proc(
	A: ^Matrix( $T),
	B: ^Matrix( T),
	C: ^Matrix( T),
	alpha: T,
	beta: T,
	transA: blas.CBLAS_TRANSPOSE = .NoTrans,
	transB: blas.CBLAS_TRANSPOSE = .NoTrans,
) where is_complex(T) {
	// Get dimensions based on transpose operations
	m := i64(C.rows) // Rows of C
	n := i64(C.cols) // Columns of C

	// K is the shared dimension between A and B
	k: i64
	if transA == .NoTrans {
		k = i64(A.cols)
		assert(A.rows == int(m), "A rows must match C rows")
	} else {
		k = i64(A.rows)
		assert(A.cols == int(m), "A columns must match C rows when transposed")
	}

	if transB == .NoTrans {
		assert(B.rows == int(k), "B rows must match A columns")
		assert(B.cols == int(n), "B columns must match C columns")
	} else {
		assert(B.cols == int(k), "B columns must match A columns when transposed")
		assert(B.rows == int(n), "B rows must match C columns when transposed")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)
	ldc := i64(C.ld)
	alpha, beta := alpha, beta

	when T == complex64 {
		blas.cblas_cgemm3m(
			blas.CBLAS_ORDER.ColMajor,
			transA,
			transB,
			m,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		blas.cblas_zgemm3m(
			blas.CBLAS_ORDER.ColMajor,
			transA,
			transB,
			m,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for gemm3m")
	}
}

// Triangular matrix multiplication with transpose: C = alpha*A*B^T + beta*C or C = alpha*A^T*B + beta*C
// Computes matrix multiplication and stores only the upper or lower triangle of the result.
// Useful when the result is known to be symmetric or when only one triangle is needed.
// Supported types: f32, f64, complex64, complex128
m_mul_triangular :: proc(
	A: ^Matrix( $T),
	B: ^Matrix( T),
	C: ^Matrix( T),
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
	transA: blas.CBLAS_TRANSPOSE = .NoTrans,
	transB: blas.CBLAS_TRANSPOSE = .NoTrans,
) where is_float(T) ||
	is_complex(T) {
	assert(C.rows == C.cols, "C must be square")

	m := i64(C.rows) // Size of C (square)

	// K is the shared dimension between A and B
	k: i64
	if transA == .NoTrans {
		k = i64(A.cols)
		assert(A.rows == int(m), "A rows must match C dimension")
	} else {
		k = i64(A.rows)
		assert(A.cols == int(m), "A columns must match C dimension when transposed")
	}

	if transB == .NoTrans {
		assert(B.rows == int(k), "B rows must match A's inner dimension")
		assert(B.cols == int(m), "B columns must match C dimension")
	} else {
		assert(B.cols == int(k), "B columns must match A's inner dimension when transposed")
		assert(B.rows == int(m), "B rows must match C dimension when transposed")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)
	ldc := i64(C.ld)

	when T == f32 {
		blas.cblas_sgemmt(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			transA,
			transB,
			m,
			k,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == f64 {
		blas.cblas_dgemmt(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			transA,
			transB,
			m,
			k,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		blas.cblas_cgemmt(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			transA,
			transB,
			m,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		blas.cblas_zgemmt(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			transA,
			transB,
			m,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for gemmt")
	}
}

// ===================================================================================
// SYMMETRIC/HERMITIAN MATRIX MULTIPLICATION
// ===================================================================================

// Symmetric matrix multiplication: C = alpha*A*B + beta*C or C = alpha*B*A + beta*C
// Where A is symmetric and only upper or lower triangle is referenced.
// The side parameter determines if A is on the left (A*B) or right (B*A).
// Supported types: f32, f64, complex64, complex128
m_symm :: proc(
	A: ^Matrix( $T), // Symmetric matrix
	B: ^Matrix( T),
	C: ^Matrix( T),
	alpha: T,
	beta: T,
	side: blas.CBLAS_SIDE = .Left,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_float(T) || is_complex(T) {
	m := i64(C.rows)
	n := i64(C.cols)

	if side == .Left {
		// C = alpha*A*B + beta*C, A is m×m
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(m), "A dimension must match C rows")
		assert(B.rows == int(m), "B rows must match C rows")
		assert(B.cols == int(n), "B columns must match C columns")
	} else {
		// C = alpha*B*A + beta*C, A is n×n
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(n), "A dimension must match C columns")
		assert(B.rows == int(m), "B rows must match C rows")
		assert(B.cols == int(n), "B columns must match C columns")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)
	ldc := i64(C.ld)

	when T == f32 {
		blas.cblas_ssymm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == f64 {
		blas.cblas_dsymm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		blas.cblas_csymm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		blas.cblas_zsymm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for symm")
	}
}

// ===================================================================================
// SYMMETRIC/HERMITIAN RANK-K UPDATES
// ===================================================================================

// Symmetric rank-k update: C = alpha*A*A^T + beta*C or C = alpha*A^T*A + beta*C
// Updates a symmetric matrix C with the product of A and its transpose.
// Only the upper or lower triangle of C is computed and stored.
// Useful for computing Gram matrices, covariance matrices, etc.
// Supported types: f32, f64, complex64, complex128
m_syrk :: proc(
	A: ^Matrix( $T),
	C: ^Matrix( T), // Symmetric output matrix
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {
	assert(C.rows == C.cols, "C must be square")

	n := i64(C.rows)
	k: i64

	if trans == .NoTrans {
		// C = alpha*A*A^T + beta*C, A is n×k
		k = i64(A.cols)
		assert(A.rows == int(n), "A rows must match C dimension")
	} else {
		// C = alpha*A^T*A + beta*C, A is k×n
		k = i64(A.rows)
		assert(A.cols == int(n), "A columns must match C dimension when transposed")
	}

	lda := i64(A.ld)
	ldc := i64(C.ld)

	when T == f32 {
		blas.cblas_ssyrk(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == f64 {
		blas.cblas_dsyrk(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		blas.cblas_csyrk(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		blas.cblas_zsyrk(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for syrk")
	}
}

// Symmetric rank-2k update: C = alpha*(A*B^T + B*A^T) + beta*C or C = alpha*(A^T*B + B^T*A) + beta*C
// Updates a symmetric matrix C with the sum of two rank-k products.
// Only the upper or lower triangle of C is computed and stored.
// More general than syrk, useful for various matrix factorizations.
// Supported types: f32, f64, complex64, complex128
m_syr2k :: proc(
	A: ^Matrix( $T),
	B: ^Matrix( T),
	C: ^Matrix( T), // Symmetric output matrix
	alpha: T,
	beta: T,
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {
	assert(C.rows == C.cols, "C must be square")

	n := i64(C.rows)
	k: i64

	if trans == .NoTrans {
		// C = alpha*(A*B^T + B*A^T) + beta*C, A and B are n×k
		k = i64(A.cols)
		assert(A.rows == int(n), "A rows must match C dimension")
		assert(B.rows == int(n), "B rows must match C dimension")
		assert(B.cols == int(k), "B columns must match A columns")
	} else {
		// C = alpha*(A^T*B + B^T*A) + beta*C, A and B are k×n
		k = i64(A.rows)
		assert(A.cols == int(n), "A columns must match C dimension when transposed")
		assert(B.cols == int(n), "B columns must match C dimension when transposed")
		assert(B.rows == int(k), "B rows must match A rows when transposed")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)
	ldc := i64(C.ld)

	when T == f32 {
		blas.cblas_ssyr2k(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == f64 {
		blas.cblas_dsyr2k(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		blas.cblas_csyr2k(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		blas.cblas_zsyr2k(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for syr2k")
	}
}

// ===================================================================================
// TRIANGULAR MATRIX OPERATIONS
// ===================================================================================

// Triangular matrix multiply: B = alpha*op(A)*B or B = alpha*B*op(A)
// Where A is triangular (upper or lower) and may be unit triangular.
// The result overwrites B. Side determines if A is on left or right.
// Supported types: f32, f64, complex64, complex128
m_trmm :: proc(
	A: ^Matrix( $T), // Triangular matrix
	B: ^Matrix( T), // Input/output matrix
	alpha: T,
	side: blas.CBLAS_SIDE = .Left,
	uplo: blas.CBLAS_UPLO = .Upper,
	transA: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {
	m := i64(B.rows)
	n := i64(B.cols)

	if side == .Left {
		// B = alpha*op(A)*B, A is m×m
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(m), "A dimension must match B rows")
	} else {
		// B = alpha*B*op(A), A is n×n
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(n), "A dimension must match B columns")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)

	when T == f32 {
		blas.cblas_strmm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == f64 {
		blas.cblas_dtrmm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == complex64 {
		alpha := alpha
		blas.cblas_ctrmm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == complex128 {
		alpha := alpha
		blas.cblas_ztrmm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else {
		#panic("Unsupported type for trmm")
	}
}

// Triangular solve with multiple right-hand sides: op(A)*X = alpha*B or X*op(A) = alpha*B
// Solves a system of equations where A is triangular and B contains multiple RHS vectors.
// The solution X overwrites B. Side determines if A is on left or right.
// Supported types: f32, f64, complex64, complex128
m_trsm :: proc(
	A: ^Matrix( $T), // Triangular coefficient matrix
	B: ^Matrix( T), // Right-hand side(s) / solution matrix
	alpha: T,
	side: blas.CBLAS_SIDE = .Left,
	uplo: blas.CBLAS_UPLO = .Upper,
	transA: blas.CBLAS_TRANSPOSE = .NoTrans,
	diag: blas.CBLAS_DIAG = .NonUnit,
) where intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {
	m := i64(B.rows)
	n := i64(B.cols)

	if side == .Left {
		// op(A)*X = alpha*B, A is m×m
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(m), "A dimension must match B rows")
	} else {
		// X*op(A) = alpha*B, A is n×n
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(n), "A dimension must match B columns")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)

	when T == f32 {
		blas.cblas_strsm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == f64 {
		blas.cblas_dtrsm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == complex64 {
		alpha := alpha
		blas.cblas_ctrsm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else when T == complex128 {
		alpha := alpha
		blas.cblas_ztrsm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			transA,
			diag,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
		)
	} else {
		#panic("Unsupported type for trsm")
	}
}

// Hermitian matrix multiplication: C = alpha*A*B + beta*C or C = alpha*B*A + beta*C
// Where A is Hermitian and only upper or lower triangle is referenced.
// The side parameter determines if A is on the left (A*B) or right (B*A).
// Only available for complex types since Hermitian matrices are complex.
// Supported types: complex64, complex128
m_hemm :: proc(
	A: ^Matrix( $T), // Hermitian matrix
	B: ^Matrix( T),
	C: ^Matrix( T),
	alpha: T,
	beta: T,
	side: blas.CBLAS_SIDE = .Left,
	uplo: blas.CBLAS_UPLO = .Upper,
) where is_complex(T) {
	m := i64(C.rows)
	n := i64(C.cols)

	if side == .Left {
		// C = alpha*A*B + beta*C, A is m×m
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(m), "A dimension must match C rows")
		assert(B.rows == int(m), "B rows must match C rows")
		assert(B.cols == int(n), "B columns must match C columns")
	} else {
		// C = alpha*B*A + beta*C, A is n×n
		assert(A.rows == A.cols, "A must be square")
		assert(A.rows == int(n), "A dimension must match C columns")
		assert(B.rows == int(m), "B rows must match C rows")
		assert(B.cols == int(n), "B columns must match C columns")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)
	ldc := i64(C.ld)
	alpha, beta := alpha, beta

	when T == complex64 {
		blas.cblas_chemm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 {
		blas.cblas_zhemm(
			blas.CBLAS_ORDER.ColMajor,
			side,
			uplo,
			m,
			n,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			&beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Unsupported type for hemm - must be complex")
	}
}

// ===================================================================================
// HERMITIAN RANK-K UPDATES
// ===================================================================================

// Hermitian rank-k update: C = alpha*A*A^H + beta*C or C = alpha*A^H*A + beta*C
// Updates a Hermitian matrix C with the product of A and its conjugate transpose.
// Only the upper or lower triangle of C is computed and stored.
// Note: alpha and beta are real even for complex matrices.
// Only available for complex types.
// Supported types: complex64, complex128
m_herk :: proc(
	A: ^Matrix( $T),
	C: ^Matrix( T), // Hermitian output matrix
	alpha: $R, // Real scalar
	beta: R, // Real scalar
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
) where is_complex(T) && is_float(R) {
	assert(C.rows == C.cols, "C must be square")

	n := i64(C.rows)
	k: i64

	if trans == .NoTrans {
		// C = alpha*A*A^H + beta*C, A is n×k
		k = i64(A.cols)
		assert(A.rows == int(n), "A rows must match C dimension")
	} else {
		// C = alpha*A^H*A + beta*C, A is k×n
		k = i64(A.rows)
		assert(A.cols == int(n), "A columns must match C dimension when transposed")
	}

	lda := i64(A.ld)
	ldc := i64(C.ld)

	when T == complex64 && R == f32 {
		blas.cblas_cherk(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 && R == f64 {
		blas.cblas_zherk(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			alpha,
			raw_data(A.data),
			lda,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Type mismatch: complex64 requires f32 scalars, complex128 requires f64 scalars")
	}
}

// Hermitian rank-2k update: C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C or C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C
// Updates a Hermitian matrix C with the sum of two rank-k products.
// Only the upper or lower triangle of C is computed and stored.
// Note: beta is real even for complex matrices, alpha is complex.
// Only available for complex types.
// Supported types: complex64, complex128
m_her2k :: proc(
	A: ^Matrix( $T),
	B: ^Matrix( T),
	C: ^Matrix( T), // Hermitian output matrix
	alpha: T, // Complex scalar
	beta: $R, // Real scalar
	uplo: blas.CBLAS_UPLO = .Upper,
	trans: blas.CBLAS_TRANSPOSE = .NoTrans,
) where is_complex(T) && is_float(R) {
	assert(C.rows == C.cols, "C must be square")

	n := i64(C.rows)
	k: i64

	if trans == .NoTrans {
		// C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C, A and B are n×k
		k = i64(A.cols)
		assert(A.rows == int(n), "A rows must match C dimension")
		assert(B.rows == int(n), "B rows must match C dimension")
		assert(B.cols == int(k), "B columns must match A columns")
	} else {
		// C = alpha*A^H*B + conj(alpha)*B^H*A + beta*C, A and B are k×n
		k = i64(A.rows)
		assert(A.cols == int(n), "A columns must match C dimension when transposed")
		assert(B.cols == int(n), "B columns must match C dimension when transposed")
		assert(B.rows == int(k), "B rows must match A rows when transposed")
	}

	lda := i64(A.ld)
	ldb := i64(B.ld)
	ldc := i64(C.ld)
	alpha := alpha

	when T == complex64 && R == f32 {
		blas.cblas_cher2k(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else when T == complex128 && R == f64 {
		blas.cblas_zher2k(
			blas.CBLAS_ORDER.ColMajor,
			uplo,
			trans,
			n,
			k,
			&alpha,
			raw_data(A.data),
			lda,
			raw_data(B.data),
			ldb,
			beta,
			raw_data(C.data),
			ldc,
		)
	} else {
		#panic("Type mismatch: complex64 requires f32 beta, complex128 requires f64 beta")
	}
}
