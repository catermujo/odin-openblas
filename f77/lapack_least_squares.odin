package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// Standard Least Squares:
	// ===================================================================================

	// GELS - Least squares solver using QR or LQ factorization
	/*
		trans: ^char [in] 'N': the linear system involves A; 'C': the linear system involves A**H
		m: ^blasint [in] number of rows in A; m>=0
		n: ^blasint [in] number of columns in A; n>=0
		nrhs: ^blasint [in] number of columns in B & X; nrhs>=0
		A: []T [in,out] M-by-N matrix. On exit:
		                if m>=n: A is overwritten by details of its QR factorization (GEQRF)
		                if m<n: A is overwritten by details of its LQ factorization (GELQF)
		lda: ^blasint [in] leading dim of A; >=max(1,m)
		B: []T [in,out] On entry: M-by-NRHS if TRANS='N', or N-by-NRHS if TRANS='C'
		                On exit if INFO=0: overwritten by solution vectors, stored columnwise
		                  if TRANS='N' and m>=n: rows 1 to n of B contain least squares solution
		                  if TRANS='N' and m<n: rows 1 to N of B contain minimum norm solution
		                  if TRANS='C' and m>=n: rows 1 to M of B contain minimum norm solution
		                  if TRANS='C' and m<n: rows 1 to M of B contain least squares solution
		ldb: ^blasint [in] leading dim of B; >=max(1,max(m,n))
		work: []T [out] workspace array, dimension max(1,lwork)
		lwork: ^blasint [in] dimension of WORK array
		                     LWORK >= max(1, MN + max(MN, NRHS)) where MN = min(M,N)
		                     For optimal performance: LWORK >= max(1, MN + max(MN, NRHS)*NB) where NB is optimal block size
		info: ^Info [out] 0: Success
		                 <0: -ith argument had illegal value
		                 >0: ith diagonal element of triangular factor of A is zero (A does not have full rank)
	*/
	cgels_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	dgels_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgels_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgels_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---

	// GELST - Least squares solver using complete orthogonal factorization (most robust for rank-deficient)
	/*
		trans: ^char [in] 'N': the linear system involves A; 'C': the linear system involves A**H
		m: ^blasint [in] number of rows in A; m>=0
		n: ^blasint [in] number of columns in A; n>=0
		nrhs: ^blasint [in] number of columns in B & X; nrhs>=0
		A: []T [in,out] M-by-N matrix A. On exit, overwritten by factorization details
		lda: ^blasint [in] leading dim of A; >=max(1,m)
		B: []T [in,out] On entry: M-by-NRHS if TRANS='N', or N-by-NRHS if TRANS='C'
		                On exit: overwritten by solution matrix X
		ldb: ^blasint [in] leading dim of B; >=max(1,max(m,n))
		jpvt: []blasint [in,out] dimension N. Pivot indices for column pivoting
		rcond: ^Real [in] used to determine effective rank. Singular values <= RCOND*S(1) treated as zero
		rank: ^blasint [out] effective rank of matrix A
		work: []T [out] workspace array, dimension max(1,lwork)
		lwork: ^blasint [in] dimension of work array. Use -1 for workspace query
		info: ^Info [out] 0: Success
		                 <0: -ith argument had illegal value
		Note: GELST uses complete orthogonal factorization for maximum numerical stability
		      with rank-deficient matrices. Added in LAPACK 3.9.0
	*/
	cgelst_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, jpvt: [^]blasint, rcond: ^f32, rank: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_trans: c.size_t = 1) ---
	dgelst_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, jpvt: [^]blasint, rcond: ^f64, rank: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgelst_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, jpvt: [^]blasint, rcond: ^f32, rank: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgelst_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, jpvt: [^]blasint, rcond: ^f64, rank: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_trans: c.size_t = 1) ---


	// GELSS - Least squares solver using SVD (slower than GELSD, less memory)
	/*
		m: ^blasint [in] number of rows in A; m>=0
		n: ^blasint [in] number of columns in A; n>=0
		nrhs: ^blasint [in] number of columns in B & X; nrhs>=0
		A: []T [in,out] M-by-N matrix A. On exit, first min(m,n) rows of A are overwritten with its right singular vectors, stored rowwise
		lda: ^blasint [in] leading dim of A; >=max(1,m)
		B: []T [in,out] M-by-NRHS right hand side matrix B. On exit, B is overwritten by the N-by-NRHS solution matrix X
		ldb: ^blasint [in] leading dim of B; >=max(1,max(m,n))
		S: []Real [out] singular values of A in decreasing order. Dimension min(m,n)
		rcond: ^Real [in] used to determine effective rank of A. Singular values S(i) <= RCOND*S(1) are treated as zero
		rank: ^blasint [out] effective rank of A
		work: []T [out] workspace array, dimension max(1,lwork)
		lwork: ^blasint [in] dimension of work array. For good performance, LWORK >= 3*min(m,n) + max(2*min(m,n), max(m,n), nrhs)
		rwork: []Real [out] (complex only) real workspace array, dimension at least 5*min(m,n)
		info: ^Info [out] 0:Success; <0: -ith arg illegal; >0: SVD algorithm failed to converge
	*/
	cgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, S: [^]f32, rcond: ^f32, rank: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info) ---
	dgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, S: [^]f64, rcond: ^f64, rank: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, S: [^]f32, rcond: ^f32, rank: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgelss_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, S: [^]f64, rcond: ^f64, rank: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info) ---

	// GELSD - Least squares solver using SVD with divide-and-conquer (fastest, most memory)
	/*
		m: ^blasint [in] number of rows in A; m>=0
		n: ^blasint [in] number of columns in A; n>=0
		nrhs: ^blasint [in] number of columns in B & X; nrhs>=0
		A: []T [in,out] M-by-N matrix A. On exit, A has been destroyed
		lda: ^blasint [in] leading dim of A; >=max(1,m)
		B: []T [in,out] M-by-NRHS right hand side matrix B. On exit, B is overwritten by the N-by-NRHS solution matrix X
		ldb: ^blasint [in] leading dim of B; >=max(1,max(m,n))
		S: []Real [out] singular values of A in decreasing order. Dimension min(m,n)
		rcond: ^Real [in] used to determine effective rank of A. Singular values S(i) <= RCOND*S(1) are treated as zero.
		                  If RCOND < 0, machine precision is used instead
		rank: ^blasint [out] effective rank of A, i.e., the number of singular values which are greater than RCOND*S(1)
		work: []T [out] workspace array, dimension max(1,lwork)
		lwork: ^blasint [in] dimension of work array. For good performance, LWORK should generally be larger
		iwork: []blasint [out] integer workspace array, dimension at least LIWORK >= max(1, 3*min(m,n)*nlvl + 11*min(m,n))
		info: ^Info [out] 0:Success; <0: -ith arg illegal; >0: SVD algorithm failed to converge
	*/
	cgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, S: [^]f32, rcond: ^f32, rank: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, info: ^Info) ---
	dgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, S: [^]f64, rcond: ^f64, rank: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info) ---
	sgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, S: [^]f32, rcond: ^f32, rank: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info) ---
	zgelsd_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, S: [^]f64, rcond: ^f64, rank: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, info: ^Info) ---


	// GELSY - Least squares solver using complete orthogonal factorization (good compromise between speed and accuracy)
	/*
		m: ^blasint [in] number of rows in A; m>=0
		n: ^blasint [in] number of columns in A; n>=0
		nrhs: ^blasint [in] number of columns in B & X; nrhs>=0
		A: []T [in,out] M-by-N matrix A. On exit, A overwritten by details of its complete orthogonal factorization
		lda: ^blasint [in] leading dim of A; >=max(1,m)
		B: []T [in,out] M-by-NRHS right hand side matrix B. On exit, the N-by-NRHS solution matrix X
		ldb: ^blasint [in] leading dim of B; >=max(1,max(m,n))
		JPVT: []blasint [in,out] dimension N. On entry, if JPVT(i) != 0, the i-th column of A is permuted to front of A*P
		                         On exit, if JPVT(i) = k, then the i-th column of A*P was the k-th column of A
		rcond: ^Real [in] used to determine effective rank of A. Singular values S(i) <= RCOND*S(1) are treated as zero
		rank: ^blasint [out] effective rank of A
		work: []T [out] workspace array, dimension max(1,lwork)
		lwork: ^blasint [in] dimension of work array. For good performance, LWORK >= min(m,n) + max(2*min(m,n), n+1, min(m,n)+nrhs)
		rwork: []Real [out] (complex only) real workspace array, dimension at least 2*N
		info: ^Info [out] 0:Success; <0: -ith arg illegal
	*/
	cgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, JPVT: [^]blasint, rcond: ^f32, rank: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info) ---
	dgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, JPVT: [^]blasint, rcond: ^f64, rank: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, JPVT: [^]blasint, rcond: ^f32, rank: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgelsy_ :: proc(m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, JPVT: [^]blasint, rcond: ^f64, rank: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info) ---

	// GETSLS - Least squares solver using tall-skinny or short-wide QR/LQ factorization with column pivoting
	/*
		trans: ^char [in] 'N': the linear system involves A; 'C': the linear system involves A**H
		m: ^blasint [in] number of rows in A; m>=0
		n: ^blasint [in] number of columns in A; n>=0
		nrhs: ^blasint [in] number of columns in B & X; nrhs>=0
		A: []T [in,out] M-by-N matrix A. On exit, overwritten by factorization details
		lda: ^blasint [in] leading dim of A; >=max(1,m)
		B: []T [in,out] Matrix B. On exit, overwritten by solution matrix X
		ldb: ^blasint [in] leading dim of B; >=max(1,max(m,n))
		work: []T [out] workspace array, dimension max(1,lwork)
		lwork: ^blasint [in] dimension of work array. Use -1 for workspace query
		info: ^Info [out] 0:Success; <0: -ith arg illegal; >0: A does not have full rank
	*/
	cgetsls_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	dgetsls_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgetsls_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgetsls_ :: proc(trans: ^char, m: ^blasint, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---

	// ===================================================================================
	// Constrained Least Squares:
	// ===================================================================================

	cgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, C: [^]complex64, D: [^]complex64, X: [^]complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, C: [^]f64, D: [^]f64, X: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, C: [^]f32, D: [^]f32, X: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgglse_ :: proc(m: ^blasint, n: ^blasint, p: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, C: [^]complex128, D: [^]complex128, X: [^]complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	cggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, D: [^]complex64, X: [^]complex64, Y: [^]complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, D: [^]f64, X: [^]f64, Y: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, D: [^]f32, X: [^]f32, Y: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zggglm_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, D: [^]complex128, X: [^]complex128, Y: [^]complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	// ===================================================================================
	// Additional Least Squares Routines
	// ===================================================================================

	// GELQT - LQ factorization with compact WY representation
	/*
		m: ^blasint [in] number of rows in A; m>=0
		n: ^blasint [in] number of columns in A; n>=0
		mb: ^blasint [in] block size for factorization; mb>=1 and mb<=min(m,n)
		A: []T [in,out] M-by-N matrix. On exit, lower triangle contains L, upper part has reflectors
		lda: ^blasint [in] leading dim of A; >=max(1,m)
		T: []T [out] T factor for compact WY representation, dimension (LDT,MIN(M,N))
		ldt: ^blasint [in] leading dim of T; >=MB
		work: []T [out] workspace array
		lwork: ^blasint [in] dimension of work array
		info: ^Info [out] 0: Success; <0: -ith argument illegal
	*/
	cgelqt_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, A: [^]complex64, lda: ^blasint, T: ^complex64, ldt: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgelqt_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, A: [^]f64, lda: ^blasint, T: ^f64, ldt: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgelqt_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, A: [^]f32, lda: ^blasint, T: ^f32, ldt: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgelqt_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, A: [^]complex128, lda: ^blasint, T: ^complex128, ldt: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

}
