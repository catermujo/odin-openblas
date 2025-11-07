package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// Initialize, copy, convert
	// https://www.netlib.org/lapack/explore-html/d4/d7e/group__set__grp.html
	// ===================================================================================
	// laset: set matrix
	claset_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, alpha: ^complex64, beta: ^complex64, A: [^]complex64, lda: ^blasint, l_uplo: c.size_t = 1) ---
	dlaset_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, alpha: ^f64, beta: ^f64, A: [^]f64, lda: ^blasint, l_uplo: c.size_t = 1) ---
	slaset_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, alpha: ^f32, beta: ^f32, A: [^]f32, lda: ^blasint, l_uplo: c.size_t = 1) ---
	zlaset_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, alpha: ^complex128, beta: ^complex128, A: [^]complex128, lda: ^blasint, l_uplo: c.size_t = 1) ---

	// larnv: random vector
	clarnv_ :: proc(idist: ^blasint, iseed: [^]blasint, n: ^blasint, X: [^]complex64) ---
	dlarnv_ :: proc(idist: ^blasint, iseed: [^]blasint, n: ^blasint, X: [^]f64) ---
	slarnv_ :: proc(idist: ^blasint, iseed: [^]blasint, n: ^blasint, X: [^]f32) ---
	zlarnv_ :: proc(idist: ^blasint, iseed: [^]blasint, n: ^blasint, X: [^]complex128) ---

	// laruv: random uniform vector

	// lacpy: copy matrix
	clacpy_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, l_uplo: c.size_t = 1) ---
	dlacpy_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, l_uplo: c.size_t = 1) ---
	slacpy_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, l_uplo: c.size_t = 1) ---
	zlacpy_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, l_uplo: c.size_t = 1) ---

	// lacp2: general matrix, convert real to complex
	clacp2_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]complex64, ldb: ^blasint, l_uplo: c.size_t = 1) ---
	zlacp2_ :: proc(uplo: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]complex128, ldb: ^blasint, l_uplo: c.size_t = 1) ---

	// lag2: general matrix, convert double <=> single
	zlag2c_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, SA: [^]complex64, ldsa: ^blasint, info: ^Info) ---
	slag2d_ :: proc(m: ^blasint, n: ^blasint, SA: [^]f32, ldsa: ^blasint, A: [^]f64, lda: ^blasint, info: ^Info) ---
	dlag2s_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, SA: [^]f32, ldsa: ^blasint, info: ^Info) ---
	clag2z_ :: proc(m: ^blasint, n: ^blasint, SA: [^]complex64, ldsa: ^blasint, A: [^]complex128, lda: ^blasint, info: ^Info) ---

	// lat2: triangular matrix, convert double <=> single

	// tfttp: triangular matrix, RFP (tf) to packed (tp)
	ctfttp_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]complex64, AP: [^]complex64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dtfttp_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]f64, AP: [^]f64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	stfttp_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]f32, AP: [^]f32, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ztfttp_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]complex128, AP: [^]complex128, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// tfttr: triangular matrix, RFP (tf) to full (tr)
	ctfttr_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]complex64, A: [^]complex64, lda: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dtfttr_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]f64, A: [^]f64, lda: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	stfttr_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]f32, A: [^]f32, lda: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ztfttr_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, ARF: [^]complex128, A: [^]complex128, lda: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// tpttf: triangular matrix, packed (tp) to RFP (tf)
	ctpttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, ARF: [^]complex64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dtpttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, ARF: [^]f64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	stpttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, ARF: [^]f32, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ztpttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, ARF: [^]complex128, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// tpttr: triangular matrix, packed (tp) to full (tr)
	ctpttr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, A: [^]complex64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dtpttr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, A: [^]f64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	stpttr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, A: [^]f32, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ztpttr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, A: [^]complex128, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// trttf: triangular matrix, full (tr) to RFP (tf)
	ctrttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ARF: [^]complex64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dtrttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ARF: [^]f64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	strttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ARF: [^]f32, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ztrttf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ARF: [^]complex128, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// trttp: triangular matrix, full (tr) to packed (tp)
	ctrttp_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, AP: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dtrttp_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, AP: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	strttp_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, AP: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	ztrttp_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, AP: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	// ===================================================================================
	// matrix norm
	// https://www.netlib.org/lapack/explore-html/db/d60/group__norm__grp.html
	// ===================================================================================
	// lange: general matrix
	clange_ :: proc(norm: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---
	dlange_ :: proc(norm: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1) -> f64 ---
	slange_ :: proc(norm: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---
	zlange_ :: proc(norm: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1) -> f64 ---

	// langb: general matrix, banded
	clangb_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex64, ldab: ^blasint, work: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---
	dlangb_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f64, ldab: ^blasint, work: [^]f64, l_norm: c.size_t = 1) -> f64 ---
	slangb_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f32, ldab: ^blasint, work: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---
	zlangb_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex128, ldab: ^blasint, work: [^]f64, l_norm: c.size_t = 1) -> f64 ---

	// langt: general matrix, tridiagonal
	clangt_ :: proc(norm: ^char, n: ^blasint, DL: [^]complex64, D: [^]complex64, DU: [^]complex64, l_norm: c.size_t = 1) -> lapack_float_return ---
	dlangt_ :: proc(norm: ^char, n: ^blasint, DL: [^]f64, D: [^]f64, DU: [^]f64, l_norm: c.size_t = 1) -> f64 ---
	slangt_ :: proc(norm: ^char, n: ^blasint, DL: [^]f32, D: [^]f32, DU: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---
	zlangt_ :: proc(norm: ^char, n: ^blasint, DL: [^]complex128, D: [^]complex128, DU: [^]complex128, l_norm: c.size_t = 1) -> f64 ---

	// lanhs: Hessenberg
	clanhs_ :: proc(norm: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---
	dlanhs_ :: proc(norm: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1) -> f64 ---
	slanhs_ :: proc(norm: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---
	zlanhs_ :: proc(norm: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1) -> f64 ---

	// lan{he,sy}: Hermitian/symmetric matrix
	clanhe_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	zlanhe_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---

	clansy_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	dlansy_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---
	slansy_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	zlansy_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---

	// lan{hf,sf}: Hermitian/symmetric matrix, RFP

	// lan{hp,sp}: Hermitian/symmetric matrix, packed
	clanhp_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	zlanhp_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---

	clansp_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	dlansp_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---
	slansp_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	zlansp_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---

	// lan{hb,sb}: Hermitian/symmetric matrix, banded
	clanhb_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, k: ^blasint, AB: [^]complex64, ldab: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	zlanhb_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, k: ^blasint, AB: [^]complex128, ldab: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---

	clansb_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, k: ^blasint, AB: [^]complex64, ldab: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	dlansb_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, k: ^blasint, AB: [^]f64, ldab: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---
	slansb_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, k: ^blasint, AB: [^]f32, ldab: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> lapack_float_return ---
	zlansb_ :: proc(norm: ^char, uplo: ^char, n: ^blasint, k: ^blasint, AB: [^]complex128, ldab: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1) -> f64 ---

	// lan{ht,st}: Hermitian/symmetric matrix, tridiagonal
	clanht_ :: proc(norm: ^char, n: ^blasint, D: [^]f32, E: [^]complex64, l_norm: c.size_t = 1) -> lapack_float_return ---
	zlanht_ :: proc(norm: ^char, n: ^blasint, D: [^]f64, E: [^]complex128, l_norm: c.size_t = 1) -> f64 ---

	dlanst_ :: proc(norm: ^char, n: ^blasint, D: [^]f64, E: [^]f64, l_norm: c.size_t = 1) -> f64 ---
	slanst_ :: proc(norm: ^char, n: ^blasint, D: [^]f32, E: [^]f32, l_norm: c.size_t = 1) -> lapack_float_return ---

	// lantr: triangular matrix
	clantr_ :: proc(norm: ^char, uplo: ^char, diag: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> lapack_float_return ---
	dlantr_ :: proc(norm: ^char, uplo: ^char, diag: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> f64 ---
	slantr_ :: proc(norm: ^char, uplo: ^char, diag: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> lapack_float_return ---
	zlantr_ :: proc(norm: ^char, uplo: ^char, diag: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> f64 ---

	// lantp: triangular matrix, packed
	clantp_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]complex64, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> lapack_float_return ---
	dlantp_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]f64, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> f64 ---
	slantp_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]f32, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> lapack_float_return ---
	zlantp_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]complex128, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> f64 ---

	// lantb: triangular matrix, banded
	clantb_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, k: ^blasint, AB: [^]complex64, ldab: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> lapack_float_return ---
	dlantb_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, k: ^blasint, AB: [^]f64, ldab: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> f64 ---
	slantb_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, k: ^blasint, AB: [^]f32, ldab: ^blasint, work: [^]f32, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> lapack_float_return ---
	zlantb_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, k: ^blasint, AB: [^]complex128, ldab: ^blasint, work: [^]f64, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) -> f64 ---

	// ===================================================================================
	// scalar operations
	// https://www.netlib.org/lapack/explore-html/db/dac/group__blas0__like__grp.html
	// ===================================================================================
	// isnan: test for NaN

	// laisnan: test for NaN, unoptimized

	// ladiv: complex divide

	// lapy2: robust sqrt( x^2 + y^2 )
	dlapy2_ :: proc(x: ^f64, y: ^f64) -> f64 ---
	slapy2_ :: proc(x: ^f32, y: ^f32) -> f32 ---

	// lapy3: robust sqrt( x^2 + y^2 + z^2 )
	dlapy3_ :: proc(x: ^f64, y: ^f64, z: ^f64) -> f64 ---
	slapy3_ :: proc(x: ^f32, y: ^f32, z: ^f32) -> f32 ---

	// larmm: scale factor to avoid overflow, step in latrs


	// ===================================================================================
	// blas level 1
	// https://www.netlib.org/lapack/explore-html/d5/dde/group__blas1__like__grp.html
	// ===================================================================================
	//  lacgv: conjugate vector
	clacgv_ :: proc(n: ^blasint, X: [^]complex64, incx: ^blasint) ---
	zlacgv_ :: proc(n: ^blasint, X: [^]complex128, incx: ^blasint) ---

	// lasrt: sort vector
	dlasrt_ :: proc(id: ^char, n: ^blasint, D: [^]f64, info: ^Info, l_id: c.size_t = 1) ---
	slasrt_ :: proc(id: ^char, n: ^blasint, D: [^]f32, info: ^Info, l_id: c.size_t = 1) ---

	// lassq: sum-of-squares, avoiding over/underflow
	classq_ :: proc(n: ^blasint, X: [^]complex64, incx: ^blasint, scale: ^f32, sumsq: ^f32) ---
	dlassq_ :: proc(n: ^blasint, X: [^]f64, incx: ^blasint, scale: ^f64, sumsq: ^f64) ---
	slassq_ :: proc(n: ^blasint, X: [^]f32, incx: ^blasint, scale: ^f32, sumsq: ^f32) ---
	zlassq_ :: proc(n: ^blasint, X: [^]complex128, incx: ^blasint, scale: ^f64, sumsq: ^f64) ---

	// rscl: scale vector by reciprocal

	// ===================================================================================
	// blas level 2
	// https://www.netlib.org/lapack/explore-html/d5/dde/group__blas2__like__grp.html
	// ===================================================================================
	// ilalc: find non-zero col

	// ilalr: find non-zero row

	// lascl: scale matrix
	clascl_ :: proc(type: ^char, kl: ^blasint, ku: ^blasint, cfrom: ^f32, cto: ^f32, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, info: ^Info, l_type: c.size_t = 1) ---
	dlascl_ :: proc(type: ^char, kl: ^blasint, ku: ^blasint, cfrom: ^f64, cto: ^f64, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, info: ^Info, l_type: c.size_t = 1) ---
	slascl_ :: proc(type: ^char, kl: ^blasint, ku: ^blasint, cfrom: ^f32, cto: ^f32, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, info: ^Info, l_type: c.size_t = 1) ---
	zlascl_ :: proc(type: ^char, kl: ^blasint, ku: ^blasint, cfrom: ^f64, cto: ^f64, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, info: ^Info, l_type: c.size_t = 1) ---

	// la_geamv: matrix-vector multiply |A| * |x|, general

	// la_gbamv: matrix-vector multiply |A| * |x|, general banded

	// la_heamv: matrix-vector multiply |A| * |x|, Hermitian/symmetric

	// lascl2: diagonal scale matrix, A = D A

	// larscl2: reciprocal diagonal scale matrix, A = D^{-1} A

	// la_wwaddw: add to double-double or single-single vector
	// ===================================================================================
	// blas level 3
	// https://www.netlib.org/lapack/explore-html/d5/dde/group__blas3__like__grp.html
	// ===================================================================================
	// lagtm: tridiagonal matrix-matrix multiply

	// lacrm: complex * real matrix-matrix multiply
	clacrm_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]f32, ldb: ^blasint, C: [^]complex64, ldc: ^blasint, rwork: [^]f32) ---
	zlacrm_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]f64, ldb: ^blasint, C: [^]complex128, ldc: ^blasint, rwork: [^]f64) ---

	// larcm: real * complex matrix-matrix multiply
	clarcm_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]complex64, ldb: ^blasint, C: [^]complex64, ldc: ^blasint, rwork: [^]f32) ---
	zlarcm_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]complex128, ldb: ^blasint, C: [^]complex128, ldc: ^blasint, rwork: [^]f64) ---

	// hfrk: Hermitian rank-k update, RFP format
	chfrk_ :: proc(transr: ^char, uplo: ^char, trans: ^char, n: ^blasint, k: ^blasint, alpha: ^f32, A: [^]complex64, lda: ^blasint, beta: ^f32, C: [^]complex64, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---
	zhfrk_ :: proc(transr: ^char, uplo: ^char, trans: ^char, n: ^blasint, k: ^blasint, alpha: ^f64, A: [^]complex128, lda: ^blasint, beta: ^f64, C: [^]complex128, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---

	// tfsm: triangular-matrix solve, RFP format
	ctfsm_ :: proc(transr: ^char, side: ^char, uplo: ^char, trans: ^char, diag: ^char, m: ^blasint, n: ^blasint, alpha: ^complex64, A: [^]complex64, B: [^]complex64, ldb: ^blasint, l_transr: c.size_t = 1, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtfsm_ :: proc(transr: ^char, side: ^char, uplo: ^char, trans: ^char, diag: ^char, m: ^blasint, n: ^blasint, alpha: ^f64, A: [^]f64, B: [^]f64, ldb: ^blasint, l_transr: c.size_t = 1, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	stfsm_ :: proc(transr: ^char, side: ^char, uplo: ^char, trans: ^char, diag: ^char, m: ^blasint, n: ^blasint, alpha: ^f32, A: [^]f32, B: [^]f32, ldb: ^blasint, l_transr: c.size_t = 1, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztfsm_ :: proc(transr: ^char, side: ^char, uplo: ^char, trans: ^char, diag: ^char, m: ^blasint, n: ^blasint, alpha: ^complex128, A: [^]complex128, B: [^]complex128, ldb: ^blasint, l_transr: c.size_t = 1, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---

	// RFP (Rectangular Full Packed) format: Level 3 BLAS-like (Computational)
	// SFRK: Symmetric rank-k update in RFP format
	dsfrk_ :: proc(transr: ^char, uplo: ^char, trans: ^char, n: ^blasint, k: ^blasint, alpha: ^f64, A: [^]f64, lda: ^blasint, beta: ^f64, C: [^]f64, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---
	ssfrk_ :: proc(transr: ^char, uplo: ^char, trans: ^char, n: ^blasint, k: ^blasint, alpha: ^f32, A: [^]f32, lda: ^blasint, beta: ^f32, C: [^]f32, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---

}
