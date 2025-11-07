package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// Standard SVD driver, A = UΣV^H
	// https://www.netlib.org/lapack/explore-html/d3/df8/group__svd__driver__grp.html
	// ===================================================================================
	// — Full —
	// Standard
	cgesvd_ :: proc(jobu: ^char, jobvt: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, S: [^]f32, U: [^]complex64, ldu: ^blasint, VT: [^]complex64, ldvt: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgesvd_ :: proc(jobu: ^char, jobvt: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, S: [^]f64, U: [^]f64, ldu: ^blasint, VT: [^]f64, ldvt: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgesvd_ :: proc(jobu: ^char, jobvt: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, S: [^]f32, U: [^]f32, ldu: ^blasint, VT: [^]f32, ldvt: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgesvd_ :: proc(jobu: ^char, jobvt: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, S: [^]f64, U: [^]complex128, ldu: ^blasint, VT: [^]complex128, ldvt: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	// QR with Pivoting
	cgesvdq_ :: proc(joba: ^char, jobp: ^char, jobr: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, S: [^]f32, U: [^]complex64, ldu: ^blasint, V: [^]complex64, ldv: ^blasint, numrank: ^blasint, iwork: [^]blasint, liwork: ^blasint, cwork: ^complex64, lcwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgesvdq_ :: proc(joba: ^char, jobp: ^char, jobr: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, S: [^]f64, U: [^]f64, ldu: ^blasint, V: [^]f64, ldv: ^blasint, numrank: ^blasint, iwork: [^]blasint, liwork: ^blasint, work: [^]f64, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgesvdq_ :: proc(joba: ^char, jobp: ^char, jobr: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, S: [^]f32, U: [^]f32, ldu: ^blasint, V: [^]f32, ldv: ^blasint, numrank: ^blasint, iwork: [^]blasint, liwork: ^blasint, work: [^]f32, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgesvdq_ :: proc(joba: ^char, jobp: ^char, jobr: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, S: [^]f64, U: [^]complex128, ldu: ^blasint, V: [^]complex128, ldv: ^blasint, numrank: ^blasint, iwork: [^]blasint, liwork: ^blasint, cwork: ^complex128, lcwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	// Divide & Conquer
	cgesdd_ :: proc(jobz: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, S: [^]f32, U: [^]complex64, ldu: ^blasint, VT: [^]complex64, ldvt: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---
	dgesdd_ :: proc(jobz: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, S: [^]f64, U: [^]f64, ldu: ^blasint, VT: [^]f64, ldvt: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---
	sgesdd_ :: proc(jobz: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, S: [^]f32, U: [^]f32, ldu: ^blasint, VT: [^]f32, ldvt: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---
	zgesdd_ :: proc(jobz: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, S: [^]f64, U: [^]complex128, ldu: ^blasint, VT: [^]complex128, ldvt: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---
	// Selective SVD
	cgesvdx_ :: proc(jobu: ^char, jobvt: ^char, range: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, ns: ^blasint, S: [^]f32, U: [^]complex64, ldu: ^blasint, VT: [^]complex64, ldvt: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgesvdx_ :: proc(jobu: ^char, jobvt: ^char, range: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, ns: ^blasint, S: [^]f64, U: [^]f64, ldu: ^blasint, VT: [^]f64, ldvt: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgesvdx_ :: proc(jobu: ^char, jobvt: ^char, range: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, ns: ^blasint, S: [^]f32, U: [^]f32, ldu: ^blasint, VT: [^]f32, ldvt: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgesvdx_ :: proc(jobu: ^char, jobvt: ^char, range: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, ns: ^blasint, S: [^]f64, U: [^]complex128, ldu: ^blasint, VT: [^]complex128, ldvt: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	// Jacobi SVD (High Accuracy)
	cgejsv_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, jobr: ^char, jobt: ^char, jobp: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, SVA: [^]f32, U: [^]complex64, ldu: ^blasint, V: [^]complex64, ldv: ^blasint, cwork: ^complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgejsv_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, jobr: ^char, jobt: ^char, jobp: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, SVA: [^]f64, U: [^]f64, ldu: ^blasint, V: [^]f64, ldv: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgejsv_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, jobr: ^char, jobt: ^char, jobp: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, SVA: [^]f32, U: [^]f32, ldu: ^blasint, V: [^]f32, ldv: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgejsv_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, jobr: ^char, jobt: ^char, jobp: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, SVA: [^]f64, U: [^]complex128, ldu: ^blasint, V: [^]complex128, ldv: ^blasint, cwork: ^complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	// Jacobi SVD Variant
	cgesvj_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, SVA: [^]f32, mv: ^blasint, V: [^]complex64, ldv: ^blasint, cwork: ^complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgesvj_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, SVA: [^]f64, mv: ^blasint, V: [^]f64, ldv: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgesvj_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, SVA: [^]f32, mv: ^blasint, V: [^]f32, ldv: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgesvj_ :: proc(joba: ^char, jobu: ^char, jobv: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, SVA: [^]f64, mv: ^blasint, V: [^]complex128, ldv: ^blasint, cwork: ^complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// — bidiagonal —

	cbdsqr_ :: proc(uplo: ^char, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: [^]f32, E: [^]f32, VT: [^]complex64, ldvt: ^blasint, U: [^]complex64, ldu: ^blasint, C: [^]complex64, ldc: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1) ---
	dbdsqr_ :: proc(uplo: ^char, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: [^]f64, E: [^]f64, VT: [^]f64, ldvt: ^blasint, U: [^]f64, ldu: ^blasint, C: [^]f64, ldc: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1) ---
	sbdsqr_ :: proc(uplo: ^char, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: [^]f32, E: [^]f32, VT: [^]f32, ldvt: ^blasint, U: [^]f32, ldu: ^blasint, C: [^]f32, ldc: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	zbdsqr_ :: proc(uplo: ^char, n: ^blasint, ncvt: ^blasint, nru: ^blasint, ncc: ^blasint, D: [^]f64, E: [^]f64, VT: [^]complex128, ldvt: ^blasint, U: [^]complex128, ldu: ^blasint, C: [^]complex128, ldc: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1) ---

	dbdsdc_ :: proc(uplo: ^char, compq: ^char, n: ^blasint, D: [^]f64, E: [^]f64, U: [^]f64, ldu: ^blasint, VT: [^]f64, ldvt: ^blasint, Q: [^]f64, IQ: ^blasint, work: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sbdsdc_ :: proc(uplo: ^char, compq: ^char, n: ^blasint, D: [^]f32, E: [^]f32, U: [^]f32, ldu: ^blasint, VT: [^]f32, ldvt: ^blasint, Q: [^]f32, IQ: ^blasint, work: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	dbdsvdx_ :: proc(uplo: ^char, jobz: ^char, range: ^char, n: ^blasint, D: [^]f64, E: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, ns: ^blasint, S: [^]f64, Z: ^f64, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sbdsvdx_ :: proc(uplo: ^char, jobz: ^char, range: ^char, n: ^blasint, D: [^]f32, E: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, ns: ^blasint, S: [^]f32, Z: ^f32, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// ===================================================================================
	// Generalized SVD driver
	// https://www.netlib.org/lapack/explore-html/df/ddb/group__ggsvd__driver__grp.html
	// ===================================================================================
	cggsvd3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: [^]complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, alpha: ^f32, beta: ^f32, U: [^]complex64, ldu: ^blasint, V: [^]complex64, ldv: ^blasint, Q: [^]complex64, ldq: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dggsvd3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: [^]f64, lda: ^blasint, B: ^f64, ldb: ^blasint, alpha: ^f64, beta: ^f64, U: [^]f64, ldu: ^blasint, V: [^]f64, ldv: ^blasint, Q: [^]f64, ldq: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sggsvd3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: [^]f32, lda: ^blasint, B: ^f32, ldb: ^blasint, alpha: ^f32, beta: ^f32, U: [^]f32, ldu: ^blasint, V: [^]f32, ldv: ^blasint, Q: [^]f32, ldq: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zggsvd3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, A: [^]complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, alpha: ^f64, beta: ^f64, U: [^]complex128, ldu: ^blasint, V: [^]complex128, ldv: ^blasint, Q: [^]complex128, ldq: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cggsvd_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: [^]complex64, lda: ^blasint, b: ^complex64, ldb: ^blasint, alpha: ^f32, beta: ^f32, u: [^]complex64, ldu: ^blasint, v: [^]complex64, ldv: ^blasint, q: [^]complex64, ldq: ^blasint, work: [^]complex64, rwork: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sggsvd_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: [^]f32, lda: ^blasint, b: ^f32, ldb: ^blasint, alpha: ^f32, beta: ^f32, u: [^]f32, ldu: ^blasint, v: [^]f32, ldv: ^blasint, q: [^]f32, ldq: ^blasint, work: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dggsvd_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: [^]f64, lda: ^blasint, b: ^f64, ldb: ^blasint, alpha: ^f64, beta: ^f64, u: [^]f64, ldu: ^blasint, v: [^]f64, ldv: ^blasint, q: [^]f64, ldq: ^blasint, work: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zggsvd_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, n: ^blasint, p: ^blasint, k: ^blasint, l: ^blasint, a: [^]complex128, lda: ^blasint, b: ^complex128, ldb: ^blasint, alpha: ^f64, beta: ^f64, u: [^]complex128, ldu: ^blasint, v: [^]complex128, ldv: ^blasint, q: [^]complex128, ldq: ^blasint, work: [^]complex128, rwork: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	sggsvp_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, a: [^]f32, lda: ^blasint, b: ^f32, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, u: [^]f32, ldu: ^blasint, v: [^]f32, ldv: ^blasint, q: [^]f32, ldq: ^blasint, iwork: [^]blasint, tau: ^f32, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dggsvp_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, a: [^]f64, lda: ^blasint, b: ^f64, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, u: [^]f64, ldu: ^blasint, v: [^]f64, ldv: ^blasint, q: [^]f64, ldq: ^blasint, iwork: [^]blasint, tau: ^f64, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	cggsvp_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, a: [^]complex64, lda: ^blasint, b: ^complex64, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, u: [^]complex64, ldu: ^blasint, v: [^]complex64, ldv: ^blasint, q: [^]complex64, ldq: ^blasint, iwork: [^]blasint, rwork: [^]f32, tau: ^complex64, work: [^]complex64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zggsvp_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, a: [^]complex128, lda: ^blasint, b: ^complex128, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, u: [^]complex128, ldu: ^blasint, v: [^]complex128, ldv: ^blasint, q: [^]complex128, ldq: ^blasint, iwork: [^]blasint, rwork: [^]f64, tau: ^complex128, work: [^]complex128, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cggsvp3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, U: [^]complex64, ldu: ^blasint, V: [^]complex64, ldv: ^blasint, Q: [^]complex64, ldq: ^blasint, iwork: [^]blasint, rwork: [^]f32, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dggsvp3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: ^f64, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, U: [^]f64, ldu: ^blasint, V: [^]f64, ldv: ^blasint, Q: [^]f64, ldq: ^blasint, iwork: [^]blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sggsvp3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: ^f32, ldb: ^blasint, tola: ^f32, tolb: ^f32, k: ^blasint, l: ^blasint, U: [^]f32, ldu: ^blasint, V: [^]f32, ldv: ^blasint, Q: [^]f32, ldq: ^blasint, iwork: [^]blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zggsvp3_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, tola: ^f64, tolb: ^f64, k: ^blasint, l: ^blasint, U: [^]complex128, ldu: ^blasint, V: [^]complex128, ldv: ^blasint, Q: [^]complex128, ldq: ^blasint, iwork: [^]blasint, rwork: [^]f64, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// ===================================================================================
	// SVD computational routines
	// https://www.netlib.org/lapack/explore-html/d8/de0/group__gesvd__comp__grp.html
	// ===================================================================================
	// Reduction to Bidiagonal
	cgebrd_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, D: [^]f32, E: [^]f32, tauq: ^complex64, taup: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgebrd_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, D: [^]f64, E: [^]f64, tauq: ^f64, taup: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgebrd_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, D: [^]f32, E: [^]f32, tauq: ^f32, taup: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgebrd_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, D: [^]f64, E: [^]f64, tauq: ^complex128, taup: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	// gebd2: reduction to bidiagonal, level 2

	// labrd: step in gebrd

	// Banded to Bidiagonal
	cgbbrd_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex64, ldab: ^blasint, D: [^]f32, E: [^]f32, Q: [^]complex64, ldq: ^blasint, PT: [^]complex64, ldpt: ^blasint, C: [^]complex64, ldc: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, _: c.size_t = 1) ---
	dgbbrd_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f64, ldab: ^blasint, D: [^]f64, E: [^]f64, Q: [^]f64, ldq: ^blasint, PT: [^]f64, ldpt: ^blasint, C: [^]f64, ldc: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1) ---
	sgbbrd_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^f32, ldab: ^blasint, D: [^]f32, E: [^]f32, Q: [^]f32, ldq: ^blasint, PT: [^]f32, ldpt: ^blasint, C: [^]f32, ldc: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	zgbbrd_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, ncc: ^blasint, kl: ^blasint, ku: ^blasint, AB: ^complex128, ldab: ^blasint, D: [^]f64, E: [^]f64, Q: [^]complex128, ldq: ^blasint, PT: [^]complex128, ldpt: ^blasint, C: [^]complex128, ldc: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, _: c.size_t = 1) ---
	// generate Q, P from gebrd
	cungbr_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1) ---
	zungbr_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1) ---
	dorgbr_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1) ---
	sorgbr_ :: proc(vect: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1) ---

	// multiply by Q, P from gebrd
	cunmbr_ :: proc(vect: ^char, side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmbr_ :: proc(vect: ^char, side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dormbr_ :: proc(vect: ^char, side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sormbr_ :: proc(vect: ^char, side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// Generalized Singular Value Decomposition (Computational)
	// https://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing.html
	// TGSJA: Compute generalized singular value decomposition
	ctgsja_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]complex64, lda: ^blasint, B: ^complex64, ldb: ^blasint, tola: ^f32, tolb: ^f32, alpha: ^f32, beta: ^f32, U: [^]complex64, ldu: ^blasint, V: [^]complex64, ldv: ^blasint, Q: [^]complex64, ldq: ^blasint, work: [^]complex64, ncycle: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dtgsja_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]f64, lda: ^blasint, B: ^f64, ldb: ^blasint, tola: ^f64, tolb: ^f64, alpha: ^f64, beta: ^f64, U: [^]f64, ldu: ^blasint, V: [^]f64, ldv: ^blasint, Q: [^]f64, ldq: ^blasint, work: [^]f64, ncycle: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	stgsja_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]f32, lda: ^blasint, B: ^f32, ldb: ^blasint, tola: ^f32, tolb: ^f32, alpha: ^f32, beta: ^f32, U: [^]f32, ldu: ^blasint, V: [^]f32, ldv: ^blasint, Q: [^]f32, ldq: ^blasint, work: [^]f32, ncycle: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	ztgsja_ :: proc(jobu: ^char, jobv: ^char, jobq: ^char, m: ^blasint, p: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]complex128, lda: ^blasint, B: ^complex128, ldb: ^blasint, tola: ^f64, tolb: ^f64, alpha: ^f64, beta: ^f64, U: [^]complex128, ldu: ^blasint, V: [^]complex128, ldv: ^blasint, Q: [^]complex128, ldq: ^blasint, work: [^]complex128, ncycle: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// — auxiliary routines —

	// gsvj0: step in gesvj

	// gsvj1: step in gesvj

	// las2: 2x2 triangular SVD

	// lasv2: 2x2 triangular SVD

	// lartgs: generate plane rotation for bidiag SVD

	// ===================================================================================
	// Generalized SVD computational routines
	// https://www.netlib.org/lapack/explore-html/df/d5e/group__ggsvd__comp__grp.html
	// ===================================================================================

	// ggsvp3: step in ggsvd

	// tgsja: generalized SVD of trapezoidal matrices, step in ggsvd3

	// lags2: 2x2 orthogonal factor, step in tgsja

	// lapll: linear dependence of 2 vectors

	// ===================================================================================
	// bidiag QR iteration routines
	// https://www.netlib.org/lapack/explore-html/d9/dd0/group__lasq__comp__grp.html
	// ===================================================================================

	// lasq1: dqds step

	// lasq2: dqds step

	// lasq3: dqds step

	// lasq4: dqds step

	// lasq5: dqds step

	// lasq6: dqds step

	// ===================================================================================
	// bidiag D&C routines
	// https://www.netlib.org/lapack/explore-html/d0/d99/group__lasd__comp__grp.html
	// ===================================================================================
	// lasd0: D&C step: top level solver

	// lasdt: D&C step: tree

	// lasd1: D&C step: merge subproblems

	// lasd2: D&C step: deflation

	// lasd3: D&C step: secular equation

	// lasd4: D&C step: secular equation nonlinear solver

	// lasd5: D&C step: secular equation, 2x2

	// lasdq: D&C step: leaf using bdsqr

	// — singular values only or factored form —

	// lasda: D&C step: top level solver

	// lasd6: D&C step: merge subproblems

	// lasd7: D&C step: deflation

	// lasd8: D&C step: secular equation
}
