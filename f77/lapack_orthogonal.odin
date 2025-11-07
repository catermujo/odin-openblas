package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// QR
	// https://www.netlib.org/lapack/explore-html/d8/d93/group__geqr__comp__grp.html
	// ===================================================================================
	// -- flexible --
	cgeqr_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, tsize: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgeqr_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, tsize: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgeqr_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, tsize: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgeqr_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, tsize: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	cgemqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, tsize: ^blasint, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgemqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, tsize: ^blasint, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgemqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, tsize: ^blasint, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgemqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, tsize: ^blasint, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// -- classic --
	cgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgeqrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	cgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, info: ^Info) ---
	dgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, info: ^Info) ---
	sgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, info: ^Info) ---
	zgeqr2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, info: ^Info) ---

	cungqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	zungqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---
	cungrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	zungrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	dorgqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sorgqr_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---

	// {un,or}g2r: generate explicit Q from geqrf, level 2

	cunmqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	dormqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sormqr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// -- with T --
	cgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, ldt: ^blasint, work: [^]complex64, info: ^Info) ---
	dgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, ldt: ^blasint, work: [^]f64, info: ^Info) ---
	sgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, ldt: ^blasint, work: [^]f32, info: ^Info) ---
	zgeqrt_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, ldt: ^blasint, work: [^]complex128, info: ^Info) ---

	cgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, ldt: ^blasint, info: ^Info) ---
	dgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, ldt: ^blasint, info: ^Info) ---
	sgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, ldt: ^blasint, info: ^Info) ---
	zgeqrt2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, ldt: ^blasint, info: ^Info) ---

	cgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, ldt: ^blasint, info: ^Info) ---
	dgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, ldt: ^blasint, info: ^Info) ---
	sgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, ldt: ^blasint, info: ^Info) ---
	zgeqrt3_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, ldt: ^blasint, info: ^Info) ---

	cgemqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: [^]complex64, ldv: ^blasint, T: [^]complex64, ldt: ^blasint, C: [^]complex64, ldc: ^blasint, work: [^]complex64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgemqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: [^]f64, ldv: ^blasint, T: [^]f64, ldt: ^blasint, C: [^]f64, ldc: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgemqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: [^]f32, ldv: ^blasint, T: [^]f32, ldt: ^blasint, C: [^]f32, ldc: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgemqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, nb: ^blasint, V: [^]complex128, ldv: ^blasint, T: [^]complex128, ldt: ^blasint, C: [^]complex128, ldc: ^blasint, work: [^]complex128, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// -- positive --
	cgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgeqrfp_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	// geqr2p: QR factor, diag( R ) ≥ 0, level 2

	// ===================================================================================
	// QR with pivoting
	// https://www.netlib.org/lapack/explore-html/da/dd7/group__geqpf__comp__grp.html
	// ===================================================================================
	cgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, JPVT: [^]blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info) ---
	dgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, JPVT: [^]blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, JPVT: [^]blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgeqp3_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, JPVT: [^]blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info) ---

	// laqp2: step of geqp3

	// laqps: step of geqp3

	// not documented right:
	sgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: [^]f32, lda: ^blasint, jpvt: [^]blasint, tau: ^f32, work: [^]f32, info: ^Info) ---
	dgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: [^]f64, lda: ^blasint, jpvt: [^]blasint, tau: ^f64, work: [^]f64, info: ^Info) ---
	cgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: [^]complex64, lda: ^blasint, jpvt: [^]blasint, tau: ^complex64, work: [^]complex64, rwork: [^]f32, info: ^Info) ---
	zgeqpf_ :: proc(m: ^blasint, n: ^blasint, a: [^]complex128, lda: ^blasint, jpvt: [^]blasint, tau: ^complex128, work: [^]complex128, rwork: [^]f64, info: ^Info) ---


	// ===================================================================================
	// QR Tall Skinny
	// https://www.netlib.org/lapack/explore-html/d3/dc8/group__getsqr__comp__grp.html
	// ===================================================================================
	// 	latsqr: tall-skinny QR factor

	// {un,or}gtsqr: generate Q from latsqr

	cgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, ldt: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, ldt: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, ldt: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgetsqrhrt_ :: proc(m: ^blasint, n: ^blasint, mb1: ^blasint, nb1: ^blasint, nb2: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, ldt: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info) ---


	cungtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, ldt: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	zungtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, ldt: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	dorgtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, ldt: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sorgtsqr_row_ :: proc(m: ^blasint, n: ^blasint, mb: ^blasint, nb: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, ldt: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---

	// larfb_gett: step in ungtsqr_row

	// lamtsqr: multiply by Q from latsqr

	// getsqrhrt: tall-skinny QR factor, with Householder reconstruction

	cunhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, ldt: ^blasint, D: [^]complex64, info: ^Info) ---
	zunhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, ldt: ^blasint, D: [^]complex128, info: ^Info) ---

	dorhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, ldt: ^blasint, D: [^]f64, info: ^Info) ---
	sorhr_col_ :: proc(m: ^blasint, n: ^blasint, nb: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, ldt: ^blasint, D: [^]f32, info: ^Info) ---

	// la{un,or}hr_col_getrfnp: LU factor without pivoting

	// la{un,or}hr_col_getrfnp2: LU factor without pivoting, level 2

	// ===================================================================================
	// QR triangular-pantagonal
	// https://www.netlib.org/lapack/explore-html/da/dd4/group__tpqr__comp__grp.html
	// ===================================================================================
	ctpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, T: [^]complex64, ldt: ^blasint, work: [^]complex64, info: ^Info) ---
	dtpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, T: [^]f64, ldt: ^blasint, work: [^]f64, info: ^Info) ---
	stpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, T: [^]f32, ldt: ^blasint, work: [^]f32, info: ^Info) ---
	ztpqrt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, nb: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, T: [^]complex128, ldt: ^blasint, work: [^]complex128, info: ^Info) ---

	ctpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, T: [^]complex64, ldt: ^blasint, info: ^Info) ---
	dtpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, T: [^]f64, ldt: ^blasint, info: ^Info) ---
	stpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, T: [^]f32, ldt: ^blasint, info: ^Info) ---
	ztpqrt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, T: [^]complex128, ldt: ^blasint, info: ^Info) ---

	ctpmqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: [^]complex64, ldv: ^blasint, T: [^]complex64, ldt: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtpmqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: [^]f64, ldv: ^blasint, T: [^]f64, ldt: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	stpmqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: [^]f32, ldv: ^blasint, T: [^]f32, ldt: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztpmqrt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, nb: ^blasint, V: [^]complex128, ldv: ^blasint, T: [^]complex128, ldt: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	ctprfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: [^]complex64, ldv: ^blasint, T: [^]complex64, ldt: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dtprfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: [^]f64, ldv: ^blasint, T: [^]f64, ldt: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	stprfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: [^]f32, ldv: ^blasint, T: [^]f32, ldt: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	ztprfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, V: [^]complex128, ldv: ^blasint, T: [^]complex128, ldt: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// ===================================================================================
	// Generalized QR
	// https://www.netlib.org/lapack/explore-html/d1/d52/group__ggqr__comp__grp.html
	// ===================================================================================
	cggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]complex64, lda: ^blasint, taua: ^complex64, B: [^]complex64, ldb: ^blasint, taub: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]f64, lda: ^blasint, taua: ^f64, B: [^]f64, ldb: ^blasint, taub: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]f32, lda: ^blasint, taua: ^f32, B: [^]f32, ldb: ^blasint, taub: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zggqrf_ :: proc(n: ^blasint, m: ^blasint, p: ^blasint, A: [^]complex128, lda: ^blasint, taua: ^complex128, B: [^]complex128, ldb: ^blasint, taub: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---


	// ===================================================================================
	// LQ
	// https://www.netlib.org/lapack/explore-html/d1/d75/group__gelq__comp__grp.html
	// ===================================================================================
	// -- flexible --
	cgelq_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, tsize: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgelq_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, tsize: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgelq_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, tsize: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgelq_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, tsize: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	cgemlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, T: [^]complex64, tsize: ^blasint, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgemlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, T: [^]f64, tsize: ^blasint, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgemlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, T: [^]f32, tsize: ^blasint, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgemlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, T: [^]complex128, tsize: ^blasint, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// -- classic --
	cgelqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgelqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgelqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgelqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	cgelq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, info: ^Info) ---
	dgelq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, info: ^Info) ---
	sgelq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, info: ^Info) ---
	zgelq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, info: ^Info) ---

	cunglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	zunglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	dorglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sorglq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---

	// {un,or}gl2: generate explicit Q, level 2, step in unglq

	cunmlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	dormlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sormlq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// {un,or}ml2: multiply by Q, level 2, step in unmlq

	// -- with T --
	// gelqt: LQ factor, with T

	// gelqt3: LQ factor, with T, recursive

	// gemlqt: multiply by Q from gelqt

	// ===================================================================================
	// LQ, short wide
	// https://www.netlib.org/lapack/explore-html/d8/d15/group__geswlq__comp__grp.html
	// ===================================================================================

	// laswlq: short-wide LQ factor

	// lamswlq: multiply by Q from laswlq

	// ===================================================================================
	// LQ, triangular-pentagonal
	// https://www.netlib.org/lapack/explore-html/de/d49/group__tplq__comp__grp.html
	// ===================================================================================

	ctplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, T: [^]complex64, ldt: ^blasint, work: [^]complex64, info: ^Info) ---
	dtplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, T: [^]f64, ldt: ^blasint, work: [^]f64, info: ^Info) ---
	stplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, T: [^]f32, ldt: ^blasint, work: [^]f32, info: ^Info) ---
	ztplqt_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, mb: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, T: [^]complex128, ldt: ^blasint, work: [^]complex128, info: ^Info) ---

	ctplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, T: [^]complex64, ldt: ^blasint, info: ^Info) ---
	dtplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, T: [^]f64, ldt: ^blasint, info: ^Info) ---
	stplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, T: [^]f32, ldt: ^blasint, info: ^Info) ---
	ztplqt2_ :: proc(m: ^blasint, n: ^blasint, l: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, T: [^]complex128, ldt: ^blasint, info: ^Info) ---

	ctpmlqt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: [^]complex64, ldv: ^blasint, T: [^]complex64, ldt: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtpmlqt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: [^]f64, ldv: ^blasint, T: [^]f64, ldt: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	stpmlqt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: [^]f32, ldv: ^blasint, T: [^]f32, ldt: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztpmlqt_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, mb: ^blasint, V: [^]complex128, ldv: ^blasint, T: [^]complex128, ldt: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// ===================================================================================
	// QL
	// https://www.netlib.org/lapack/explore-html/d3/d89/group__geql__comp__grp.html
	// ===================================================================================
	cgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgeqlf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	cgeql2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, info: ^Info) ---
	dgeql2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, info: ^Info) ---
	sgeql2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, info: ^Info) ---
	zgeql2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, info: ^Info) ---

	cungql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	zungql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	dorgql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sorgql_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---

	cunmql_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmql_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	dormql_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sormql_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// {un,or}g2l: step in ungql

	// {un,or}m2l: step in unmql

	// ===================================================================================
	// RQ
	// https://www.netlib.org/lapack/explore-html/df/d5e/group__gerq__comp__grp.html
	// ===================================================================================
	cgerqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgerqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgerqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgerqf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	cgerq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, info: ^Info) ---
	dgerq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, info: ^Info) ---
	sgerq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, info: ^Info) ---
	zgerq2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, info: ^Info) ---

	// {un,or}grq: generate explicit Q from gerqf

	cunmrq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmrq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// {un,or}mr2: step in unmrq

	// {un,or}gr2: step in ungrq

	// RQ factorization: Orthogonal matrix generation/multiplication (Computational)
	// https://www.netlib.org/lapack/explore-html/d9/da1/group__rq.html
	// ORGRQ: Generate orthogonal matrix Q from RQ factorization
	dorgrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sorgrq_ :: proc(m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---

	// ORMRQ: Multiply by orthogonal matrix Q from RQ factorization
	dormrq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sormrq_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---


	// ===================================================================================
	// Generalized RQ
	// https://www.netlib.org/lapack/explore-html/dd/dd6/group__ggrq__comp__grp.html
	// ===================================================================================
	cggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, taua: ^complex64, B: [^]complex64, ldb: ^blasint, taub: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, taua: ^f64, B: [^]f64, ldb: ^blasint, taub: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, taua: ^f32, B: [^]f32, ldb: ^blasint, taub: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zggrqf_ :: proc(m: ^blasint, p: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, taua: ^complex128, B: [^]complex128, ldb: ^blasint, taub: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	// ===================================================================================
	// RZ
	// https://www.netlib.org/lapack/explore-html/dc/dd8/group__gerz__comp__grp.html
	// ===================================================================================
	ctzrzf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dtzrzf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	stzrzf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	ztzrzf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	// latrz: RZ factor step

	cunmrz_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmrz_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	cunmtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	dormrz_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sormrz_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, k: ^blasint, l: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// {un,or}mr3: step in unmrz

	// larz: apply reflector

	// larzb: apply block reflector

	// larzt: generate T matrix

	// ===================================================================================
	// Cosine-Sine (CS) decomposition
	// https://www.netlib.org/lapack/explore-html/d7/d4d/group__gecs__comp__grp.html
	// ===================================================================================
	cbbcsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f32, phi: ^f32, U1: [^]complex64, ldu1: ^blasint, U2: [^]complex64, ldu2: ^blasint, V1T: [^]complex64, ldv1t: ^blasint, V2T: [^]complex64, ldv2t: ^blasint, B11D: [^]f32, B11E: [^]f32, B12D: [^]f32, B12E: [^]f32, B21D: [^]f32, B21E: [^]f32, B22D: [^]f32, B22E: [^]f32, rwork: [^]f32, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dbbcsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f64, phi: ^f64, U1: [^]f64, ldu1: ^blasint, U2: [^]f64, ldu2: ^blasint, V1T: [^]f64, ldv1t: ^blasint, V2T: [^]f64, ldv2t: ^blasint, B11D: [^]f64, B11E: [^]f64, B12D: [^]f64, B12E: [^]f64, b21d: [^]f64, b21e: [^]f64, b22d: [^]f64, b22e: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sbbcsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f32, phi: ^f32, U1: [^]f32, ldu1: ^blasint, U2: [^]f32, ldu2: ^blasint, V1T: [^]f32, ldv1t: ^blasint, V2T: [^]f32, ldv2t: ^blasint, B11D: [^]f32, B11E: [^]f32, B12D: [^]f32, B12E: [^]f32, B21D: [^]f32, B21E: [^]f32, B22D: [^]f32, B22E: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zbbcsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, m: ^blasint, p: ^blasint, q: ^blasint, theta: ^f64, phi: ^f64, U1: [^]complex128, ldu1: ^blasint, U2: [^]complex128, ldu2: ^blasint, V1T: [^]complex128, ldv1t: ^blasint, V2T: [^]complex128, ldv2t: ^blasint, B11D: [^]f64, B11E: [^]f64, B12D: [^]f64, B12E: [^]f64, B21D: [^]f64, B21E: [^]f64, B22D: [^]f64, B22E: [^]f64, rwork: [^]f64, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cuncsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]complex64, ldx11: ^blasint, X12: [^]complex64, ldx12: ^blasint, X21: [^]complex64, ldx21: ^blasint, X22: [^]complex64, ldx22: ^blasint, theta: ^f32, U1: [^]complex64, ldu1: ^blasint, U2: [^]complex64, ldu2: ^blasint, V1T: [^]complex64, ldv1t: ^blasint, V2T: [^]complex64, ldv2t: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zuncsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]complex128, ldx11: ^blasint, X12: [^]complex128, ldx12: ^blasint, X21: [^]complex128, ldx21: ^blasint, X22: [^]complex128, ldx22: ^blasint, theta: ^f64, U1: [^]complex128, ldu1: ^blasint, U2: [^]complex128, ldu2: ^blasint, V1T: [^]complex128, ldv1t: ^blasint, V2T: [^]complex128, ldv2t: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	dorcsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]f64, ldx11: ^blasint, X12: [^]f64, ldx12: ^blasint, X21: [^]f64, ldx21: ^blasint, X22: [^]f64, ldx22: ^blasint, theta: ^f64, U1: [^]f64, ldu1: ^blasint, U2: [^]f64, ldu2: ^blasint, V1T: [^]f64, ldv1t: ^blasint, V2T: [^]f64, ldv2t: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sorcsd_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, jobv2t: ^char, trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]f32, ldx11: ^blasint, X12: [^]f32, ldx12: ^blasint, X21: [^]f32, ldx21: ^blasint, X22: [^]f32, ldx22: ^blasint, theta: ^f32, U1: [^]f32, ldu1: ^blasint, U2: [^]f32, ldu2: ^blasint, V1T: [^]f32, ldv1t: ^blasint, V2T: [^]f32, ldv2t: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cuncsd2by1_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]complex64, ldx11: ^blasint, X21: [^]complex64, ldx21: ^blasint, theta: ^f32, U1: [^]complex64, ldu1: ^blasint, U2: [^]complex64, ldu2: ^blasint, V1T: [^]complex64, ldv1t: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zuncsd2by1_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]complex128, ldx11: ^blasint, X21: [^]complex128, ldx21: ^blasint, theta: ^f64, U1: [^]complex128, ldu1: ^blasint, U2: [^]complex128, ldu2: ^blasint, V1T: [^]complex128, ldv1t: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	dorcsd2by1_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]f64, ldx11: ^blasint, X21: [^]f64, ldx21: ^blasint, theta: ^f64, U1: [^]f64, ldu1: ^blasint, U2: [^]f64, ldu2: ^blasint, V1T: [^]f64, ldv1t: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sorcsd2by1_ :: proc(jobu1: ^char, jobu2: ^char, jobv1t: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]f32, ldx11: ^blasint, X21: [^]f32, ldx21: ^blasint, theta: ^f32, U1: [^]f32, ldu1: ^blasint, U2: [^]f32, ldu2: ^blasint, V1T: [^]f32, ldv1t: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cunbdb_ :: proc(trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]complex64, ldx11: ^blasint, X12: [^]complex64, ldx12: ^blasint, X21: [^]complex64, ldx21: ^blasint, X22: [^]complex64, ldx22: ^blasint, theta: ^f32, phi: ^f32, TAUP1: ^complex64, TAUP2: [^]complex64, TAUQ1: [^]complex64, TAUQ2: [^]complex64, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zunbdb_ :: proc(trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]complex128, ldx11: ^blasint, X12: [^]complex128, ldx12: ^blasint, X21: [^]complex128, ldx21: ^blasint, X22: [^]complex128, ldx22: ^blasint, theta: ^f64, phi: ^f64, TAUP1: ^complex128, TAUP2: [^]complex128, TAUQ1: [^]complex128, TAUQ2: [^]complex128, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	dorbdb_ :: proc(trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]f64, ldx11: ^blasint, X12: [^]f64, ldx12: ^blasint, X21: [^]f64, ldx21: ^blasint, X22: [^]f64, ldx22: ^blasint, theta: ^f64, phi: ^f64, TAUP1: ^f64, TAUP2: [^]f64, TAUQ1: [^]f64, TAUQ2: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sorbdb_ :: proc(trans: ^char, signs: ^char, m: ^blasint, p: ^blasint, q: ^blasint, X11: [^]f32, ldx11: ^blasint, X12: [^]f32, ldx12: ^blasint, X21: [^]f32, ldx21: ^blasint, X22: [^]f32, ldx22: ^blasint, theta: ^f32, phi: ^f32, TAUP1: ^f32, TAUP2: [^]f32, TAUQ1: [^]f32, TAUQ2: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// {un,or}bdb1: step in uncsd2by1

	// {un,or}bdb2: step in uncsd2by1

	// {un,or}bdb3: step in uncsd2by1

	// {un,or}bdb4: step in uncsd2by1

	// {un,or}bdb5: step in uncsd2by1

	// {un,or}bdb6: step in uncsd2by1

	// lapmr: permute rows

	// lapmt: permute cols

	// ===================================================================================
	// Householder reflectors
	// https://www.netlib.org/lapack/explore-html/d7/d09/group__reflector__aux__grp.html
	// ===================================================================================
	clarf_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]complex64, incv: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, _: c.size_t = 1) ---
	dlarf_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]f64, incv: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, _: c.size_t = 1) ---
	slarf_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]f32, incv: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, _: c.size_t = 1) ---
	zlarf_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]complex128, incv: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, _: c.size_t = 1) ---

	clarfx_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]complex64, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, _: c.size_t = 1) ---
	dlarfx_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]f64, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, _: c.size_t = 1) ---
	slarfx_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]f32, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, _: c.size_t = 1) ---
	zlarfx_ :: proc(side: ^char, m: ^blasint, n: ^blasint, V: [^]complex128, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, _: c.size_t = 1) ---

	// larfy: apply Householder reflector symmetrically (2-sided)

	clarfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, V: [^]complex64, ldv: ^blasint, T: [^]complex64, ldt: ^blasint, C: [^]complex64, ldc: ^blasint, work: [^]complex64, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dlarfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, V: [^]f64, ldv: ^blasint, T: [^]f64, ldt: ^blasint, C: [^]f64, ldc: ^blasint, work: [^]f64, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	slarfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, V: [^]f32, ldv: ^blasint, T: [^]f32, ldt: ^blasint, C: [^]f32, ldc: ^blasint, work: [^]f32, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zlarfb_ :: proc(side: ^char, trans: ^char, direct: ^char, storev: ^char, m: ^blasint, n: ^blasint, k: ^blasint, V: [^]complex128, ldv: ^blasint, T: [^]complex128, ldt: ^blasint, C: [^]complex128, ldc: ^blasint, work: [^]complex128, ldwork: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	clarfg_ :: proc(n: ^blasint, alpha: ^complex64, X: ^complex64, incx: ^blasint, tau: ^complex64) ---
	dlarfg_ :: proc(n: ^blasint, alpha: ^f64, X: ^f64, incx: ^blasint, tau: ^f64) ---
	slarfg_ :: proc(n: ^blasint, alpha: ^f32, X: ^f32, incx: ^blasint, tau: ^f32) ---
	zlarfg_ :: proc(n: ^blasint, alpha: ^complex128, X: ^complex128, incx: ^blasint, tau: ^complex128) ---

	// larfgp: generate Householder reflector, beta ≥ 0

	clarft_ :: proc(direct: ^char, storev: ^char, n: ^blasint, k: ^blasint, V: [^]complex64, ldv: ^blasint, tau: ^complex64, T: [^]complex64, ldt: ^blasint, _: c.size_t = 1, _: c.size_t = 1) ---
	dlarft_ :: proc(direct: ^char, storev: ^char, n: ^blasint, k: ^blasint, V: [^]f64, ldv: ^blasint, tau: ^f64, T: [^]f64, ldt: ^blasint, _: c.size_t = 1, _: c.size_t = 1) ---
	slarft_ :: proc(direct: ^char, storev: ^char, n: ^blasint, k: ^blasint, V: [^]f32, ldv: ^blasint, tau: ^f32, T: [^]f32, ldt: ^blasint, _: c.size_t = 1, _: c.size_t = 1) ---
	zlarft_ :: proc(direct: ^char, storev: ^char, n: ^blasint, k: ^blasint, V: [^]complex128, ldv: ^blasint, tau: ^complex128, T: [^]complex128, ldt: ^blasint, _: c.size_t = 1, _: c.size_t = 1) ---


	// ===================================================================================
	// Givens/Jacobi plane rotations
	// https://www.netlib.org/lapack/explore-html/da/d81/group__rot__aux__grp.html
	// ===================================================================================

	// lartg: generate plane rotation, more accurate than BLAS rot

	dlartgp_ :: proc(f: ^f64, g: ^f64, cs: ^f64, sn: ^f64, r: ^f64) ---
	slartgp_ :: proc(f: ^f32, g: ^f32, cs: ^f32, sn: ^f32, r: ^f32) ---

	dlasr_ :: proc(side: ^char, pivot: ^char, direct: ^char, m: ^blasint, n: ^blasint, c_: ^f64, s: ^f64, A: [^]f64, lda: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	slasr_ :: proc(side: ^char, pivot: ^char, direct: ^char, m: ^blasint, n: ^blasint, c_: ^f32, s: ^f32, A: [^]f32, lda: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	clasr_ :: proc(side: ^char, pivot: ^char, direct: ^char, m: ^blasint, n: ^blasint, c_: ^f32, s: ^f32, A: [^]complex64, lda: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zlasr_ :: proc(side: ^char, pivot: ^char, direct: ^char, m: ^blasint, n: ^blasint, c_: ^f64, s: ^f64, A: [^]complex128, lda: ^blasint, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// largv: generate vector of plane rotations

	// lartv: apply vector of plane rotations to vectors

	// lar2v: apply vector of plane rotations to 2x2 matrices

	// lacrt: apply plane rotation (unused?)
}
