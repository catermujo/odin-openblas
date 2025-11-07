package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// Standard eig driver, AV = VΛ
	// https://www.netlib.org/lapack/explore-html/db/d88/group__heev__driver__grp.html
	// ===================================================================================

	// — full —
	cheev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, W: [^]f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, W: [^]f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, W: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, W: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cheevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, W: [^]f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, W: [^]f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, W: [^]f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, W: [^]f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cheevr_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheevr_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyevr_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyevr_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cheevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---


	// — full, 2-stage —
	cheev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, W: [^]f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, W: [^]f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, W: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, W: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---


	cheevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, W: [^]f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, W: [^]f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, W: [^]f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, W: [^]f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cheevr_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheevr_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyevr_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyevr_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cheevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zheevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsyevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssyevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// — packed —

	chpev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhpev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dspev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	sspev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chpevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhpevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dspevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	sspevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chpevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhpevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dspevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	sspevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// — banded —

	chbev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbev_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chbevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbevd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chbevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, Q: [^]complex64, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, Q: [^]complex128, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, Q: [^]f64, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbevx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, Q: [^]f32, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// — banded, 2nd-stage —

	chbev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbev_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chbevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbevd_2stage_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chbevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, Q: [^]complex64, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, Q: [^]complex128, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, Q: [^]f64, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbevx_2stage_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, Q: [^]f32, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// — tridiagonal —

	dstev_ :: proc(jobz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, info: ^Info, l_jobz: c.size_t = 1) ---
	sstev_ :: proc(jobz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, info: ^Info, l_jobz: c.size_t = 1) ---

	dstevd_ :: proc(jobz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1) ---
	sstevd_ :: proc(jobz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1) ---

	dstevr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f64, E: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1) ---
	sstevr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f32, E: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1) ---

	dstevx_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f64, E: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1) ---
	sstevx_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f32, E: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1) ---

	cpteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]f32, info: ^Info, l_compz: c.size_t = 1) ---
	dpteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1) ---
	spteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	zpteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1) ---

	dstebz_ :: proc(range: ^char, order: ^char, n: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, D: [^]f64, E: [^]f64, m: ^blasint, nsplit: ^blasint, W: [^]f64, IBLOCK: ^blasint, ISPLIT: ^blasint, work: [^]f64, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sstebz_ :: proc(range: ^char, order: ^char, n: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, D: [^]f32, E: [^]f32, m: ^blasint, nsplit: ^blasint, W: [^]f32, IBLOCK: ^blasint, ISPLIT: ^blasint, work: [^]f32, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	dsterf_ :: proc(n: ^blasint, D: [^]f64, E: [^]f64, info: ^Info) ---
	ssterf_ :: proc(n: ^blasint, D: [^]f32, E: [^]f32, info: ^Info) ---

	cstedc_ :: proc(compz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1) ---
	dstedc_ :: proc(compz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1) ---
	sstedc_ :: proc(compz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1) ---
	zstedc_ :: proc(compz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1) ---

	cstegr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f32, E: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dstegr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f64, E: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sstegr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f32, E: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zstegr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f64, E: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, ISUPPZ: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	cstein_ :: proc(n: ^blasint, D: [^]f32, E: [^]f32, m: ^blasint, W: [^]f32, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: [^]complex64, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info) ---
	dstein_ :: proc(n: ^blasint, D: [^]f64, E: [^]f64, m: ^blasint, W: [^]f64, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: [^]f64, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info) ---
	sstein_ :: proc(n: ^blasint, D: [^]f32, E: [^]f32, m: ^blasint, W: [^]f32, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: [^]f32, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info) ---
	zstein_ :: proc(n: ^blasint, D: [^]f64, E: [^]f64, m: ^blasint, W: [^]f64, IBLOCK: ^blasint, ISPLIT: ^blasint, Z: [^]complex128, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info) ---


	cstemr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f32, E: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dstemr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f64, E: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sstemr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f32, E: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zstemr_ :: proc(jobz: ^char, range: ^char, n: ^blasint, D: [^]f64, E: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, nzc: ^blasint, ISUPPZ: ^blasint, tryrac: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---


	csteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	dsteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1) ---
	ssteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f32, E: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	zsteqr_ :: proc(compz: ^char, n: ^blasint, D: [^]f64, E: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1) ---

	// ===================================================================================
	// Generalized eig driver, AV = BVΛ, etc.
	// https://www.netlib.org/lapack/explore-html/d2/dae/group__hegv__driver__grp.html
	// ===================================================================================
	// — full —

	chegv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, W: [^]f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhegv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, W: [^]f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsygv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, W: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssygv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, W: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chegv_2stage_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, W: [^]f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhegv_2stage_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, W: [^]f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsygv_2stage_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, W: [^]f64, work: [^]f64, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssygv_2stage_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, W: [^]f32, work: [^]f32, lwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chegvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, W: [^]f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhegvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, W: [^]f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsygvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, W: [^]f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssygvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, W: [^]f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chegvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhegvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsygvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssygvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// — packed —

	chpgv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, BP: [^]complex64, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhpgv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, BP: [^]complex128, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dspgv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, BP: [^]f64, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	sspgv_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, BP: [^]f32, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chpgvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, BP: [^]complex64, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhpgvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, BP: [^]complex128, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dspgvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, BP: [^]f64, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	sspgvd_ :: proc(itype: ^blasint, jobz: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, BP: [^]f32, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chpgvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]complex64, BP: [^]complex64, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhpgvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]complex128, BP: [^]complex128, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dspgvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]f64, BP: [^]f64, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	sspgvx_ :: proc(itype: ^blasint, jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, AP: [^]f32, BP: [^]f32, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// — banded —

	chbgv_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbgv_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbgv_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbgv_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chbgvd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbgvd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbgvd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbgvd_ :: proc(jobz: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chbgvx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, Q: [^]complex64, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, rwork: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbgvx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, Q: [^]complex128, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, rwork: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbgvx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, Q: [^]f64, ldq: ^blasint, vl: ^f64, vu: ^f64, il: ^blasint, iu: ^blasint, abstol: ^f64, m: ^blasint, W: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbgvx_ :: proc(jobz: ^char, range: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, Q: [^]f32, ldq: ^blasint, vl: ^f32, vu: ^f32, il: ^blasint, iu: ^blasint, abstol: ^f32, m: ^blasint, W: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, iwork: [^]blasint, IFAIL: ^blasint, info: ^Info, l_jobz: c.size_t = 1, l_range: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// ===================================================================================
	// Eig computational routines
	// https://www.netlib.org/lapack/explore-html/d2/d91/group__heev__comp__grp.html
	// ===================================================================================

	// — full —

	ddisna_ :: proc(job: ^char, m: ^blasint, n: ^blasint, D: [^]f64, SEP: [^]f64, info: ^Info, l_job: c.size_t = 1) ---
	sdisna_ :: proc(job: ^char, m: ^blasint, n: ^blasint, D: [^]f32, SEP: [^]f32, info: ^Info, l_job: c.size_t = 1) ---

	chetrd_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, D: [^]f32, E: [^]f32, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrd_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, D: [^]f64, E: [^]f64, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	dsytrd_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, D: [^]f64, E: [^]f64, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrd_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, D: [^]f32, E: [^]f32, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// {he,sy}td2: reduction to tridiagonal, level 2

	// latrd: step in hetrd

	cungtr_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zungtr_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	dorgtr_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	sorgtr_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// unmtr: multiply by Q from hetrd

	dormtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---
	sormtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---

	chetrd_2stage_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, D: [^]f32, E: [^]f32, tau: ^complex64, HOUS2: [^]complex64, lhous2: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhetrd_2stage_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, D: [^]f64, E: [^]f64, tau: ^complex128, HOUS2: [^]complex128, lhous2: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsytrd_2stage_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, D: [^]f64, E: [^]f64, tau: ^f64, HOUS2: [^]f64, lhous2: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssytrd_2stage_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, D: [^]f32, E: [^]f32, tau: ^f32, HOUS2: [^]f32, lhous2: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// {he,sy}trd_he2hb: full to band (1st stage)

	// {he,sy}trd_hb2st: band to tridiagonal (2nd stage)

	// {hb,sb}2st_kernels: band to tridiagonal (2nd stage)

	// lae2: 2x2 eig, step in steqr, stemr

	// laesy: 2x2 eig

	// laev2: 2x2 eig

	// lagtf: LU factor of (T - λI)

	// lagts: LU solve of (T - λI) x = y

	// — packed —

	// {hp,sp}trd: reduction to tridiagonal
	chptrd_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, D: [^]f32, E: [^]f32, tau: ^complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhptrd_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, D: [^]f64, E: [^]f64, tau: ^complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	dsptrd_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, D: [^]f64, E: [^]f64, tau: ^f64, info: ^Info, l_uplo: c.size_t = 1) ---
	ssptrd_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, D: [^]f32, E: [^]f32, tau: ^f32, info: ^Info, l_uplo: c.size_t = 1) ---

	// {up,op}gtr: generate Q from hetrd
	cupgtr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, tau: ^complex64, Q: [^]complex64, ldq: ^blasint, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zupgtr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, tau: ^complex128, Q: [^]complex128, ldq: ^blasint, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	dopgtr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, tau: ^f64, Q: [^]f64, ldq: ^blasint, work: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	sopgtr_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, tau: ^f32, Q: [^]f32, ldq: ^blasint, work: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---

	// {up,op}mtr: multiply by Q from hptrd
	cupmtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, AP: [^]complex64, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, info: ^Info, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---
	zupmtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, AP: [^]complex128, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, info: ^Info, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---

	dopmtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, AP: [^]f64, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, info: ^Info, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---
	sopmtr_ :: proc(side: ^char, uplo: ^char, trans: ^char, m: ^blasint, n: ^blasint, AP: [^]f32, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, info: ^Info, l_side: c.size_t = 1, l_uplo: c.size_t = 1, l_trans: c.size_t = 1) ---

	// — banded —

	chbtrd_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, D: [^]f32, E: [^]f32, Q: [^]complex64, ldq: ^blasint, work: [^]complex64, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbtrd_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, D: [^]f64, E: [^]f64, Q: [^]complex128, ldq: ^blasint, work: [^]complex128, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbtrd_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, D: [^]f64, E: [^]f64, Q: [^]f64, ldq: ^blasint, work: [^]f64, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbtrd_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, D: [^]f32, E: [^]f32, Q: [^]f32, ldq: ^blasint, work: [^]f32, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// ===================================================================================
	// Generalized eig computational routines
	// https://www.netlib.org/lapack/explore-html/df/da2/group__hegv__comp__grp.html
	// ===================================================================================
	chegst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhegst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	dsygst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssygst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// {he,sy}gs2: reduction to standard form, level 2

	chpgst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, AP: [^]complex64, BP: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhpgst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, AP: [^]complex128, BP: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	dspgst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, AP: [^]f64, BP: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	sspgst_ :: proc(itype: ^blasint, uplo: ^char, n: ^blasint, AP: [^]f32, BP: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---

	chbgst_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex64, ldab: ^blasint, BB: ^complex64, ldbb: ^blasint, X: ^complex64, ldx: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhbgst_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]complex128, ldab: ^blasint, BB: ^complex128, ldbb: ^blasint, X: ^complex128, ldx: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---

	dsbgst_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f64, ldab: ^blasint, BB: ^f64, ldbb: ^blasint, X: ^f64, ldx: ^blasint, work: [^]f64, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssbgst_ :: proc(vect: ^char, uplo: ^char, n: ^blasint, ka: ^blasint, kb: ^blasint, AB: [^]f32, ldab: ^blasint, BB: ^f32, ldbb: ^blasint, X: ^f32, ldx: ^blasint, work: [^]f32, info: ^Info, l_vect: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cpbstf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dpbstf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spbstf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpbstf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// Mixed precision conversion routines (Auxiliary)
	// https://www.netlib.org/lapack/explore-html/d8/dcc/group__lag2.html
	// LAG2*: Convert between single and double precision

	// ===================================================================================
	// tridiag bisection routines
	// https://www.netlib.org/lapack/explore-html/d3/d27/group__stev__comp__grp.html
	// ===================================================================================
	// laebz: counts eigvals <= value

	// laneg: Sturm count

	// ===================================================================================
	// tridiag divide and conquer (D&C) routines
	// https://www.netlib.org/lapack/explore-html/d6/d90/group__laed__comp__grp.html
	// ===================================================================================
	// laed0: D&C step: top level solver

	// laed1: D&C step: merge subproblems

	// laed2: D&C step: deflation

	// laed3: D&C step: secular equation

	// laed4: D&C step: secular equation nonlinear solver

	// laed5: D&C step: secular equation, 2x2

	// laed6: D&C step: secular equation Newton step

	// lamrg: permutation to merge 2 sorted lists

	// — eig value only or update Q —

	// laed7: D&C step: merge subproblems

	// laed8: D&C step: deflation

	// laed9: D&C step: secular equation

	// laeda: D&C step: z vector

	// ===================================================================================
	// tridiag RRR routines
	// https://www.netlib.org/lapack/explore-html/d4/d49/group__larr__comp__grp.html
	// ===================================================================================
	// larra: step in stemr

	// larrb: step in stemr

	// larrc: step in stemr

	// larrd: step in stemr, tridiag eig

	// larre: step in stemr

	// larrf: step in stemr, find relative robust representation (RRR)

	// larrj: step in stemr, refine eigval estimates

	// larrk: step in stemr, compute one eigval

	// larrr: step in stemr, test to do expensive tridiag eig algorithm

	// larrv: eig tridiagonal, step in stemr & stegr

	// lar1v: step in larrv, hence stemr & stegr
}
