package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// Standard eig driver, AV = VΛ
	// https://www.netlib.org/lapack/explore-html/dc/d78/group__geev__driver__grp.html
	// ===================================================================================

	cgeev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, W: [^]complex64, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgeev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, WR: [^]f64, WI: [^]f64, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgeev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, WR: [^]f32, WI: [^]f32, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgeev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, W: [^]complex128, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	cgeevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, W: [^]complex64, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgeevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, WR: [^]f64, WI: [^]f64, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgeevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, WR: [^]f32, WI: [^]f32, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, abnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgeevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, W: [^]complex128, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, abnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cgees_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_C_SELECT1, n: ^blasint, A: [^]complex64, lda: ^blasint, sdim: ^blasint, W: [^]complex64, VS: [^]complex64, ldvs: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgees_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_D_SELECT2, n: ^blasint, A: [^]f64, lda: ^blasint, sdim: ^blasint, WR: [^]f64, WI: [^]f64, VS: [^]f64, ldvs: ^blasint, work: [^]f64, lwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgees_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_S_SELECT2, n: ^blasint, A: [^]f32, lda: ^blasint, sdim: ^blasint, WR: [^]f32, WI: [^]f32, VS: [^]f32, ldvs: ^blasint, work: [^]f32, lwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgees_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_Z_SELECT1, n: ^blasint, A: [^]complex128, lda: ^blasint, sdim: ^blasint, W: [^]complex128, VS: [^]complex128, ldvs: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	cgeesx_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_C_SELECT1, sense: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, sdim: ^blasint, W: [^]complex64, VS: [^]complex64, ldvs: ^blasint, rconde: ^f32, rcondv: ^f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgeesx_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_D_SELECT2, sense: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, sdim: ^blasint, WR: [^]f64, WI: [^]f64, VS: [^]f64, ldvs: ^blasint, rconde: ^f64, rcondv: ^f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgeesx_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_S_SELECT2, sense: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, sdim: ^blasint, WR: [^]f32, WI: [^]f32, VS: [^]f32, ldvs: ^blasint, rconde: ^f32, rcondv: ^f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgeesx_ :: proc(jobvs: ^char, sort: ^char, select: LAPACK_Z_SELECT1, sense: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, sdim: ^blasint, W: [^]complex128, VS: [^]complex128, ldvs: ^blasint, rconde: ^f64, rcondv: ^f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// ===================================================================================
	// Generalized eig driver
	// https://www.netlib.org/lapack/explore-html/d5/d81/group__ggev__driver__grp.html
	// ===================================================================================

	cggev3_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dggev3_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sggev3_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zggev3_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	cggev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dggev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sggev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zggev_ :: proc(jobvl: ^char, jobvr: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	cggevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dggevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sggevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, abnrm: ^f32, bbnrm: ^f32, rconde: ^f32, rcondv: ^f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zggevx_ :: proc(balanc: ^char, jobvl: ^char, jobvr: ^char, sense: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, abnrm: ^f64, bbnrm: ^f64, rconde: ^f64, rcondv: ^f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cgges3_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_C_SELECT2, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, sdim: ^blasint, alpha: ^complex64, beta: ^complex64, VSL: [^]complex64, ldvsl: ^blasint, VSR: [^]complex64, ldvsr: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgges3_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_D_SELECT3, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, sdim: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VSL: [^]f64, ldvsl: ^blasint, VSR: [^]f64, ldvsr: ^blasint, work: [^]f64, lwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgges3_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_S_SELECT3, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, sdim: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VSL: [^]f32, ldvsl: ^blasint, VSR: [^]f32, ldvsr: ^blasint, work: [^]f32, lwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgges3_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_Z_SELECT2, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, sdim: ^blasint, alpha: ^complex128, beta: ^complex128, VSL: [^]complex128, ldvsl: ^blasint, VSR: [^]complex128, ldvsr: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cgges_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_C_SELECT2, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, sdim: ^blasint, alpha: ^complex64, beta: ^complex64, VSL: [^]complex64, ldvsl: ^blasint, VSR: [^]complex64, ldvsr: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgges_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_D_SELECT3, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, sdim: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VSL: [^]f64, ldvsl: ^blasint, VSR: [^]f64, ldvsr: ^blasint, work: [^]f64, lwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgges_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_S_SELECT3, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, sdim: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VSL: [^]f32, ldvsl: ^blasint, VSR: [^]f32, ldvsr: ^blasint, work: [^]f32, lwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgges_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_Z_SELECT2, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, sdim: ^blasint, alpha: ^complex128, beta: ^complex128, VSL: [^]complex128, ldvsl: ^blasint, VSR: [^]complex128, ldvsr: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cggesx_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_C_SELECT2, sense: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, sdim: ^blasint, alpha: ^complex64, beta: ^complex64, VSL: [^]complex64, ldvsl: ^blasint, VSR: [^]complex64, ldvsr: ^blasint, rconde: ^f32, rcondv: ^f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, iwork: [^]blasint, liwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dggesx_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_D_SELECT3, sense: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, sdim: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, VSL: [^]f64, ldvsl: ^blasint, VSR: [^]f64, ldvsr: ^blasint, rconde: ^f64, rcondv: ^f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sggesx_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_S_SELECT3, sense: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, sdim: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, VSL: [^]f32, ldvsl: ^blasint, VSR: [^]f32, ldvsr: ^blasint, rconde: ^f32, rcondv: ^f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zggesx_ :: proc(jobvsl: ^char, jobvsr: ^char, sort: ^char, selctg: LAPACK_Z_SELECT2, sense: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, sdim: ^blasint, alpha: ^complex128, beta: ^complex128, VSL: [^]complex128, ldvsl: ^blasint, VSR: [^]complex128, ldvsr: ^blasint, rconde: ^f64, rcondv: ^f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, iwork: [^]blasint, liwork: ^blasint, BWORK: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// ===================================================================================
	// DMD driver, Dynamic Mode Decomposition
	// https://www.netlib.org/lapack/explore-html/de/dbe/group__gedmd.html
	// ===================================================================================
	cgedmd_ :: proc(jobs: ^u8, jobz: ^u8, jobr: ^u8, jobf: ^u8, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^complex64, ldx: ^blasint, y: ^complex64, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, eigs: ^complex64, z: [^]complex64, ldz: ^blasint, res: ^f32, b: [^]complex64, ldb: ^blasint, w: [^]complex64, ldw: ^blasint, s: [^]complex64, lds: ^blasint, zwork: ^complex64, lzwork: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgedmd_ :: proc(jobs: ^u8, jobz: ^u8, jobr: ^u8, jobf: ^u8, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^f64, ldx: ^blasint, y: ^f64, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, reig: ^f64, imeig: ^f64, z: [^]f64, ldz: ^blasint, res: ^f64, b: [^]f64, ldb: ^blasint, w: [^]f64, ldw: ^blasint, s: [^]f64, lds: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgedmd_ :: proc(jobs: ^u8, jobz: ^u8, jobr: ^u8, jobf: ^u8, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^f32, ldx: ^blasint, y: ^f32, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, reig: ^f32, imeig: ^f32, z: [^]f32, ldz: ^blasint, res: ^f32, b: [^]f32, ldb: ^blasint, w: [^]f32, ldw: ^blasint, s: [^]f32, lds: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgedmd_ :: proc(jobs: ^u8, jobz: ^u8, jobr: ^u8, jobf: ^u8, whtsvd: ^blasint, m: ^blasint, n: ^blasint, x: ^complex128, ldx: ^blasint, y: ^complex128, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, eigs: ^complex128, z: [^]complex128, ldz: ^blasint, res: ^f64, b: [^]complex128, ldb: ^blasint, w: [^]complex128, ldw: ^blasint, s: [^]complex128, lds: ^blasint, zwork: ^complex128, lzwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cgedmdq_ :: proc(jobs: ^char, jobz: ^char, jobr: ^char, jobq: ^char, jobt: ^char, jobf: ^char, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^complex64, ldf: ^blasint, x: ^complex64, ldx: ^blasint, y: ^complex64, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, eigs: ^complex64, z: [^]complex64, ldz: ^blasint, res: ^f32, b: [^]complex64, ldb: ^blasint, v: ^complex64, ldv: ^blasint, s: [^]complex64, lds: ^blasint, zwork: ^complex64, lzwork: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dgedmdq_ :: proc(jobs: ^char, jobz: ^char, jobr: ^char, jobq: ^char, jobt: ^char, jobf: ^char, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^f64, ldf: ^blasint, x: ^f64, ldx: ^blasint, y: ^f64, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, reig: ^f64, imeig: ^f64, z: [^]f64, ldz: ^blasint, res: ^f64, b: [^]f64, ldb: ^blasint, v: ^f64, ldv: ^blasint, s: [^]f64, lds: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	sgedmdq_ :: proc(jobs: ^char, jobz: ^char, jobr: ^char, jobq: ^char, jobt: ^char, jobf: ^char, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^f32, ldf: ^blasint, x: ^f32, ldx: ^blasint, y: ^f32, ldy: ^blasint, nrnk: ^blasint, tol: ^f32, k: ^blasint, reig: ^f32, imeig: ^f32, z: [^]f32, ldz: ^blasint, res: ^f32, b: [^]f32, ldb: ^blasint, v: ^f32, ldv: ^blasint, s: [^]f32, lds: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zgedmdq_ :: proc(jobs: ^char, jobz: ^char, jobr: ^char, jobq: ^char, jobt: ^char, jobf: ^char, whtsvd: ^blasint, m: ^blasint, n: ^blasint, f: ^complex128, ldf: ^blasint, x: ^complex128, ldx: ^blasint, y: ^complex128, ldy: ^blasint, nrnk: ^blasint, tol: ^f64, k: ^blasint, eigs: ^complex128, z: [^]complex128, ldz: ^blasint, res: ^f64, b: [^]complex128, ldb: ^blasint, v: ^complex128, ldv: ^blasint, s: [^]complex128, lds: ^blasint, zwork: ^complex128, lzwork: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// ===================================================================================
	// Eig computational routines
	// https://www.netlib.org/lapack/explore-html/d8/d01/group__geev__comp__grp.html
	// ===================================================================================
	cgebal_ :: proc(job: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, info: ^Info, _: c.size_t = 1) ---
	dgebal_ :: proc(job: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, info: ^Info, _: c.size_t = 1) ---
	sgebal_ :: proc(job: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, info: ^Info, _: c.size_t = 1) ---
	zgebal_ :: proc(job: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, info: ^Info, _: c.size_t = 1) ---

	cgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgehrd_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	// gehd2: reduction to Hessenberg, level 2

	// lahr2: step in gehrd

	cunghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	zunghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	dorghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sorghr_ :: proc(n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, work: [^]f32, lwork: ^blasint, info: ^Info) ---

	cunmhr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex64, lda: ^blasint, tau: ^complex64, C: [^]complex64, ldc: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zunmhr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex128, lda: ^blasint, tau: ^complex128, C: [^]complex128, ldc: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	dormhr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f64, lda: ^blasint, tau: ^f64, C: [^]f64, ldc: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sormhr_ :: proc(side: ^char, trans: ^char, m: ^blasint, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f32, lda: ^blasint, tau: ^f32, C: [^]f32, ldc: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	cgebak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, m: ^blasint, V: ^complex64, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgebak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, m: ^blasint, V: ^f64, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgebak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f32, m: ^blasint, V: ^f32, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgebak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, scale: ^f64, m: ^blasint, V: ^complex128, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	chseqr_ :: proc(job: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]complex64, ldh: ^blasint, W: [^]complex64, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dhseqr_ :: proc(job: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]f64, ldh: ^blasint, WR: [^]f64, WI: [^]f64, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	shseqr_ :: proc(job: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]f32, ldh: ^blasint, WR: [^]f32, WI: [^]f32, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zhseqr_ :: proc(job: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]complex128, ldh: ^blasint, W: [^]complex128, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	chsein_ :: proc(side: ^char, eigsrc: ^char, initv: ^char, select: ^blasint, n: ^blasint, H: [^]complex64, ldh: ^blasint, W: [^]complex64, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex64, rwork: [^]f32, IFAILL: [^]blasint, IFAILR: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dhsein_ :: proc(side: ^char, eigsrc: ^char, initv: ^char, select: ^blasint, n: ^blasint, H: [^]f64, ldh: ^blasint, WR: [^]f64, WI: [^]f64, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f64, IFAILL: [^]blasint, IFAILR: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	shsein_ :: proc(side: ^char, eigsrc: ^char, initv: ^char, select: ^blasint, n: ^blasint, H: [^]f32, ldh: ^blasint, WR: [^]f32, WI: [^]f32, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f32, IFAILL: [^]blasint, IFAILR: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zhsein_ :: proc(side: ^char, eigsrc: ^char, initv: ^char, select: ^blasint, n: ^blasint, H: [^]complex128, ldh: ^blasint, W: [^]complex128, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex128, rwork: [^]f64, IFAILL: [^]blasint, IFAILR: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	ctrevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]complex64, ldt: ^blasint, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtrevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]f64, ldt: ^blasint, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	strevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]f32, ldt: ^blasint, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztrevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]complex128, ldt: ^blasint, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	ctrevc3_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]complex64, ldt: ^blasint, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtrevc3_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]f64, ldt: ^blasint, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	strevc3_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]f32, ldt: ^blasint, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztrevc3_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]complex128, ldt: ^blasint, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, lrwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// laln2: 1x1 or 2x2 solve, step in trevc

	ctrsyl_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, C: [^]complex64, ldc: ^blasint, scale: ^f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtrsyl_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, C: [^]f64, ldc: ^blasint, scale: ^f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	strsyl_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, C: [^]f32, ldc: ^blasint, scale: ^f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztrsyl_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, C: [^]complex128, ldc: ^blasint, scale: ^f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	ctrsyl3_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, C: [^]complex64, ldc: ^blasint, scale: ^f32, swork: [^]f32, ldswork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtrsyl3_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, C: [^]f64, ldc: ^blasint, scale: ^f64, iwork: [^]blasint, liwork: ^blasint, swork: [^]f64, ldswork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	strsyl3_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, C: [^]f32, ldc: ^blasint, scale: ^f32, iwork: [^]blasint, liwork: ^blasint, swork: [^]f32, ldswork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztrsyl3_ :: proc(trana: ^char, tranb: ^char, isgn: ^blasint, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, C: [^]complex128, ldc: ^blasint, scale: ^f64, swork: [^]f64, ldswork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// lasy2: Sylvester equation

	ctrsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]complex64, ldt: ^blasint, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, S: [^]f32, SEP: [^]f32, mm: ^blasint, m: ^blasint, work: [^]complex64, ldwork: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtrsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]f64, ldt: ^blasint, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, S: [^]f64, SEP: [^]f64, mm: ^blasint, m: ^blasint, work: [^]f64, ldwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	strsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]f32, ldt: ^blasint, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, S: [^]f32, SEP: [^]f32, mm: ^blasint, m: ^blasint, work: [^]f32, ldwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztrsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, T: [^]complex128, ldt: ^blasint, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, S: [^]f64, SEP: [^]f64, mm: ^blasint, m: ^blasint, work: [^]complex128, ldwork: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// laqtr: quasi-triangular solve

	ctrexc_ :: proc(compq: ^char, n: ^blasint, T: [^]complex64, ldt: ^blasint, Q: [^]complex64, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info, _: c.size_t = 1) ---
	dtrexc_ :: proc(compq: ^char, n: ^blasint, T: [^]f64, ldt: ^blasint, Q: [^]f64, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1) ---
	strexc_ :: proc(compq: ^char, n: ^blasint, T: [^]f32, ldt: ^blasint, Q: [^]f32, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	ztrexc_ :: proc(compq: ^char, n: ^blasint, T: [^]complex128, ldt: ^blasint, Q: [^]complex128, ldq: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info, _: c.size_t = 1) ---

	ctrsen_ :: proc(job: ^char, compq: ^char, select: ^blasint, n: ^blasint, T: [^]complex64, ldt: ^blasint, Q: [^]complex64, ldq: ^blasint, W: [^]complex64, m: ^blasint, s: [^]f32, sep: [^]f32, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtrsen_ :: proc(job: ^char, compq: ^char, select: ^blasint, n: ^blasint, T: [^]f64, ldt: ^blasint, Q: [^]f64, ldq: ^blasint, WR: [^]f64, WI: [^]f64, m: ^blasint, s: [^]f64, sep: [^]f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	strsen_ :: proc(job: ^char, compq: ^char, select: ^blasint, n: ^blasint, T: [^]f32, ldt: ^blasint, Q: [^]f32, ldq: ^blasint, WR: [^]f32, WI: [^]f32, m: ^blasint, s: [^]f32, sep: [^]f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztrsen_ :: proc(job: ^char, compq: ^char, select: ^blasint, n: ^blasint, T: [^]complex128, ldt: ^blasint, Q: [^]complex128, ldq: ^blasint, W: [^]complex128, m: ^blasint, s: [^]f64, sep: [^]f64, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// laexc: reorder Schur form

	// lanv2: 2x2 Schur factor

	// — hseqr auxiliary —

	// laein: eigvec by Hessenberg inverse iteration

	// lahqr: eig of Hessenberg, step in hseqr

	// laqr0: eig of Hessenberg, step in hseqr

	// laqr1: step in hseqr

	// laqr2: step in hseqr

	// laqr3: step in hseqr

	// laqr4: eig of Hessenberg, step in hseqr

	// laqr5: step in hseqr

	// iparmq: set parameters for hseqr

	// — ggev3, gges3 auxiliary —

	// laqz0: step in ggev3, gges3

	// laqz1: step in ggev3, gges3

	// laqz2: step in ggev3, gges3

	// laqz3: step in ggev3, gges3

	// laqz4: step in ggev3, gges3

	// ===================================================================================
	// Generalized eig computational routines
	// https://www.netlib.org/lapack/explore-html/d3/db5/group__ggev__comp__grp.html
	// ===================================================================================

	cggbal_ :: proc(job: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	dggbal_ :: proc(job: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, work: [^]f64, info: ^Info, _: c.size_t = 1) ---
	sggbal_ :: proc(job: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, work: [^]f32, info: ^Info, _: c.size_t = 1) ---
	zggbal_ :: proc(job: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, work: [^]f64, info: ^Info, _: c.size_t = 1) ---

	cgghrd_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, Q: [^]complex64, ldq: ^blasint, Z: [^]complex64, ldz: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgghrd_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, Q: [^]f64, ldq: ^blasint, Z: [^]f64, ldz: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgghrd_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, Q: [^]f32, ldq: ^blasint, Z: [^]f32, ldz: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgghrd_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, Q: [^]complex128, ldq: ^blasint, Z: [^]complex128, ldz: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	cgghd3_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, Q: [^]complex64, ldq: ^blasint, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dgghd3_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, Q: [^]f64, ldq: ^blasint, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sgghd3_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, Q: [^]f32, ldq: ^blasint, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zgghd3_ :: proc(compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, Q: [^]complex128, ldq: ^blasint, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	chgeqz_ :: proc(job: ^char, compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]complex64, ldh: ^blasint, T: [^]complex64, ldt: ^blasint, alpha: ^complex64, beta: ^complex64, Q: [^]complex64, ldq: ^blasint, Z: [^]complex64, ldz: ^blasint, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dhgeqz_ :: proc(job: ^char, compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]f64, ldh: ^blasint, T: [^]f64, ldt: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, Q: [^]f64, ldq: ^blasint, Z: [^]f64, ldz: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	shgeqz_ :: proc(job: ^char, compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]f32, ldh: ^blasint, T: [^]f32, ldt: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, Q: [^]f32, ldq: ^blasint, Z: [^]f32, ldz: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zhgeqz_ :: proc(job: ^char, compq: ^char, compz: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, H: [^]complex128, ldh: ^blasint, T: [^]complex128, ldt: ^blasint, alpha: ^complex128, beta: ^complex128, Q: [^]complex128, ldq: ^blasint, Z: [^]complex128, ldz: ^blasint, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	cggbak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, m: ^blasint, V: ^complex64, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dggbak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, m: ^blasint, V: ^f64, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	sggbak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f32, rscale: ^f32, m: ^blasint, V: ^f32, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zggbak_ :: proc(job: ^char, side: ^char, n: ^blasint, ilo: ^blasint, ihi: ^blasint, lscale: ^f64, rscale: ^f64, m: ^blasint, V: ^complex128, ldv: ^blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	ctgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, alpha: ^complex64, beta: ^complex64, Q: [^]complex64, ldq: ^blasint, Z: [^]complex64, ldz: ^blasint, m: ^blasint, pl: ^f32, pr: ^f32, DIF: [^]f32, work: [^]complex64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info) ---
	dtgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, alphar: ^f64, alphai: ^f64, beta: ^f64, Q: [^]f64, ldq: ^blasint, Z: [^]f64, ldz: ^blasint, m: ^blasint, pl: ^f64, pr: ^f64, DIF: [^]f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info) ---
	stgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, alphar: ^f32, alphai: ^f32, beta: ^f32, Q: [^]f32, ldq: ^blasint, Z: [^]f32, ldz: ^blasint, m: ^blasint, pl: ^f32, pr: ^f32, DIF: [^]f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info) ---
	ztgsen_ :: proc(ijob: ^blasint, wantq: ^blasint, wantz: ^blasint, select: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, alpha: ^complex128, beta: ^complex128, Q: [^]complex128, ldq: ^blasint, Z: [^]complex128, ldz: ^blasint, m: ^blasint, pl: ^f64, pr: ^f64, DIF: [^]f64, work: [^]complex128, lwork: ^blasint, iwork: [^]blasint, liwork: ^blasint, info: ^Info) ---


	ctgsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, S: [^]f32, DIF: [^]f32, mm: ^blasint, m: ^blasint, work: [^]complex64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtgsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, S: [^]f64, DIF: [^]f64, mm: ^blasint, m: ^blasint, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	stgsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, S: [^]f32, DIF: [^]f32, mm: ^blasint, m: ^blasint, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztgsna_ :: proc(job: ^char, howmny: ^char, select: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, S: [^]f64, DIF: [^]f64, mm: ^blasint, m: ^blasint, work: [^]complex128, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	ctgsyl_ :: proc(trans: ^char, ijob: ^blasint, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, C: [^]complex64, ldc: ^blasint, D: ^complex64, ldd: ^blasint, E: ^complex64, lde: ^blasint, F: ^complex64, ldf: ^blasint, dif: [^]f32, scale: ^f32, work: [^]complex64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---
	dtgsyl_ :: proc(trans: ^char, ijob: ^blasint, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, C: [^]f64, ldc: ^blasint, D: ^f64, ldd: ^blasint, E: ^f64, lde: ^blasint, F: ^f64, ldf: ^blasint, dif: [^]f64, scale: ^f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---
	stgsyl_ :: proc(trans: ^char, ijob: ^blasint, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, C: [^]f32, ldc: ^blasint, D: ^f32, ldd: ^blasint, E: ^f32, lde: ^blasint, F: ^f32, ldf: ^blasint, dif: [^]f32, scale: ^f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---
	ztgsyl_ :: proc(trans: ^char, ijob: ^blasint, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, C: [^]complex128, ldc: ^blasint, D: ^complex128, ldd: ^blasint, E: ^complex128, lde: ^blasint, F: ^complex128, ldf: ^blasint, dif: [^]f64, scale: ^f64, work: [^]complex128, lwork: ^blasint, iwork: [^]blasint, info: ^Info, _: c.size_t = 1) ---

	// tgsy2: Sylvester equation panel (?)

	// {un,or}m22: multiply by banded Q, step in gghd3

	// lagv2: 2x2 generalized Schur factor

	ctgevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, S: [^]complex64, lds: ^blasint, P: ^complex64, ldp: ^blasint, VL: [^]complex64, ldvl: ^blasint, VR: [^]complex64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex64, rwork: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	dtgevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, S: [^]f64, lds: ^blasint, P: ^f64, ldp: ^blasint, VL: [^]f64, ldvl: ^blasint, VR: [^]f64, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	stgevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, S: [^]f32, lds: ^blasint, P: ^f32, ldp: ^blasint, VL: [^]f32, ldvl: ^blasint, VR: [^]f32, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	ztgevc_ :: proc(side: ^char, howmny: ^char, select: ^blasint, n: ^blasint, S: [^]complex128, lds: ^blasint, P: ^complex128, ldp: ^blasint, VL: [^]complex128, ldvl: ^blasint, VR: [^]complex128, ldvr: ^blasint, mm: ^blasint, m: ^blasint, work: [^]complex128, rwork: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	ctgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, Q: [^]complex64, ldq: ^blasint, Z: [^]complex64, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info) ---
	dtgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, Q: [^]f64, ldq: ^blasint, Z: [^]f64, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	stgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, Q: [^]f32, ldq: ^blasint, Z: [^]f32, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	ztgexc_ :: proc(wantq: ^blasint, wantz: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, Q: [^]complex128, ldq: ^blasint, Z: [^]complex128, ldz: ^blasint, ifst: ^blasint, ilst: ^blasint, info: ^Info) ---

	// tgex2: reorder generalized Schur form
}
