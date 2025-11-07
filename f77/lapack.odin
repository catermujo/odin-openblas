package f77

import "core:c"

char :: byte

FORTRAN_STRLEN :: c.size_t

lapack_float_return :: f32

/* Callback logical functions of one, two, or three arguments are used
*  to select eigenvalues to sort to the top left of the Schur form.
*  The value is selected if function returns TRUE (non-zero). */
LAPACK_S_SELECT2 :: proc "c" (_: ^f32, _: ^f32) -> blasint

LAPACK_S_SELECT3 :: proc "c" (_: ^f32, _: ^f32, _: ^f32) -> blasint

LAPACK_D_SELECT2 :: proc "c" (_: ^f64, _: ^f64) -> blasint

LAPACK_D_SELECT3 :: proc "c" (_: ^f64, _: ^f64, _: ^f64) -> blasint

LAPACK_C_SELECT1 :: proc "c" (_: ^complex64) -> blasint

LAPACK_C_SELECT2 :: proc "c" (_: ^complex64, _: ^complex64) -> blasint

LAPACK_Z_SELECT1 :: proc "c" (_: ^complex128) -> blasint

LAPACK_Z_SELECT2 :: proc "c" (_: ^complex128, _: ^complex128) -> blasint

// Testing & Random stuff i have not found a home for yet:
@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// Test matrix generation routines (Testing/Timing)
	// LAGGE: Generate general matrix with specified eigenvalues
	clagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: [^]f32, A: [^]complex64, lda: ^blasint, iseed: [^]blasint, work: [^]complex64, info: ^Info) ---
	dlagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: [^]f64, A: [^]f64, lda: ^blasint, iseed: [^]blasint, work: [^]f64, info: ^Info) ---
	slagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: [^]f32, A: [^]f32, lda: ^blasint, iseed: [^]blasint, work: [^]f32, info: ^Info) ---
	zlagge_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, D: [^]f64, A: [^]complex128, lda: ^blasint, iseed: [^]blasint, work: [^]complex128, info: ^Info) ---

	// LAGHE: Generate Hermitian matrix with specified eigenvalues
	claghe_ :: proc(n: ^blasint, k: [^]blasint, D: [^]f32, A: [^]complex64, lda: ^blasint, iseed: [^]blasint, work: [^]complex64, info: ^Info) ---
	zlaghe_ :: proc(n: ^blasint, k: [^]blasint, D: [^]f64, A: [^]complex128, lda: ^blasint, iseed: [^]blasint, work: [^]complex128, info: ^Info) ---

	// LAGSY: Generate symmetric matrix with specified eigenvalues
	clagsy_ :: proc(n: ^blasint, k: [^]blasint, D: [^]f32, A: [^]complex64, lda: ^blasint, iseed: [^]blasint, work: [^]complex64, info: ^Info) ---
	dlagsy_ :: proc(n: ^blasint, k: [^]blasint, D: [^]f64, A: [^]f64, lda: ^blasint, iseed: [^]blasint, work: [^]f64, info: ^Info) ---
	slagsy_ :: proc(n: ^blasint, k: [^]blasint, D: [^]f32, A: [^]f32, lda: ^blasint, iseed: [^]blasint, work: [^]f32, info: ^Info) ---
	zlagsy_ :: proc(n: ^blasint, k: [^]blasint, D: [^]f64, A: [^]complex128, lda: ^blasint, iseed: [^]blasint, work: [^]complex128, info: ^Info) ---

	// Permutation routines (Auxiliary)
	// LAPMR: Permute rows of a matrix
	clapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]complex64, ldx: ^blasint, K: [^]blasint) ---
	dlapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]f64, ldx: ^blasint, K: [^]blasint) ---
	slapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]f32, ldx: ^blasint, K: [^]blasint) ---
	zlapmr_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]complex128, ldx: ^blasint, K: [^]blasint) ---

	// LAPMT: Permute columns of a matrix
	clapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]complex64, ldx: ^blasint, K: [^]blasint) ---
	dlapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]f64, ldx: ^blasint, K: [^]blasint) ---
	slapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]f32, ldx: ^blasint, K: [^]blasint) ---
	zlapmt_ :: proc(forwrd: ^blasint, m: ^blasint, n: ^blasint, X: [^]complex128, ldx: ^blasint, K: [^]blasint) ---

	// Test matrix generation: Random orthogonal matrices (Testing/Timing)
	// LAROR: Generate random orthogonal matrix
	dlaror_ :: proc(side: ^char, init: ^char, m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, iseed: [^]blasint, X: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	slaror_ :: proc(side: ^char, init: ^char, m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, iseed: [^]blasint, X: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	claror_ :: proc(side: ^char, init: ^char, m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, iseed: [^]blasint, X: [^]complex64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---
	zlaror_ :: proc(side: ^char, init: ^char, m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, iseed: [^]blasint, X: [^]complex128, info: ^Info, _: c.size_t = 1, _: c.size_t = 1) ---

	// LAROT: Apply rotation to matrix rows/columns (Testing)
	dlarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f64, s: ^f64, A: [^]f64, lda: ^blasint, xleft: ^f64, xright: ^f64) ---
	slarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f32, s: ^f32, A: [^]f32, lda: ^blasint, xleft: ^f32, xright: ^f32) ---
	clarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f32, s: ^complex64, A: [^]complex64, lda: ^blasint, xleft: ^complex64, xright: ^complex64) ---
	zlarot_ :: proc(lrows: ^blasint, lleft: ^blasint, lright: ^blasint, nl: ^blasint, c: ^f64, s: ^complex128, A: [^]complex128, lda: ^blasint, xleft: ^complex128, xright: ^complex128) ---

	// LARTGS: Generate rotation with given singular values (Auxiliary)
	dlartgs_ :: proc(x: ^f64, y: ^f64, sigma: ^f64, cs: ^f64, sn: ^f64) ---
	slartgs_ :: proc(x: ^f32, y: ^f32, sigma: ^f32, cs: ^f32, sn: ^f32) ---

	// Test matrix generation with specified singular/eigenvalue distribution (Testing/Timing)
	// LATMS: Generate matrix with specified singular values
	clatms_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f32, mode: ^blasint, cond: ^f32, dmax: ^f32, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]complex64, lda: ^blasint, work: [^]complex64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	dlatms_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f64, mode: ^blasint, cond: ^f64, dmax: ^f64, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]f64, lda: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	slatms_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f32, mode: ^blasint, cond: ^f32, dmax: ^f32, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]f32, lda: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zlatms_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f64, mode: ^blasint, cond: ^f64, dmax: ^f64, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]complex128, lda: ^blasint, work: [^]complex128, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// LATMT: Generate matrix with specified singular values and rank
	dlatmt_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f64, mode: ^blasint, cond: ^f64, dmax: ^f64, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]f64, lda: ^blasint, work: [^]f64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	slatmt_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f32, mode: ^blasint, cond: ^f32, dmax: ^f32, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]f32, lda: ^blasint, work: [^]f32, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	clatmt_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f32, mode: ^blasint, cond: ^f32, dmax: ^f32, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]complex64, lda: ^blasint, work: [^]complex64, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---
	zlatmt_ :: proc(m: ^blasint, n: ^blasint, dist: ^char, iseed: [^]blasint, sym: ^char, D: [^]f64, mode: ^blasint, cond: ^f64, dmax: ^f64, rank: ^blasint, kl: ^blasint, ku: ^blasint, pack: ^char, A: [^]complex128, lda: ^blasint, work: [^]complex128, info: ^Info, _: c.size_t = 1, _: c.size_t = 1, _: c.size_t = 1) ---

	// Version information (Auxiliary)
	// ILAVER: Get LAPACK version
	ilaver_ :: proc(vers_major: ^blasint, vers_minor: ^blasint, vers_patch: ^blasint) -> i32 ---
}
