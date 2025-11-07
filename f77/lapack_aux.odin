package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// Initialize, copy, convert
	// https://www.netlib.org/lapack/explore-html/db/d1b/group__aux__grp.html
	// ===================================================================================
	// lsame: string comparison
	lsame_ :: proc(ca: ^char, cb: ^char, lca: blasint, lcb: blasint, _: c.size_t = 1, _: c.size_t = 1) -> i32 ---

	// lsamen: string comparison

	// roundup_lwork: fix rounding integer to float

	// second: wall clock timer


	// ===================================================================================
	// Parameters
	// https://www.netlib.org/lapack/explore-html/dc/db0/group__params__grp.html
	// ===================================================================================
	// lamch: machine parameters

	// lamc1: ??
	// lamc2: ??
	// lamc3: ??
	// lamc4: ??
	// lamc5: ??
	// labad: over/underflow on obsolete pre-IEEE machines
	// ilaver: LAPACK version
	// ilaenv: tuning parameters
	// ilaenv2stage: tuning parameters for 2-stage eig
	// iparam2stage: sets parameters for 2-stage eig
	// ieeeck: verify inf and NaN are safe
	// la_constants: Fortran 95 module of constants
	// — BLAST constants —
	// iladiag: diag string to BLAST const
	// ilaprec: precision string to BLAST const
	// ilatrans: trans string to BLAST const
	// ilauplo: uplo string to BLAST const
	// la_transtype: BLAST const to string

	// ===================================================================================
	// Error reporting
	// https://www.netlib.org/lapack/explore-html/d0/d91/group__xerbla__grp.html
	// ===================================================================================

	// xerbla: error reporting

	// xerbla_array: error reporting, callable from C
}
