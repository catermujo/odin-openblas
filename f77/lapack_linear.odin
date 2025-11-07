package f77

import "core:c"

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	// ===================================================================================
	// LU: GENERAL MATRICES - Driver Routines
	// https://www.netlib.org/lapack/explore-html/d1/d84/group__gesv__driver__grp.html
	//
	// ===================================================================================

	// GESV - Simple driver for solving general linear systems AX = B
	// Uses LU decomposition with partial pivoting
	// Parameters:
	//   n: Order of matrix A (n >= 0)
	//   nrhs: Number of right-hand sides (nrhs >= 0)
	//   A: Input/Output matrix [n x n], overwritten with L and U factors
	//   lda: Leading dimension of A (lda >= max(1,n))
	//   ipiv: Output pivot indices [n]
	//   B: Input RHS matrix [n x nrhs], Output solution matrix X
	//   ldb: Leading dimension of B (ldb >= max(1,n))
	//   info: Status (0=success, <0=bad arg, >0=singular at info)
	// Re-Decl from Blas:
	// cgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info) -> i32 ---
	// dgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info) -> i32 ---
	// sgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info) -> i32 ---
	// zgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info) -> i32 ---

	// GESVX - Expert driver with error bounds and equilibration
	// Solves AX = B with optional equilibration and condition estimation
	// Parameters:
	//   fact: 'N'=new factorization, 'F'=use given factorization, 'E'=equilibrate
	//   trans: 'N'=no transpose, 'T'=transpose, 'C'=conjugate transpose
	//   n: Order of matrix A
	//   nrhs: Number of right-hand sides
	//   A: Input matrix [n x n], may be equilibrated
	//   lda: Leading dimension of A
	//   AF: Input/Output LU factorization [n x n]
	//   ldaf: Leading dimension of AF
	//   ipiv: Pivot indices [n]
	//   equed: Equilibration status: 'N'=none, 'R'=row, 'C'=col, 'B'=both
	//   R: Row scale factors [n]
	//   C: Column scale factors [n]
	//   B: Input RHS matrix [n x nrhs]
	//   ldb: Leading dimension of B
	//   X: Output solution matrix [n x nrhs]
	//   ldx: Leading dimension of X
	//   rcond: Reciprocal condition number estimate
	//   ferr: Forward error bounds for each solution vector [nrhs]
	//   berr: Component-wise relative backward error bounds [nrhs]
	//   work: Workspace array
	//   rwork/iwork: Real/integer workspace (complex versions use rwork)
	//   info: Status
	cgesvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	dgesvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	sgesvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	zgesvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---

	// GESVXX - Extra-precise iterative refinement expert driver
	// Most accurate solver with multiple error bounds
	// Parameters:
	//   fact: 'N'=new, 'F'=use given, 'E'=equilibrate
	//   trans: 'N'=none, 'T'=transpose, 'C'=conjugate
	//   n, nrhs: Matrix order and number of RHS
	//   A, AF: Input matrix and LU factorization
	//   ipiv: Pivot indices
	//   equed: Equilibration status
	//   R, C: Row and column scale factors
	//   B: RHS matrix
	//   X: Solution matrix
	//   rcond: Condition number estimate
	//   rpvgrw: Reciprocal pivot growth factor
	//   berr: Component-wise backward error
	//   n_err_bnds: Number of error bounds (usually 3)
	//   err_bnds_norm: Forward error bounds for each RHS [nrhs x n_err_bnds]
	//   err_bnds_comp: Component-wise error bounds [nrhs x n_err_bnds]
	//   nparams: Number of parameters in params array
	//   params: Algorithm parameters (see documentation)
	cgesvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	dgesvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	sgesvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	zgesvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---

	// DSGESV/ZCGESV - Mixed precision iterative refinement
	// Solves in double precision, refines solution using single precision
	// Faster than pure double precision for large systems
	// Parameters:
	//   n, nrhs: Matrix order and number of RHS
	//   A: Double precision matrix, overwritten with factors
	//   ipiv: Pivot indices
	//   B: Double precision RHS
	//   X: Double precision solution (refined)
	//   work: Double precision workspace
	//   swork: Single precision workspace for refinement
	//   iter: Number of refinement iterations performed
	//   info: Status
	dsgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, work: [^]f64, swork: [^]f32, iter: ^blasint, info: ^Info) ---
	zcgesv_ :: proc(n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, work: [^]complex128, swork: [^]complex64, rwork: [^]f64, iter: ^blasint, info: ^Info) ---

	// ===================================================================================
	// BANDED MATRICES - Driver Routines
	// ===================================================================================

	// GBSV - Simple driver for banded linear systems
	// Uses LU decomposition with partial pivoting for banded matrices
	// Parameters:
	//   n: Order of matrix A
	//   kl: Number of subdiagonals
	//   ku: Number of superdiagonals
	//   nrhs: Number of right-hand sides
	//   AB: Banded matrix in band storage [ldab x n], overwritten with LU
	//   ldab: Leading dimension of AB (ldab >= 2*kl+ku+1)
	//   ipiv: Pivot indices [n]
	//   B: RHS matrix [n x nrhs], overwritten with solution
	//   ldb: Leading dimension of B
	//   info: Status
	cgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info) ---
	dgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info) ---
	sgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info) ---
	zgbsv_ :: proc(n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info) ---

	// GBSVX - Expert driver for banded systems with error bounds
	// Solves banded systems with equilibration and condition estimation
	// Parameters:
	//   fact: 'N'=new, 'F'=use given factorization, 'E'=equilibrate
	//   trans: 'N'=none, 'T'=transpose, 'C'=conjugate
	//   n, kl, ku: Order and band widths
	//   nrhs: Number of right-hand sides
	//   AB: Banded matrix in band storage
	//   AFB: LU factorization in band storage
	//   ipiv: Pivot indices
	//   equed: Equilibration status
	//   R, C: Row and column scale factors
	//   B, X: RHS and solution matrices
	//   rcond: Reciprocal condition number
	//   ferr, berr: Forward and backward error bounds
	cgbsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, AFB: [^]complex64, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	dgbsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, AFB: [^]f64, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	sgbsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, AFB: [^]f32, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	zgbsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, AFB: [^]complex128, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---

	// GBSVXX - Extra-precise iterative refinement for banded systems
	// Most accurate banded solver with multiple error bounds
	// Similar to GESVXX but for banded matrices
	cgbsvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, AFB: [^]complex64, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	dgbsvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, AFB: [^]f64, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	sgbsvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, AFB: [^]f32, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f32, C: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	zgbsvxx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, AFB: [^]complex128, ldafb: ^blasint, ipiv: [^]blasint, equed: ^char, R: [^]f64, C: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---

	// ===================================================================================
	// TRIDIAGONAL MATRICES - Driver Routines
	// ===================================================================================

	// GTSV - Simple driver for tridiagonal linear systems
	// Uses Gaussian elimination with partial pivoting
	// Parameters:
	//   n: Order of matrix A
	//   nrhs: Number of right-hand sides
	//   DL: Subdiagonal elements [n-1], overwritten
	//   D: Diagonal elements [n], overwritten
	//   DU: Superdiagonal elements [n-1], overwritten
	//   B: RHS matrix [n x nrhs], overwritten with solution
	//   ldb: Leading dimension of B
	//   info: Status
	cgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: [^]complex64, D: [^]complex64, DU: [^]complex64, B: [^]complex64, ldb: ^blasint, info: ^Info) ---
	dgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: [^]f64, D: [^]f64, DU: [^]f64, B: [^]f64, ldb: ^blasint, info: ^Info) ---
	sgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: [^]f32, D: [^]f32, DU: [^]f32, B: [^]f32, ldb: ^blasint, info: ^Info) ---
	zgtsv_ :: proc(n: ^blasint, nrhs: ^blasint, DL: [^]complex128, D: [^]complex128, DU: [^]complex128, B: [^]complex128, ldb: ^blasint, info: ^Info) ---

	// GTSVX - Expert driver for tridiagonal systems with error bounds
	// Solves tridiagonal systems with condition estimation
	// Parameters:
	//   fact: 'N'=new factorization, 'F'=use given factorization
	//   trans: 'N'=none, 'T'=transpose, 'C'=conjugate
	//   n, nrhs: Order and number of RHS
	//   DL, D, DU: Tridiagonal matrix diagonals
	//   DLF, DF, DUF, DU2: LU factorization of tridiagonal matrix
	//   ipiv: Pivot indices
	//   B: RHS matrix
	//   X: Solution matrix
	//   rcond: Reciprocal condition number
	//   ferr, berr: Forward and backward error bounds
	cgtsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]complex64, D: [^]complex64, DU: [^]complex64, DLF: [^]complex64, DF: [^]complex64, DUF: [^]complex64, DU2: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1) ---
	dgtsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]f64, D: [^]f64, DU: [^]f64, DLF: [^]f64, DF: [^]f64, DUF: [^]f64, DU2: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1) ---
	sgtsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]f32, D: [^]f32, DU: [^]f32, DLF: [^]f32, DF: [^]f32, DUF: [^]f32, DU2: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1) ---
	zgtsvx_ :: proc(fact: ^char, trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]complex128, D: [^]complex128, DU: [^]complex128, DLF: [^]complex128, DF: [^]complex128, DUF: [^]complex128, DU2: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_trans: c.size_t = 1) ---

	// ===================================================================================
	// LU: computational routines (factor, cond, etc.)
	// https://www.netlib.org/lapack/explore-html/d4/d1e/group__gesv__comp__grp.html
	// ===================================================================================

	// GECON - Estimate the reciprocal condition number of a general matrix
	// Uses the LU factorization computed by GETRF
	// Parameters:
	//   norm: '1'=1-norm, 'I'=infinity norm
	//   n: Order of matrix A
	//   A: LU factorization from GETRF [n x n]
	//   lda: Leading dimension of A
	//   anorm: Norm of the original matrix A
	//   rcond: Reciprocal condition number (output)
	//   work: Workspace array
	//   rwork/iwork: Real/integer workspace
	//   info: Status
	cgecon_ :: proc(norm: ^byte, n: ^blasint, A: [^]complex64, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_norm: c.size_t = 1) ---
	dgecon_ :: proc(norm: ^byte, n: ^blasint, A: [^]f64, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1) ---
	sgecon_ :: proc(norm: ^byte, n: ^blasint, A: [^]f32, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1) ---
	zgecon_ :: proc(norm: ^byte, n: ^blasint, A: [^]complex128, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_norm: c.size_t = 1) ---

	// ===================================================================================
	// COMPUTATIONAL ROUTINES - LU Factorization
	// ===================================================================================

	// GETRF - Compute LU factorization with partial pivoting
	// Core factorization routine used by GESV drivers
	// Parameters:
	//   m, n: Dimensions of matrix A
	//   A: Matrix [m x n], overwritten with L and U factors
	//   lda: Leading dimension of A
	//   ipiv: Pivot indices [min(m,n)]
	//   info: Status (>0 means U(info,info) is singular)
	// cgetrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---
	// dgetrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---
	// sgetrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---
	// zgetrf_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---

	// GETRF2 - Recursive LU factorization
	// More efficient than GETRF for large matrices on modern architectures
	// Uses recursive blocked algorithm with better cache performance
	cgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, info: ^Info) ---
	dgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, info: ^Info) ---
	sgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, info: ^Info) ---
	zgetrf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, info: ^Info) ---

	// GETF2 - Unblocked LU factorization
	// Used internally by GETRF for small matrices or panel factorization
	// Direct implementation without blocking for better performance on small matrices
	// cgetf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---
	// dgetf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---
	// sgetf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---
	// zgetf2_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, info: ^Info) -> i32 ---

	// GETRS - Solve linear system using LU factorization from GETRF
	// Forward and backward substitution with pivoting
	// Parameters:
	//   trans: 'N'=no transpose, 'T'=transpose, 'C'=conjugate transpose
	//   n: Order of matrix A
	//   nrhs: Number of right-hand sides
	//   A: LU factorization from GETRF [n x n]
	//   lda: Leading dimension of A
	//   ipiv: Pivot indices from GETRF
	//   B: RHS matrix [n x nrhs], overwritten with solution
	//   ldb: Leading dimension of B
	//   info: Status
	// cgetrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// dgetrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// sgetrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---
	// zgetrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, _: c.size_t) -> i32 ---

	// GETRI - Compute inverse of a matrix using LU factorization
	// Uses the LU factorization computed by GETRF
	// Parameters:
	//   n: Order of matrix A
	//   A: LU factorization from GETRF, overwritten with inverse
	//   lda: Leading dimension of A
	//   ipiv: Pivot indices from GETRF
	//   work: Workspace array
	//   lwork: Size of work array (use -1 for query)
	//   info: Status
	cgetri_ :: proc(n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info) ---
	dgetri_ :: proc(n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, work: [^]f64, lwork: ^blasint, info: ^Info) ---
	sgetri_ :: proc(n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, work: [^]f32, lwork: ^blasint, info: ^Info) ---
	zgetri_ :: proc(n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info) ---

	// GERFS - Iterative refinement for linear system solution
	// Improves the computed solution and provides error bounds
	// Parameters:
	//   trans: 'N'=no transpose, 'T'=transpose, 'C'=conjugate
	//   n, nrhs: Order and number of RHS
	//   A: Original matrix [n x n]
	//   AF: LU factorization from GETRF
	//   ipiv: Pivot indices from GETRF
	//   B: Original RHS matrix
	//   X: Computed solution, improved on output
	//   ferr: Forward error bounds for each solution [nrhs]
	//   berr: Backward error bounds [nrhs]
	cgerfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_trans: c.size_t = 1) ---
	dgerfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgerfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgerfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_trans: c.size_t = 1) ---

	// GERFSX - Extra-precise iterative refinement with error bounds
	// Most accurate refinement with multiple error estimates
	// Parameters:
	//   trans: 'N'=none, 'T'=transpose, 'C'=conjugate
	//   equed: Equilibration status from GESVXX
	//   n, nrhs: Order and number of RHS
	//   A: Original or equilibrated matrix
	//   AF: LU factorization
	//   ipiv: Pivot indices
	//   R, C: Row and column scale factors
	//   B: RHS matrix
	//   X: Solution matrix (refined)
	//   rcond: Reciprocal condition number
	//   berr: Component-wise backward error
	//   n_err_bnds: Number of error bounds
	//   err_bnds_norm, err_bnds_comp: Error bound arrays
	//   nparams, params: Algorithm parameters
	cgerfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, R: [^]f32, C: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	dgerfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, R: [^]f64, C: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	sgerfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, R: [^]f32, C: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	zgerfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, R: [^]f64, C: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---

	// ===================================================================================
	// EQUILIBRATION ROUTINES
	// ===================================================================================

	// GEEQU - Compute row and column equilibration scale factors
	// Used to improve conditioning before solving linear systems
	// Parameters:
	//   m, n: Dimensions of matrix A
	//   A: Input matrix [m x n]
	//   lda: Leading dimension of A
	//   R: Row scale factors [m] (output)
	//   C: Column scale factors [n] (output)
	//   rowcnd: Ratio of smallest to largest row scale (output)
	//   colcnd: Ratio of smallest to largest column scale (output)
	//   amax: Absolute value of largest matrix element (output)
	//   info: Status (>0 means row/col info has zero norm)
	cgeequ_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgeequ_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgeequ_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgeequ_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---

	// GEEQUB - Compute row and column equilibration with better algorithm
	// More robust than GEEQU, reduces over/underflow risk
	// Uses base-radix representation to avoid overflow
	// Parameters same as GEEQU
	cgeequb_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex64, lda: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgeequb_ :: proc(m: ^blasint, n: ^blasint, A: [^]f64, lda: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgeequb_ :: proc(m: ^blasint, n: ^blasint, A: [^]f32, lda: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgeequb_ :: proc(m: ^blasint, n: ^blasint, A: [^]complex128, lda: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---

	// LAQGE - Apply row and column equilibration to a matrix
	// Scales matrix A by diag(R)*A*diag(C)
	// MISSING

	// LASWP - Apply row interchanges to a matrix
	// Performs a series of row swaps specified by pivot vector
	// Parameters:
	//   n: Number of columns in matrix A
	//   A: Matrix to be permuted
	//   lda: Leading dimension of A
	//   k1, k2: First and last row to be interchanged
	//   ipiv: Pivot indices (for i=k1..k2, row i swapped with ipiv[i])
	//   incx: Increment for pivot indices (1 or -1)
	// claswp_ :: proc(n: ^blasint, A: [^]complex64, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: [^]blasint, incx: ^blasint) -> i32 ---
	// dlaswp_ :: proc(n: ^blasint, A: [^]f64, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: [^]blasint, incx: ^blasint) -> i32 ---
	// slaswp_ :: proc(n: ^blasint, A: [^]f32, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: [^]blasint, incx: ^blasint) -> i32 ---
	// zlaswp_ :: proc(n: ^blasint, A: [^]complex128, lda: ^blasint, k1: ^blasint, k2: ^blasint, ipiv: [^]blasint, incx: ^blasint) -> i32 ---


	// getc2: triangular factor, with complete pivoting

	// gesc2: triangular solve using factor, with complete pivoting

	// latdf: Dif-estimate with complete pivoting LU, step in tgsen

	// la_gercond: Skeel condition number estimate

	// la_gerpvgrw: reciprocal pivot growth

	// la_gerfsx_extended: step in gerfsx

	// ===================================================================================
	// BANDED MATRICES - Computational Routines
	// ===================================================================================

	// GBCON - Estimate reciprocal condition number of banded matrix
	// Uses the LU factorization computed by GBTRF
	// Parameters:
	//   norm: '1'=1-norm, 'I'=infinity norm
	//   n: Order of matrix A
	//   kl, ku: Number of sub/superdiagonals
	//   AB: LU factorization from GBTRF in band storage
	//   ldab: Leading dimension of AB
	//   ipiv: Pivot indices from GBTRF
	//   anorm: Norm of the original matrix A
	//   rcond: Reciprocal condition number (output)
	//   work: Workspace array
	//   rwork/iwork: Real/integer workspace
	//   info: Status
	cgbcon_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex64, ldab: ^blasint, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_norm: c.size_t = 1) ---
	dgbcon_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f64, ldab: ^blasint, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1) ---
	sgbcon_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f32, ldab: ^blasint, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1) ---
	zgbcon_ :: proc(norm: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex128, ldab: ^blasint, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_norm: c.size_t = 1) ---

	// GBTRF - LU factorization of banded matrix
	// Computes LU decomposition with partial pivoting for banded matrices
	// Parameters:
	//   m, n: Dimensions of matrix A
	//   kl, ku: Number of sub/superdiagonals
	//   AB: Banded matrix in band storage, overwritten with L and U factors
	//   ldab: Leading dimension of AB (ldab >= 2*kl+ku+1)
	//   ipiv: Pivot indices (output)
	//   info: Status (>0 means U(info,info) is singular)
	cgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex64, ldab: ^blasint, ipiv: [^]blasint, info: ^Info) ---
	dgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f64, ldab: ^blasint, ipiv: [^]blasint, info: ^Info) ---
	sgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f32, ldab: ^blasint, ipiv: [^]blasint, info: ^Info) ---
	zgbtrf_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex128, ldab: ^blasint, ipiv: [^]blasint, info: ^Info) ---

	// GBTF2 - Unblocked LU factorization of banded matrix (MISSING)
	// Used internally by GBTRF for small matrices or panels

	// GBTRS - Solve banded system using LU factorization
	// Solves AX = B using the LU factorization from GBTRF
	// Parameters:
	//   trans: 'N'=no transpose, 'T'=transpose, 'C'=conjugate
	//   n: Order of matrix A
	//   kl, ku: Number of sub/superdiagonals
	//   nrhs: Number of right-hand sides
	//   AB: LU factorization from GBTRF
	//   ldab: Leading dimension of AB
	//   ipiv: Pivot indices from GBTRF
	//   B: RHS matrix [n x nrhs], overwritten with solution
	//   ldb: Leading dimension of B
	//   info: Status
	cgbtrs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	dgbtrs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgbtrs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgbtrs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---

	// GBRFS - Iterative refinement for banded system solution
	// Improves the computed solution and provides error bounds
	// Parameters:
	//   trans: 'N'=no transpose, 'T'=transpose, 'C'=conjugate
	//   n: Order of matrix A
	//   kl, ku: Number of sub/superdiagonals
	//   nrhs: Number of right-hand sides
	//   AB: Original banded matrix
	//   AFB: LU factorization from GBTRF
	//   ipiv: Pivot indices from GBTRF
	//   B: Original RHS matrix
	//   X: Computed solution, improved on output
	//   ferr: Forward error bounds [nrhs]
	//   berr: Backward error bounds [nrhs]
	cgbrfs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, AFB: [^]complex64, ldafb: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_trans: c.size_t = 1) ---
	dgbrfs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, AFB: [^]f64, ldafb: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgbrfs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, AFB: [^]f32, ldafb: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgbrfs_ :: proc(trans: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, AFB: [^]complex128, ldafb: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_trans: c.size_t = 1) ---

	// GBRFSX - Extra-precise iterative refinement for banded systems
	// Most accurate refinement with multiple error estimates
	// Similar to GERFSX but for banded matrices
	// Parameters include equilibration status, scale factors,
	// multiple error bounds, and algorithm parameters
	cgbrfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, AFB: [^]complex64, ldafb: ^blasint, ipiv: [^]blasint, R: [^]f32, C: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	dgbrfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, AFB: [^]f64, ldafb: ^blasint, ipiv: [^]blasint, R: [^]f64, C: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	sgbrfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, AFB: [^]f32, ldafb: ^blasint, ipiv: [^]blasint, R: [^]f32, C: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---
	zgbrfsx_ :: proc(trans: ^char, equed: ^char, n: ^blasint, kl: ^blasint, ku: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, AFB: [^]complex128, ldafb: ^blasint, ipiv: [^]blasint, R: [^]f64, C: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_trans: c.size_t = 1, l_equed: c.size_t = 1) ---

	// GBEQU - Compute equilibration scale factors for banded matrix
	// Used to improve conditioning of banded systems
	// Parameters:
	//   m, n: Dimensions of matrix A
	//   kl, ku: Number of sub/superdiagonals
	//   AB: Banded matrix in band storage
	//   ldab: Leading dimension of AB
	//   R: Row scale factors [m] (output)
	//   C: Column scale factors [n] (output)
	//   rowcnd: Ratio of smallest to largest row scale
	//   colcnd: Ratio of smallest to largest column scale
	//   amax: Absolute value of largest matrix element
	//   info: Status (>0 means row/col info has zero norm)
	cgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex64, ldab: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f64, ldab: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f32, ldab: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgbequ_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex128, ldab: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---

	// GBEQUB - Improved equilibration for banded matrices
	// More robust than GBEQU, reduces over/underflow risk
	// Uses base-radix representation to avoid overflow
	// Parameters same as GBEQU
	cgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex64, ldab: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	dgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f64, ldab: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---
	sgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]f32, ldab: ^blasint, R: [^]f32, C: [^]f32, rowcnd: ^f32, colcnd: ^f32, amax: ^f32, info: ^Info) ---
	zgbequb_ :: proc(m: ^blasint, n: ^blasint, kl: ^blasint, ku: ^blasint, AB: [^]complex128, ldab: ^blasint, R: [^]f64, C: [^]f64, rowcnd: ^f64, colcnd: ^f64, amax: ^f64, info: ^Info) ---

	// LAQGB - Apply row and column equilibration to banded matrix (MISSING)
	// Scales banded matrix AB by diag(R)*AB*diag(C)

	// LA_GBRCOND - Skeel condition number estimate for banded matrices (MISSING)
	// Component-wise condition number estimation

	// LA_GBRPVGRW - Reciprocal pivot growth for banded matrices (MISSING)
	// Measures stability of the LU factorization

	// LA_GBRFSX_EXTENDED - Extended precision refinement step (MISSING)
	// Used internally by GBRFSX

	// ===================================================================================
	// TRIDIAGONAL MATRICES - Computational Routines
	// ===================================================================================

	// GTCON - Estimate reciprocal condition number of tridiagonal matrix
	// Uses the LU factorization computed by GTTRF
	// Parameters:
	//   norm: '1'=1-norm, 'I'=infinity norm
	//   n: Order of matrix A
	//   DL, D, DU: Tridiagonal matrix diagonals
	//   DU2: Second superdiagonal from factorization
	//   ipiv: Pivot indices from GTTRF
	//   anorm: Norm of the original matrix
	//   rcond: Reciprocal condition number (output)
	//   work: Workspace array
	//   info: Status
	cgtcon_ :: proc(norm: ^char, n: ^blasint, DL: [^]complex64, D: [^]complex64, DU: [^]complex64, DU2: [^]complex64, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, info: ^Info, l_norm: c.size_t = 1) ---
	dgtcon_ :: proc(norm: ^char, n: ^blasint, DL: [^]f64, D: [^]f64, DU: [^]f64, DU2: [^]f64, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1) ---
	sgtcon_ :: proc(norm: ^char, n: ^blasint, DL: [^]f32, D: [^]f32, DU: [^]f32, DU2: [^]f32, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1) ---
	zgtcon_ :: proc(norm: ^char, n: ^blasint, DL: [^]complex128, D: [^]complex128, DU: [^]complex128, DU2: [^]complex128, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, info: ^Info, l_norm: c.size_t = 1) ---

	// GTTRF - LU factorization of tridiagonal matrix
	// Computes LU decomposition with partial pivoting
	// Parameters:
	//   n: Order of matrix A
	//   DL: Subdiagonal [n-1], overwritten with L factors
	//   D: Diagonal [n], overwritten with U diagonal
	//   DU: Superdiagonal [n-1], overwritten with U factors
	//   DU2: Second superdiagonal of U [n-2] (output)
	//   ipiv: Pivot indices (output)
	//   info: Status (>0 means U(info,info) is singular)
	cgttrf_ :: proc(n: ^blasint, DL: [^]complex64, D: [^]complex64, DU: [^]complex64, DU2: [^]complex64, ipiv: [^]blasint, info: ^Info) ---
	dgttrf_ :: proc(n: ^blasint, DL: [^]f64, D: [^]f64, DU: [^]f64, DU2: [^]f64, ipiv: [^]blasint, info: ^Info) ---
	sgttrf_ :: proc(n: ^blasint, DL: [^]f32, D: [^]f32, DU: [^]f32, DU2: [^]f32, ipiv: [^]blasint, info: ^Info) ---
	zgttrf_ :: proc(n: ^blasint, DL: [^]complex128, D: [^]complex128, DU: [^]complex128, DU2: [^]complex128, ipiv: [^]blasint, info: ^Info) ---

	// GTTRS - Solve tridiagonal system using LU factorization
	// Solves AX = B using the LU factorization from GTTRF
	// Parameters:
	//   trans: 'N'=no transpose, 'T'=transpose, 'C'=conjugate
	//   n: Order of matrix A
	//   nrhs: Number of right-hand sides
	//   DL, D, DU, DU2: LU factorization from GTTRF
	//   ipiv: Pivot indices from GTTRF
	//   B: RHS matrix [n x nrhs], overwritten with solution
	//   ldb: Leading dimension of B
	//   info: Status
	cgttrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]complex64, D: [^]complex64, DU: [^]complex64, DU2: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	dgttrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]f64, D: [^]f64, DU: [^]f64, DU2: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgttrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]f32, D: [^]f32, DU: [^]f32, DU2: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgttrs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]complex128, D: [^]complex128, DU: [^]complex128, DU2: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_trans: c.size_t = 1) ---

	// GTTS2 - Solve one tridiagonal system using LU factorization (MISSING)
	// Single RHS version of GTTRS

	// GTRFS - Iterative refinement for tridiagonal system solution
	// Improves the computed solution and provides error bounds
	// Parameters:
	//   trans: 'N'=no transpose, 'T'=transpose, 'C'=conjugate
	//   n, nrhs: Order and number of RHS
	//   DL, D, DU: Original tridiagonal matrix
	//   DLF, DF, DUF, DU2: LU factorization from GTTRF
	//   ipiv: Pivot indices from GTTRF
	//   B: Original RHS matrix
	//   X: Computed solution, improved on output
	//   ferr: Forward error bounds [nrhs]
	//   berr: Backward error bounds [nrhs]
	cgtrfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]complex64, D: [^]complex64, DU: [^]complex64, DLF: [^]complex64, DF: [^]complex64, DUF: [^]complex64, DU2: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_trans: c.size_t = 1) ---
	dgtrfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]f64, D: [^]f64, DU: [^]f64, DLF: [^]f64, DF: [^]f64, DUF: [^]f64, DU2: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1) ---
	sgtrfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]f32, D: [^]f32, DU: [^]f32, DLF: [^]f32, DF: [^]f32, DUF: [^]f32, DU2: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_trans: c.size_t = 1) ---
	zgtrfs_ :: proc(trans: ^char, n: ^blasint, nrhs: ^blasint, DL: [^]complex128, D: [^]complex128, DU: [^]complex128, DLF: [^]complex128, DF: [^]complex128, DUF: [^]complex128, DU2: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_trans: c.size_t = 1) ---

	// ===================================================================================
	// Cholesky: Hermitian/symmetric positive definite matrix, driver
	// https://www.netlib.org/lapack/explore-html/de/db1/group__posv__driver__grp.html
	// ===================================================================================
	// full
	// POSV: Simple driver, solves A*X = B using Cholesky factorization
	// A is HPD (Hermitian positive definite) or SPD (symmetric positive definite)
	cposv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dposv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	sposv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zposv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// POSVX: Expert driver with equilibration, condition estimation, and error bounds
	// fact='N': factorize and solve, fact='F': use pre-factored AF
	// equed: 'N'=no equilibration, 'Y'=equilibrated
	cposvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, equed: ^char, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	dposvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, equed: ^char, S: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	sposvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, equed: ^char, S: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zposvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, equed: ^char, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	// POSVXX: Extra precise driver with componentwise error bounds
	// Provides forward and backward error bounds for each solution component
	cposvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, equed: ^char, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	dposvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, equed: ^char, S: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	sposvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, equed: ^char, S: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zposvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, equed: ^char, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	// DSPOSV/ZCPOSV: Mixed precision iterative refinement
	// Solves in double precision, refines using single precision for better performance
	dsposv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, work: [^]f64, swork: [^]f32, iter: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zcposv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, work: [^]complex128, swork: [^]complex64, rwork: [^]f64, iter: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// packed
	// PPSV: Solves A*X = B using packed storage format for positive definite matrices
	// AP contains upper or lower triangle in packed format (n*(n+1)/2 elements)
	cppsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dppsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	sppsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zppsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// PPSVX: Expert driver for packed positive definite matrices
	cppsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, AFP: [^]complex64, equed: ^char, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	dppsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, AFP: [^]f64, equed: ^char, S: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	sppsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, AFP: [^]f32, equed: ^char, S: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zppsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, AFP: [^]complex128, equed: ^char, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	// banded
	// PBSV: Solves A*X = B for banded positive definite matrices
	// AB stores the banded matrix with kd super/sub-diagonals
	cpbsv_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dpbsv_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spbsv_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpbsv_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// PBSVX: Expert driver for banded positive definite matrices
	cpbsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, AFB: [^]complex64, ldafb: ^blasint, equed: ^char, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	dpbsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, AFB: [^]f64, ldafb: ^blasint, equed: ^char, S: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	spbsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, AFB: [^]f32, ldafb: ^blasint, equed: ^char, S: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zpbsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, AFB: [^]complex128, ldafb: ^blasint, equed: ^char, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	// tridiagonal
	// PTSV: Solves A*X = B for tridiagonal positive definite matrices
	// D: diagonal elements, E: off-diagonal elements
	cptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]complex64, B: [^]complex64, ldb: ^blasint, info: ^Info) ---
	dptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]f64, B: [^]f64, ldb: ^blasint, info: ^Info) ---
	sptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]f32, B: [^]f32, ldb: ^blasint, info: ^Info) ---
	zptsv_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]complex128, B: [^]complex128, ldb: ^blasint, info: ^Info) ---

	// PTSVX: Expert driver for tridiagonal positive definite matrices
	cptsvx_ :: proc(fact: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]complex64, DF: [^]f32, EF: [^]complex64, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1) ---
	dptsvx_ :: proc(fact: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]f64, DF: [^]f64, EF: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, info: ^Info, l_fact: c.size_t = 1) ---
	sptsvx_ :: proc(fact: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]f32, DF: [^]f32, EF: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, info: ^Info, l_fact: c.size_t = 1) ---
	zptsvx_ :: proc(fact: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]complex128, DF: [^]f64, EF: [^]complex128, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1) ---

	// ===================================================================================
	// Cholesky: computational routines (factor, cond, etc.)
	// https://www.netlib.org/lapack/explore-html/dd/dce/group__posv__comp__grp.html
	// ===================================================================================
	// full
	// POCON: Estimates reciprocal condition number of positive definite matrix
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// A [in]: factored matrix from POTRF
	// lda [in]: leading dimension of A >= n
	// anorm [in]: 1-norm of original matrix
	// rcond [out]: reciprocal condition number
	// work [workspace]: size at least 3*n (real) or 2*n (complex)
	// rwork [workspace]: size at least n (complex only)
	// iwork [workspace]: size at least n (real only)
	// info [out]: 0 on success
	cpocon_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dpocon_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spocon_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpocon_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	// POTRF: Cholesky factorization (blocked algorithm)
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// A [in,out]: on entry, positive definite matrix; on exit, Cholesky factor
	// lda [in]: leading dimension of A >= n
	// info [out]: 0 on success, >0 if A(i,i) is not positive
	// cpotrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) -> i32 ---
	// dpotrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) -> i32 ---
	// spotrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) -> i32 ---
	// zpotrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) -> i32 ---

	// POTRF2: Cholesky factorization (recursive algorithm)
	// Parameters same as POTRF
	cpotrf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dpotrf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spotrf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpotrf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// POTF2: Cholesky factorization (unblocked algorithm)
	// Parameters same as POTRF
	// cpotf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// dpotf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// spotf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// zpotf2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// POTRS: Solve A*X = B using Cholesky factorization
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// nrhs [in]: number of right-hand sides
	// A [in]: Cholesky factor from POTRF
	// lda [in]: leading dimension of A >= n
	// B [in,out]: on entry, RHS matrix; on exit, solution X
	// ldb [in]: leading dimension of B >= n
	// info [out]: 0 on success
	// cpotrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// dpotrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// spotrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// zpotrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// POTRI: Compute inverse of positive definite matrix
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// A [in,out]: on entry, Cholesky factor; on exit, inverse
	// lda [in]: leading dimension of A >= n
	// info [out]: 0 on success
	// cpotri_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// dpotri_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// spotri_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	// zpotri_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// PORFS: Refine solution and compute error bounds
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// nrhs [in]: number of right-hand sides
	// A [in]: original positive definite matrix
	// lda [in]: leading dimension of A >= n
	// AF [in]: Cholesky factor from POTRF
	// ldaf [in]: leading dimension of AF >= n
	// B [in]: right-hand side matrix
	// ldb [in]: leading dimension of B >= n
	// X [in,out]: on entry, solution from POTRS; on exit, refined solution
	// ldx [in]: leading dimension of X >= n
	// ferr [out]: forward error bounds for each solution
	// berr [out]: backward error bounds for each solution
	// work [workspace]: size at least 3*n (real) or 2*n (complex)
	// rwork [workspace]: size at least n (complex only)
	// iwork [workspace]: size at least n (real only)
	// info [out]: 0 on success
	cporfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dporfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	sporfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zporfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	// Cholesky with pivoting for rank-deficient matrices (Computational)
	// https://www.netlib.org/lapack/explore-html/dd/dce/group__posv__comp__grp.html
	// PSTRF: Cholesky factorization with complete pivoting
	cpstrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, piv: [^]blasint, rank: ^blasint, tol: ^f32, work: [^]f32, info: ^Info, _: c.size_t) ---
	dpstrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, piv: [^]blasint, rank: ^blasint, tol: ^f64, work: [^]f64, info: ^Info, _: c.size_t) ---
	spstrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, piv: [^]blasint, rank: ^blasint, tol: ^f32, work: [^]f32, info: ^Info, _: c.size_t) ---
	zpstrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, piv: [^]blasint, rank: ^blasint, tol: ^f64, work: [^]f64, info: ^Info, _: c.size_t) ---


	// PORFSX: Extended refinement with multiple error bounds
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// equed [in]: 'N' no equilibration, 'Y' equilibrated
	// n [in]: order of matrix
	// nrhs [in]: number of right-hand sides
	// A [in]: original positive definite matrix (possibly equilibrated)
	// lda [in]: leading dimension of A >= n
	// AF [in]: Cholesky factor from POTRF
	// ldaf [in]: leading dimension of AF >= n
	// S [in]: scale factors from POEQU (if equed='Y')
	// B [in]: right-hand side matrix (possibly equilibrated)
	// ldb [in]: leading dimension of B >= n
	// X [in,out]: on entry, solution from POTRS; on exit, refined solution
	// ldx [in]: leading dimension of X >= n
	// rcond [out]: reciprocal condition number
	// berr [out]: backward error for each solution
	// n_err_bnds [in]: number of error bounds to compute
	// err_bnds_norm [out]: error bounds for normwise error
	// err_bnds_comp [out]: error bounds for componentwise error
	// nparams [in]: number of parameters in params
	// params [in]: algorithm parameters
	// work [workspace]: size at least 4*n (real) or 2*n (complex)
	// rwork [workspace]: size at least 3*n (complex only)
	// iwork [workspace]: size at least n (real only)
	// info [out]: 0 on success
	cporfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	dporfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, S: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	sporfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, S: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zporfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	// POEQU: Compute equilibration scale factors
	// n [in]: order of matrix
	// A [in]: positive definite matrix
	// lda [in]: leading dimension of A >= n
	// S [out]: scale factors, S(i) = 1/sqrt(A(i,i))
	// scond [out]: ratio of smallest to largest scale factor
	// amax [out]: absolute value of largest matrix element
	// info [out]: 0 on success, >0 if A(i,i) <= 0
	cpoequ_ :: proc(n: ^blasint, A: [^]complex64, lda: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	dpoequ_ :: proc(n: ^blasint, A: [^]f64, lda: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info) ---
	spoequ_ :: proc(n: ^blasint, A: [^]f32, lda: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	zpoequ_ :: proc(n: ^blasint, A: [^]complex128, lda: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info) ---

	// POEQUB: Compute equilibration scale factors (base 2 scaling)
	// Parameters same as POEQU but uses powers of 2 for scaling
	cpoequb_ :: proc(n: ^blasint, A: [^]complex64, lda: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	dpoequb_ :: proc(n: ^blasint, A: [^]f64, lda: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info) ---
	spoequb_ :: proc(n: ^blasint, A: [^]f32, lda: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info) ---
	zpoequb_ :: proc(n: ^blasint, A: [^]complex128, lda: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info) ---

	// laqhe: row/col scale matrix

	// la_porcond: Skeel condition number estimate

	// la_porpvgrw: reciprocal pivot growth

	// la_porfsx_extended: step in porfsx

	// packed
	// PPCON: Estimate reciprocal condition number (packed storage)
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// AP [in]: packed Cholesky factor from PPTRF
	// anorm [in]: 1-norm of original matrix
	// rcond [out]: reciprocal condition number
	// work [workspace]: size at least 3*n (real) or 2*n (complex)
	// rwork [workspace]: size at least n (complex only)
	// iwork [workspace]: size at least n (real only)
	// info [out]: 0 on success
	cppcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, anorm: ^f32, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dppcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	sppcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zppcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, anorm: ^f64, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	// PPTRF: Cholesky factorization (packed storage)
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// AP [in,out]: on entry, packed positive definite matrix; on exit, Cholesky factor
	// info [out]: 0 on success, >0 if A(i,i) is not positive
	cpptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dpptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	spptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zpptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	// PPTRS: Solve A*X = B using Cholesky factorization (packed storage)
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// nrhs [in]: number of right-hand sides
	// AP [in]: packed Cholesky factor from PPTRF
	// B [in,out]: on entry, RHS matrix; on exit, solution X
	// ldb [in]: leading dimension of B >= n
	// info [out]: 0 on success
	cpptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dpptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// PPTRI: Compute inverse (packed storage)
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// AP [in,out]: on entry, Cholesky factor; on exit, inverse
	// info [out]: 0 on success
	cpptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dpptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	spptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zpptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	// PPRFS: Refine solution and compute error bounds (packed storage)
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// nrhs [in]: number of right-hand sides
	// AP [in]: original packed positive definite matrix
	// AFP [in]: packed Cholesky factor from PPTRF
	// B [in]: right-hand side matrix
	// ldb [in]: leading dimension of B >= n
	// X [in,out]: on entry, solution from PPTRS; on exit, refined solution
	// ldx [in]: leading dimension of X >= n
	// ferr [out]: forward error bounds for each solution
	// berr [out]: backward error bounds for each solution
	// work [workspace]: size at least 3*n (real) or 2*n (complex)
	// rwork [workspace]: size at least n (complex only)
	// iwork [workspace]: size at least n (real only)
	// info [out]: 0 on success
	cpprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, AFP: [^]complex64, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dpprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, AFP: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, AFP: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, AFP: [^]complex128, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	// PPEQU: Compute equilibration scale factors (packed storage)
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// AP [in]: packed positive definite matrix
	// S [out]: scale factors, S(i) = 1/sqrt(A(i,i))
	// scond [out]: ratio of smallest to largest scale factor
	// amax [out]: absolute value of largest matrix element
	// info [out]: 0 on success, >0 if A(i,i) <= 0
	cppequ_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dppequ_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info, l_uplo: c.size_t = 1) ---
	sppequ_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zppequ_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info, l_uplo: c.size_t = 1) ---

	//  	laqhp: row/col scale matrix

	// rfp
	// PFTRF: Cholesky factorization (RFP storage)
	// transr [in]: 'N' normal format, 'T' transpose/conjugate format
	// uplo [in]: 'U' upper triangle, 'L' lower triangle
	// n [in]: order of matrix
	// A [in,out]: on entry, RFP positive definite matrix; on exit, Cholesky factor
	// info [out]: 0 on success, >0 if A(i,i) is not positive
	cpftrf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dpftrf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]f64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	spftrf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]f32, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zpftrf_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cpftrs_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, B: [^]complex64, ldb: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dpftrs_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, B: [^]f64, ldb: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	spftrs_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, B: [^]f32, ldb: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zpftrs_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, B: [^]complex128, ldb: ^blasint, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cpftri_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]complex64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dpftri_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]f64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	spftri_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]f32, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zpftri_ :: proc(transr: ^char, uplo: ^char, n: ^blasint, A: [^]complex128, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// banded
	cpbcon_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dpbcon_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spbcon_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpbcon_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	cpbtrf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dpbtrf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spbtrf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpbtrf_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// pbtf2: triangular factor panel, level 2

	cpbtrs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dpbtrs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spbtrs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpbtrs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	cpbrfs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, AFB: [^]complex64, ldafb: ^blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dpbrfs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, AFB: [^]f64, ldafb: ^blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	spbrfs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, AFB: [^]f32, ldafb: ^blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zpbrfs_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, AFB: [^]complex128, ldafb: ^blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	cpbequ_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dpbequ_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info, l_uplo: c.size_t = 1) ---
	spbequ_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zpbequ_ :: proc(uplo: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, info: ^Info, l_uplo: c.size_t = 1) ---


	//  laqhb: row/col scale matrix

	// tridiagonal
	cptcon_ :: proc(n: ^blasint, D: [^]f32, E: [^]complex64, anorm: ^f32, rcond: ^f32, rwork: [^]f32, info: ^Info) ---
	dptcon_ :: proc(n: ^blasint, D: [^]f64, E: [^]f64, anorm: ^f64, rcond: ^f64, work: [^]f64, info: ^Info) ---
	sptcon_ :: proc(n: ^blasint, D: [^]f32, E: [^]f32, anorm: ^f32, rcond: ^f32, work: [^]f32, info: ^Info) ---
	zptcon_ :: proc(n: ^blasint, D: [^]f64, E: [^]complex128, anorm: ^f64, rcond: ^f64, rwork: [^]f64, info: ^Info) ---

	cpttrf_ :: proc(n: ^blasint, D: [^]f32, E: [^]complex64, info: ^Info) ---
	dpttrf_ :: proc(n: ^blasint, D: [^]f64, E: [^]f64, info: ^Info) ---
	spttrf_ :: proc(n: ^blasint, D: [^]f32, E: [^]f32, info: ^Info) ---
	zpttrf_ :: proc(n: ^blasint, D: [^]f64, E: [^]complex128, info: ^Info) ---

	cpttrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]complex64, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dpttrs_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]f64, B: [^]f64, ldb: ^blasint, info: ^Info) ---
	spttrs_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]f32, B: [^]f32, ldb: ^blasint, info: ^Info) ---
	zpttrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]complex128, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	//  ptts2: triangular solve using factor, unblocked

	cptrfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]complex64, DF: [^]f32, EF: [^]complex64, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dptrfs_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]f64, DF: [^]f64, EF: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, info: ^Info) ---
	sptrfs_ :: proc(n: ^blasint, nrhs: ^blasint, D: [^]f32, E: [^]f32, DF: [^]f32, EF: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, info: ^Info) ---
	zptrfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, D: [^]f64, E: [^]complex128, DF: [^]f64, EF: [^]complex128, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	// ===================================================================================
	// LDL: Hermitian/symmetric indefinite matrix, driver
	// https://www.netlib.org/lapack/explore-html/d6/d90/group__hesv__driver__grp.html
	// ===================================================================================
	// full, rook pivoting
	chesv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhesv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chesv_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhesv_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chesv_rk_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhesv_rk_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chesvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhesvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---

	chesvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zhesvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	csysv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsysv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssysv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsysv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csysv_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsysv_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssysv_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsysv_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csysv_rk_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsysv_rk_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, E: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssysv_rk_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, E: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsysv_rk_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csysvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, lwork: ^blasint, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dsysvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, lwork: ^blasint, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	ssysvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, lwork: ^blasint, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zsysvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, lwork: ^blasint, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---

	csysvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	dsysvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, S: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	ssysvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, S: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, rpvgrw: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zsysvxx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, equed: ^char, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, rpvgrw: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	// packed, rook pivoting
	chpsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhpsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chpsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, AFP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zhpsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, AFP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---

	cspsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dspsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	sspsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zspsv_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	cspsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, AFP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	dspsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, AFP: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	sspsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, AFP: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---
	zspsvx_ :: proc(fact: ^char, uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, AFP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_fact: c.size_t = 1, l_uplo: c.size_t = 1) ---

	// full, aasen
	chesv_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhesv_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	chesv_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, TB: [^]complex64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhesv_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, TB: [^]complex128, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csysv_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsysv_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssysv_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsysv_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csysv_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, TB: [^]complex64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsysv_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, TB: [^]f64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssysv_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, TB: [^]f32, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsysv_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, TB: [^]complex128, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// ===================================================================================
	// LDL: computational routines (factor, cond, etc.)
	// https://www.netlib.org/lapack/explore-html/d1/d7c/group__hesv__comp__grp.html
	// ===================================================================================

	checon_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhecon_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---
	checon_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhecon_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	csycon_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dsycon_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssycon_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsycon_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	csycon_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dsycon_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, E: [^]f64, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssycon_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, E: [^]f32, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsycon_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	chetrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrf_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// la{he,sy}f: step in hetrf

	// {he,sy}tf2: triangular factor, level 2

	chetrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chetri_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetri_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	csytri_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytri_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, work: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytri_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, work: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytri_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	cherfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zherfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	cherfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zherfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	csyrfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dsyrfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssyrfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsyrfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	csyrfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, AF: [^]complex64, ldaf: ^blasint, ipiv: [^]blasint, S: [^]f32, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	dsyrfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, AF: [^]f64, ldaf: ^blasint, ipiv: [^]blasint, S: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	ssyrfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, AF: [^]f32, ldaf: ^blasint, ipiv: [^]blasint, S: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, rcond: ^f32, berr: ^f32, n_err_bnds: ^blasint, err_bnds_norm: ^f32, err_bnds_comp: ^f32, nparams: ^blasint, params: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---
	zsyrfsx_ :: proc(uplo: ^char, equed: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, AF: [^]complex128, ldaf: ^blasint, ipiv: [^]blasint, S: [^]f64, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, rcond: ^f64, berr: ^f64, n_err_bnds: ^blasint, err_bnds_norm: ^f64, err_bnds_comp: ^f64, nparams: ^blasint, params: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_equed: c.size_t = 1) ---

	cheequb_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zheequb_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	csyequb_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dsyequb_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, work: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	ssyequb_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, S: [^]f32, scond: ^f32, amax: ^f32, work: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zsyequb_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, S: [^]f64, scond: ^f64, amax: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	csyconv_ :: proc(uplo: ^char, way: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, E: [^]complex64, info: ^Info, l_uplo: c.size_t = 1, l_way: c.size_t = 1) ---
	dsyconv_ :: proc(uplo: ^char, way: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, E: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_way: c.size_t = 1) ---
	ssyconv_ :: proc(uplo: ^char, way: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, E: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_way: c.size_t = 1) ---
	zsyconv_ :: proc(uplo: ^char, way: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, E: [^]complex128, info: ^Info, l_uplo: c.size_t = 1, l_way: c.size_t = 1) ---

	chetri2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetri2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chetri2x_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, nb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetri2x_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, nb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chetri_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetri_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytri2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytri2_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytri2_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytri2_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytri2x_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, nb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytri2x_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, work: [^]f64, nb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytri2x_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, work: [^]f32, nb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytri2x_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, nb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytri_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytri_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, E: [^]f64, ipiv: [^]blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytri_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, E: [^]f32, ipiv: [^]blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytri_3_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	//  {he,sy}tri_3x: inverse

	chetrs2_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrs2_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrs2_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrs2_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrs2_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrs2_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	chetrs_3_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrs_3_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrs_3_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrs_3_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, E: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrs_3_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, E: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrs_3_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	cheswapr_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, i1: ^blasint, i2: ^blasint, l_uplo: c.size_t = 1) ---
	zheswapr_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, i1: ^blasint, i2: ^blasint, l_uplo: c.size_t = 1) ---

	csyswapr_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, i1: ^blasint, i2: ^blasint, l_uplo: c.size_t = 1) ---
	dsyswapr_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, i1: ^blasint, i2: ^blasint, l_uplo: c.size_t = 1) ---
	ssyswapr_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, i1: ^blasint, i2: ^blasint, l_uplo: c.size_t = 1) ---
	zsyswapr_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, i1: ^blasint, i2: ^blasint, l_uplo: c.size_t = 1) ---

	// la_hercond: Skeel condition number estimate

	// la_herfsx_extended: step in herfsx

	// la_herpvgrw: reciprocal pivot growth

	// packed, rook v1
	chpcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhpcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	cspcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dspcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	sspcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, ipiv: [^]blasint, anorm: ^f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zspcon_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, ipiv: [^]blasint, anorm: ^f64, rcond: ^f64, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	chptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, ipiv: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, ipiv: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, ipiv: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, ipiv: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, ipiv: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsptrf_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, ipiv: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsptrs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, ipiv: [^]blasint, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	zhptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, ipiv: [^]blasint, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	csptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex64, ipiv: [^]blasint, work: [^]complex64, info: ^Info, l_uplo: c.size_t = 1) ---
	dsptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f64, ipiv: [^]blasint, work: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---
	ssptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]f32, ipiv: [^]blasint, work: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zsptri_ :: proc(uplo: ^char, n: ^blasint, AP: [^]complex128, ipiv: [^]blasint, work: [^]complex128, info: ^Info, l_uplo: c.size_t = 1) ---

	chprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, AFP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	zhprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, AFP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	csprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, AFP: [^]complex64, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1) ---
	dsprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, AFP: [^]f64, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, AFP: [^]f32, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsprfs_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, AFP: [^]complex128, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1) ---

	// full, rook v2
	// {he,sy}con_rook: condition number estimate

	chetrf_rook_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrf_rook_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrf_rook_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrf_rook_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrf_rook_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrf_rook_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	//  la{he,sy}f_rook: triangular factor step

	// {he,sy}tf2_rook: triangular factor, level 2

	chetrs_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrs_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrs_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrs_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrs_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrs_rook_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// {he,sy}tri_rook: triangular inverse

	// full, rook v3
	chetrf_rk_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrf_rk_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrf_rk_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, E: [^]complex64, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrf_rk_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, E: [^]f64, ipiv: [^]blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrf_rk_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, E: [^]f32, ipiv: [^]blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrf_rk_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, E: [^]complex128, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// la{he,sy}f_rk: triangular factor step
	// {he,sy}tf2_rk: triangular factor, level 2

	// syconvf: convert to/from hetrf to hetrf_rk format

	// syconvf_rook: convert to/from hetrf_rook to hetrf_rk format

	// full, aasen
	chetrf_aa_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrf_aa_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrf_aa_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrf_aa_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrf_aa_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrf_aa_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// la{he,sy}f_aa: triangular factor partial factor

	chetrs_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrs_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrs_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, ipiv: [^]blasint, B: [^]complex64, ldb: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrs_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, ipiv: [^]blasint, B: [^]f64, ldb: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrs_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, ipiv: [^]blasint, B: [^]f32, ldb: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrs_aa_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, ipiv: [^]blasint, B: [^]complex128, ldb: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---


	// full, aasen blocked 2-stage
	chetrf_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, TB: [^]complex64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrf_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, TB: [^]complex128, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrf_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, TB: [^]complex64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, work: [^]complex64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrf_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, TB: [^]f64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, work: [^]f64, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrf_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, TB: [^]f32, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, work: [^]f32, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrf_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, TB: [^]complex128, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, work: [^]complex128, lwork: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	chetrs_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, TB: [^]complex64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zhetrs_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, TB: [^]complex128, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	csytrs_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, TB: [^]complex64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	dsytrs_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, TB: [^]f64, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	ssytrs_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, TB: [^]f32, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---
	zsytrs_aa_2stage_ :: proc(uplo: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, TB: [^]complex128, ltb: ^blasint, ipiv: [^]blasint, ipiv2: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1) ---

	// ===================================================================================
	// Triangular computational routines (solve, cond, etc.)
	// https://www.netlib.org/lapack/explore-html/dc/dab/group__trsv__comp__grp.html
	// ===================================================================================
	// full
	ctrcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]complex64, lda: ^blasint, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtrcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]f64, lda: ^blasint, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	strcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]f32, lda: ^blasint, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztrcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]complex128, lda: ^blasint, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---

	ctrtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtrtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	strtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztrtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---

	// latrs: triangular solve with robust scaling

	// latrs3: triangular solve with robust scaling, level 3

	// Note: trtri functions are already declared in blas.odin

	// trti2: triangular inverse, level 2

	ctrrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex64, lda: ^blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtrrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f64, lda: ^blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	strrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]f32, lda: ^blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztrrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, A: [^]complex128, lda: ^blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---

	// lauum: triangular multiply: U^H U

	// lauu2: triangular multiply: U^H U, level 2

	// packed
	ctpcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]complex64, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtpcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]f64, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	stpcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]f32, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztpcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, AP: [^]complex128, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---

	ctptrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtptrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	stptrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztptrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---

	ctptri_ :: proc(uplo: ^char, diag: ^char, n: ^blasint, AP: [^]complex64, info: ^Info, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtptri_ :: proc(uplo: ^char, diag: ^char, n: ^blasint, AP: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	stptri_ :: proc(uplo: ^char, diag: ^char, n: ^blasint, AP: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztptri_ :: proc(uplo: ^char, diag: ^char, n: ^blasint, AP: [^]complex128, info: ^Info, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---

	ctprfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex64, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtprfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f64, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	stprfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]f32, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztprfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, nrhs: ^blasint, AP: [^]complex128, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---

	// rfp
	ctftri_ :: proc(transr: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]complex64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtftri_ :: proc(transr: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]f64, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	stftri_ :: proc(transr: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]f32, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztftri_ :: proc(transr: ^char, uplo: ^char, diag: ^char, n: ^blasint, A: [^]complex128, info: ^Info, l_transr: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---

	// banded
	ctbcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex64, ldab: ^blasint, rcond: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtbcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, kd: ^blasint, AB: [^]f64, ldab: ^blasint, rcond: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	stbcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, kd: ^blasint, AB: [^]f32, ldab: ^blasint, rcond: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztbcon_ :: proc(norm: ^char, uplo: ^char, diag: ^char, n: ^blasint, kd: ^blasint, AB: [^]complex128, ldab: ^blasint, rcond: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_norm: c.size_t = 1, l_uplo: c.size_t = 1, l_diag: c.size_t = 1) ---

	ctbtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, B: [^]complex64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtbtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, B: [^]f64, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	stbtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, B: [^]f32, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztbtrs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, B: [^]complex128, ldb: ^blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---

	// latbs: triangular solve with scaling

	ctbrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex64, ldab: ^blasint, B: [^]complex64, ldb: ^blasint, X: [^]complex64, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]complex64, rwork: [^]f32, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	dtbrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f64, ldab: ^blasint, B: [^]f64, ldb: ^blasint, X: [^]f64, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]f64, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	stbrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]f32, ldab: ^blasint, B: [^]f32, ldb: ^blasint, X: [^]f32, ldx: ^blasint, ferr: ^f32, berr: ^f32, work: [^]f32, iwork: [^]blasint, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---
	ztbrfs_ :: proc(uplo: ^char, trans: ^char, diag: ^char, n: ^blasint, kd: ^blasint, nrhs: ^blasint, AB: [^]complex128, ldab: ^blasint, B: [^]complex128, ldb: ^blasint, X: [^]complex128, ldx: ^blasint, ferr: ^f64, berr: ^f64, work: [^]complex128, rwork: [^]f64, info: ^Info, l_uplo: c.size_t = 1, l_trans: c.size_t = 1, l_diag: c.size_t = 1) ---

	// ===================================================================================
	// Auxiliary routines
	// https://www.netlib.org/lapack/explore-html/d1/d1e/group__solve__aux__grp.html
	// ===================================================================================

	// lacn2: 1-norm estimate, e.g., || A^{-1} ||_1 in gecon
	clacn2_ :: proc(n: ^blasint, V: ^complex64, X: [^]complex64, est: ^f32, kase: ^blasint, ISAVE: ^blasint) ---
	dlacn2_ :: proc(n: ^blasint, V: ^f64, X: [^]f64, ISGN: ^blasint, est: ^f64, kase: ^blasint, ISAVE: ^blasint) ---
	slacn2_ :: proc(n: ^blasint, V: ^f32, X: [^]f32, ISGN: ^blasint, est: ^f32, kase: ^blasint, ISAVE: ^blasint) ---
	zlacn2_ :: proc(n: ^blasint, V: ^complex128, X: [^]complex128, est: ^f64, kase: ^blasint, ISAVE: ^blasint) ---

	// lacon: 1-norm estimate, e.g., || A^{-1} ||_1 in gecon, old

	// la_lin_berr: backward error
}
