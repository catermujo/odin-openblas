package openblas

import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:math"
import "core:slice"

// ===================================================================================
// SINGULAR VALUE DECOMPOSITION
// Decompose matrices as A = U*Sigma*V^T
// ===================================================================================

m_svd :: proc {
	m_svd_f32_c64,
	m_svd_f64_c128,
}

m_svd_qr :: proc {
	m_svd_qr_f32_c64,
	m_svd_qr_f64_c128,
}

m_svd_select :: proc {
	m_svd_select_f32_c64,
	m_svd_select_f64_c128,
}

m_svd_divide :: proc {
	m_svd_divide_f32_c64,
	m_svd_divide_f64_c128,
}

m_cs_decomp :: proc {
	m_cs_decomp_f32_f64,
	m_cs_decomp_c64_c128,
}

// Jacobi SVD computation options
JacobiSVDMode :: struct {
	// Job options
	compute_u:      bool, // Compute left singular vectors
	compute_v:      bool, // Compute right singular vectors

	// Algorithm options
	preprocess:     bool, // Apply preprocessing
	transpose_hint: bool, // Hint that A^T may be more efficient
	perturb:        bool, // Apply controlled perturbation for rank detection

	// Accuracy options
	high_accuracy:  bool, // Request highest accuracy mode
	restrict_range: bool, // Restrict range of matrix for conditioning
}

// Compute SVD using Jacobi method (highest accuracy)
// Especially good for small matrices and when high accuracy is needed
m_svd_jacobi :: proc {
	m_svd_jacobi_f32_c64,
	m_svd_jacobi_f64_c128,
}

m_svd_jacobi_variant :: proc {
	m_svd_jacobi_variant_f32_c64,
	m_svd_jacobi_variant_f64_c128,
}

// ===================================================================================
// STANDARD SVD
// ===================================================================================

// Compute SVD using standard algorithm
// A = U * Sigma * V^T
// Combined f32 real and complex64 SVD
m_svd_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
	allocator := context.allocator,
) -> (
	S: []f32,
	U: Matrix(T),
	VT: Matrix(T),
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (transposed)
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Determine job parameters
	jobu_c: cstring
	jobvt_c: cstring

	if !compute_u {
		jobu_c = "N" // Don't compute U
	} else if full_matrices && m > min_mn {
		jobu_c = "A" // All m columns of U
	} else {
		jobu_c = "S" // First min(m,n) columns of U
	}

	if !compute_vt {
		jobvt_c = "N" // Don't compute VT
	} else if full_matrices && n > min_mn {
		jobvt_c = "A" // All n rows of VT
	} else {
		jobvt_c = "S" // First min(m,n) rows of VT
	}

	// Allocate singular values (always real)
	S = make([]f32, min_mn)

	// Allocate U and VT based on job
	ldu: Blas_Int = 1
	if compute_u {
		u_cols := full_matrices && m > min_mn ? int(m) : int(min_mn)
		U = make_matrix(T, int(m), u_cols, .General)
		ldu = U.ld
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		vt_rows := full_matrices && n > min_mn ? int(n) : int(min_mn)
		VT = make_matrix(T, vt_rows, int(n), .General)
		ldvt = VT.ld
	}

	// Complex versions need real workspace for singular values
	rwork: []f32
	if T == complex64 {
		rwork = make([]f32, 5 * min_mn)
		defer delete(rwork)
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f32 {
		lapack.sgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		lwork = Blas_Int(real(work_query))

		// Allocate workspace
		work := make([]f32, lwork)
		defer delete(work)

		// Compute SVD
		lapack.sgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == complex64 {
		lapack.cgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
			1,
			1,
		)
		lwork = Blas_Int(real(work_query))

		// Allocate workspace
		work := make([]complex64, lwork)
		defer delete(work)

		// Compute SVD
		lapack.cgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	return S, U, VT, info
}

// Combined f64 real and complex128 SVD
m_svd_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
	allocator := context.allocator,
) -> (
	S: []f64,
	U: Matrix(T),
	VT: Matrix(T),
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (transposed)
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Determine job parameters
	jobu_c: cstring
	jobvt_c: cstring

	if !compute_u {
		jobu_c = "N"
	} else if full_matrices && m > min_mn {
		jobu_c = "A"
	} else {
		jobu_c = "S"
	}

	if !compute_vt {
		jobvt_c = "N"
	} else if full_matrices && n > min_mn {
		jobvt_c = "A"
	} else {
		jobvt_c = "S"
	}

	// Allocate singular values (always real)
	S = make([]f64, min_mn)

	// Allocate U and VT
	ldu: Blas_Int = 1
	if compute_u {
		u_cols := full_matrices && m > min_mn ? int(m) : int(min_mn)
		U = make_matrix(T, int(m), u_cols, .General)
		ldu = U.ld
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		vt_rows := full_matrices && n > min_mn ? int(n) : int(min_mn)
		VT = make_matrix(T, vt_rows, int(n), .General)
		ldvt = VT.ld
	}

	// Complex versions need real workspace for singular values
	rwork: []f64
	if T == complex128 {
		rwork = make([]f64, 5 * min_mn)
		defer delete(rwork)
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f64 {
		lapack.dgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := make([]f64, lwork)
		defer delete(work)

		// Compute SVD
		lapack.dgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		lapack.zgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			raw_data(rwork),
			&info,
			1,
			1,
		)
		lwork = Blas_Int(real(work_query))

		// Allocate workspace
		work := make([]complex128, lwork)
		defer delete(work)

		// Compute SVD
		lapack.zgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	return S, U, VT, info
}


// ===================================================================================
// QR-BASED SVD WITH PIVOTING
// ===================================================================================

// SVD using QR factorization with column pivoting
// High accuracy and rank-revealing, especially for rank-deficient matrices
// Combined f32 real and complex64 SVD with QR
m_svd_qr_f32_c64 :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true, // Use high accuracy mode
	pivot: bool = true, // Use pivoting
	rank_reveal: bool = true, // Compute numerical rank
	allocator := context.allocator,
) -> (
	S: []f32,
	U: Matrix(T),
	V: Matrix(T),
	numrank: Blas_Int,
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (NOT transposed)// Numerical rank
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("M") // H=high, M=medium accuracy
	jobp_c := pivot ? cstring("P") : cstring("N") // P=pivot, N=no pivot
	jobr_c := rank_reveal ? cstring("R") : cstring("N") // R=rank revealing
	jobu_c := compute_u ? cstring("U") : cstring("N") // U=compute U
	jobv_c := compute_v ? cstring("V") : cstring("N") // V=compute V

	// Allocate singular values (always real for SVD)
	S = make([]f32, min_mn)

	// Allocate U and V
	ldu: Blas_Int = 1
	if compute_u {
		U = make_matrix(T, m, m, .General)
		ldu = U.ld
	}

	ldv: Blas_Int = 1
	if compute_v {
		V = make_matrix(T, int(n), int(n), .General)
		ldv = V.ld
	}

	// Query for optimal workspace sizes
	liwork: Blas_Int = -1
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	iwork_query: Blas_Int
	work_query: T
	rwork_query: f32

	when T == f32 {
		lapack.sgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			&iwork_query,
			&liwork,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		liwork = iwork_query
		lwork = Blas_Int(work_query)
		lrwork = Blas_Int(rwork_query)

		// Allocate workspaces
		iwork := make([]Blas_Int, liwork)
		work := make([]f32, lwork)
		rwork := make([]f32, lrwork)
		defer delete(iwork)
		defer delete(work)
		defer delete(rwork)

		// Compute SVD
		lapack.sgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			raw_data(iwork),
			&liwork,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
	} else when T == complex64 {
		cwork_query: complex64

		lapack.cgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			&iwork_query,
			&liwork,
			&cwork_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		liwork = iwork_query
		lwork = Blas_Int(real(cwork_query))
		lrwork = Blas_Int(rwork_query)

		// Allocate workspaces
		iwork := make([]Blas_Int, liwork)
		cwork := make([]complex64, lwork)
		rwork := make([]f32, lrwork)
		defer delete(iwork)
		defer delete(cwork)
		defer delete(rwork)

		// Compute SVD
		lapack.cgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			raw_data(iwork),
			&liwork,
			raw_data(cwork),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
	}

	// Resize outputs based on numerical rank if requested
	if rank_reveal && numrank < min_mn {
		S = S[:numrank]
		if compute_u {
			U.cols = numrank
		}
		if compute_v {
			V.cols = numrank
		}
	}

	return S, U, V, numrank, info
}

// Combined f64 real and complex128 SVD with QR
m_svd_qr_f64_c128 :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true, // Use high accuracy mode
	pivot: bool = true, // Use pivoting
	rank_reveal: bool = true, // Compute numerical rank
	allocator := context.allocator,
) -> (
	S: []f64,
	U: Matrix(T),
	V: Matrix(T),
	numrank: Blas_Int,
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (NOT transposed)// Numerical rank
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("M") // H=high, M=medium accuracy
	jobp_c := pivot ? cstring("P") : cstring("N") // P=pivot, N=no pivot
	jobr_c := rank_reveal ? cstring("R") : cstring("N") // R=rank revealing
	jobu_c := compute_u ? cstring("U") : cstring("N") // U=compute U
	jobv_c := compute_v ? cstring("V") : cstring("N") // V=compute V

	// Allocate singular values (always real for SVD)
	S = make([]f64, min_mn)

	// Allocate U and V
	ldu: Blas_Int = 1
	if compute_u {
		U = make_matrix(T, m, m, .General)
		ldu = U.ld
	}

	ldv: Blas_Int = 1
	if compute_v {
		V = make_matrix(T, int(n), int(n), .General)
		ldv = V.ld
	}

	// Query for optimal workspace sizes
	liwork: Blas_Int = -1
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	iwork_query: Blas_Int
	work_query: T
	rwork_query: f64

	when T == f64 {
		lapack.dgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			&iwork_query,
			&liwork,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		liwork = iwork_query
		lwork = Blas_Int(work_query)
		lrwork = Blas_Int(rwork_query)

		// Allocate workspaces
		iwork := make([]Blas_Int, liwork)
		work := make([]f64, lwork)
		rwork := make([]f64, lrwork)
		defer delete(iwork)
		defer delete(work)
		defer delete(rwork)

		// Compute SVD
		lapack.dgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			raw_data(iwork),
			&liwork,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
	} else when T == complex128 {
		zwork_query: complex128

		lapack.zgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			&iwork_query,
			&liwork,
			&zwork_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		liwork = iwork_query
		lwork = Blas_Int(real(zwork_query))
		lrwork = Blas_Int(rwork_query)

		// Allocate workspaces
		iwork := make([]Blas_Int, liwork)
		zwork := make([]complex128, lwork)
		rwork := make([]f64, lrwork)
		defer delete(iwork)
		defer delete(zwork)
		defer delete(rwork)

		// Compute SVD
		lapack.zgesvdq_(
			joba_c,
			jobp_c,
			jobr_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&numrank,
			raw_data(iwork),
			&liwork,
			raw_data(zwork),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
	}

	// Resize outputs based on numerical rank if requested
	if rank_reveal && numrank < min_mn {
		S = S[:numrank]
		if compute_u {
			U.cols = numrank
		}
		if compute_v {
			V.cols = numrank
		}
	}

	return S, U, V, numrank, info
}

// ===================================================================================
// SELECTIVE SVD
// ===================================================================================

// Compute selected singular values and vectors
// Can compute subset by index range or value range
// Combined f32 real and complex64 selective SVD
m_svd_select_f32_c64 :: proc(
	A: ^Matrix($T),
	range_mode: string = "A", // "A"=all, "V"=by value, "I"=by index
	vl: f32 = 0, // Lower bound (for range="V")
	vu: f32 = 0, // Upper bound (for range="V")
	il: Blas_Int = 1, // Lower index (for range="I")
	iu: Blas_Int = -1, // Upper index (for range="I", -1 = min(m,n))
	compute_u: bool = true,
	compute_vt: bool = true,
	allocator := context.allocator,
) -> (
	S: []f32,
	U: Matrix(T),
	VT: Matrix(T),
	ns: Blas_Int,
	info: Info, // Selected singular values (always real)// Selected left singular vectors// Selected right singular vectors (transposed)// Number of singular values found
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set range parameter
	range_c := cstring(raw_data(range_mode))

	// Set job parameters
	jobu_c := compute_u ? cstring("V") : cstring("N")
	jobvt_c := compute_vt ? cstring("V") : cstring("N")

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size
	max_ns: Blas_Int
	if range_mode == "A" {
		max_ns = min_mn
	} else if range_mode == "I" {
		max_ns = iu_val - il + 1
	} else {
		max_ns = min_mn // Conservative estimate for value range
	}

	// Allocate singular values (always real for SVD)
	S = make([]f32, max_ns)

	// Allocate U and VT
	ldu: Blas_Int = 1
	if compute_u {
		U = make_matrix(T, m, max_ns, .General)
		ldu = U.ld
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		VT = make_matrix(T, max_ns, n, .General)
		ldvt = VT.ld
	}

	// Allocate integer workspace
	iwork := make([]Blas_Int, 12 * min_mn)
	defer delete(iwork)

	// Complex versions need real workspace
	rwork: []f32
	if T == complex64 {
		rwork_size :=
			min_mn *
			max(
				Blas_Int(5) * min_mn + Blas_Int(7),
				Blas_Int(2) * max_ns * (Blas_Int(3) * max_ns + Blas_Int(1)),
			)
		rwork = make([]f32, rwork_size)
		defer delete(rwork)
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f32 {
		lapack.sgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := make([]T, lwork)
		defer delete(work)

		// Compute SVD
		lapack.sgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	} else when T == complex64 {
		cwork_query: complex64
		lapack.cgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&cwork_query,
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
		lwork = Blas_Int(real(cwork_query))

		// Allocate workspace
		cwork := make([]complex64, lwork)
		defer delete(cwork)

		// Compute SVD
		lapack.cgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(cwork),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	// Resize outputs to actual number found
	if ns < max_ns {
		S = S[:ns]
		if compute_u {
			U.cols = ns
		}
		if compute_vt {
			VT.rows = ns
		}
	}

	return S, U, VT, ns, info
}

// Combined f64 real and complex128 selective SVD
m_svd_select_f64_c128 :: proc(
	A: ^Matrix($T),
	range_mode: string = "A", // "A"=all, "V"=by value, "I"=by index
	vl: f64 = 0, // Lower bound (for range="V")
	vu: f64 = 0, // Upper bound (for range="V")
	il: Blas_Int = 1, // Lower index (for range="I")
	iu: Blas_Int = -1, // Upper index (for range="I", -1 = min(m,n))
	compute_u: bool = true,
	compute_vt: bool = true,
	allocator := context.allocator,
) -> (
	S: []f64,
	U: Matrix(T),
	VT: Matrix(T),
	ns: Blas_Int,
	info: Info, // Selected singular values (always real)// Selected left singular vectors// Selected right singular vectors (transposed)// Number of singular values found
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set range parameter
	range_c := cstring(raw_data(range_mode))

	// Set job parameters
	jobu_c := compute_u ? cstring("V") : cstring("N")
	jobvt_c := compute_vt ? cstring("V") : cstring("N")

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size
	max_ns: Blas_Int
	if range_mode == "A" {
		max_ns = min_mn
	} else if range_mode == "I" {
		max_ns = iu_val - il + 1
	} else {
		max_ns = min_mn // Conservative estimate for value range
	}

	// Allocate singular values (always real for SVD)
	S = make([]f64, max_ns)

	// Allocate U and VT
	ldu: Blas_Int = 1
	if compute_u {
		U = make_matrix(T, m, max_ns, .General)
		ldu = U.ld
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		VT = make_matrix(T, max_ns, n, .General)
		ldvt = VT.ld
	}

	// Allocate integer workspace
	iwork := make([]Blas_Int, 12 * min_mn)
	defer delete(iwork)

	// Complex versions need real workspace
	rwork: []f64
	if T == complex128 {
		rwork_size :=
			min_mn *
			max(
				Blas_Int(5) * min_mn + Blas_Int(7),
				Blas_Int(2) * max_ns * (Blas_Int(3) * max_ns + Blas_Int(1)),
			)
		rwork = make([]f64, rwork_size)
		defer delete(rwork)
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f64 {
		lapack.dgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := make([]f64, lwork)
		defer delete(work)

		// Compute SVD
		lapack.dgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	} else when T == complex128 {
		zwork_query: complex128
		lapack.zgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&zwork_query,
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
		lwork = Blas_Int(real(zwork_query))

		// Allocate workspace
		zwork := make([]complex128, lwork)
		defer delete(zwork)

		// Compute SVD
		lapack.zgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl,
			&vu,
			&il,
			&iu_val,
			&ns,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(zwork),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	// Resize outputs to actual number found
	if ns < max_ns {
		S = S[:ns]
		if compute_u {
			U.cols = ns
		}
		if compute_vt {
			VT.rows = ns
		}
	}

	return S, U, VT, ns, info
}

// DIVIDE-AND-CONQUER SVD
// ===================================================================================

// Compute SVD using divide-and-conquer algorithm
// Faster than standard SVD for large matrices
// Combined f32 real and complex64 divide-and-conquer SVD
m_svd_divide_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
	allocator := context.allocator,
) -> (
	S: []f32,
	U: Matrix(T),
	VT: Matrix(T),
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (transposed)
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Determine job parameter
	jobz_c: cstring
	if !compute_u && !compute_vt {
		jobz_c = "N" // Only singular values
	} else if full_matrices {
		jobz_c = "A" // All columns of U and V
	} else {
		jobz_c = "S" // Min(m,n) columns of U and V
	}

	// Allocate singular values (always real for SVD)
	S = make([]f32, min_mn)

	// Allocate U and VT based on job
	ldu: Blas_Int = 1
	if compute_u {
		u_rows := int(m)
		u_cols := full_matrices ? int(m) : int(min_mn)
		U = make_matrix(T, u_rows, u_cols, .General)
		ldu = U.ld
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		vt_rows := full_matrices ? int(n) : int(min_mn)
		vt_cols := int(n)
		VT = make_matrix(T, vt_rows, vt_cols, .General)
		ldvt = VT.ld
	}

	// Allocate integer workspace
	iwork := make([]Blas_Int, 8 * min_mn)
	defer delete(iwork)

	// Complex versions need real workspace
	rwork: []f32
	if T == complex64 {
		rwork_size :=
			min_mn * max(Blas_Int(5) * min_mn + Blas_Int(7), Blas_Int(2) * min_mn + Blas_Int(1))
		if full_matrices {
			rwork_size = max(rwork_size, Blas_Int(5) * min_mn * min_mn + Blas_Int(5) * min_mn)
		}
		rwork = make([]f32, rwork_size)
		defer delete(rwork)
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f32 {
		lapack.sgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := make([]f32, lwork)
		defer delete(work)

		// Compute SVD
		lapack.sgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
		)
	} else when T == complex64 {
		cwork_query: complex64
		lapack.cgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&cwork_query,
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
		)
		lwork = Blas_Int(real(cwork_query))

		// Allocate workspace
		cwork := make([]complex64, lwork)
		defer delete(cwork)

		// Compute SVD
		lapack.cgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(cwork),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return S, U, VT, info
}

// Combined f64 real and complex128 divide-and-conquer SVD
m_svd_divide_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
	allocator := context.allocator,
) -> (
	S: []f64,
	U: Matrix(T),
	VT: Matrix(T),
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (transposed)
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Determine job parameter
	jobz_c: cstring
	if !compute_u && !compute_vt {
		jobz_c = "N" // Only singular values
	} else if full_matrices {
		jobz_c = "A" // All columns of U and V
	} else {
		jobz_c = "S" // Min(m,n) columns of U and V
	}

	// Allocate singular values (always real for SVD)
	S = make([]f64, min_mn)

	// Allocate U and VT based on job
	ldu: Blas_Int = 1
	if compute_u {
		u_rows := int(m)
		u_cols := full_matrices ? int(m) : int(min_mn)
		U = make_matrix(T, u_rows, u_cols, .General)
		ldu = U.ld
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		vt_rows := full_matrices ? int(n) : int(min_mn)
		vt_cols := int(n)
		VT = make_matrix(T, vt_rows, vt_cols, .General)
		ldvt = VT.ld
	}

	// Allocate integer workspace
	iwork := make([]Blas_Int, 8 * min_mn)
	defer delete(iwork)

	// Complex versions need real workspace
	rwork: []f64
	if T == complex128 {
		rwork_size :=
			min_mn * max(Blas_Int(5) * min_mn + Blas_Int(7), Blas_Int(2) * min_mn + Blas_Int(1))
		if full_matrices {
			rwork_size = max(rwork_size, Blas_Int(5) * min_mn * min_mn + Blas_Int(5) * min_mn)
		}
		rwork = make([]f64, rwork_size)
		defer delete(rwork)
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f64 {
		lapack.dgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
		)
		lwork = Blas_Int(work_query)

		// Allocate workspace
		work := make([]f64, lwork)
		defer delete(work)

		// Compute SVD
		lapack.dgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
		)
	} else when T == complex128 {
		zwork_query: complex128
		lapack.zgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			&zwork_query,
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
		)
		lwork = Blas_Int(real(zwork_query))

		// Allocate workspace
		zwork := make([]complex128, lwork)
		defer delete(zwork)

		// Compute SVD
		lapack.zgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u ? raw_data(U.data) : nil,
			&ldu,
			compute_vt ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(zwork),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return S, U, VT, info
}

// BIDIAGONAL MATRIX OPERATIONS
// ===================================================================================

// CS Decomposition of unitary/orthogonal matrix partitioned into 2x2 blocks
// Computes the CS decomposition of an orthogonal/unitary matrix in bidiagonal-block form
// Combined f32 and f64 CS decomposition
m_cs_decomp_f32_f64 :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	trans: bool = false,
	compute_u1: bool = true, // Compute U1
	compute_u2: bool = true, // Compute U2
	compute_v1t: bool = true, // Compute V1T
	compute_v2t: bool = true, // Compute V2T
	allocator := context.allocator,
) -> (
	theta: []T,
	phi: []T,
	info: Info,
) where T == f32 || T == f64 {
	// Prepare parameters
	U1_shape := [2]Blas_Int{U1.rows, U1.cols}
	U2_shape := [2]Blas_Int{U2.rows, U2.cols}
	V1T_shape := [2]Blas_Int{V1T.rows, V1T.cols}
	V2T_shape := [2]Blas_Int{V2T.rows, V2T.cols}
	m, p, q, r, ldu1, ldu2, ldv1t, ldv2t, jobu1_c, jobu2_c, jobv1t_c, jobv2t_c, trans_c :=
		cs_decomp_prepare(
			U1_shape,
			U2_shape,
			V1T_shape,
			V2T_shape,
			compute_u1,
			compute_u2,
			compute_v1t,
			compute_v2t,
			trans,
		)

	// Allocate angles
	theta = make([]T, r)
	phi = make([]T, r - 1)

	// Allocate diagonal and off-diagonal elements
	b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e := make_bidiag_real_arrays(T, r)
	defer delete_bidiag_real_arrays(b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e)

	when T == f32 {
		lwork: Blas_Int = -1
		work_query: f32
		lapack.sbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		lwork = Blas_Int(work_query)
		work := make([]f32, lwork)
		defer delete(work)

		lapack.sbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

	} else when T == f64 {
		lwork: Blas_Int = -1
		work_query: f64
		lapack.dbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		lwork = Blas_Int(work_query)
		work := make([]f64, lwork)
		defer delete(work)

		lapack.dbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

	}

	return theta, phi, info
}

// Combined complex64 and complex128 CS decomposition
m_cs_decomp_c64_c128 :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	trans: bool = false,
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
	allocator := context.allocator,
) -> (
	theta: []f64,
	phi: []f64,
	info: Info,
) where T == complex64 ||
	T == complex128 {
	// Prepare parameters
	U1_shape := [2]Blas_Int{U1.rows, U1.cols}
	U2_shape := [2]Blas_Int{U2.rows, U2.cols}
	V1T_shape := [2]Blas_Int{V1T.rows, V1T.cols}
	V2T_shape := [2]Blas_Int{V2T.rows, V2T.cols}
	m, p, q, r, ldu1, ldu2, ldv1t, ldv2t, jobu1_c, jobu2_c, jobv1t_c, jobv2t_c, trans_c :=
		cs_decomp_prepare(
			U1_shape,
			U2_shape,
			V1T_shape,
			V2T_shape,
			compute_u1,
			compute_u2,
			compute_v1t,
			compute_v2t,
			trans,
		)

	// Always allocate output as f64 for consistency
	theta_out := make([]f64, r, allocator)
	phi_out := make([]f64, r - 1, allocator)

	// Allocate real workspace
	lrwork := (8 * r) if trans else (7 * r)

	when T == complex64 {
		// Allocate working arrays in f32 precision
		theta := make([]f32, r, context.temp_allocator)
		phi := make([]f32, r - 1, context.temp_allocator)

		// Allocate diagonal and off-diagonal elements
		b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e := make_bidiag_real_arrays(f32, r)
		defer delete_bidiag_real_arrays(b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e)

		// Allocate real workspace
		rwork := make([]f32, lrwork, context.temp_allocator)

		lwork: Blas_Int = -1
		work_query: complex64

		lapack.cbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(rwork),
			&lrwork,
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		lwork = Blas_Int(real(work_query))
		work := make([]complex64, lwork)
		defer delete(work)

		lapack.cbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(rwork),
			&lrwork,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		// Copy f32 results to f64 output
		for i in 0 ..< len(theta) {
			theta_out[i] = f64(theta[i])
		}
		for i in 0 ..< len(phi) {
			phi_out[i] = f64(phi[i])
		}
	} else when T == complex128 {
		// Allocate working arrays in f64 precision
		theta := make([]f64, r, context.temp_allocator)
		phi := make([]f64, r - 1, context.temp_allocator)

		// Allocate diagonal and off-diagonal elements
		b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e := make_bidiag_real_arrays(f64, r)
		defer delete_bidiag_real_arrays(b11d, b11e, b12d, b12e, b21d, b21e, b22d, b22e)

		// Allocate real workspace
		rwork := make([]f64, lrwork, context.temp_allocator)

		lwork: Blas_Int = -1
		work_query: complex128

		lapack.zbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(rwork),
			&lrwork,
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		lwork = Blas_Int(real(work_query))
		work := make([]complex128, lwork)
		defer delete(work)

		lapack.zbbcsd_(
			jobu1_c,
			jobu2_c,
			jobv1t_c,
			jobv2t_c,
			trans_c,
			&m,
			&p,
			&q,
			raw_data(theta),
			raw_data(phi),
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			raw_data(b11d),
			raw_data(b11e),
			raw_data(b12d),
			raw_data(b12e),
			raw_data(b21d),
			raw_data(b21e),
			raw_data(b22d),
			raw_data(b22e),
			raw_data(rwork),
			&lrwork,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)

		// Copy f64 results directly to output
		copy(theta_out, theta)
		copy(phi_out, phi)
	}

	return theta_out, phi_out, info
}

// CS Decomposition for complex64 matrices
// ===================================================================================
// JACOBI SVD (HIGH ACCURACY)
// ===================================================================================

// Combined f32 and complex64 Jacobi SVD
m_svd_jacobi_f32_c64 :: proc(
	A: ^Matrix($T),
	options: JacobiSVDMode = {compute_u = true, compute_v = true, high_accuracy = true},
	allocator := context.allocator,
) -> (
	S: []f32,
	U: Matrix(T),
	V: Matrix(T),
	work_stat: []f32,
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (not V^T)// Statistics about the computation
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Prepare job parameters
	joba_c := options.high_accuracy ? cstring("H") : cstring("C") // H=high accuracy, C=controlled accuracy
	jobu_c := options.compute_u ? cstring("U") : cstring("N")
	jobv_c := options.compute_v ? cstring("V") : cstring("N")
	jobr_c := options.restrict_range ? cstring("R") : cstring("N")
	jobt_c := options.transpose_hint ? cstring("T") : cstring("N")
	jobp_c := options.perturb ? cstring("P") : cstring("N")

	// Allocate singular values
	S = make([]T, min(m, n))

	// Allocate singular vector matrices
	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if options.compute_u {
		U = make_matrix(T, m, m)
		ldu = U.ld
	}
	if options.compute_v {
		V = make_matrix(T, int(n), int(n))
		ldv = V.ld
	}

	// Allocate integer workspace
	iwork_size := m + 3 * n
	iwork := make([]i32, iwork_size)
	defer delete(iwork)

	// Query for optimal workspace
	lwork: Blas_Int = -1
	work_query: T

	when T == f32 {
		lapack.sgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query) + 1 // Add 1 for safety
		work := make([]T, lwork)
		defer delete(work)

		// Perform SVD
		lapack.sgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Extract statistics (first 7 elements of work array)
		work_stat = make([]f32, 7, allocator)
		copy(work_stat, work[:7])

	} else when T == complex64 {
		// Complex64 case
		work_query: complex64
		lrwork: Blas_Int = -1
		rwork_query: f32

		lapack.cgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Allocate workspaces
		lwork = Blas_Int(real(work_query)) + 1
		work := make([]complex64, lwork, context.temp_allocator)
		defer delete(work)

		lrwork = Blas_Int(rwork_query) + 1
		rwork := make([]f32, lrwork, context.temp_allocator)
		defer delete(rwork)

		// Perform SVD
		lapack.cgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Extract statistics from real work array
		work_stat = make([]f32, 7, allocator)
		copy(work_stat, rwork[:7])
	}

	return S, U, V, work_stat, info
}

// Combined f64 and complex128 Jacobi SVD
m_svd_jacobi_f64_c128 :: proc(
	A: ^Matrix($T),
	options: JacobiSVDMode = {compute_u = true, compute_v = true, high_accuracy = true},
	allocator := context.allocator,
) -> (
	S: []f64,
	U: Matrix(T),
	V: Matrix(T),
	work_stat: []f64,
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (not V^T)// Statistics about the computation
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Prepare job parameters
	joba_c := options.high_accuracy ? cstring("H") : cstring("C")
	jobu_c := options.compute_u ? cstring("U") : cstring("N")
	jobv_c := options.compute_v ? cstring("V") : cstring("N")
	jobr_c := options.restrict_range ? cstring("R") : cstring("N")
	jobt_c := options.transpose_hint ? cstring("T") : cstring("N")
	jobp_c := options.perturb ? cstring("P") : cstring("N")

	// Allocate singular values (always real for SVD)
	S = make([]f64, min(m, n), allocator)

	// Allocate singular vector matrices
	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if options.compute_u {
		U = make_matrix(T, m, m, .General, allocator)
		ldu = U.ld
	}
	if options.compute_v {
		V = make_matrix(T, n, n, .General, allocator)
		ldv = V.ld
	}

	// Allocate integer workspace
	iwork_size := m + 3 * n
	iwork := make([]Blas_Int, iwork_size, context.temp_allocator)
	defer delete(iwork)

	// Query for optimal workspace
	lwork: Blas_Int = -1

	when T == f64 {
		// Real f64 case
		work_query: f64
		lapack.dgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query) + 1
		work := make([]f64, lwork, context.temp_allocator)
		defer delete(work)

		// Perform SVD
		lapack.dgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Extract statistics
		work_stat = make([]f64, 7, allocator)
		copy(work_stat, work[:7])
	} else when T == complex128 {
		// Complex128 case
		work_query: complex128
		lrwork: Blas_Int = -1
		rwork_query: f64

		lapack.zgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Allocate workspaces
		lwork = Blas_Int(real(work_query)) + 1
		work := make([]complex128, lwork, context.temp_allocator)
		defer delete(work)

		lrwork = Blas_Int(rwork_query) + 1
		rwork := make([]f64, lrwork, context.temp_allocator)
		defer delete(rwork)

		// Perform SVD
		lapack.zgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			options.compute_u ? raw_data(U.data) : nil,
			&ldu,
			options.compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			raw_data(iwork),
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)

		// Extract statistics from real work array
		work_stat = make([]f64, 7, allocator)
		copy(work_stat, rwork[:7])
	}

	return S, U, V, work_stat, info
}

// ===================================================================================
// JACOBI SVD VARIANT (GESVJ)
// ===================================================================================

// Compute SVD using Jacobi method variant (gesvj)
// Computes the SVD directly with V instead of V^T
// Good for matrices with well-conditioned columns
// Combined f32 and complex64 Jacobi variant SVD
m_svd_jacobi_variant_f32_c64 :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
	upper_triangular: bool = false, // If A is already upper triangular
	allocator := context.allocator,
) -> (
	S: []f32,
	U: Matrix(T),
	V: Matrix(T),
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (not V^T)
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Job parameters
	joba_c := upper_triangular ? cstring("U") : cstring("G") // U=upper triangular, G=general
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")

	// Allocate singular values (always real)
	S = make([]f32, n, allocator)

	// Allocate singular vectors
	mv := n // Number of rows for V
	if compute_u && m < n {
		mv = m // If U is computed and m < n, V has m rows
	}

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if compute_u {
		U = make_matrix(T, m, n, .General, allocator)
		ldu = U.ld
	}
	if compute_v {
		V = make_matrix(T, mv, n, .General, allocator)
		ldv = V.ld
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1

	when T == f32 {
		// Real f32 case
		work_query: f32
		lapack.sgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := make([]f32, lwork, context.temp_allocator)
		defer delete(work)

		// Perform SVD
		lapack.sgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
		)
	} else when T == complex64 {
		// Complex64 case
		work_query: complex64
		lrwork := max(6, m + n)
		rwork := make([]f32, lrwork, context.temp_allocator)
		defer delete(rwork)

		lapack.cgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(real(work_query))
		work := make([]complex64, lwork, context.temp_allocator)
		defer delete(work)

		// Perform SVD
		lapack.cgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
		)
	}

	// Extract U from modified A if requested
	if compute_u {
		copy(U.data, A.data[:m * n])
	}

	return S, U, V, info
}

// Combined f64 and complex128 Jacobi variant SVD
m_svd_jacobi_variant_f64_c128 :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
	upper_triangular: bool = false, // If A is already upper triangular
	allocator := context.allocator,
) -> (
	S: []f64,
	U: Matrix(T),
	V: Matrix(T),
	info: Info, // Singular values (always real)// Left singular vectors// Right singular vectors (not V^T)
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Job parameters
	joba_c := upper_triangular ? cstring("U") : cstring("G")
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")

	// Allocate singular values (always real)
	S = make([]f64, n, allocator)

	// Allocate singular vectors
	mv := n
	if compute_u && m < n {
		mv = m
	}

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if compute_u {
		U = make_matrix(T, m, n, .General, allocator)
		ldu = U.ld
	}
	if compute_v {
		V = make_matrix(T, mv, n, .General, allocator)
		ldv = V.ld
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1

	when T == f64 {
		// Real f64 case
		work_query: f64

		lapack.dgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(work_query)
		work := make([]f64, lwork, context.temp_allocator)
		defer delete(work)

		// Perform SVD
		lapack.dgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
		)
	} else when T == complex128 {
		// Complex128 case
		work_query: complex128
		lrwork := max(6, m + n)
		rwork := make([]f64, lrwork, context.temp_allocator)
		defer delete(rwork)

		lapack.zgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			&work_query,
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
		)

		// Allocate workspace
		lwork = Blas_Int(real(work_query))
		work := make([]complex128, lwork, context.temp_allocator)
		defer delete(work)

		// Perform SVD
		lapack.zgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			&mv,
			compute_v ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&lrwork,
			&info,
			1,
			1,
			1,
		)
	}

	// Extract U from modified A if requested
	if compute_u {
		copy(U.data, A.data[:m * n])
	}

	return S, U, V, info
}
