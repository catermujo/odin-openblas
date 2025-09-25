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

svd :: proc {
	svd_f32_c64,
	svd_f64_c128,
}

svd_qr :: proc {
	svd_qr_f32_c64,
	svd_qr_f64_c128,
}

svd_select :: proc {
	svd_select_f32_c64,
	svd_select_f64_c128,
}

svd_divide :: proc {
	svd_divide_f32_c64,
	svd_divide_f64_c128,
}

cs_decomp :: proc {
	cs_decomp_f32_f64,
	cs_decomp_c64_c128,
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
svd_jacobi :: proc {
	svd_jacobi_f32_c64,
	svd_jacobi_f64_c128,
}

svd_jacobi_variant :: proc {
	svd_jacobi_variant_f32_c64,
	svd_jacobi_variant_f64_c128,
}

// ===================================================================================
// STANDARD SVD
// ===================================================================================

// Query result sizes for SVD
query_result_sizes_svd :: proc(
	m: int,
	n: int,
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	S_size: int,
	U_rows: int,
	U_cols: int,
	VT_rows: int,
	VT_cols: int,
) {
	min_mn := min(m, n)

	S_size = min_mn

	if compute_u {
		U_rows = m
		U_cols = full_matrices && m > min_mn ? m : min_mn
	}

	if compute_vt {
		VT_rows = full_matrices && n > min_mn ? n : min_mn
		VT_cols = n
	}

	return S_size, U_rows, U_cols, VT_rows, VT_cols
}

// Query workspace size for SVD (both real and complex)
// For real types, rwork_size will be 0
query_workspace_svd :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	work_size: int,
	rwork_size: int,
	info: Info,
) where is_float(T) ||
	is_complex(T) {
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

	// Query for optimal workspace
	lwork: Blas_Int = -1
	ldu: Blas_Int = 1
	ldvt: Blas_Int = 1

	dummy_s := [1]T{}

	when T == f32 {
		work_query: f32
		lapack.sgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil,
			&ldu,
			nil,
			&ldvt,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0 // Not used for real types
	} else when T == f64 {
		work_query: f64
		lapack.dgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil,
			&ldu,
			nil,
			&ldvt,
			&work_query,
			&lwork,
			&info,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0 // Not used for real types
	} else when T == complex64 {
		work_query: complex64
		dummy_rwork := [1]f32{}

		lapack.cgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil,
			&ldu,
			nil,
			&ldvt,
			&work_query,
			&lwork,
			&dummy_rwork[0],
			&info,
			1,
			1,
		)
		work_size = int(real(work_query))
		rwork_size = 5 * int(min_mn) // Complex types need real workspace
	} else when T == complex128 {
		work_query: complex128
		dummy_rwork := [1]f64{}

		lapack.zgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil,
			&ldu,
			nil,
			&ldvt,
			&work_query,
			&lwork,
			&dummy_rwork[0],
			&info,
			1,
			1,
		)
		work_size = int(real(work_query))
		rwork_size = 5 * int(min_mn) // Complex types need real workspace
	}

	return work_size, rwork_size, info
}
// Compute SVD using standard algorithm
// A = U * Sigma * V^T
// Combined f32 real and complex64 SVD (pre-allocated arrays)
svd_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f32, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	work: []T = nil, // Workspace (pre-allocated, optional)
	rwork: []f32 = nil, // Real workspace for complex (pre-allocated, optional)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify S array size
	assert(len(S) >= int(min_mn), "S array too small")

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

	// Set up U and VT pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions
		u_cols := full_matrices && m > min_mn ? int(m) : int(min_mn)
		assert(U.rows == m && U.cols >= Blas_Int(u_cols), "U matrix dimensions incorrect")
	}

	ldvt: Blas_Int = 1
	if compute_vt && VT != nil {
		ldvt = VT.ld
		// Verify VT dimensions
		vt_rows := full_matrices && n > min_mn ? int(n) : int(min_mn)
		assert(VT.rows >= Blas_Int(vt_rows) && VT.cols == n, "VT matrix dimensions incorrect")
	}

	// Verify workspace size
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd to get size)")
	lwork := Blas_Int(len(work))

	when T == f32 {
		// Compute SVD - real version doesn't need rwork
		lapack.sgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == complex64 {
		// Complex versions need real workspace
		assert(len(rwork) >= 5 * int(min_mn), "rwork array too small (need at least 5*min(m,n))")

		// Compute SVD
		lapack.cgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	ok := info == 0
	return info, ok
}

// Combined f64 real and complex128 SVD (pre-allocated arrays)
svd_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f64, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	work: []T = nil, // Workspace (pre-allocated, optional)
	rwork: []f64 = nil, // Real workspace for complex (pre-allocated, optional)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify S array size
	assert(len(S) >= int(min_mn), "S array too small")

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

	// Set up U and VT pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions
		u_cols := full_matrices && m > min_mn ? int(m) : int(min_mn)
		assert(U.rows == m && U.cols >= Blas_Int(u_cols), "U matrix dimensions incorrect")
	}

	ldvt: Blas_Int = 1
	if compute_vt && VT != nil {
		ldvt = VT.ld
		// Verify VT dimensions
		vt_rows := full_matrices && n > min_mn ? int(n) : int(min_mn)
		assert(VT.rows >= Blas_Int(vt_rows) && VT.cols == n, "VT matrix dimensions incorrect")
	}

	// Verify workspace size
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd to get size)")
	lwork := Blas_Int(len(work))

	when T == f64 {
		// Compute SVD - real version doesn't need rwork
		lapack.dgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
		)
	} else when T == complex128 {
		// Complex versions need real workspace
		assert(len(rwork) >= 5 * int(min_mn), "rwork array too small (need at least 5*min(m,n))")

		// Compute SVD
		lapack.zgesvd_(
			jobu_c,
			jobvt_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			&info,
			1,
			1,
		)
	}

	return info, info == 0
}


// ===================================================================================
// QR-BASED SVD WITH PIVOTING
// ===================================================================================

// Query result sizes for QR-based SVD
query_result_sizes_svd_qr :: proc(
	m: int,
	n: int,
	compute_u: bool = true,
	compute_v: bool = true,
) -> (
	S_size: int,
	U_rows: int,
	U_cols: int,
	V_rows: int,
	V_cols: int,
	iwork_size: int,
) {
	min_mn := min(m, n)

	S_size = min_mn

	if compute_u {
		U_rows = m
		U_cols = m
	}

	if compute_v {
		V_rows = n
		V_cols = n
	}

	// Integer workspace size
	iwork_size = 3 * min_mn

	return S_size, U_rows, U_cols, V_rows, V_cols, iwork_size
}

// Query workspace sizes for QR-based SVD
// For real types, rwork also used (unlike standard SVD)
query_workspace_svd_qr :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true,
	pivot: bool = true,
	rank_reveal: bool = true,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	info: Info,
) where T == f32 ||
	T == f64 ||
	T == complex64 ||
	T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("M")
	jobp_c := pivot ? cstring("P") : cstring("N")
	jobr_c := rank_reveal ? cstring("R") : cstring("N")
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")

	// Integer workspace is always needed
	iwork_size = 3 * int(min_mn)

	// Query for optimal workspace
	liwork: Blas_Int = -1
	lwork: Blas_Int = -1
	lrwork: Blas_Int = -1
	iwork_query: Blas_Int
	numrank: Blas_Int

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1

	when T == f32 {
		work_query: f32
		rwork_query: f32
		dummy_s := [1]f32{}

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
			&dummy_s[0],
			nil,
			&ldu,
			nil,
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

		work_size = int(work_query)
		rwork_size = int(rwork_query)

	} else when T == f64 {
		work_query: f64
		rwork_query: f64
		dummy_s := [1]f64{}

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
			&dummy_s[0],
			nil,
			&ldu,
			nil,
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

		work_size = int(work_query)
		rwork_size = int(rwork_query)

	} else when T == complex64 {
		cwork_query: complex64
		rwork_query: f32
		dummy_s := [1]f32{}

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
			&dummy_s[0],
			nil,
			&ldu,
			nil,
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

		work_size = int(real(cwork_query))
		rwork_size = int(rwork_query)

	} else when T == complex128 {
		zwork_query: complex128
		rwork_query: f64
		dummy_s := [1]f64{}

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
			&dummy_s[0],
			nil,
			&ldu,
			nil,
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

		work_size = int(real(zwork_query))
		rwork_size = int(rwork_query)
	}

	// Use the queried iwork size
	if iwork_query > 0 {
		iwork_size = int(iwork_query)
	}

	return work_size, rwork_size, iwork_size, info
}
// SVD using QR factorization with column pivoting
// High accuracy and rank-revealing, especially for rank-deficient matrices
// Combined f32 real and complex64 SVD with QR (pre-allocated arrays)
// Note: Caller should resize S, U.cols, V.cols based on numrank if rank_reveal is true
svd_qr_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f32, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T), // Right singular vectors NOT transposed (pre-allocated, optional)
	iwork: []Blas_Int = nil, // Integer workspace (pre-allocated, optional)
	work: []T = nil, // Workspace (pre-allocated, optional)
	rwork: []f32 = nil, // Real workspace (pre-allocated, optional - used for both real and complex)
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true, // Use high accuracy mode
	pivot: bool = true, // Use pivoting
	rank_reveal: bool = true, // Compute numerical rank
) -> (
	numrank: Blas_Int,
	info: Info,
	ok: bool, // Numerical rank
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify S array size
	assert(len(S) >= int(min_mn), "S array too small")

	// Set job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("M") // H=high, M=medium accuracy
	jobp_c := pivot ? cstring("P") : cstring("N") // P=pivot, N=no pivot
	jobr_c := rank_reveal ? cstring("R") : cstring("N") // R=rank revealing
	jobu_c := compute_u ? cstring("U") : cstring("N") // U=compute U
	jobv_c := compute_v ? cstring("V") : cstring("N") // V=compute V

	// Set up U and V pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions - full m x m matrix for QR-based SVD
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect (should be m x m)")
	}

	ldv: Blas_Int = 1
	if compute_v && V != nil {
		ldv = V.ld
		// Verify V dimensions - full n x n matrix for QR-based SVD
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect (should be n x n)")
	}

	// Verify workspace sizes - QR-based SVD needs all three workspace arrays even for real types
	assert(len(iwork) > 0, "iwork array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(rwork) > 0, "rwork array must be provided (use query_workspace_svd_qr to get size)")

	liwork := Blas_Int(len(iwork))
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))

	when T == f32 {
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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_v && V != nil ? raw_data(V.data) : nil,
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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_v && V != nil ? raw_data(V.data) : nil,
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
	}

	return numrank, info, info == 0
}

// Combined f64 real and complex128 SVD with QR (pre-allocated arrays)
// Note: Caller should resize S, U.cols, V.cols based on numrank if rank_reveal is true
svd_qr_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f64, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T), // Right singular vectors NOT transposed (pre-allocated, optional)
	iwork: []Blas_Int = nil, // Integer workspace (pre-allocated, optional)
	work: []T = nil, // Workspace (pre-allocated, optional)
	rwork: []f64 = nil, // Real workspace (pre-allocated, optional - used for both real and complex)
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true, // Use high accuracy mode
	pivot: bool = true, // Use pivoting
	rank_reveal: bool = true, // Compute numerical rank
) -> (
	numrank: Blas_Int,
	info: Info,
	ok: bool, // Numerical rank
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify S array size
	assert(len(S) >= int(min_mn), "S array too small")

	// Set job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("M") // H=high, M=medium accuracy
	jobp_c := pivot ? cstring("P") : cstring("N") // P=pivot, N=no pivot
	jobr_c := rank_reveal ? cstring("R") : cstring("N") // R=rank revealing
	jobu_c := compute_u ? cstring("U") : cstring("N") // U=compute U
	jobv_c := compute_v ? cstring("V") : cstring("N") // V=compute V

	// Set up U and V pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions - full m x m matrix for QR-based SVD
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect (should be m x m)")
	}

	ldv: Blas_Int = 1
	if compute_v && V != nil {
		ldv = V.ld
		// Verify V dimensions - full n x n matrix for QR-based SVD
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect (should be n x n)")
	}

	// Verify workspace sizes - QR-based SVD needs all three workspace arrays even for real types
	assert(len(iwork) > 0, "iwork array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(work) > 0, "work array must be provided (use query_workspace_svd_qr to get size)")
	assert(len(rwork) > 0, "rwork array must be provided (use query_workspace_svd_qr to get size)")

	liwork := Blas_Int(len(iwork))
	lwork := Blas_Int(len(work))
	lrwork := Blas_Int(len(rwork))

	when T == f64 {
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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_v && V != nil ? raw_data(V.data) : nil,
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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_v && V != nil ? raw_data(V.data) : nil,
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
	}

	return numrank, info, info == 0
}

// ===================================================================================
// SELECTIVE SVD
// ===================================================================================
// Query result sizes for selective SVD
// Returns sizes based on the selection criteria
query_result_sizes_svd_select :: proc(
	m: int,
	n: int,
	range_mode: SVDRangeOption = .ALL,
	il: Blas_Int = 1, // Lower index (for range=.INDEX)
	iu: Blas_Int = -1, // Upper index (for range=.INDEX, -1 = min(m,n))
	compute_u: bool = true,
	compute_vt: bool = true,
) -> (
	max_ns: int,
	S_size: int,
	U_rows: int,
	U_cols: int,
	VT_rows: int,
	VT_cols: int, // Maximum number of singular values that could be found
) {
	min_mn := Blas_Int(min(m, n))

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size
	switch range_mode {
	case .ALL:
		max_ns = int(min_mn)
	case .INDEX:
		max_ns = int(iu_val - il + 1)
	case .VALUE:
		max_ns = int(min_mn) // Conservative estimate for value range
	}

	S_size = max_ns

	if compute_u {
		U_rows = m
		U_cols = max_ns
	}

	if compute_vt {
		VT_rows = max_ns
		VT_cols = n
	}

	return
}

// Query workspace size for selective SVD
query_workspace_svd_select :: proc(
	A: ^Matrix($T),
	range_mode: SVDRangeOption = .ALL,
	vl: $F, // Lower bound (for range=.VALUE)
	vu: F, // Upper bound (for range=.VALUE)
	il: Blas_Int = 1, // Lower index (for range=.INDEX)
	iu: Blas_Int = -1, // Upper index (for range=.INDEX, -1 = min(m,n))
	compute_u: bool = true,
	compute_vt: bool = true,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	info: Info,
) where (T == f32 && F == f32) || (T == f64 && F == f64) || (T == complex64 && F == f32) || (T == complex128 && F == f64) {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set range parameter
	range_c := svd_range_to_cstring(range_mode)

	// Set job parameters
	jobu_c := compute_u ? cstring("V") : cstring("N")
	jobvt_c := compute_vt ? cstring("V") : cstring("N")

	// Adjust upper index if needed
	iu_val := iu
	if iu_val < 0 {
		iu_val = min_mn
	}

	// Determine maximum output size for rwork calculation
	max_ns: Blas_Int
	switch range_mode {
	case .ALL:
		max_ns = min_mn
	case .INDEX:
		max_ns = iu_val - il + 1
	case .VALUE:
		max_ns = min_mn // Conservative estimate for value range
	}

	// Integer workspace is always the same
	iwork_size = int(12 * min_mn)

	// Set up dummy output arrays for query
	ldu: Blas_Int = 1
	if compute_u {
		ldu = m
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		ldvt = max_ns
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	ns: Blas_Int
	dummy_iwork := [1]Blas_Int{}

	when T == f32 {
		work_query: f32
		dummy_s := [1]f32{}
		vl_f32 := f32(vl)
		vu_f32 := f32(vu)

		lapack.sgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f32,
			&vu_f32,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real version doesn't need rwork

	} else when T == f64 {
		work_query: f64
		dummy_s := [1]f64{}
		vl_f64 := f64(vl)
		vu_f64 := f64(vu)

		lapack.dgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f64,
			&vu_f64,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real version doesn't need rwork

	} else when T == complex64 {
		cwork_query: complex64
		dummy_s := [1]f32{}
		dummy_rwork := [1]f32{}
		vl_f32 := f32(vl)
		vu_f32 := f32(vu)

		lapack.cgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f32,
			&vu_f32,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&cwork_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
		)
		work_size = int(real(cwork_query))
		// Complex version needs real workspace
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * max_ns * (3 * max_ns + 1)))

	} else when T == complex128 {
		zwork_query: complex128
		dummy_s := [1]f64{}
		dummy_rwork := [1]f64{}
		vl_f64 := f64(vl)
		vu_f64 := f64(vu)

		lapack.zgesvdx_(
			jobu_c,
			jobvt_c,
			range_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&vl_f64,
			&vu_f64,
			&il,
			&iu_val,
			&ns,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&zwork_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
		)
		work_size = int(real(zwork_query))
		// Complex version needs real workspace
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * max_ns * (3 * max_ns + 1)))
	}

	return work_size, rwork_size, iwork_size, info
}
// Compute selected singular values and vectors
// Can compute subset by index range or value range
// Combined f32 real and complex64 selective SVD
svd_select_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f32, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex (pre-allocated, optional)
	range_mode: SVDRangeOption = .ALL,
	vl: f32 = 0, // Lower bound (for range=.VALUE)
	vu: f32 = 0, // Upper bound (for range=.VALUE)
	il: Blas_Int = 1, // Lower index (for range=.INDEX)
	iu: Blas_Int = -1, // Upper index (for range=.INDEX, -1 = min(m,n))
	compute_u: bool = true,
	compute_vt: bool = true,
) -> (
	ns: Blas_Int,
	info: Info,
	ok: bool, // Number of singular values found
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set range parameter
	range_c := svd_range_to_cstring(range_mode)

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
	switch range_mode {
	case .ALL:
		max_ns = min_mn
	case .INDEX:
		max_ns = iu_val - il + 1
	case .VALUE:
		max_ns = min_mn // Conservative estimate for value range
	}

	// Verify S array size
	assert(len(S) >= int(max_ns), "S array too small for potential output")

	// Set up U and VT pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions
		assert(U.rows == m && U.cols >= max_ns, "U matrix dimensions incorrect")
	}

	ldvt: Blas_Int = 1
	if compute_vt && VT != nil {
		ldvt = VT.ld
		// Verify VT dimensions
		assert(VT.rows >= max_ns && VT.cols == n, "VT matrix dimensions incorrect")
	}

	// Verify workspace sizes
	assert(len(iwork) >= int(12 * min_mn), "iwork array too small (need at least 12*min(m,n))")
	assert(
		len(work) > 0,
		"work array must be provided (use query_workspace_svd_select to get size)",
	)
	lwork := Blas_Int(len(work))

	when T == f32 {
		// Real version doesn't need rwork
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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
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
		// Complex version needs real workspace
		max_rwork := min_mn * max(5 * min_mn + 7, 2 * max_ns * (3 * max_ns + 1))
		assert(len(rwork) >= int(max_rwork), "rwork array too small for complex selective SVD")

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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	// Note: Caller should resize S, U.cols, VT.rows based on ns if needed

	ok = info == 0
	return ns, info, ok
}

// Combined f64 real and complex128 selective SVD
svd_select_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f64, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex (pre-allocated, optional)
	range_mode: SVDRangeOption = .ALL,
	vl: f64 = 0, // Lower bound (for range=.VALUE)
	vu: f64 = 0, // Upper bound (for range=.VALUE)
	il: Blas_Int = 1, // Lower index (for range=.INDEX)
	iu: Blas_Int = -1, // Upper index (for range=.INDEX, -1 = min(m,n))
	compute_u: bool = true,
	compute_vt: bool = true,
) -> (
	ns: Blas_Int,
	info: Info,
	ok: bool, // Number of singular values found
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Set range parameter
	range_c := svd_range_to_cstring(range_mode)

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
	switch range_mode {
	case .ALL:
		max_ns = min_mn
	case .INDEX:
		max_ns = iu_val - il + 1
	case .VALUE:
		max_ns = min_mn // Conservative estimate for value range
	}

	// Verify S array size
	assert(len(S) >= int(max_ns), "S array too small for potential output")

	// Set up U and VT pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions
		assert(U.rows == m && U.cols >= max_ns, "U matrix dimensions incorrect")
	}

	ldvt: Blas_Int = 1
	if compute_vt && VT != nil {
		ldvt = VT.ld
		// Verify VT dimensions
		assert(VT.rows >= max_ns && VT.cols == n, "VT matrix dimensions incorrect")
	}

	// Verify workspace sizes
	assert(len(iwork) >= int(12 * min_mn), "iwork array too small (need at least 12*min(m,n))")
	assert(
		len(work) > 0,
		"work array must be provided (use query_workspace_svd_select to get size)",
	)
	lwork := Blas_Int(len(work))

	when T == f64 {
		// Real version doesn't need rwork
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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
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
		// Complex version needs real workspace
		max_rwork := min_mn * max(5 * min_mn + 7, 2 * max_ns * (3 * max_ns + 1))
		assert(len(rwork) >= int(max_rwork), "rwork array too small for complex selective SVD")

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
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
			1,
			1,
		)
	}

	// Note: Caller should resize S, U.cols, VT.rows based on ns if needed

	ok = info == 0
	return ns, info, ok
}

// DIVIDE-AND-CONQUER SVD
// ===================================================================================

// Query result sizes for divide-and-conquer SVD
// Same as regular SVD result sizes
query_result_sizes_svd_divide :: proc(
	m: int,
	n: int,
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	S_size: int,
	U_rows: int,
	U_cols: int,
	VT_rows: int,
	VT_cols: int,
) {
	min_mn := min(m, n)

	S_size = min_mn

	if compute_u {
		U_rows = m
		U_cols = full_matrices ? m : min_mn
	}

	if compute_vt {
		VT_rows = full_matrices ? n : min_mn
		VT_cols = n
	}

	return
}

// Query workspace size for divide-and-conquer SVD
query_workspace_svd_divide :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	info: Info,
) where is_float(T) ||
	is_complex(T) {
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

	// Integer workspace is always the same
	iwork_size = int(8 * min_mn)

	// Set up dummy arrays for query
	ldu: Blas_Int = 1
	if compute_u {
		ldu = m
	}

	ldvt: Blas_Int = 1
	if compute_vt {
		ldvt = full_matrices ? n : min_mn
	}

	// Query for optimal workspace
	lwork: Blas_Int = -1
	dummy_iwork := [1]Blas_Int{}

	when T == f32 {
		work_query: f32
		dummy_s := [1]f32{}

		lapack.sgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real version doesn't need rwork

	} else when T == f64 {
		work_query: f64
		dummy_s := [1]f64{}

		lapack.dgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0 // Real version doesn't need rwork

	} else when T == complex64 {
		cwork_query: complex64
		dummy_s := [1]f32{}
		dummy_rwork := [1]f32{}

		lapack.cgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&cwork_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
			1,
		)
		work_size = int(real(cwork_query))
		// Complex version needs real workspace
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * min_mn + 1))
		if full_matrices {
			rwork_size = max(rwork_size, int(5 * min_mn * min_mn + 5 * min_mn))
		}

	} else when T == complex128 {
		zwork_query: complex128
		dummy_s := [1]f64{}
		dummy_rwork := [1]f64{}

		lapack.zgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			&dummy_s[0],
			nil, // U
			&ldu,
			nil, // VT
			&ldvt,
			&zwork_query,
			&lwork,
			&dummy_rwork[0],
			&dummy_iwork[0],
			&info,
			1,
		)
		work_size = int(real(zwork_query))
		// Complex version needs real workspace
		rwork_size = int(min_mn * max(5 * min_mn + 7, 2 * min_mn + 1))
		if full_matrices {
			rwork_size = max(rwork_size, int(5 * min_mn * min_mn + 5 * min_mn))
		}
	}

	return work_size, rwork_size, iwork_size, info
}

// Compute SVD using divide-and-conquer algorithm
// Faster than standard SVD for large matrices
// Combined f32 real and complex64 divide-and-conquer SVD
svd_divide_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f32, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex (pre-allocated, optional)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify S array size
	assert(len(S) >= int(min_mn), "S array too small")

	// Determine job parameter
	jobz_c: cstring
	if !compute_u && !compute_vt {
		jobz_c = "N" // Only singular values
	} else if full_matrices {
		jobz_c = "A" // All columns of U and V
	} else {
		jobz_c = "S" // Min(m,n) columns of U and V
	}

	// Set up U and VT pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions
		u_cols := full_matrices ? m : min_mn
		assert(U.rows == m && U.cols >= u_cols, "U matrix dimensions incorrect")
	}

	ldvt: Blas_Int = 1
	if compute_vt && VT != nil {
		ldvt = VT.ld
		// Verify VT dimensions
		vt_rows := full_matrices ? n : min_mn
		assert(VT.rows >= vt_rows && VT.cols == n, "VT matrix dimensions incorrect")
	}

	// Verify workspace sizes
	assert(len(iwork) >= int(8 * min_mn), "iwork array too small (need at least 8*min(m,n))")
	assert(
		len(work) > 0,
		"work array must be provided (use query_workspace_svd_divide to get size)",
	)
	lwork := Blas_Int(len(work))

	when T == f32 {
		// Real version doesn't need rwork
		// Compute SVD
		lapack.sgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
		)
	} else when T == complex64 {
		// Complex version needs real workspace
		rwork_size := min_mn * max(5 * min_mn + 7, 2 * min_mn + 1)
		if full_matrices {
			rwork_size = max(rwork_size, 5 * min_mn * min_mn + 5 * min_mn)
		}
		assert(
			len(rwork) >= int(rwork_size),
			"rwork array too small for complex divide-and-conquer SVD",
		)

		// Compute SVD
		lapack.cgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return info, info == 0
}

// Combined f64 real and complex128 divide-and-conquer SVD
svd_divide_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f64, // Singular values (pre-allocated)
	U: ^Matrix(T), // Left singular vectors (pre-allocated, optional)
	VT: ^Matrix(T), // Right singular vectors transposed (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex (pre-allocated, optional)
	compute_u: bool = true,
	compute_vt: bool = true,
	full_matrices: bool = false,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify S array size
	assert(len(S) >= int(min_mn), "S array too small")

	// Determine job parameter
	jobz_c: cstring
	if !compute_u && !compute_vt {
		jobz_c = "N" // Only singular values
	} else if full_matrices {
		jobz_c = "A" // All columns of U and V
	} else {
		jobz_c = "S" // Min(m,n) columns of U and V
	}

	// Set up U and VT pointers
	ldu: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		// Verify U dimensions
		u_cols := full_matrices ? m : min_mn
		assert(U.rows == m && U.cols >= u_cols, "U matrix dimensions incorrect")
	}

	ldvt: Blas_Int = 1
	if compute_vt && VT != nil {
		ldvt = VT.ld
		// Verify VT dimensions
		vt_rows := full_matrices ? n : min_mn
		assert(VT.rows >= vt_rows && VT.cols == n, "VT matrix dimensions incorrect")
	}

	// Verify workspace sizes
	assert(len(iwork) >= int(8 * min_mn), "iwork array too small (need at least 8*min(m,n))")
	assert(
		len(work) > 0,
		"work array must be provided (use query_workspace_svd_divide to get size)",
	)
	lwork := Blas_Int(len(work))

	when T == f64 {
		// Real version doesn't need rwork
		// Compute SVD
		lapack.dgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(iwork),
			&info,
			1,
		)
	} else when T == complex128 {
		// Complex version needs real workspace
		rwork_size := min_mn * max(5 * min_mn + 7, 2 * min_mn + 1)
		if full_matrices {
			rwork_size = max(rwork_size, 5 * min_mn * min_mn + 5 * min_mn)
		}
		assert(
			len(rwork) >= int(rwork_size),
			"rwork array too small for complex divide-and-conquer SVD",
		)

		// Compute SVD
		lapack.zgesdd_(
			jobz_c,
			&m,
			&n,
			raw_data(A.data),
			&lda,
			raw_data(S),
			compute_u && U != nil ? raw_data(U.data) : nil,
			&ldu,
			compute_vt && VT != nil ? raw_data(VT.data) : nil,
			&ldvt,
			raw_data(work),
			&lwork,
			raw_data(rwork),
			raw_data(iwork),
			&info,
			1,
		)
	}

	return info, info == 0
}

// BIDIAGONAL MATRIX OPERATIONS
// ===================================================================================

// Helper function to prepare CS decomposition parameters
cs_decomp_prepare :: proc(
	U1_shape: [2]Blas_Int,
	U2_shape: [2]Blas_Int,
	V1T_shape: [2]Blas_Int,
	V2T_shape: [2]Blas_Int,
	compute_u1: bool,
	compute_u2: bool,
	compute_v1t: bool,
	compute_v2t: bool,
	trans: bool,
) -> (
	m: Blas_Int,
	p: Blas_Int,
	q: Blas_Int,
	r: Blas_Int,
	ldu1: Blas_Int,
	ldu2: Blas_Int,
	ldv1t: Blas_Int,
	ldv2t: Blas_Int,
	jobu1_c: cstring,
	jobu2_c: cstring,
	jobv1t_c: cstring,
	jobv2t_c: cstring,
	trans_c: cstring,
) {
	// Extract dimensions
	if trans {
		m = U1_shape[1] + U2_shape[1]
		p = U1_shape[1]
		q = U2_shape[1]
	} else {
		m = U1_shape[0] + U2_shape[0]
		p = U1_shape[0]
		q = U2_shape[0]
	}

	r = min(min(p, q), min(m - q, m - p))

	// Set leading dimensions
	ldu1 = U1_shape[1] if U1_shape[1] > 0 else 1
	ldu2 = U2_shape[1] if U2_shape[1] > 0 else 1
	ldv1t = V1T_shape[1] if V1T_shape[1] > 0 else 1
	ldv2t = V2T_shape[1] if V2T_shape[1] > 0 else 1

	// Set job parameters
	jobu1_c = compute_u1 ? cstring("Y") : cstring("N")
	jobu2_c = compute_u2 ? cstring("Y") : cstring("N")
	jobv1t_c = compute_v1t ? cstring("Y") : cstring("N")
	jobv2t_c = compute_v2t ? cstring("Y") : cstring("N")
	trans_c = trans ? cstring("T") : cstring("N")

	return
}

// Helper to allocate bidiagonal arrays for real types
make_bidiag_real_arrays :: proc(
	$T: typeid,
	r: Blas_Int,
) -> (
	b11d: []T,
	b11e: []T,
	b12d: []T,
	b12e: []T,
	b21d: []T,
	b21e: []T,
	b22d: []T,
	b22e: []T,
) {
	b11d = make([]T, r)
	b11e = make([]T, r - 1)
	b12d = make([]T, r)
	b12e = make([]T, r - 1)
	b21d = make([]T, r)
	b21e = make([]T, r - 1)
	b22d = make([]T, r)
	b22e = make([]T, r - 1)
	return
}

// Helper to delete bidiagonal arrays
delete_bidiag_real_arrays :: proc(
	b11d: $T,
	b11e: T,
	b12d: T,
	b12e: T,
	b21d: T,
	b21e: T,
	b22d: T,
	b22e: T,
) {
	delete(b11d)
	delete(b11e)
	delete(b12d)
	delete(b12e)
	delete(b21d)
	delete(b21e)
	delete(b22d)
	delete(b22e)
}

// Query result sizes for CS decomposition
query_result_sizes_cs_decomp :: proc(
	p: int, // Rows of U1
	q: int, // Rows of U2
) -> (
	theta_size: int,
	phi_size: int,
	r: int, // min(p, q)
) {
	r = min(p, q)
	theta_size = r
	phi_size = max(0, r - 1)
	return
}

// Query workspace size for CS decomposition
query_workspace_cs_decomp :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	trans: bool = false,
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
) -> (
	work_size: int,
	rwork_size: int,
	info: Info,
) where T == f32 ||
	T == f64 ||
	T == complex64 ||
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

	// Create dummy arrays - use f64 for all, transmute for f32
	dummy_theta := [1]f64{}
	dummy_phi := [1]f64{}
	dummy_b11d := [1]f64{}
	dummy_b11e := [1]f64{}
	dummy_b12d := [1]f64{}
	dummy_b12e := [1]f64{}
	dummy_b21d := [1]f64{}
	dummy_b21e := [1]f64{}
	dummy_b22d := [1]f64{}
	dummy_b22e := [1]f64{}
	dummy_rwork := [1]f64{}

	lwork: Blas_Int = -1

	when T == f32 {
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
			cast(^f32)&dummy_theta[0],
			cast(^f32)&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			cast(^f32)&dummy_b11d[0],
			cast(^f32)&dummy_b11e[0],
			cast(^f32)&dummy_b12d[0],
			cast(^f32)&dummy_b12e[0],
			cast(^f32)&dummy_b21d[0],
			cast(^f32)&dummy_b21e[0],
			cast(^f32)&dummy_b22d[0],
			cast(^f32)&dummy_b22e[0],
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == f64 {
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
			&dummy_theta[0],
			&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			&dummy_b11d[0],
			&dummy_b11e[0],
			&dummy_b12d[0],
			&dummy_b12e[0],
			&dummy_b21d[0],
			&dummy_b21e[0],
			&dummy_b22d[0],
			&dummy_b22e[0],
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == complex64 {
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
			cast(^f32)&dummy_theta[0],
			cast(^f32)&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			cast(^f32)&dummy_b11d[0],
			cast(^f32)&dummy_b11e[0],
			cast(^f32)&dummy_b12d[0],
			cast(^f32)&dummy_b12e[0],
			cast(^f32)&dummy_b21d[0],
			cast(^f32)&dummy_b21e[0],
			cast(^f32)&dummy_b22d[0],
			cast(^f32)&dummy_b22e[0],
			cast(^f32)&dummy_rwork[0],
			(trans ? 8 * r : 7 * r),
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(real(work_query))
		rwork_size = int(trans ? 8 * r : 7 * r)

	} else when T == complex128 {
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
			&dummy_theta[0],
			&dummy_phi[0],
			raw_data(U1.data),
			&ldu1,
			raw_data(U2.data),
			&ldu2,
			raw_data(V1T.data),
			&ldv1t,
			raw_data(V2T.data),
			&ldv2t,
			&dummy_b11d[0],
			&dummy_b11e[0],
			&dummy_b12d[0],
			&dummy_b12e[0],
			&dummy_b21d[0],
			&dummy_b21e[0],
			&dummy_b22d[0],
			&dummy_b22e[0],
			&dummy_rwork[0],
			(trans ? 8 * r : 7 * r),
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(real(work_query))
		rwork_size = int(trans ? 8 * r : 7 * r)
	}

	return work_size, rwork_size, info
}

// CS Decomposition of unitary/orthogonal matrix partitioned into 2x2 blocks
// Computes the CS decomposition of an orthogonal/unitary matrix in bidiagonal-block form
// Combined f32 and f64 CS decomposition
// Note: Bidiagonal arrays (b11d, b11e, etc.) are workspace arrays - the CS decomposition
// transforms the input matrices in-place
cs_decomp_f32_f64 :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	theta: []T, // Cosines (pre-allocated, size r)
	phi: []T, // Sines (pre-allocated, size r-1)
	work: []T, // Workspace (pre-allocated)
	// Bidiagonal workspace arrays (pre-allocated)
	b11d: []T, // Size r
	b11e: []T, // Size r-1
	b12d: []T, // Size r
	b12e: []T, // Size r-1
	b21d: []T, // Size r
	b21e: []T, // Size r-1
	b22d: []T, // Size r
	b22e: []T, // Size r-1
	trans: bool = false,
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
) -> (
	info: Info,
	ok: bool,
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

	// Verify array sizes
	assert(len(theta) >= int(r), "theta array too small (need at least r)")
	assert(len(phi) >= int(max(0, r - 1)), "phi array too small (need at least r-1)")
	assert(
		len(work) > 0,
		"work array must be provided (use query_workspace_cs_decomp to get size)",
	)

	// Verify bidiagonal workspace arrays
	assert(len(b11d) >= int(r), "b11d array too small (need at least r)")
	assert(len(b11e) >= int(max(0, r - 1)), "b11e array too small (need at least r-1)")
	assert(len(b12d) >= int(r), "b12d array too small (need at least r)")
	assert(len(b12e) >= int(max(0, r - 1)), "b12e array too small (need at least r-1)")
	assert(len(b21d) >= int(r), "b21d array too small (need at least r)")
	assert(len(b21e) >= int(max(0, r - 1)), "b21e array too small (need at least r-1)")
	assert(len(b22d) >= int(r), "b22d array too small (need at least r)")
	assert(len(b22e) >= int(max(0, r - 1)), "b22e array too small (need at least r-1)")

	lwork := Blas_Int(len(work))

	when T == f32 {
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

	return info, info == 0
}

// Combined complex64 and complex128 CS decomposition
// Note: For complex types, theta and phi are always f32 for complex64 and f64 for complex128
cs_decomp_c64_c128 :: proc(
	U1: ^Matrix($T),
	U2: ^Matrix(T),
	V1T: ^Matrix(T),
	V2T: ^Matrix(T),
	theta: []$F, // Cosines (pre-allocated, size r) - f32 for complex64, f64 for complex128
	phi: []F, // Sines (pre-allocated, size r-1) - f32 for complex64, f64 for complex128
	work: []T, // Complex workspace (pre-allocated)
	rwork: []F, // Real workspace (pre-allocated)
	// Bidiagonal workspace arrays (pre-allocated) - real type matches theta/phi
	b11d: []F, // Size r
	b11e: []F, // Size r-1
	b12d: []F, // Size r
	b12e: []F, // Size r-1
	b21d: []F, // Size r
	b21e: []F, // Size r-1
	b22d: []F, // Size r
	b22e: []F, // Size r-1
	trans: bool = false,
	compute_u1: bool = true,
	compute_u2: bool = true,
	compute_v1t: bool = true,
	compute_v2t: bool = true,
) -> (
	info: Info,
	ok: bool,
) where (T == complex64 && F == f32) || (T == complex128 && F == f64) {
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

	// Verify array sizes
	assert(len(theta) >= int(r), "theta array too small (need at least r)")
	assert(len(phi) >= int(max(0, r - 1)), "phi array too small (need at least r-1)")
	assert(
		len(work) > 0,
		"work array must be provided (use query_workspace_cs_decomp to get size)",
	)

	// Verify real workspace
	lrwork := Blas_Int((8 * r) if trans else (7 * r))
	assert(len(rwork) >= int(lrwork), "rwork array too small")

	// Verify bidiagonal workspace arrays
	assert(len(b11d) >= int(r), "b11d array too small (need at least r)")
	assert(len(b11e) >= int(max(0, r - 1)), "b11e array too small (need at least r-1)")
	assert(len(b12d) >= int(r), "b12d array too small (need at least r)")
	assert(len(b12e) >= int(max(0, r - 1)), "b12e array too small (need at least r-1)")
	assert(len(b21d) >= int(r), "b21d array too small (need at least r)")
	assert(len(b21e) >= int(max(0, r - 1)), "b21e array too small (need at least r-1)")
	assert(len(b22d) >= int(r), "b22d array too small (need at least r)")
	assert(len(b22e) >= int(max(0, r - 1)), "b22e array too small (need at least r-1)")

	lwork := Blas_Int(len(work))

	when T == complex64 {
		// For complex64, theta and phi are f32 arrays
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
	} else when T == complex128 {
		// For complex128, theta and phi are f64 arrays

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
	}

	return info, info == 0
}

// ===================================================================================
// JACOBI SVD (HIGH ACCURACY) - WORKSPACE QUERIES
// ===================================================================================

// Query workspace size for Jacobi SVD (high accuracy)
query_workspace_svd_jacobi :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true,
	restrict_range: bool = false,
	transpose_hint: bool = false,
	perturb: bool = false,
) -> (
	work_size: int,
	rwork_size: int,
	iwork_size: int,
	info: Info, // For complex types
) where is_float(T) || is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Prepare job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("C")
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobr_c := restrict_range ? cstring("R") : cstring("N")
	jobt_c := transpose_hint ? cstring("T") : cstring("N")
	jobp_c := perturb ? cstring("P") : cstring("N")

	// Integer workspace size
	iwork_size = int(m + 3 * n)

	// Dummy arrays for queries
	dummy_s := [1]f64{}
	dummy_iwork := [1]Blas_Int{}

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	lwork: Blas_Int = -1

	when T == f32 {
		work_query: f32

		lapack.sgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			nil, // A data
			&lda,
			cast(^f32)&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == f64 {
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
			nil, // A data
			&lda,
			&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == complex64 {
		work_query: complex64
		rwork_query: f32
		lrwork: Blas_Int = -1

		lapack.cgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			nil, // A data
			&lda,
			cast(^f32)&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)

	} else when T == complex128 {
		work_query: complex128
		rwork_query: f64
		lrwork: Blas_Int = -1

		lapack.zgejsv_(
			joba_c,
			jobu_c,
			jobv_c,
			jobr_c,
			jobt_c,
			jobp_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			nil, // U data
			&ldu,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&rwork_query,
			&lrwork,
			&dummy_iwork[0],
			&info,
			1,
			1,
			1,
			1,
			1,
			1,
		)
		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
	}

	return work_size, rwork_size, iwork_size, info
}

// Query result sizes for Jacobi SVD
query_result_sizes_svd_jacobi :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
) -> (
	s_size: int,
	u_rows: int,
	u_cols: int,
	v_rows: int,
	v_cols: int,
	work_stat_size: int,
) where is_float(T) ||
	is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)
	min_mn := min(m, n)

	s_size = min_mn
	work_stat_size = 7 // Statistics array size

	if compute_u {
		u_rows = m
		u_cols = m
	}

	if compute_v {
		v_rows = n
		v_cols = n
	}

	return s_size, u_rows, u_cols, v_rows, v_cols, work_stat_size
}

// ===================================================================================
// JACOBI SVD (HIGH ACCURACY)
// ===================================================================================

// Combined f32 and complex64 Jacobi SVD (pre-allocated arrays)
svd_jacobi_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f32, // Singular values (pre-allocated)
	U: ^Matrix(T) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T) = nil, // Right singular vectors NOT transposed (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work_stat: []f32, // Statistics array (pre-allocated, size 7)
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true,
	restrict_range: bool = false,
	transpose_hint: bool = false,
	perturb: bool = false,
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify array sizes
	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(iwork) >= int(m + 3 * n), "iwork array too small")
	assert(len(work_stat) >= 7, "work_stat array too small (need at least 7)")

	// Prepare job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("C") // H=high accuracy, C=controlled accuracy
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobr_c := restrict_range ? cstring("R") : cstring("N")
	jobt_c := transpose_hint ? cstring("T") : cstring("N")
	jobp_c := perturb ? cstring("P") : cstring("N")

	// Set up matrix dimensions
	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect")
	}
	if compute_v && V != nil {
		ldv = V.ld
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect")
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
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
			(compute_u && U != nil) ? raw_data(U.data) : nil,
			&ldu,
			(compute_v && V != nil) ? raw_data(V.data) : nil,
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

		// Extract statistics from work array
		copy(work_stat, work[:7])

	} else when T == complex64 {
		// Complex version requires rwork
		assert(len(rwork) > 0, "rwork array must be provided for complex types")
		lrwork := Blas_Int(len(rwork))

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
			(compute_u && U != nil) ? raw_data(U.data) : nil,
			&ldu,
			(compute_v && V != nil) ? raw_data(V.data) : nil,
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

		// Extract statistics from rwork array
		copy(work_stat, rwork[:7])
	}

	return info, info == 0
}

// Combined f64 and complex128 Jacobi SVD (pre-allocated arrays)
svd_jacobi_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f64, // Singular values (pre-allocated)
	U: ^Matrix(T) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T) = nil, // Right singular vectors NOT transposed (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex (pre-allocated, optional)
	iwork: []Blas_Int, // Integer workspace (pre-allocated)
	work_stat: []f64, // Statistics array (pre-allocated, size 7)
	compute_u: bool = true,
	compute_v: bool = true,
	high_accuracy: bool = true,
	restrict_range: bool = false,
	transpose_hint: bool = false,
	perturb: bool = false,
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld
	min_mn := min(m, n)

	// Verify array sizes
	assert(len(S) >= int(min_mn), "S array too small")
	assert(len(work) > 0, "work array must be provided")
	assert(len(iwork) >= int(m + 3 * n), "iwork array too small")
	assert(len(work_stat) >= 7, "work_stat array too small (need at least 7)")

	// Prepare job parameters
	joba_c := high_accuracy ? cstring("H") : cstring("C") // H=high accuracy, C=controlled accuracy
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")
	jobr_c := restrict_range ? cstring("R") : cstring("N")
	jobt_c := transpose_hint ? cstring("T") : cstring("N")
	jobp_c := perturb ? cstring("P") : cstring("N")

	// Set up matrix dimensions
	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == m, "U matrix dimensions incorrect")
	}
	if compute_v && V != nil {
		ldv = V.ld
		assert(V.rows == n && V.cols == n, "V matrix dimensions incorrect")
	}

	lwork := Blas_Int(len(work))

	when T == f64 {
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
			(compute_u && U != nil) ? raw_data(U.data) : nil,
			&ldu,
			(compute_v && V != nil) ? raw_data(V.data) : nil,
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
		copy(work_stat, work[:7])
	} else when T == complex128 {
		// Complex version requires rwork
		assert(len(rwork) > 0, "rwork array must be provided for complex types")
		lrwork := Blas_Int(len(rwork))

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
			(compute_u && U != nil) ? raw_data(U.data) : nil,
			&ldu,
			(compute_v && V != nil) ? raw_data(V.data) : nil,
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

		// Extract statistics from rwork array
		copy(work_stat, rwork[:7])
	}

	return info, info == 0
}

// ===================================================================================
// JACOBI SVD VARIANT (GESVJ) - WORKSPACE QUERIES
// ===================================================================================

// Query workspace size for Jacobi SVD variant (gesvj)
// rwork_size: For complex types only
query_workspace_svd_jacobi_variant :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
	upper_triangular: bool = false,
) -> (
	work_size: int,
	rwork_size: int,
	info: Info,
) where is_float(T) ||
	is_complex(T) {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Job parameters
	joba_c := upper_triangular ? cstring("U") : cstring("G") // U=upper triangular, G=general
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")

	// Determine V dimensions
	mv := n
	if compute_u && m < n {
		mv = m
	}

	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	lwork: Blas_Int = -1

	// Dummy arrays for queries
	dummy_s := [1]f64{}

	when T == f32 {
		work_query: f32

		lapack.sgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			cast(^f32)&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == f64 {
		work_query: f64

		lapack.dgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			&info,
			1,
			1,
			1,
		)
		work_size = int(work_query)
		rwork_size = 0

	} else when T == complex64 {
		work_query: complex64
		rwork_size = max(6, int(m + n)) // Real workspace for complex variant

		lapack.cgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			cast(^f32)&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			nil, // rwork dummy
			&info,
			1,
			1,
			1,
		)
		work_size = int(real(work_query))

	} else when T == complex128 {
		work_query: complex128
		rwork_size = max(6, int(m + n)) // Real workspace for complex variant

		lapack.zgesvj_(
			joba_c,
			jobu_c,
			jobv_c,
			&m,
			&n,
			nil, // A data
			&lda,
			&dummy_s[0],
			&mv,
			nil, // V data
			&ldv,
			&work_query,
			&lwork,
			nil, // rwork dummy
			&info,
			1,
			1,
			1,
		)
		work_size = int(real(work_query))
	}

	return work_size, rwork_size, info
}

// Query result sizes for Jacobi SVD variant
query_result_sizes_svd_jacobi_variant :: proc(
	A: ^Matrix($T),
	compute_u: bool = true,
	compute_v: bool = true,
) -> (
	s_size: int,
	u_rows: int,
	u_cols: int,
	v_rows: int,
	v_cols: int,
) where is_float(T) ||
	is_complex(T) {
	m := int(A.rows)
	n := int(A.cols)

	s_size = n

	if compute_u {
		u_rows = m
		u_cols = n
	}

	if compute_v {
		v_rows = n
		if compute_u && m < n {
			v_rows = m
		}
		v_cols = n
	}

	return s_size, u_rows, u_cols, v_rows, v_cols
}

// ===================================================================================
// JACOBI SVD VARIANT (GESVJ)
// ===================================================================================

// Compute SVD using Jacobi method variant (gesvj)
// Computes the SVD directly with V instead of V^T
// Good for matrices with well-conditioned columns
// Combined f32 and complex64 Jacobi variant SVD (pre-allocated arrays)
svd_jacobi_variant_f32_c64 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f32, // Singular values (pre-allocated)
	U: ^Matrix(T) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T) = nil, // Right singular vectors (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated)
	rwork: []f32 = nil, // Real workspace for complex (pre-allocated, optional)
	compute_u: bool = true,
	compute_v: bool = true,
	upper_triangular: bool = false, // If A is already upper triangular
) -> (
	info: Info,
	ok: bool,
) where T == f32 || T == complex64 {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Verify array sizes
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) > 0, "work array must be provided")

	// Job parameters
	joba_c := upper_triangular ? cstring("U") : cstring("G") // U=upper triangular, G=general
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")

	// Determine V dimensions
	mv := n // Number of rows for V
	if compute_u && m < n {
		mv = m // If U is computed and m < n, V has m rows
	}

	// Set up matrix dimensions
	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == n, "U matrix dimensions incorrect")
	}
	if compute_v && V != nil {
		ldv = V.ld
		assert(V.rows == mv && V.cols == n, "V matrix dimensions incorrect")
	}

	lwork := Blas_Int(len(work))

	when T == f32 {
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
			(compute_v && V != nil) ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
		)
	} else when T == complex64 {
		// Complex version requires rwork
		assert(len(rwork) > 0, "rwork array must be provided for complex types")
		lrwork := Blas_Int(len(rwork))

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
			(compute_v && V != nil) ? raw_data(V.data) : nil,
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
	if compute_u && U != nil {
		copy(U.data, A.data[:m * n])
	}

	return info, info == 0
}

// Combined f64 and complex128 Jacobi variant SVD (pre-allocated arrays)
svd_jacobi_variant_f64_c128 :: proc(
	A: ^Matrix($T), // Input matrix (overwritten)
	S: []f64, // Singular values (pre-allocated)
	U: ^Matrix(T) = nil, // Left singular vectors (pre-allocated, optional)
	V: ^Matrix(T) = nil, // Right singular vectors (pre-allocated, optional)
	work: []T, // Workspace (pre-allocated)
	rwork: []f64 = nil, // Real workspace for complex (pre-allocated, optional)
	compute_u: bool = true,
	compute_v: bool = true,
	upper_triangular: bool = false, // If A is already upper triangular
) -> (
	info: Info,
	ok: bool,
) where T == f64 || T == complex128 {
	m := A.rows
	n := A.cols
	lda := A.ld

	// Verify array sizes
	assert(len(S) >= int(n), "S array too small")
	assert(len(work) > 0, "work array must be provided")

	// Job parameters
	joba_c := upper_triangular ? cstring("U") : cstring("G")
	jobu_c := compute_u ? cstring("U") : cstring("N")
	jobv_c := compute_v ? cstring("V") : cstring("N")

	// Determine V dimensions
	mv := n
	if compute_u && m < n {
		mv = m
	}

	// Set up matrix dimensions
	ldu: Blas_Int = 1
	ldv: Blas_Int = 1
	if compute_u && U != nil {
		ldu = U.ld
		assert(U.rows == m && U.cols == n, "U matrix dimensions incorrect")
	}
	if compute_v && V != nil {
		ldv = V.ld
		assert(V.rows == mv && V.cols == n, "V matrix dimensions incorrect")
	}

	lwork := Blas_Int(len(work))

	when T == f64 {
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
			(compute_v && V != nil) ? raw_data(V.data) : nil,
			&ldv,
			raw_data(work),
			&lwork,
			&info,
			1,
			1,
			1,
		)
	} else when T == complex128 {
		// Complex version requires rwork
		assert(len(rwork) > 0, "rwork array must be provided for complex types")
		lrwork := Blas_Int(len(rwork))

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
			(compute_v && V != nil) ? raw_data(V.data) : nil,
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
	if compute_u && U != nil {
		copy(U.data, A.data[:m * n])
	}

	return info, info == 0
}
