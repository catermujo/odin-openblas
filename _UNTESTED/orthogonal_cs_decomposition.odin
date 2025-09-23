package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// LAPACK VERSION, ORTHOGONAL TRANSFORMATIONS, AND CS DECOMPOSITION
// ===================================================================================

// Packed orthogonal matrix generation proc group
m_generate_packed_orthogonal :: proc {
	m_generate_packed_orthogonal_f64,
	m_generate_packed_orthogonal_f32,
}

// Packed orthogonal matrix multiplication proc group
m_multiply_packed_orthogonal :: proc {
	m_multiply_packed_orthogonal_f64,
	m_multiply_packed_orthogonal_f32,
}

// Bidiagonal CS decomposition proc group
m_bidiagonal_cs_decomposition :: proc {
	m_bidiagonal_cs_decomposition_f64,
	m_bidiagonal_cs_decomposition_f32,
}

// CS decomposition proc group
m_cs_decomposition :: proc {
	m_cs_decomposition_f64,
	m_cs_decomposition_f32,
}

// CS decomposition 2x1 proc group
m_cs_decomposition_2x1 :: proc {
	m_cs_decomposition_2x1_f64,
	m_cs_decomposition_2x1_f32,
}

// ===================================================================================
// LAPACK VERSION INFORMATION
// ===================================================================================

// LAPACK version information
LapackVersion :: struct {
	major: int,
	minor: int,
	patch: int,
}

// Get LAPACK version
get_lapack_version :: proc() -> LapackVersion {
	major, minor, patch: Blas_Int
	lapack.ilaver_(&major, &minor, &patch)
	return LapackVersion{major = int(major), minor = int(minor), patch = int(patch)}
}

// ===================================================================================
// PACKED ORTHOGONAL MATRIX OPERATIONS
// ===================================================================================

// Generate orthogonal matrix from packed storage (f64)
// Generates Q from packed representation computed by DSPTRD
m_generate_packed_orthogonal_f64 :: proc(
	AP: []f64, // Packed symmetric matrix (from DSPTRD)
	tau: []f64, // Scalar factors (from DSPTRD)
	Q: ^Matrix(f64), // Output orthogonal matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	n := Q.rows
	if Q.rows != Q.cols {
		panic("Q must be square")
	}
	if len(tau) != n - 1 {
		panic("tau array must have length n-1")
	}
	expected_ap_size := n * (n + 1) / 2
	if len(AP) != expected_ap_size {
		panic("AP array size must be n*(n+1)/2")
	}

	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	ldq := Blas_Int(Q.ld)

	// Allocate workspace
	work := make([]f64, n - 1, context.temp_allocator)

	info_val: Info
	lapack.dopgtr_(
		uplo_c,
		&n_val,
		raw_data(AP),
		raw_data(tau),
		raw_data(Q.data),
		&ldq,
		raw_data(work),
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Generate orthogonal matrix from packed storage (f32)
// Generates Q from packed representation computed by SSPTRD
m_generate_packed_orthogonal_f32 :: proc(
	AP: []f32, // Packed symmetric matrix (from SSPTRD)
	tau: []f32, // Scalar factors (from SSPTRD)
	Q: ^Matrix(f32), // Output orthogonal matrix
	uplo_upper := true, // Upper or lower triangular storage
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	n := Q.rows
	if Q.rows != Q.cols {
		panic("Q must be square")
	}
	if len(tau) != n - 1 {
		panic("tau array must have length n-1")
	}
	expected_ap_size := n * (n + 1) / 2
	if len(AP) != expected_ap_size {
		panic("AP array size must be n*(n+1)/2")
	}

	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	ldq := Blas_Int(Q.ld)

	// Allocate workspace
	work := make([]f32, n - 1, context.temp_allocator)

	info_val: Info
	lapack.sopgtr_(
		uplo_c,
		&n_val,
		raw_data(AP),
		raw_data(tau),
		raw_data(Q.data),
		&ldq,
		raw_data(work),
		&info_val,
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Multiply matrix by packed orthogonal matrix (f64)
// Multiplies C by Q or Q^T where Q is stored in packed format
m_multiply_packed_orthogonal_f64 :: proc(
	AP: []f64, // Packed orthogonal matrix representation
	tau: []f64, // Scalar factors
	C: ^Matrix(f64), // Matrix to multiply
	left_multiply := true, // Apply Q from left (true) or right (false)
	uplo_upper := true, // Upper or lower triangular storage
	transpose := false, // Apply Q^T instead of Q
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	m := C.rows
	n := C.cols
	matrix_size := left_multiply ? m : n

	expected_ap_size := matrix_size * (matrix_size + 1) / 2
	if len(AP) != expected_ap_size {
		panic("AP array size inconsistent with matrix dimensions")
	}
	if len(tau) != matrix_size - 1 {
		panic("tau array must have appropriate length")
	}

	side_c := "L" if left_multiply else "R"
	uplo_c := "U" if uplo_upper else "L"
	trans_c := "T" if transpose else "N"
	m_val := Blas_Int(m)
	n_val := Blas_Int(n)
	ldc := Blas_Int(C.ld)

	// Allocate workspace
	work_size := left_multiply ? n : m
	work := make([]f64, work_size, context.temp_allocator)

	info_val: Info
	lapack.dopmtr_(
		side_c,
		uplo_c,
		trans_c,
		&m_val,
		&n_val,
		raw_data(AP),
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&info_val,
		len(side_c),
		len(uplo_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// Multiply matrix by packed orthogonal matrix (f32)
// Multiplies C by Q or Q^T where Q is stored in packed format
m_multiply_packed_orthogonal_f32 :: proc(
	AP: []f32, // Packed orthogonal matrix representation
	tau: []f32, // Scalar factors
	C: ^Matrix(f32), // Matrix to multiply
	left_multiply := true, // Apply Q from left (true) or right (false)
	uplo_upper := true, // Upper or lower triangular storage
	transpose := false, // Apply Q^T instead of Q
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	m := C.rows
	n := C.cols
	matrix_size := left_multiply ? m : n

	expected_ap_size := matrix_size * (matrix_size + 1) / 2
	if len(AP) != expected_ap_size {
		panic("AP array size inconsistent with matrix dimensions")
	}
	if len(tau) != matrix_size - 1 {
		panic("tau array must have appropriate length")
	}

	side_c := "L" if left_multiply else "R"
	uplo_c := "U" if uplo_upper else "L"
	trans_c := "T" if transpose else "N"
	m_val := Blas_Int(m)
	n_val := Blas_Int(n)
	ldc := Blas_Int(C.ld)

	// Allocate workspace
	work_size := left_multiply ? n : m
	work := make([]f32, work_size, context.temp_allocator)

	info_val: Info
	lapack.sopmtr_(
		side_c,
		uplo_c,
		trans_c,
		&m_val,
		&n_val,
		raw_data(AP),
		raw_data(tau),
		raw_data(C.data),
		&ldc,
		raw_data(work),
		&info_val,
		len(side_c),
		len(uplo_c),
		len(trans_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// CS DECOMPOSITION UTILITIES
// ===================================================================================

// Block matrix structure for CS decomposition
CSDecompositionBlocks :: struct($T: typeid) {
	X11: ^Matrix(T), // (p x q) upper-left block
	X12: ^Matrix(T), // (p x m-q) upper-right block
	X21: ^Matrix(T), // (m-p x q) lower-left block
	X22: ^Matrix(T), // (m-p x m-q) lower-right block
}

// CS decomposition result structure
CSDecompositionResult :: struct($T: typeid) {
	theta: []T, // Cosine-sine angles
	phi:   []T, // Additional angles (for ORBDB)
	U1:    ^Matrix(T), // Left orthogonal matrix 1
	U2:    ^Matrix(T), // Left orthogonal matrix 2
	V1T:   ^Matrix(T), // Right orthogonal matrix 1 (transposed)
	V2T:   ^Matrix(T), // Right orthogonal matrix 2 (transposed)
	TAUP1: []T, // Householder scalars for P1
	TAUP2: []T, // Householder scalars for P2
	TAUQ1: []T, // Householder scalars for Q1
	TAUQ2: []T, // Householder scalars for Q2
}

// Bidiagonal CS decomposition (f64)
// Computes the CS decomposition of a bidiagonal matrix
m_bidiagonal_cs_decomposition_f64 :: proc(
	blocks: CSDecompositionBlocks(f64),
	result: ^CSDecompositionResult(f64),
	transpose := false, // Transpose the matrix
	positive_signs := true, // Use positive signs convention
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Extract dimensions
	p := blocks.X11.rows
	q := blocks.X11.cols
	m := p + blocks.X21.rows

	// Validate block dimensions
	if blocks.X12.rows != p || blocks.X21.cols != q {
		panic("Inconsistent block dimensions")
	}

	trans_c := "T" if transpose else "N"
	signs_c := "+" if positive_signs else "-"
	m_val := Blas_Int(m)
	p_val := Blas_Int(p)
	q_val := Blas_Int(q)
	ldx11 := Blas_Int(blocks.X11.ld)
	ldx12 := Blas_Int(blocks.X12.ld)
	ldx21 := Blas_Int(blocks.X21.ld)
	ldx22 := Blas_Int(blocks.X22.ld)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.dorbdb_(
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		raw_data(result.phi),
		raw_data(result.TAUP1),
		raw_data(result.TAUP2),
		raw_data(result.TAUQ1),
		raw_data(result.TAUQ2),
		&work_query,
		&lwork,
		&info_val,
		len(trans_c),
		len(signs_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorbdb_(
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		raw_data(result.phi),
		raw_data(result.TAUP1),
		raw_data(result.TAUP2),
		raw_data(result.TAUQ1),
		raw_data(result.TAUQ2),
		raw_data(work),
		&lwork,
		&info_val,
		len(trans_c),
		len(signs_c),
	)

	return info_val == 0, info_val
}

// Bidiagonal CS decomposition (f32)
// Computes the CS decomposition of a bidiagonal matrix
m_bidiagonal_cs_decomposition_f32 :: proc(
	blocks: CSDecompositionBlocks(f32),
	result: ^CSDecompositionResult(f32),
	transpose := false, // Transpose the matrix
	positive_signs := true, // Use positive signs convention
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Extract dimensions
	p := blocks.X11.rows
	q := blocks.X11.cols
	m := p + blocks.X21.rows

	// Validate block dimensions
	if blocks.X12.rows != p || blocks.X21.cols != q {
		panic("Inconsistent block dimensions")
	}

	trans_c := "T" if transpose else "N"
	signs_c := "+" if positive_signs else "-"
	m_val := Blas_Int(m)
	p_val := Blas_Int(p)
	q_val := Blas_Int(q)
	ldx11 := Blas_Int(blocks.X11.ld)
	ldx12 := Blas_Int(blocks.X12.ld)
	ldx21 := Blas_Int(blocks.X21.ld)
	ldx22 := Blas_Int(blocks.X22.ld)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	info_val: Info

	lapack.sorbdb_(
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		raw_data(result.phi),
		raw_data(result.TAUP1),
		raw_data(result.TAUP2),
		raw_data(result.TAUQ1),
		raw_data(result.TAUQ2),
		&work_query,
		&lwork,
		&info_val,
		len(trans_c),
		len(signs_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorbdb_(
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		raw_data(result.phi),
		raw_data(result.TAUP1),
		raw_data(result.TAUP2),
		raw_data(result.TAUQ1),
		raw_data(result.TAUQ2),
		raw_data(work),
		&lwork,
		&info_val,
		len(trans_c),
		len(signs_c),
	)

	return info_val == 0, info_val
}

// Full CS decomposition (f64)
// Computes the complete CS decomposition with orthogonal matrices
m_cs_decomposition_f64 :: proc(
	blocks: CSDecompositionBlocks(f64),
	result: ^CSDecompositionResult(f64),
	compute_u1 := true, // Compute U1 matrix
	compute_u2 := true, // Compute U2 matrix
	compute_v1t := true, // Compute V1T matrix
	compute_v2t := true, // Compute V2T matrix
	transpose := false, // Transpose the matrix
	positive_signs := true, // Use positive signs convention
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Extract dimensions
	p := blocks.X11.rows
	q := blocks.X11.cols
	m := p + blocks.X21.rows

	jobu1_c := "Y" if compute_u1 else "N"
	jobu2_c := "Y" if compute_u2 else "N"
	jobv1t_c := "Y" if compute_v1t else "N"
	jobv2t_c := "Y" if compute_v2t else "N"
	trans_c := "T" if transpose else "N"
	signs_c := "+" if positive_signs else "-"

	m_val := Blas_Int(m)
	p_val := Blas_Int(p)
	q_val := Blas_Int(q)
	ldx11 := Blas_Int(blocks.X11.ld)
	ldx12 := Blas_Int(blocks.X12.ld)
	ldx21 := Blas_Int(blocks.X21.ld)
	ldx22 := Blas_Int(blocks.X22.ld)

	// Set up matrix pointers and leading dimensions
	ldu1 := Blas_Int(compute_u1 ? result.U1.ld : 1)
	ldu2 := Blas_Int(compute_u2 ? result.U2.ld : 1)
	ldv1t := Blas_Int(compute_v1t ? result.V1T.ld : 1)
	ldv2t := Blas_Int(compute_v2t ? result.V2T.ld : 1)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	iwork := make([]Blas_Int, m - min(p, q), context.temp_allocator)
	info_val: Info

	u1_ptr := raw_data(result.U1.data) if compute_u1 else nil
	u2_ptr := raw_data(result.U2.data) if compute_u2 else nil
	v1t_ptr := raw_data(result.V1T.data) if compute_v1t else nil
	v2t_ptr := raw_data(result.V2T.data) if compute_v2t else nil

	lapack.dorcsd_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		jobv2t_c,
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		v2t_ptr,
		&ldv2t,
		&work_query,
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
		len(jobv2t_c),
		len(trans_c),
		len(signs_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorcsd_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		jobv2t_c,
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		v2t_ptr,
		&ldv2t,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
		len(jobv2t_c),
		len(trans_c),
		len(signs_c),
	)

	return info_val == 0, info_val
}

// Full CS decomposition (f32)
// Computes the complete CS decomposition with orthogonal matrices
m_cs_decomposition_f32 :: proc(
	blocks: CSDecompositionBlocks(f32),
	result: ^CSDecompositionResult(f32),
	compute_u1 := true, // Compute U1 matrix
	compute_u2 := true, // Compute U2 matrix
	compute_v1t := true, // Compute V1T matrix
	compute_v2t := true, // Compute V2T matrix
	transpose := false, // Transpose the matrix
	positive_signs := true, // Use positive signs convention
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Extract dimensions
	p := blocks.X11.rows
	q := blocks.X11.cols
	m := p + blocks.X21.rows

	jobu1_c := "Y" if compute_u1 else "N"
	jobu2_c := "Y" if compute_u2 else "N"
	jobv1t_c := "Y" if compute_v1t else "N"
	jobv2t_c := "Y" if compute_v2t else "N"
	trans_c := "T" if transpose else "N"
	signs_c := "+" if positive_signs else "-"

	m_val := Blas_Int(m)
	p_val := Blas_Int(p)
	q_val := Blas_Int(q)
	ldx11 := Blas_Int(blocks.X11.ld)
	ldx12 := Blas_Int(blocks.X12.ld)
	ldx21 := Blas_Int(blocks.X21.ld)
	ldx22 := Blas_Int(blocks.X22.ld)

	// Set up matrix pointers and leading dimensions
	ldu1 := Blas_Int(compute_u1 ? result.U1.ld : 1)
	ldu2 := Blas_Int(compute_u2 ? result.U2.ld : 1)
	ldv1t := Blas_Int(compute_v1t ? result.V1T.ld : 1)
	ldv2t := Blas_Int(compute_v2t ? result.V2T.ld : 1)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	iwork := make([]Blas_Int, m - min(p, q), context.temp_allocator)
	info_val: Info

	u1_ptr := raw_data(result.U1.data) if compute_u1 else nil
	u2_ptr := raw_data(result.U2.data) if compute_u2 else nil
	v1t_ptr := raw_data(result.V1T.data) if compute_v1t else nil
	v2t_ptr := raw_data(result.V2T.data) if compute_v2t else nil

	lapack.sorcsd_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		jobv2t_c,
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		v2t_ptr,
		&ldv2t,
		&work_query,
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
		len(jobv2t_c),
		len(trans_c),
		len(signs_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorcsd_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		jobv2t_c,
		trans_c,
		signs_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(blocks.X11.data),
		&ldx11,
		raw_data(blocks.X12.data),
		&ldx12,
		raw_data(blocks.X21.data),
		&ldx21,
		raw_data(blocks.X22.data),
		&ldx22,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		v2t_ptr,
		&ldv2t,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
		len(jobv2t_c),
		len(trans_c),
		len(signs_c),
	)

	return info_val == 0, info_val
}

// 2x1 CS decomposition (f64)
// Simplified CS decomposition for 2x1 block structure
m_cs_decomposition_2x1_f64 :: proc(
	X11: ^Matrix(f64), // Upper block
	X21: ^Matrix(f64), // Lower block
	result: ^CSDecompositionResult(f64),
	compute_u1 := true, // Compute U1 matrix
	compute_u2 := true, // Compute U2 matrix
	compute_v1t := true, // Compute V1T matrix
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Extract dimensions
	p := X11.rows
	q := X11.cols
	m := p + X21.rows

	if X21.cols != q {
		panic("Block dimensions must be consistent")
	}

	jobu1_c := "Y" if compute_u1 else "N"
	jobu2_c := "Y" if compute_u2 else "N"
	jobv1t_c := "Y" if compute_v1t else "N"

	m_val := Blas_Int(m)
	p_val := Blas_Int(p)
	q_val := Blas_Int(q)
	ldx11 := Blas_Int(X11.ld)
	ldx21 := Blas_Int(X21.ld)

	// Set up matrix pointers and leading dimensions
	ldu1 := Blas_Int(compute_u1 ? result.U1.ld : 1)
	ldu2 := Blas_Int(compute_u2 ? result.U2.ld : 1)
	ldv1t := Blas_Int(compute_v1t ? result.V1T.ld : 1)

	// Query workspace size
	work_query: f64
	lwork := Blas_Int(-1)
	iwork := make([]Blas_Int, m - min(p, q), context.temp_allocator)
	info_val: Info

	u1_ptr := raw_data(result.U1.data) if compute_u1 else nil
	u2_ptr := raw_data(result.U2.data) if compute_u2 else nil
	v1t_ptr := raw_data(result.V1T.data) if compute_v1t else nil

	lapack.dorcsd2by1_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(X11.data),
		&ldx11,
		raw_data(X21.data),
		&ldx21,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		&work_query,
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f64, lwork, context.temp_allocator)

	lapack.dorcsd2by1_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(X11.data),
		&ldx11,
		raw_data(X21.data),
		&ldx21,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
	)

	return info_val == 0, info_val
}

// 2x1 CS decomposition (f32)
// Simplified CS decomposition for 2x1 block structure
m_cs_decomposition_2x1_f32 :: proc(
	X11: ^Matrix(f32), // Upper block
	X21: ^Matrix(f32), // Lower block
	result: ^CSDecompositionResult(f32),
	compute_u1 := true, // Compute U1 matrix
	compute_u2 := true, // Compute U2 matrix
	compute_v1t := true, // Compute V1T matrix
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Extract dimensions
	p := X11.rows
	q := X11.cols
	m := p + X21.rows

	if X21.cols != q {
		panic("Block dimensions must be consistent")
	}

	jobu1_c := "Y" if compute_u1 else "N"
	jobu2_c := "Y" if compute_u2 else "N"
	jobv1t_c := "Y" if compute_v1t else "N"

	m_val := Blas_Int(m)
	p_val := Blas_Int(p)
	q_val := Blas_Int(q)
	ldx11 := Blas_Int(X11.ld)
	ldx21 := Blas_Int(X21.ld)

	// Set up matrix pointers and leading dimensions
	ldu1 := Blas_Int(compute_u1 ? result.U1.ld : 1)
	ldu2 := Blas_Int(compute_u2 ? result.U2.ld : 1)
	ldv1t := Blas_Int(compute_v1t ? result.V1T.ld : 1)

	// Query workspace size
	work_query: f32
	lwork := Blas_Int(-1)
	iwork := make([]Blas_Int, m - min(p, q), context.temp_allocator)
	info_val: Info

	u1_ptr := raw_data(result.U1.data) if compute_u1 else nil
	u2_ptr := raw_data(result.U2.data) if compute_u2 else nil
	v1t_ptr := raw_data(result.V1T.data) if compute_v1t else nil

	lapack.sorcsd2by1_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(X11.data),
		&ldx11,
		raw_data(X21.data),
		&ldx21,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		&work_query,
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
	)

	if info_val != 0 {
		return false, info_val
	}

	// Allocate workspace and compute
	lwork = Blas_Int(work_query)
	work := make([]f32, lwork, context.temp_allocator)

	lapack.sorcsd2by1_(
		jobu1_c,
		jobu2_c,
		jobv1t_c,
		&m_val,
		&p_val,
		&q_val,
		raw_data(X11.data),
		&ldx11,
		raw_data(X21.data),
		&ldx21,
		raw_data(result.theta),
		u1_ptr,
		&ldu1,
		u2_ptr,
		&ldu2,
		v1t_ptr,
		&ldv1t,
		raw_data(work),
		&lwork,
		raw_data(iwork),
		&info_val,
		len(jobu1_c),
		len(jobu2_c),
		len(jobv1t_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Create CS decomposition result structure
create_cs_decomposition_result :: proc(
	$T: typeid,
	m, p, q: int,
	allocator := context.allocator,
) -> CSDecompositionResult(T) {
	min_pq := min(p, q)
	return CSDecompositionResult(T) {
		theta = make([]T, min_pq, allocator),
		phi   = make([]T, q, allocator),
		U1    = nil, // Will be allocated based on options
		U2    = nil,
		V1T   = nil,
		V2T   = nil,
		TAUP1 = make([]T, p, allocator),
		TAUP2 = make([]T, m - p, allocator),
		TAUQ1 = make([]T, q, allocator),
		TAUQ2 = make([]T, m - q, allocator),
	}
}

// Delete CS decomposition result
delete_cs_decomposition_result :: proc(result: ^CSDecompositionResult($T)) {
	if result.theta != nil do delete(result.theta)
	if result.phi != nil do delete(result.phi)
	if result.U1 != nil do delete_matrix(result.U1)
	if result.U2 != nil do delete_matrix(result.U2)
	if result.V1T != nil do delete_matrix(result.V1T)
	if result.V2T != nil do delete_matrix(result.V2T)
	if result.TAUP1 != nil do delete(result.TAUP1)
	if result.TAUP2 != nil do delete(result.TAUP2)
	if result.TAUQ1 != nil do delete(result.TAUQ1)
	if result.TAUQ2 != nil do delete(result.TAUQ2)
}
