package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// RECTANGULAR FULL PACKED (RFP) FORMAT CHOLESKY FACTORIZATION AND INVERSION
// ===================================================================================

// RFP Cholesky factorization proc group
m_cholesky_rfp :: proc {
	m_cholesky_rfp_c64,
	m_cholesky_rfp_f64,
	m_cholesky_rfp_f32,
	m_cholesky_rfp_c128,
}

// RFP matrix inversion proc group
m_invert_rfp :: proc {
	m_invert_rfp_c64,
	m_invert_rfp_f64,
	m_invert_rfp_f32,
	m_invert_rfp_c128,
}

// RFP triangular solve proc group
m_solve_rfp :: proc {
	m_solve_rfp_c64,
	m_solve_rfp_f64,
	m_solve_rfp_f32,
	m_solve_rfp_c128,
}

// ===================================================================================
// RFP FORMAT PARAMETERS
// ===================================================================================


// Convert RFP transpose to LAPACK character
_rfp_transpose_to_char :: proc(transr: RFPTranspose) -> cstring {
	switch transr {
	case .Normal:
		return "N"
	case .Transpose:
		return "T"
	case:
		return "N"
	}
}

// ===================================================================================
// RFP CHOLESKY FACTORIZATION IMPLEMENTATION
// ===================================================================================

// Cholesky factorization in RFP format (c64)
// Computes the Cholesky factorization of a positive definite matrix in RFP format
m_cholesky_rfp_c64 :: proc(
	A: []complex64, // Matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.cpftrf_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// Cholesky factorization in RFP format (f64)
// Computes the Cholesky factorization of a positive definite matrix in RFP format
m_cholesky_rfp_f64 :: proc(
	A: []f64, // Matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.dpftrf_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// Cholesky factorization in RFP format (f32)
// Computes the Cholesky factorization of a positive definite matrix in RFP format
m_cholesky_rfp_f32 :: proc(
	A: []f32, // Matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.spftrf_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// Cholesky factorization in RFP format (c128)
// Computes the Cholesky factorization of a positive definite matrix in RFP format
m_cholesky_rfp_c128 :: proc(
	A: []complex128, // Matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.zpftrf_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// ===================================================================================
// RFP MATRIX INVERSION IMPLEMENTATION
// ===================================================================================

// Matrix inversion in RFP format (c64)
// Computes the inverse of a positive definite matrix using Cholesky factorization
m_invert_rfp_c64 :: proc(
	A: []complex64, // Factorized matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.cpftri_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// Matrix inversion in RFP format (f64)
// Computes the inverse of a positive definite matrix using Cholesky factorization
m_invert_rfp_f64 :: proc(
	A: []f64, // Factorized matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.dpftri_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// Matrix inversion in RFP format (f32)
// Computes the inverse of a positive definite matrix using Cholesky factorization
m_invert_rfp_f32 :: proc(
	A: []f32, // Factorized matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.spftri_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// Matrix inversion in RFP format (c128)
// Computes the inverse of a positive definite matrix using Cholesky factorization
m_invert_rfp_c128 :: proc(
	A: []complex128, // Factorized matrix in RFP format (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	info_val: Info

	lapack.zpftri_(transr_c, uplo_c, &n_val, raw_data(A), &info_val, len(transr_c), len(uplo_c))

	return info_val == 0, info_val
}

// ===================================================================================
// RFP TRIANGULAR SOLVE IMPLEMENTATION
// ===================================================================================

// Solve triangular system in RFP format (c64)
// Solves A*X = B or A^T*X = B where A is in RFP format from Cholesky factorization
m_solve_rfp_c64 :: proc(
	A: []complex64, // Factored matrix in RFP format
	B: ^Matrix(complex64), // Right-hand side matrix (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}
	if B.rows != n {
		panic("B matrix dimension mismatch")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.cpftrs_(
		transr_c,
		uplo_c,
		&n_val,
		&nrhs_val,
		raw_data(A),
		cast(^complex64)B.data,
		&ldb,
		&info_val,
		len(transr_c),
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve triangular system in RFP format (f64)
// Solves A*X = B or A^T*X = B where A is in RFP format from Cholesky factorization
m_solve_rfp_f64 :: proc(
	A: []f64, // Factored matrix in RFP format
	B: ^Matrix(f64), // Right-hand side matrix (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}
	if B.rows != n {
		panic("B matrix dimension mismatch")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.dpftrs_(
		transr_c,
		uplo_c,
		&n_val,
		&nrhs_val,
		raw_data(A),
		cast(^f64)B.data,
		&ldb,
		&info_val,
		len(transr_c),
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve triangular system in RFP format (f32)
// Solves A*X = B or A^T*X = B where A is in RFP format from Cholesky factorization
m_solve_rfp_f32 :: proc(
	A: []f32, // Factored matrix in RFP format
	B: ^Matrix(f32), // Right-hand side matrix (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}
	if B.rows != n {
		panic("B matrix dimension mismatch")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.spftrs_(
		transr_c,
		uplo_c,
		&n_val,
		&nrhs_val,
		raw_data(A),
		cast(^f32)B.data,
		&ldb,
		&info_val,
		len(transr_c),
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// Solve triangular system in RFP format (c128)
// Solves A*X = B or A^T*X = B where A is in RFP format from Cholesky factorization
m_solve_rfp_c128 :: proc(
	A: []complex128, // Factored matrix in RFP format
	B: ^Matrix(complex128), // Right-hand side matrix (input/output)
	n: int, // Matrix dimension
	transr := RFPTranspose.Normal, // RFP transpose format
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(A) < expected_size {
		panic("RFP array too small for matrix dimension")
	}
	if B.rows != n {
		panic("B matrix dimension mismatch")
	}

	transr_c := _rfp_transpose_to_char(transr)
	uplo_c := "U" if uplo_upper else "L"
	n_val := Blas_Int(n)
	nrhs_val := Blas_Int(B.cols)
	ldb := Blas_Int(B.stride)
	info_val: Info

	lapack.zpftrs_(
		transr_c,
		uplo_c,
		&n_val,
		&nrhs_val,
		raw_data(A),
		cast(^complex128)B.data,
		&ldb,
		&info_val,
		len(transr_c),
		len(uplo_c),
	)

	return info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// RFP matrix structure
RFPMatrix :: struct($T: typeid) {
	data:       []T, // RFP format data
	n:          int, // Matrix dimension
	transr:     RFPTranspose, // Transpose format
	uplo_upper: bool, // Upper or lower triangular
}

// Create RFP matrix from standard matrix
create_rfp_matrix :: proc(
	A: ^Matrix($T),
	transr := RFPTranspose.Normal,
	uplo_upper := true,
	allocator := context.allocator,
) -> RFPMatrix(T) {
	if A.rows != A.cols {
		panic("Matrix must be square for RFP format")
	}

	n := A.rows
	size := n * (n + 1) / 2
	data := make([]T, size, allocator)

	// Convert to RFP format
	convert_to_rfp(A, data, n, transr, uplo_upper)

	return RFPMatrix(T){data = data, n = n, transr = transr, uplo_upper = uplo_upper}
}

// Convert RFP matrix back to standard format
extract_from_rfp :: proc(rfp: ^RFPMatrix($T), allocator := context.allocator) -> Matrix(T) {
	A := make_matrix(T, rfp.n, rfp.n, .General, allocator)

	// Extract from RFP format
	extract_rfp_to_matrix(rfp.data, &A, rfp.n, rfp.transr, rfp.uplo_upper)

	return A
}

// Complete Cholesky factorization and inversion workflow
cholesky_invert_rfp :: proc(
	A: ^Matrix($T), // Input positive definite matrix
	allocator := context.allocator,
) -> (
	A_inv: Matrix(T),
	success: bool,
) {
	// Convert to RFP format
	rfp := create_rfp_matrix(A, .Normal, true, allocator)

	// Factor the matrix
	when T == complex64 {
		success, _ := m_cholesky_rfp_c64(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
		if !success {
			delete(rfp.data)
			return Matrix(T){}, false
		}
		// Invert the matrix
		success, _ = m_invert_rfp_c64(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
	} else when T == f64 {
		success, _ := m_cholesky_rfp_f64(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
		if !success {
			delete(rfp.data)
			return Matrix(T){}, false
		}
		success, _ = m_invert_rfp_f64(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
	} else when T == f32 {
		success, _ := m_cholesky_rfp_f32(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
		if !success {
			delete(rfp.data)
			return Matrix(T){}, false
		}
		success, _ = m_invert_rfp_f32(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
	} else when T == complex128 {
		success, _ := m_cholesky_rfp_c128(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
		if !success {
			delete(rfp.data)
			return Matrix(T){}, false
		}
		success, _ = m_invert_rfp_c128(rfp.data, rfp.n, rfp.transr, rfp.uplo_upper, allocator)
	} else {
		panic("Unsupported type for RFP operations")
	}

	if !success {
		delete(rfp.data)
		return Matrix(T){}, false
	}

	// Convert back to standard format
	A_inv = extract_from_rfp(&rfp, allocator)
	delete(rfp.data)

	return A_inv, true
}

// Factor only (for reuse in solving systems)
factor_rfp :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	rfp_factor: RFPMatrix(T),
	success: bool,
) {
	// Convert to RFP format
	rfp_factor = create_rfp_matrix(A, .Normal, true, allocator)

	// Factor the matrix
	when T == complex64 {
		success, _ := m_cholesky_rfp_c64(
			rfp_factor.data,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
	} else when T == f64 {
		success, _ := m_cholesky_rfp_f64(
			rfp_factor.data,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
	} else when T == f32 {
		success, _ := m_cholesky_rfp_f32(
			rfp_factor.data,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
	} else when T == complex128 {
		success, _ := m_cholesky_rfp_c128(
			rfp_factor.data,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
	} else {
		panic("Unsupported type for RFP factorization")
	}

	if !success {
		delete(rfp_factor.data)
	}

	return rfp_factor, success
}

// Solve linear system using pre-factored RFP matrix
solve_with_rfp_factor :: proc(
	rfp_factor: ^RFPMatrix($T),
	B: ^Matrix(T),
	allocator := context.allocator,
) -> bool {
	// Use the appropriate solve function
	when T == complex64 {
		success, _ := m_solve_rfp_c64(
			rfp_factor.data,
			B,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
		return success
	} else when T == f64 {
		success, _ := m_solve_rfp_f64(
			rfp_factor.data,
			B,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
		return success
	} else when T == f32 {
		success, _ := m_solve_rfp_f32(
			rfp_factor.data,
			B,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
		return success
	} else when T == complex128 {
		success, _ := m_solve_rfp_c128(
			rfp_factor.data,
			B,
			rfp_factor.n,
			rfp_factor.transr,
			rfp_factor.uplo_upper,
			allocator,
		)
		return success
	} else {
		panic("Unsupported type for RFP solve")
	}
}

// Complete solve workflow: factor and solve
solve_rfp :: proc(A: ^Matrix($T), B: ^Matrix(T), allocator := context.allocator) -> bool {
	// Factor the matrix
	rfp_factor, success := factor_rfp(A, allocator)
	if !success {
		return false
	}
	defer delete_rfp_matrix(&rfp_factor)

	// Solve the system
	return solve_with_rfp_factor(&rfp_factor, B, allocator)
}

// Get RFP storage size for n×n matrix
get_rfp_size :: proc(n: int) -> int {
	return n * (n + 1) / 2
}

// Check if matrix dimension is suitable for RFP format
is_rfp_compatible :: proc(n: int) -> bool {
	// RFP format works best for certain dimensions
	// Generally works for all n, but some sizes are more efficient
	return n > 0
}

// Convert standard matrix to RFP format (simplified implementation)
convert_to_rfp :: proc(A: ^Matrix($T), rfp: []T, n: int, transr: RFPTranspose, uplo_upper: bool) {
	// Simplified conversion - actual RFP layout is complex
	// This is a placeholder for the actual rectangular full packed format
	idx := 0
	if uplo_upper {
		for j in 0 ..< n {
			for i in 0 ..= j {
				rfp[idx] = matrix_get(A, i, j)
				idx += 1
			}
		}
	} else {
		for j in 0 ..< n {
			for i in j ..< n {
				rfp[idx] = matrix_get(A, i, j)
				idx += 1
			}
		}
	}
}

// Extract RFP format to standard matrix (simplified implementation)
extract_rfp_to_matrix :: proc(
	rfp: []$T,
	A: ^Matrix(T),
	n: int,
	transr: RFPTranspose,
	uplo_upper: bool,
) {
	// Simplified extraction - actual RFP layout is complex
	idx := 0
	if uplo_upper {
		for j in 0 ..< n {
			for i in 0 ..= j {
				matrix_set(A, i, j, rfp[idx])
				// Symmetric/Hermitian: also set the lower triangle
				if i != j {
					when T == complex64 || T == complex128 {
						matrix_set(A, j, i, conj(rfp[idx]))
					} else {
						matrix_set(A, j, i, rfp[idx])
					}
				}
				idx += 1
			}
		}
	} else {
		for j in 0 ..< n {
			for i in j ..< n {
				matrix_set(A, i, j, rfp[idx])
				// Symmetric/Hermitian: also set the upper triangle
				if i != j {
					when T == complex64 || T == complex128 {
						matrix_set(A, j, i, conj(rfp[idx]))
					} else {
						matrix_set(A, j, i, rfp[idx])
					}
				}
				idx += 1
			}
		}
	}
}

// Memory usage comparison
rfp_memory_savings :: proc(n: int) -> f64 {
	full_size := f64(n * n)
	rfp_size := f64(n * (n + 1) / 2)
	return (full_size - rfp_size) / full_size * 100.0
}

// Delete RFP matrix
delete_rfp_matrix :: proc(rfp: ^RFPMatrix($T)) {
	if rfp.data != nil {
		delete(rfp.data)
	}
}
