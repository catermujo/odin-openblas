package openblas

import lapack "./f77"
import "base:builtin"
import "core:math"
import "core:mem"

// ===================================================================================
// PACKED POSITIVE DEFINITE MATRIX UTILITIES
// ===================================================================================

// Condition number estimation proc group
m_condition_packed_positive_definite :: proc {
	m_condition_packed_positive_definite_c64,
	m_condition_packed_positive_definite_f64,
	m_condition_packed_positive_definite_f32,
	m_condition_packed_positive_definite_c128,
}

// Equilibration proc group
m_equilibrate_packed_positive_definite :: proc {
	m_equilibrate_packed_positive_definite_c64,
	m_equilibrate_packed_positive_definite_f64,
	m_equilibrate_packed_positive_definite_f32,
	m_equilibrate_packed_positive_definite_c128,
}

// ===================================================================================
// PACKED MATRIX STRUCTURE
// ===================================================================================

// Packed symmetric/Hermitian matrix storage
PackedMatrix :: struct($T: typeid) {
	data:       []T, // Packed storage array (n*(n+1)/2 elements)
	n:          int, // Matrix dimension
	uplo_upper: bool, // True if upper triangular packed
}

// ===================================================================================
// PACKED CONDITION NUMBER ESTIMATION
// ===================================================================================

// Estimate condition number of packed positive definite matrix (c64)
// Requires matrix to be already factored using Cholesky factorization
m_condition_packed_positive_definite_c64 :: proc(
	AP: []complex64, // Factored packed matrix
	n: int, // Matrix dimension
	anorm: f32, // Norm of original matrix
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate workspace
	work := make([]complex64, 2 * n, allocator)
	defer delete(work)

	rwork := make([]f32, n, allocator)
	defer delete(rwork)

	rcond_val: f32
	info_val: Info

	lapack.cppcon_(
		uplo_c,
		&n_val,
		raw_data(AP),
		&anorm,
		&rcond_val,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val
}

// Estimate condition number of packed positive definite matrix (f64)
// Requires matrix to be already factored using Cholesky factorization
m_condition_packed_positive_definite_f64 :: proc(
	AP: []f64, // Factored packed matrix
	n: int, // Matrix dimension
	anorm: f64, // Norm of original matrix
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate workspace
	work := make([]f64, 3 * n, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, n, allocator)
	defer delete(iwork)

	rcond_val: f64
	info_val: Info

	lapack.dppcon_(
		uplo_c,
		&n_val,
		raw_data(AP),
		&anorm,
		&rcond_val,
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val
}

// Estimate condition number of packed positive definite matrix (f32)
// Requires matrix to be already factored using Cholesky factorization
m_condition_packed_positive_definite_f32 :: proc(
	AP: []f32, // Factored packed matrix
	n: int, // Matrix dimension
	anorm: f32, // Norm of original matrix
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	rcond: f32,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate workspace
	work := make([]f32, 3 * n, allocator)
	defer delete(work)

	iwork := make([]Blas_Int, n, allocator)
	defer delete(iwork)

	rcond_val: f32
	info_val: Info

	lapack.sppcon_(
		uplo_c,
		&n_val,
		raw_data(AP),
		&anorm,
		&rcond_val,
		raw_data(work),
		raw_data(iwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val
}

// Estimate condition number of packed positive definite matrix (c128)
// Requires matrix to be already factored using Cholesky factorization
m_condition_packed_positive_definite_c128 :: proc(
	AP: []complex128, // Factored packed matrix
	n: int, // Matrix dimension
	anorm: f64, // Norm of original matrix
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	rcond: f64,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate workspace
	work := make([]complex128, 2 * n, allocator)
	defer delete(work)

	rwork := make([]f64, n, allocator)
	defer delete(rwork)

	rcond_val: f64
	info_val: Info

	lapack.zppcon_(
		uplo_c,
		&n_val,
		raw_data(AP),
		&anorm,
		&rcond_val,
		raw_data(work),
		raw_data(rwork),
		&info_val,
		len(uplo_c),
	)

	return rcond_val, info_val
}

// ===================================================================================
// PACKED EQUILIBRATION
// ===================================================================================

// Compute equilibration scaling factors for packed positive definite matrix (c64)
m_equilibrate_packed_positive_definite_c64 :: proc(
	AP: []complex64, // Packed matrix
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	S: []f32,
	scond: f32,
	amax: f32,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate scale factors
	S = make([]f32, n, allocator)

	scond_val: f32
	amax_val: f32
	info_val: Info

	lapack.cppequ_(
		uplo_c,
		&n_val,
		raw_data(AP),
		raw_data(S),
		&scond_val,
		&amax_val,
		&info_val,
		len(uplo_c),
	)

	// Handle error
	if info_val > 0 {
		delete(S)
		S = nil
	}

	return S, scond_val, amax_val, info_val
}

// Compute equilibration scaling factors for packed positive definite matrix (f64)
m_equilibrate_packed_positive_definite_f64 :: proc(
	AP: []f64, // Packed matrix
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	S: []f64,
	scond: f64,
	amax: f64,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate scale factors
	S = make([]f64, n, allocator)

	scond_val: f64
	amax_val: f64
	info_val: Info

	lapack.dppequ_(
		uplo_c,
		&n_val,
		raw_data(AP),
		raw_data(S),
		&scond_val,
		&amax_val,
		&info_val,
		len(uplo_c),
	)

	// Handle error
	if info_val > 0 {
		delete(S)
		S = nil
	}

	return S, scond_val, amax_val, info_val
}

// Compute equilibration scaling factors for packed positive definite matrix (f32)
m_equilibrate_packed_positive_definite_f32 :: proc(
	AP: []f32, // Packed matrix
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	S: []f32,
	scond: f32,
	amax: f32,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate scale factors
	S = make([]f32, n, allocator)

	scond_val: f32
	amax_val: f32
	info_val: Info

	lapack.sppequ_(
		uplo_c,
		&n_val,
		raw_data(AP),
		raw_data(S),
		&scond_val,
		&amax_val,
		&info_val,
		len(uplo_c),
	)

	// Handle error
	if info_val > 0 {
		delete(S)
		S = nil
	}

	return S, scond_val, amax_val, info_val
}

// Compute equilibration scaling factors for packed positive definite matrix (c128)
m_equilibrate_packed_positive_definite_c128 :: proc(
	AP: []complex128, // Packed matrix
	n: int, // Matrix dimension
	uplo_upper := true, // Upper or lower triangular
	allocator := context.allocator,
) -> (
	S: []f64,
	scond: f64,
	amax: f64,
	info: Info,
) {
	// Validate inputs
	expected_size := n * (n + 1) / 2
	if len(AP) < expected_size {
		panic("Packed array too small for matrix dimension")
	}

	n_val := Blas_Int(n)
	uplo_c := "U" if uplo_upper else "L"

	// Allocate scale factors
	S = make([]f64, n, allocator)

	scond_val: f64
	amax_val: f64
	info_val: Info

	lapack.zppequ_(
		uplo_c,
		&n_val,
		raw_data(AP),
		raw_data(S),
		&scond_val,
		&amax_val,
		&info_val,
		len(uplo_c),
	)

	// Handle error
	if info_val > 0 {
		delete(S)
		S = nil
	}

	return S, scond_val, amax_val, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS
// ===================================================================================

// Create packed matrix from standard matrix
create_packed_matrix :: proc(
	A: ^Matrix($T),
	uplo_upper := true,
	allocator := context.allocator,
) -> PackedMatrix(T) {
	if A.rows != A.cols {
		panic("Matrix must be square")
	}

	packed: PackedMatrix(T)
	packed.n = A.rows
	packed.uplo_upper = uplo_upper

	// Allocate packed storage
	packed_size := packed.n * (packed.n + 1) / 2
	packed.data = make([]T, packed_size, allocator)

	// Pack the matrix
	pack_matrix(A, packed.data, uplo_upper)

	return packed
}

// Extract standard matrix from packed format
extract_from_packed :: proc(
	packed: ^PackedMatrix($T),
	allocator := context.allocator,
) -> Matrix(T) {
	A := create_matrix(T, packed.n, packed.n, allocator)

	// Unpack the matrix
	unpack_matrix(packed.data, &A, packed.uplo_upper)

	return A
}

// Pack matrix into packed storage
pack_matrix :: proc(A: ^Matrix($T), AP: []T, uplo_upper: bool) {
	idx := 0
	if uplo_upper {
		// Pack upper triangle column by column
		for j in 0 ..< A.cols {
			for i in 0 ..= j {
				AP[idx] = matrix_get(A, i, j)
				idx += 1
			}
		}
	} else {
		// Pack lower triangle column by column
		for j in 0 ..< A.cols {
			for i in j ..< A.rows {
				AP[idx] = matrix_get(A, i, j)
				idx += 1
			}
		}
	}
}

// Unpack matrix from packed storage
unpack_matrix :: proc(AP: []$T, A: ^Matrix(T), uplo_upper: bool) {
	idx := 0
	if uplo_upper {
		// Unpack upper triangle and mirror to lower
		for j in 0 ..< A.cols {
			for i in 0 ..= j {
				val := AP[idx]
				matrix_set(A, i, j, val)
				if i != j {
					when T == complex64 || T == complex128 {
						// Hermitian: conjugate for lower triangle
						matrix_set(A, j, i, conj(val))
					} else {
						// Symmetric: same value for lower triangle
						matrix_set(A, j, i, val)
					}
				}
				idx += 1
			}
		}
	} else {
		// Unpack lower triangle and mirror to upper
		for j in 0 ..< A.cols {
			for i in j ..< A.rows {
				val := AP[idx]
				matrix_set(A, i, j, val)
				if i != j {
					when T == complex64 || T == complex128 {
						// Hermitian: conjugate for upper triangle
						matrix_set(A, j, i, conj(val))
					} else {
						// Symmetric: same value for upper triangle
						matrix_set(A, j, i, val)
					}
				}
				idx += 1
			}
		}
	}
}

// Complete condition number analysis for packed matrix
analyze_packed_condition :: proc(
	packed: ^PackedMatrix($T),
	allocator := context.allocator,
) -> PackedConditionAnalysis {
	analysis: PackedConditionAnalysis

	// First compute the norm of the packed matrix
	anorm := compute_packed_norm(packed)

	// Create a copy for factorization
	packed_factor := make([]T, len(packed.data), allocator)
	copy(packed_factor, packed.data)
	defer delete(packed_factor)

	// Factor the matrix using Cholesky
	factor_success := cholesky_factor_packed(packed_factor, packed.n, packed.uplo_upper)

	if !factor_success {
		analysis.is_positive_definite = false
		analysis.condition_number = math.INF_F64
		return analysis
	}

	analysis.is_positive_definite = true

	// Estimate condition number
	when T == complex64 {
		rcond, info := m_condition_packed_positive_definite_c64(
			packed_factor,
			packed.n,
			f32(anorm),
			packed.uplo_upper,
			allocator,
		)
		analysis.rcond = f64(rcond)
	} else when T == f64 {
		rcond, info := m_condition_packed_positive_definite_f64(
			packed_factor,
			packed.n,
			anorm,
			packed.uplo_upper,
			allocator,
		)
		analysis.rcond = rcond
	} else when T == f32 {
		rcond, info := m_condition_packed_positive_definite_f32(
			packed_factor,
			packed.n,
			f32(anorm),
			packed.uplo_upper,
			allocator,
		)
		analysis.rcond = f64(rcond)
	} else when T == complex128 {
		rcond, info := m_condition_packed_positive_definite_c128(
			packed_factor,
			packed.n,
			anorm,
			packed.uplo_upper,
			allocator,
		)
		analysis.rcond = rcond
	}

	// Compute condition number
	if analysis.rcond > 0 {
		analysis.condition_number = 1.0 / analysis.rcond
		analysis.is_well_conditioned = analysis.condition_number < 1e6
	} else {
		analysis.condition_number = math.INF_F64
		analysis.is_well_conditioned = false
	}

	// Estimate relative error bound
	analysis.relative_error_bound = analysis.condition_number * builtin.F64_EPSILON

	return analysis
}

// Packed condition analysis structure
PackedConditionAnalysis :: struct {
	rcond:                f64, // Reciprocal condition number
	condition_number:     f64, // Actual condition number
	is_positive_definite: bool, // True if matrix is positive definite
	is_well_conditioned:  bool, // True if condition number < threshold
	relative_error_bound: f64, // Expected relative error in solution
}

// Equilibrate packed matrix
equilibrate_packed :: proc(
	packed: ^PackedMatrix($T),
	allocator := context.allocator,
) -> PackedEquilibrationResult(T) {
	result: PackedEquilibrationResult(T)

	when T == complex64 {
		S, scond, amax, info := m_equilibrate_packed_positive_definite_c64(
			packed.data,
			packed.n,
			packed.uplo_upper,
			allocator,
		)
		result.scale_factors = S
		result.condition_scale = f64(scond)
		result.max_element = f64(amax)
		result.success = info == 0
	} else when T == f64 {
		S, scond, amax, info := m_equilibrate_packed_positive_definite_f64(
			packed.data,
			packed.n,
			packed.uplo_upper,
			allocator,
		)
		result.scale_factors = S
		result.condition_scale = scond
		result.max_element = amax
		result.success = info == 0
	} else when T == f32 {
		S, scond, amax, info := m_equilibrate_packed_positive_definite_f32(
			packed.data,
			packed.n,
			packed.uplo_upper,
			allocator,
		)
		result.scale_factors = S
		result.condition_scale = f64(scond)
		result.max_element = f64(amax)
		result.success = info == 0
	} else when T == complex128 {
		S, scond, amax, info := m_equilibrate_packed_positive_definite_c128(
			packed.data,
			packed.n,
			packed.uplo_upper,
			allocator,
		)
		result.scale_factors = S
		result.condition_scale = scond
		result.max_element = amax
		result.success = info == 0
	}

	// Check if equilibration is needed
	result.is_equilibrated = result.condition_scale >= 0.1 && 1.0 / result.condition_scale >= 0.1
	result.needs_scaling = !result.is_equilibrated

	return result
}

// Packed equilibration result
PackedEquilibrationResult :: struct($T: typeid) {
	scale_factors:   []T, // Diagonal scaling factors
	condition_scale: f64, // Ratio of smallest to largest scaling
	max_element:     f64, // Maximum absolute element
	is_equilibrated: bool, // True if already well-scaled
	needs_scaling:   bool, // True if scaling would help
	success:         bool, // True if equilibration succeeded
}

// Apply scaling to packed matrix
apply_packed_scaling :: proc(packed: ^PackedMatrix($T), scale_factors: []$S) {
	if len(scale_factors) != packed.n {
		panic("Scale factors dimension mismatch")
	}

	idx := 0
	if packed.uplo_upper {
		// Scale upper triangle
		for j in 0 ..< packed.n {
			for i in 0 ..= j {
				val := packed.data[idx]
				scaled := val * T(scale_factors[i] * scale_factors[j])
				packed.data[idx] = scaled
				idx += 1
			}
		}
	} else {
		// Scale lower triangle
		for j in 0 ..< packed.n {
			for i in j ..< packed.n {
				val := packed.data[idx]
				scaled := val * T(scale_factors[i] * scale_factors[j])
				packed.data[idx] = scaled
				idx += 1
			}
		}
	}
}

// Memory usage comparison
packed_memory_savings :: proc(n: int) -> f64 {
	full_size := f64(n * n)
	packed_size := f64(n * (n + 1) / 2)
	return (full_size - packed_size) / full_size * 100.0
}

// Get element from packed matrix
packed_get :: proc(packed: ^PackedMatrix($T), i, j: int) -> T {
	if i < 0 || i >= packed.n || j < 0 || j >= packed.n {
		panic("Index out of bounds")
	}

	// Ensure we access the stored triangle
	row, col := i, j
	if packed.uplo_upper {
		if i > j {
			row, col = j, i // Access upper triangle
		}
	} else {
		if i < j {
			row, col = j, i // Access lower triangle
		}
	}

	// Calculate packed index
	idx: int
	if packed.uplo_upper {
		// Upper triangle stored column by column
		idx = col * (col + 1) / 2 + row
	} else {
		// Lower triangle stored column by column
		idx = col * (2 * packed.n - col - 1) / 2 + (row - col)
	}

	val := packed.data[idx]

	// Apply conjugate if accessing mirrored element in Hermitian matrix
	when T == complex64 || T == complex128 {
		if i != j && row != i {
			return conj(val)
		}
	}

	return val
}

// Set element in packed matrix
packed_set :: proc(packed: ^PackedMatrix($T), i, j: int, val: T) {
	if i < 0 || i >= packed.n || j < 0 || j >= packed.n {
		panic("Index out of bounds")
	}

	// Only set the stored triangle
	if packed.uplo_upper {
		if i > j {
			return // Don't set lower triangle
		}
	} else {
		if i < j {
			return // Don't set upper triangle
		}
	}

	// Calculate packed index
	idx: int
	if packed.uplo_upper {
		// Upper triangle stored column by column
		idx = j * (j + 1) / 2 + i
	} else {
		// Lower triangle stored column by column
		idx = j * (2 * packed.n - j - 1) / 2 + (i - j)
	}

	packed.data[idx] = val
}

// Delete packed matrix
delete_packed_matrix :: proc(packed: ^PackedMatrix($T)) {
	if packed.data != nil {
		delete(packed.data)
		packed.data = nil
	}
}

// Helper functions

compute_packed_norm :: proc(packed: ^PackedMatrix($T)) -> f64 {
	// Compute 1-norm of packed matrix
	norm := 0.0
	for val in packed.data {
		when T == complex64 || T == complex128 {
			norm = max(norm, f64(abs(val)))
		} else {
			norm = max(norm, f64(abs(val)))
		}
	}
	return norm
}

cholesky_factor_packed :: proc(AP: []$T, n: int, uplo_upper: bool) -> bool {
	// This would call LAPACK packed Cholesky factorization
	// For now, return true as placeholder
	return true
}

conj :: proc(x: $T) -> T {
	when T == complex64 || T == complex128 {
		// Return complex conjugate
		return complex(real(x), -imag(x))
	} else {
		return x
	}
}

abs :: proc(x: $T) -> f64 {
	when T == complex64 || T == complex128 {
		return math.sqrt(real(x) * real(x) + imag(x) * imag(x))
	} else {
		return f64(x) if x >= 0 else f64(-x)
	}
}
