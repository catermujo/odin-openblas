package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"
import "core:slice"

// ===================================================================================
// LAPACK TEST MATRIX GENERATION
// ===================================================================================

m_generate_test_matrix :: proc {
	m_generate_test_matrix_f32_f64,
	m_generate_test_matrix_c64_c128,
}

m_generate_hilbert :: proc {
	m_generate_hilbert_f32_f64,
	m_generate_hilbert_c64_c128,
}

m_generate_test_matrix_eigenvalues :: proc {
	m_generate_test_matrix_eigenvalues_f32_f64,
	m_generate_test_matrix_eigenvalues_c64_c128,
}

// ===================================================================================
// TEST MATRIX GENERATION PARAMETERS
// ===================================================================================

// Comprehensive parameters for test matrix generation
TestMatrixParams :: struct {
	// Matrix dimensions
	rows, cols:       int,

	// Random distribution
	distribution:     RandomDistribution,

	// Random seed (4 integers)
	seed:             [4]int,

	// Matrix structure
	symmetry:         MatrixSymmetry,

	// Singular values/eigenvalues
	singular_values:  []f64, // D array - controls spectrum

	// Condition number control
	mode:             int, // Type of matrix to generate (1-6)
	condition_number: f64, // Target condition number
	max_element:      f64, // Maximum element magnitude

	// Banding parameters
	lower_bandwidth:  int, // kl - lower bandwidth
	upper_bandwidth:  int, // ku - upper bandwidth

	// Storage format
	packing:          MatrixPacking,
}

// Default test matrix parameters
default_test_matrix_params :: proc() -> TestMatrixParams {
	return TestMatrixParams {
		rows = 10,
		cols = 10,
		distribution = .Uniform,
		seed = {1, 2, 3, 4},
		symmetry = .None,
		singular_values = nil,
		mode = 1,
		condition_number = 1.0,
		max_element = 1.0,
		lower_bandwidth = 0,
		upper_bandwidth = 0,
		packing = .No_Packing,
	}
}

// ===================================================================================
// TEST MATRIX GENERATION IMPLEMENTATION
// ===================================================================================

// Generate test matrix for f32 and f64
m_generate_test_matrix_f32_f64 :: proc(
	A: ^Matrix($T),
	params: TestMatrixParams,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) where T == f32 ||
	T == f64 {
	// Validate parameters
	assert(
		params.distribution != .ComplexUniform,
		"ComplexUniform distribution is not supported by DLATMS/SLATMS - use Uniform, UniformMinus1To1, or Normal",
	)
	assert(
		A.rows == Blas_Int(params.rows) && A.cols == Blas_Int(params.cols),
		"Matrix dimensions must match parameters",
	)

	// Prepare parameters
	m := Blas_Int(params.rows)
	n := Blas_Int(params.cols)
	dist_c := distribution_to_cstring(params.distribution)

	// Setup seed array
	iseed := make([]Blas_Int, 4, context.temp_allocator)
	for i in 0 ..< 4 {
		iseed[i] = Blas_Int(params.seed[i])
	}

	sym_c := _symmetry_to_char(params.symmetry)

	// Setup singular values array
	D: []T
	if params.singular_values != nil {
		D = make([]T, len(params.singular_values), context.temp_allocator)
		for i in 0 ..< len(params.singular_values) {
			D[i] = T(params.singular_values[i])
		}
	} else {
		D = make([]T, max(params.rows, params.cols), context.temp_allocator)
		// Initialize with default values
		for i in 0 ..< len(D) {
			D[i] = 1.0
		}
	}

	mode := Blas_Int(params.mode)
	cond := T(params.condition_number)
	dmax := T(params.max_element)
	kl := Blas_Int(params.lower_bandwidth)
	ku := Blas_Int(params.upper_bandwidth)
	pack_c := _packing_to_char(params.packing)
	lda := A.ld

	// Allocate workspace
	work := make([]T, 3 * max(params.rows, params.cols), context.temp_allocator)

	info_val: Info
	when T == f32 {
		lapack.slatms_(
			&m,
			&n,
			dist_c,
			raw_data(iseed),
			sym_c,
			raw_data(D),
			&mode,
			&cond,
			&dmax,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	} else when T == f64 {
		lapack.dlatms_(
			&m,
			&n,
			dist_c,
			raw_data(iseed),
			sym_c,
			raw_data(D),
			&mode,
			&cond,
			&dmax,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	}

	return info_val == 0, info_val
}

// Generate test matrix for complex64 and complex128
m_generate_test_matrix_c64_c128 :: proc(
	A: ^Matrix($T),
	params: TestMatrixParams,
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) where T == complex64 ||
	T == complex128 {
	// Validate parameters
	assert(
		params.distribution != .ComplexUniform,
		"ComplexUniform distribution is not supported by CLATMS/ZLATMS - use Uniform, UniformMinus1To1, or Normal",
	)
	assert(
		A.rows == Blas_Int(params.rows) && A.cols == Blas_Int(params.cols),
		"Matrix dimensions must match parameters",
	)

	// Prepare parameters
	m := Blas_Int(params.rows)
	n := Blas_Int(params.cols)
	dist_c := distribution_to_cstring(params.distribution)

	// Setup seed array
	iseed := make([]Blas_Int, 4, context.temp_allocator)
	for i in 0 ..< 4 {
		iseed[i] = Blas_Int(params.seed[i])
	}

	sym_c := _symmetry_to_char(params.symmetry)

	mode := Blas_Int(params.mode)
	kl := Blas_Int(params.lower_bandwidth)
	ku := Blas_Int(params.upper_bandwidth)
	pack_c := _packing_to_char(params.packing)
	lda := A.ld

	// Allocate workspace
	work := make([]T, 3 * max(params.rows, params.cols), context.temp_allocator)

	info_val: Info
	when T == complex64 {
		// Setup singular values array
		D := make([]f32, max(params.rows, params.cols), context.temp_allocator)
		if params.singular_values != nil {
			for i in 0 ..< min(len(params.singular_values), len(D)) {
				D[i] = f32(params.singular_values[i])
			}
		} else {
			for i in 0 ..< len(D) {
				D[i] = 1.0
			}
		}
		cond := f32(params.condition_number)
		dmax := f32(params.max_element)

		lapack.clatms_(
			&m,
			&n,
			dist_c,
			raw_data(iseed),
			sym_c,
			raw_data(D),
			&mode,
			&cond,
			&dmax,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	} else when T == complex128 {
		// Setup singular values array
		D := make([]f64, max(params.rows, params.cols), context.temp_allocator)
		if params.singular_values != nil {
			for i in 0 ..< min(len(params.singular_values), len(D)) {
				D[i] = params.singular_values[i]
			}
		} else {
			for i in 0 ..< len(D) {
				D[i] = 1.0
			}
		}
		cond := params.condition_number
		dmax := params.max_element

		lapack.zlatms_(
			&m,
			&n,
			dist_c,
			raw_data(iseed),
			sym_c,
			raw_data(D),
			&mode,
			&cond,
			&dmax,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	}

	return info_val == 0, info_val
}

// ===================================================================================
// HILBERT MATRIX GENERATION
// ===================================================================================

// Generate Hilbert matrix for f32 and f64
// Hilbert matrix H[i,j] = 1/(i+j-1), extremely ill-conditioned
m_generate_hilbert_f32_f64 :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	info: Info,
) where T == f32 ||
	T == f64 {
	// Validate input
	assert(A.rows == A.cols, "Hilbert matrix must be square")

	n := A.rows
	// Fill Hilbert matrix: H[i,j] = 1/(i+j+1)  (0-indexed)
	for i in 0 ..< n {
		for j in 0 ..< n {
			val := T(1.0) / T(i + j + 1)
			matrix_set(A, i, j, val)
		}
	}

	return 0
}

// Generate Hilbert matrix for complex64 and complex128
// For complex, we use the same formula with real values
m_generate_hilbert_c64_c128 :: proc(
	A: ^Matrix($T),
	allocator := context.allocator,
) -> (
	info: Info,
) where T == complex64 ||
	T == complex128 {
	// Validate input
	assert(A.rows == A.cols, "Hilbert matrix must be square")

	n := A.rows
	// Fill Hilbert matrix: H[i,j] = 1/(i+j+1)  (0-indexed)
	for i in 0 ..< n {
		for j in 0 ..< n {
			when T == complex64 {
				val := complex64(1.0 / f32(i + j + 1))
				matrix_set(A, i, j, val)
			} else when T == complex128 {
				val := complex128(1.0 / f64(i + j + 1))
				matrix_set(A, i, j, val)
			}
		}
	}

	return 0
}

// ===================================================================================
// EIGENVALUE-CONTROLLED MATRIX GENERATION (USING DLATMT)
// ===================================================================================

// Generate test matrix with specified eigenvalues for f32 and f64
m_generate_test_matrix_eigenvalues_f32_f64 :: proc(
	A: ^Matrix($T),
	eigenvalues: []T,
	rank: int = -1, // -1 means full rank
	distribution: RandomDistribution = .UniformMinus1To1,
	seed: [4]Blas_Int = {1, 2, 3, 5},
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) where T == f32 || T == f64 {
	// Validate parameters
	assert(
		distribution != .ComplexUniform,
		"ComplexUniform distribution is not supported by DLATMT/SLATMT - use Uniform, UniformMinus1To1, or Normal",
	)
	n := int(A.rows)
	assert(A.rows == A.cols, "Matrix must be square for eigenvalue specification")
	assert(len(eigenvalues) == n, "Number of eigenvalues must match matrix dimension")

	// Prepare parameters
	m_val := A.rows
	n_val := A.cols
	dist_c := distribution_to_cstring(distribution)

	// Setup seed array
	iseed := [4]Blas_Int{seed[0], seed[1], seed[2], seed[3]}

	sym_c := cstring("N") // Non-symmetric
	mode := Blas_Int(0) // Use provided eigenvalues directly
	cond := T(1.0)
	dmax := T(1.0)
	rank_val := Blas_Int(rank < 0 ? n : rank)
	kl := Blas_Int(n - 1) // Full matrix
	ku := Blas_Int(n - 1) // Full matrix
	pack_c := cstring("N") // No packing
	lda := A.ld

	// Allocate workspace (larger for DLATMT)
	work := make([]T, 5 * n, context.temp_allocator)
	defer delete(work)

	info_val: Info
	when T == f32 {
		lapack.slatmt_(
			&m_val,
			&n_val,
			dist_c,
			&iseed[0],
			sym_c,
			raw_data(eigenvalues),
			&mode,
			&cond,
			&dmax,
			&rank_val,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	} else when T == f64 {
		lapack.dlatmt_(
			&m_val,
			&n_val,
			dist_c,
			&iseed[0],
			sym_c,
			raw_data(eigenvalues),
			&mode,
			&cond,
			&dmax,
			&rank_val,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	}

	return info_val == 0, info_val
}

// Generate test matrix with specified eigenvalues for complex64 and complex128
m_generate_test_matrix_eigenvalues_c64_c128 :: proc(
	A: ^Matrix($T),
	eigenvalues: []T, // Complex eigenvalues
	rank: int = -1, // -1 means full rank
	distribution: RandomDistribution = .UniformMinus1To1,
	seed: [4]Blas_Int = {1, 2, 3, 5},
	allocator := context.allocator,
) -> (
	success: bool,
	info: Info,
) where T == complex64 || T == complex128 {
	// Validate parameters
	assert(
		distribution != .ComplexUniform,
		"ComplexUniform distribution is not supported by CLATMT/ZLATMT - use Uniform, UniformMinus1To1, or Normal",
	)
	n := int(A.rows)
	assert(A.rows == A.cols, "Matrix must be square for eigenvalue specification")
	assert(len(eigenvalues) == n, "Number of eigenvalues must match matrix dimension")

	// Prepare parameters
	m_val := A.rows
	n_val := A.cols
	dist_c := distribution_to_cstring(distribution)

	// Setup seed array
	iseed := [4]Blas_Int{seed[0], seed[1], seed[2], seed[3]}

	sym_c := cstring("N") // Non-Hermitian
	mode := Blas_Int(0) // Use provided eigenvalues directly
	rank_val := Blas_Int(rank < 0 ? n : rank)
	kl := Blas_Int(n - 1) // Full matrix
	ku := Blas_Int(n - 1) // Full matrix
	pack_c := cstring("N") // No packing
	lda := A.ld

	// Allocate workspace (larger for DLATMT)
	work := make([]T, 5 * n, context.temp_allocator)
	defer delete(work)

	info_val: Info
	when T == complex64 {
		// Convert complex eigenvalues to real array for CLATMT
		// (CLATMT expects real eigenvalues for non-Hermitian matrices)
		D := make([]f32, n, context.temp_allocator)
		defer delete(D)
		for i in 0 ..< n {
			D[i] = abs(eigenvalues[i])
		}
		cond := f32(1.0)
		dmax := f32(1.0)

		lapack.clatmt_(
			&m_val,
			&n_val,
			dist_c,
			&iseed[0],
			sym_c,
			raw_data(D),
			&mode,
			&cond,
			&dmax,
			&rank_val,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	} else when T == complex128 {
		// Convert complex eigenvalues to real array for ZLATMT
		// (ZLATMT expects real eigenvalues for non-Hermitian matrices)
		D := make([]f64, n, context.temp_allocator)
		defer delete(D)
		for i in 0 ..< n {
			D[i] = abs(eigenvalues[i])
		}
		cond := f64(1.0)
		dmax := f64(1.0)

		lapack.zlatmt_(
			&m_val,
			&n_val,
			dist_c,
			&iseed[0],
			sym_c,
			raw_data(D),
			&mode,
			&cond,
			&dmax,
			&rank_val,
			&kl,
			&ku,
			pack_c,
			raw_data(A.data),
			&lda,
			raw_data(work),
			&info_val,
			len(dist_c),
			len(sym_c),
			len(pack_c),
		)
	}

	return info_val == 0, info_val
}

// ===================================================================================
// CONVENIENCE FUNCTIONS FOR COMMON TEST MATRICES
// ===================================================================================

// Generate well-conditioned random matrix
generate_random_matrix :: proc(
	$T: typeid,
	rows, cols: int,
	seed: [4]Blas_Int = {1, 2, 3, 4},
	allocator := context.allocator,
) -> Matrix(T) where is_float(T) ||
	is_complex(T) {
	A := make_matrix(T, rows, cols, .General, allocator)
	params := TestMatrixParams {
		rows             = rows,
		cols             = cols,
		distribution     = .UniformMinus1To1,
		seed             = seed,
		symmetry         = .None,
		mode             = 1,
		condition_number = 1.0,
		max_element      = 1.0,
		packing          = .No_Packing,
	}

	when T == f32 || T == f64 {
		success, _ := m_generate_test_matrix_f32_f64(&A, params, allocator)
	} else when T == complex64 || T == complex128 {
		success, _ := m_generate_test_matrix_c64_c128(&A, params, allocator)
	}

	return A
}

// Generate ill-conditioned matrix with specified condition number
generate_ill_conditioned_matrix :: proc(
	$T: typeid,
	size: int,
	condition_number: f64,
	seed: [4]Blas_Int = {1, 2, 3, 4},
	allocator := context.allocator,
) -> Matrix(T) {
	A := make_matrix(T, size, size, .General, allocator)
	params := TestMatrixParams {
		rows             = size,
		cols             = size,
		distribution     = .Uniform,
		seed             = seed,
		symmetry         = .None,
		mode             = 4, // Geometric distribution of singular values
		condition_number = condition_number,
		max_element      = 1.0,
		packing          = .No_Packing,
	}

	when T == f32 || T == f64 {
		success, _ := m_generate_test_matrix_f32_f64(&A, params, allocator)
	} else when T == complex64 || T == complex128 {
		success, _ := m_generate_test_matrix_c64_c128(&A, params, allocator)
	}

	return A
}

// Generate symmetric positive definite matrix
generate_spd_matrix :: proc(
	$T: typeid,
	size: int,
	condition_number: f64 = 1.0,
	seed: [4]Blas_Int = {1, 2, 3, 4},
	allocator := context.allocator,
) -> Matrix(T) {
	A := make_matrix(T, size, size, .Symmetric, allocator)

	symmetry_type: MatrixSymmetry
	when T == complex64 || T == complex128 {
		symmetry_type = .Hermitian_Pos_Def
	} else {
		symmetry_type = .Positive_Definite
	}

	params := TestMatrixParams {
		rows             = size,
		cols             = size,
		distribution     = .Uniform,
		seed             = seed,
		symmetry         = symmetry_type,
		mode             = 1,
		condition_number = condition_number,
		max_element      = 1.0,
		packing          = .No_Packing,
	}

	when T == f32 || T == f64 {
		success, _ := m_generate_test_matrix_f32_f64(&A, params, allocator)
	} else when T == complex64 || T == complex128 {
		success, _ := m_generate_test_matrix_c64_c128(&A, params, allocator)
	}

	return A
}

// Generate banded matrix
generate_banded_matrix :: proc(
	$T: typeid,
	size: int,
	lower_bandwidth, upper_bandwidth: int,
	condition_number: f64 = 1.0,
	seed: [4]Blas_Int = {1, 2, 3, 4},
	allocator := context.allocator,
) -> Matrix(T) {
	A := make_banded_matrix(T, size, size, lower_bandwidth, upper_bandwidth, allocator)
	params := TestMatrixParams {
		rows             = size,
		cols             = size,
		distribution     = .UniformMinus1To1,
		seed             = seed,
		symmetry         = .None,
		mode             = 1,
		condition_number = condition_number,
		max_element      = 1.0,
		lower_bandwidth  = lower_bandwidth,
		upper_bandwidth  = upper_bandwidth,
		packing          = .Banded,
	}

	when T == f32 || T == f64 {
		success, _ := m_generate_test_matrix_f32_f64(&A, params, allocator)
	} else when T == complex64 || T == complex128 {
		success, _ := m_generate_test_matrix_c64_c128(&A, params, allocator)
	}

	return A
}

// Generate matrix with specific singular value distribution
generate_matrix_with_spectrum :: proc(
	$T: typeid,
	rows, cols: int,
	singular_values: []f64,
	seed: [4]Blas_Int = {1, 2, 3, 4},
	allocator := context.allocator,
) -> Matrix(T) {
	A := make_matrix(T, rows, cols, .General, allocator)
	params := TestMatrixParams {
		rows             = rows,
		cols             = cols,
		distribution     = .Uniform,
		seed             = seed,
		symmetry         = .None,
		singular_values  = singular_values,
		mode             = 6, // Use provided singular values
		condition_number = 1.0,
		max_element      = 1.0,
		packing          = .No_Packing,
	}

	when T == f32 || T == f64 {
		success, _ := m_generate_test_matrix_f32_f64(&A, params, allocator)
	} else when T == complex64 || T == complex128 {
		success, _ := m_generate_test_matrix_c64_c128(&A, params, allocator)
	}

	return A
}

// Generate Hilbert matrix (extremely ill-conditioned test matrix)
generate_hilbert_matrix :: proc(
	$T: typeid,
	size: int,
	allocator := context.allocator,
) -> Matrix(T) {
	A := make_matrix(T, size, size, .General, allocator)

	when T == f32 || T == f64 {
		_ = m_generate_hilbert_f32_f64(&A, allocator)
	} else when T == complex64 || T == complex128 {
		_ = m_generate_hilbert_c64_c128(&A, allocator)
	} else {
		panic("Unsupported type for Hilbert matrix generation")
	}

	return A
}

// Generate matrix with specified eigenvalues using DLATMT
generate_matrix_with_eigenvalues :: proc(
	$T: typeid,
	eigenvalues: []T,
	rank: int = -1,
	seed: [4]Blas_Int = {1, 2, 3, 5},
	allocator := context.allocator,
) -> Matrix(T) {
	n := len(eigenvalues)
	A := make_matrix(T, n, n, .General, allocator)

	when T == f32 || T == f64 {
		success, _ := m_generate_test_matrix_eigenvalues_f32_f64(
			&A,
			eigenvalues,
			rank,
			.UniformMinus1To1,
			seed,
			allocator,
		)
		if !success {
			panic("Failed to generate matrix with specified eigenvalues")
		}
	} else when T == complex64 || T == complex128 {
		success, _ := m_generate_test_matrix_eigenvalues_c64_c128(
			&A,
			eigenvalues,
			rank,
			.UniformMinus1To1,
			seed,
			allocator,
		)
		if !success {
			panic("Failed to generate matrix with specified eigenvalues")
		}
	} else {
		panic("Unsupported type for eigenvalue-controlled matrix generation")
	}

	return A
}

// Generate rank-deficient matrix with specified rank
generate_rank_deficient_matrix :: proc(
	$T: typeid,
	size: int,
	rank: int,
	seed: [4]Blas_Int = {1, 2, 3, 5},
	allocator := context.allocator,
) -> Matrix(T) {
	assert(rank > 0 && rank <= size, "Rank must be positive and <= size")

	// Create eigenvalues with appropriate number of zeros
	eigenvalues := make([]T, size, context.temp_allocator)
	defer delete(eigenvalues)

	// Set first 'rank' eigenvalues to non-zero, rest to zero
	for i in 0 ..< rank {
		when T == f32 {
			eigenvalues[i] = f32(1.0 / f32(i + 1))
		} else when T == f64 {
			eigenvalues[i] = f64(1.0 / f64(i + 1))
		} else when T == complex64 {
			eigenvalues[i] = complex64(1.0 / f32(i + 1), 0)
		} else when T == complex128 {
			eigenvalues[i] = complex128(1.0 / f64(i + 1), 0)
		}
	}
	// Rest are already zeros from make

	return generate_matrix_with_eigenvalues(T, eigenvalues, rank, seed, allocator)
}
