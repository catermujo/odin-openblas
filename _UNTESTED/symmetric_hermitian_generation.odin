package openblas

import lapack "./f77"
import "base:builtin"
import "core:mem"

// ===================================================================================
// SYMMETRIC AND HERMITIAN RANDOM MATRIX GENERATION
// ===================================================================================

// Generate random Hermitian matrix proc group
m_generate_random_hermitian :: proc {
	m_generate_random_hermitian_c64,
	m_generate_random_hermitian_c128,
}

// Generate random symmetric matrix proc group
m_generate_random_symmetric :: proc {
	m_generate_random_symmetric_c64,
	m_generate_random_symmetric_f64,
	m_generate_random_symmetric_f32,
	m_generate_random_symmetric_c128,
}

// Generate random Hermitian matrix (c64)
m_generate_random_hermitian_c64 :: proc(A: ^Matrix(complex64), D: []f32, seed: []int = nil, allocator := context.allocator) -> (success: bool, info: Info) {
	// Validate matrix
	if A.format != .Hermitian {
		panic("Matrix must be in Hermitian format")
	}
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix size")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	uplo_c := "U"
	if A.storage.hermitian.uplo != nil {
		uplo_c = A.storage.hermitian.uplo
	}

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work := make([]complex64, 2 * n, context.temp_allocator)

	info_val: Info
	lapack.claghe_(
		&n,
		&n, // k parameter (number of eigenvalues)
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random Hermitian matrix (c128)
m_generate_random_hermitian_c128 :: proc(A: ^Matrix(complex128), D: []f64, seed: []int = nil, allocator := context.allocator) -> (success: bool, info: Info) {
	// Validate matrix
	if A.format != .Hermitian {
		panic("Matrix must be in Hermitian format")
	}
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix size")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	uplo_c := "U"
	if A.storage.hermitian.uplo != nil {
		uplo_c = A.storage.hermitian.uplo
	}

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work := make([]complex128, 2 * n, context.temp_allocator)

	info_val: Info
	lapack.zlaghe_(
		&n,
		&n, // k parameter (number of eigenvalues)
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random symmetric matrix (c64)
m_generate_random_symmetric_c64 :: proc(A: ^Matrix(complex64), D: []f32, seed: []int = nil, allocator := context.allocator) -> (success: bool, info: Info) {
	// Validate matrix
	if A.format != .Symmetric {
		panic("Matrix must be in symmetric format")
	}
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix size")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	uplo_c := "U"
	if A.storage.symmetric.uplo != nil {
		uplo_c = A.storage.symmetric.uplo
	}

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work := make([]complex64, 2 * n, context.temp_allocator)

	info_val: Info
	lapack.clagsy_(
		&n,
		&n, // k parameter (number of eigenvalues)
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random symmetric matrix (f64)
m_generate_random_symmetric_f64 :: proc(A: ^Matrix(f64), D: []f64, seed: []int = nil, allocator := context.allocator) -> (success: bool, info: Info) {
	// Validate matrix
	if A.format != .Symmetric {
		panic("Matrix must be in symmetric format")
	}
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix size")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	uplo_c := "U"
	if A.storage.symmetric.uplo != nil {
		uplo_c = A.storage.symmetric.uplo
	}

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work := make([]f64, 2 * n, context.temp_allocator)

	info_val: Info
	lapack.dlagsy_(
		&n,
		&n, // k parameter (number of eigenvalues)
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random symmetric matrix (f32)
m_generate_random_symmetric_f32 :: proc(A: ^Matrix(f32), D: []f32, seed: []int = nil, allocator := context.allocator) -> (success: bool, info: Info) {
	// Validate matrix
	if A.format != .Symmetric {
		panic("Matrix must be in symmetric format")
	}
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix size")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	uplo_c := "U"
	if A.storage.symmetric.uplo != nil {
		uplo_c = A.storage.symmetric.uplo
	}

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work := make([]f32, 2 * n, context.temp_allocator)

	info_val: Info
	lapack.slagsy_(
		&n,
		&n, // k parameter (number of eigenvalues)
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}

// Generate random symmetric matrix (c128)
m_generate_random_symmetric_c128 :: proc(A: ^Matrix(complex128), D: []f64, seed: []int = nil, allocator := context.allocator) -> (success: bool, info: Info) {
	// Validate matrix
	if A.format != .Symmetric {
		panic("Matrix must be in symmetric format")
	}
	if A.rows != A.cols {
		panic("Matrix must be square")
	}
	if len(D) != A.rows {
		panic("Diagonal scaling array must match matrix size")
	}

	n := Blas_Int(A.rows)
	lda := Blas_Int(A.ld)
	uplo_c := "U"
	if A.storage.symmetric.uplo != nil {
		uplo_c = A.storage.symmetric.uplo
	}

	// Setup random seed
	iseed: []Blas_Int
	if seed != nil && len(seed) == 4 {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		for i in 0 ..< 4 {
			iseed[i] = Blas_Int(seed[i])
		}
	} else {
		iseed = make([]Blas_Int, 4, context.temp_allocator)
		iseed[0] = 1
		iseed[1] = 2
		iseed[2] = 3
		iseed[3] = 4
	}

	// Allocate workspace
	work := make([]complex128, 2 * n, context.temp_allocator)

	info_val: Info
	lapack.zlagsy_(
		&n,
		&n, // k parameter (number of eigenvalues)
		raw_data(D),
		raw_data(A.data),
		&lda,
		raw_data(iseed),
		raw_data(work),
		&info_val,
	)

	return info_val == 0, info_val
}
