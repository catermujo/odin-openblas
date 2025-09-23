package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// GENERALIZED SYMMETRIC EIGENVALUE PROBLEMS
// ============================================================================

// Problem type for generalized eigenvalue problems
GeneralizedProblemType :: enum {
	AX_LBX = 1, // A*x = λ*B*x
	ABX_LX = 2, // A*B*x = λ*x
	BAX_LX = 3, // B*A*x = λ*x
}

// ============================================================================
// REDUCTION TO STANDARD FORM
// ============================================================================

// Reduction result
ReductionResult :: struct {
	reduction_complete:     bool, // True if reduction was successful
	b_is_positive_definite: bool, // True if B was positive definite
}

// Double precision reduction to standard form
dsygst :: proc(
	itype: GeneralizedProblemType,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64), // Matrix A (modified in place)
	b: Matrix(f64), // Cholesky factored B from dpotrf
) -> (
	result: ReductionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.dsygst_(&itype_int, uplo_cstring, &n_int, a.data, &lda, b.data, &ldb, &info_int, 1)

	info = Info(info_int)

	// Fill result
	result.reduction_complete = info == .OK
	result.b_is_positive_definite = info == .OK // Assumes B was already factored successfully

	return
}

// Single precision reduction to standard form
ssygst :: proc(
	itype: GeneralizedProblemType,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	b: Matrix(f32),
) -> (
	result: ReductionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.ssygst_(&itype_int, uplo_cstring, &n_int, a.data, &lda, b.data, &ldb, &info_int, 1)

	info = Info(info_int)

	// Fill result
	result.reduction_complete = info == .OK
	result.b_is_positive_definite = info == .OK

	return
}

sygst :: proc {
	dsygst,
	ssygst,
}

// ============================================================================
// GENERALIZED EIGENVALUE SOLVERS - QR ALGORITHM
// ============================================================================

// Generalized eigenvalue result
GeneralizedEigenResult :: struct($T: typeid) {
	eigenvalues:            []T, // Computed eigenvalues (sorted)
	eigenvectors:           Matrix(T), // Eigenvector matrix (if requested)
	b_is_positive_definite: bool, // True if B was positive definite
	min_eigenvalue:         f64, // Smallest eigenvalue
	max_eigenvalue:         f64, // Largest eigenvalue
	condition_number:       f64, // max|λ|/min|λ|
	all_positive:           bool, // True if all eigenvalues > 0
}

// Double precision generalized eigenvalue solver (QR)
dsygv :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64), // Matrix A (eigenvectors on output if jobz == EIGENVECTORS)
	b: Matrix(f64), // Matrix B (Cholesky factor on output)
	w: []f64 = nil, // Eigenvalues (size n)
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenResult(f64),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	lwork_int := Blas_Int(lwork)
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Workspace query
	if lwork == -1 {
		work_query: f64

		lapack.dsygv_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			raw_data(w),
			&work_query,
			&lwork_int,
			&info_int,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size = dsygv(itype, jobz, uplo, n, a, b, w, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsygv_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	if info == .OK && n > 0 {
		result.eigenvalues = w
		if jobz == .EIGENVECTORS {
			result.eigenvectors = a
		}
		result.b_is_positive_definite = true

		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[n - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	} else if info > n {
		// B is not positive definite
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	return
}

// Single precision generalized eigenvalue solver (QR)
ssygv :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	b: Matrix(f32),
	w: []f32 = nil,
	work: []f32 = nil,
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenResult(f32),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	lwork_int := Blas_Int(lwork)
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Workspace query
	if lwork == -1 {
		work_query: f32

		lapack.ssygv_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			raw_data(w),
			&work_query,
			&lwork_int,
			&info_int,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size = ssygv(itype, jobz, uplo, n, a, b, w, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssygv_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	if info == .OK && n > 0 {
		result.eigenvalues = w
		if jobz == .EIGENVECTORS {
			result.eigenvectors = a
		}
		result.b_is_positive_definite = true

		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	} else if info > n {
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	return
}

// Double precision 2-stage generalized eigenvalue solver
dsygv_2stage :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	b: Matrix(f64),
	w: []f64 = nil,
	work: []f64 = nil,
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenResult(f64),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	lwork_int := Blas_Int(lwork)
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Workspace query
	if lwork == -1 {
		work_query: f64

		lapack.dsygv_2stage_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			raw_data(w),
			&work_query,
			&lwork_int,
			&info_int,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size = dsygv_2stage(itype, jobz, uplo, n, a, b, w, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsygv_2stage_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	if info == .OK && n > 0 {
		result.eigenvalues = w
		if jobz == .EIGENVECTORS {
			result.eigenvectors = a
		}
		result.b_is_positive_definite = true

		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[n - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	} else if info > n {
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	return
}

// Single precision 2-stage generalized eigenvalue solver
ssygv_2stage :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	b: Matrix(f32),
	w: []f32 = nil,
	work: []f32 = nil,
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenResult(f32),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	lwork_int := Blas_Int(lwork)
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Workspace query
	if lwork == -1 {
		work_query: f32

		lapack.ssygv_2stage_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			raw_data(w),
			&work_query,
			&lwork_int,
			&info_int,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size = ssygv_2stage(itype, jobz, uplo, n, a, b, w, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssygv_2stage_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	if info == .OK && n > 0 {
		result.eigenvalues = w
		if jobz == .EIGENVECTORS {
			result.eigenvectors = a
		}
		result.b_is_positive_definite = true

		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	} else if info > n {
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	return
}

sygv :: proc {
	dsygv,
	ssygv,
}
sygv_2stage :: proc {
	dsygv_2stage,
	ssygv_2stage,
}

// ============================================================================
// GENERALIZED EIGENVALUE SOLVERS - DIVIDE AND CONQUER
// ============================================================================

// Double precision generalized eigenvalue solver (divide-and-conquer)
dsygvd :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	b: Matrix(f64),
	w: []f64 = nil,
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dsygvd_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			raw_data(w),
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
			1,
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
		return result, Info(info_int), work_size, iwork_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size, _ = dsygvd(itype, jobz, uplo, n, a, b, w, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = dsygvd(itype, jobz, uplo, n, a, b, w, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dsygvd_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	if info == .OK && n > 0 {
		result.eigenvalues = w
		if jobz == .EIGENVECTORS {
			result.eigenvectors = a
		}
		result.b_is_positive_definite = true

		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[n - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	} else if info > n {
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision generalized eigenvalue solver (divide-and-conquer)
ssygvd :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	b: Matrix(f32),
	w: []f32 = nil,
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)
	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.ssygvd_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			raw_data(w),
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
			1,
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
		return result, Info(info_int), work_size, iwork_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size, _ = ssygvd(itype, jobz, uplo, n, a, b, w, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = ssygvd(itype, jobz, uplo, n, a, b, w, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssygvd_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		raw_data(w),
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	if info == .OK && n > 0 {
		result.eigenvalues = w
		if jobz == .EIGENVECTORS {
			result.eigenvectors = a
		}
		result.b_is_positive_definite = true

		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	} else if info > n {
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

sygvd :: proc {
	dsygvd,
	ssygvd,
}

// ============================================================================
// GENERALIZED EIGENVALUE SOLVERS - BISECTION AND INVERSE ITERATION
// ============================================================================

// Selective generalized eigenvalue result
SelectiveGeneralizedEigenResult :: struct($T: typeid) {
	eigenvalues:            []T, // Computed eigenvalues
	eigenvectors:           Matrix(T), // Eigenvector matrix (if requested)
	num_found:              int, // Number of eigenvalues found
	failed_indices:         []Blas_Int, // Indices of failed eigenvectors
	num_failures:           int, // Number of failed convergences
	all_converged:          bool, // True if all eigenvectors converged
	b_is_positive_definite: bool, // True if B was positive definite
	min_eigenvalue:         f64, // Smallest eigenvalue
	max_eigenvalue:         f64, // Largest eigenvalue
	condition_number:       f64, // max|λ|/min|λ|
}

// Double precision generalized eigenvalue solver (bisection)
dsygvx :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	b: Matrix(f64),
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (size 5*n)
	ifail: []Blas_Int = nil, // Failed indices (size n)
	allocator := context.allocator,
) -> (
	result: SelectiveGeneralizedEigenResult(f64),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	range_char: u8
	switch range {
	case .ALL:
		range_char = 'A'
	case .VALUE:
		range_char = 'V'
	case .INDEX:
		range_char = 'I'
	}
	range_cstring := cstring(&range_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Allocate integer workspace if not provided
	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array if not provided
	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, n, allocator)
	}

	// Workspace query
	if lwork == -1 {
		work_query: f64

		lapack.dsygvx_(
			&itype_int,
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_int,
			raw_data(iwork),
			raw_data(ifail),
			&info_int,
			1,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size = dsygvx(
			itype,
			jobz,
			range,
			uplo,
			n,
			a,
			b,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			lwork = -1,
		)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsygvx_(
		&itype_int,
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]

		// Analyze eigenvalues
		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[result.num_found - 1]

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[result.num_found - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	// Check for failures
	if info <= n {
		result.b_is_positive_definite = true
		if info > 0 && jobz == .EIGENVECTORS {
			result.num_failures = int(info)
			result.failed_indices = ifail[:result.num_failures]
			result.all_converged = false
		} else {
			result.all_converged = true
		}
	} else {
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	return
}

// Single precision generalized eigenvalue solver (bisection)
ssygvx :: proc(
	itype: GeneralizedProblemType,
	jobz: JobzOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	b: Matrix(f32),
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SelectiveGeneralizedEigenResult(f32),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix A too small")
	assert(b.rows >= n && b.cols >= n, "Matrix B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	range_char: u8
	switch range {
	case .ALL:
		range_char = 'A'
	case .VALUE:
		range_char = 'V'
	case .INDEX:
		range_char = 'I'
	}
	range_cstring := cstring(&range_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	ldb := Blas_Int(b.stride)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Allocate integer workspace if not provided
	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array if not provided
	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, n, allocator)
	}

	// Workspace query
	if lwork == -1 {
		work_query: f32

		lapack.ssygvx_(
			&itype_int,
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			b.data,
			&ldb,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_int,
			raw_data(iwork),
			raw_data(ifail),
			&info_int,
			1,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size = ssygvx(
			itype,
			jobz,
			range,
			uplo,
			n,
			a,
			b,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			lwork = -1,
		)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssygvx_(
		&itype_int,
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		b.data,
		&ldb,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
		1,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]

		// Analyze eigenvalues
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[result.num_found - 1])

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[result.num_found - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	// Check for failures
	if info <= n {
		result.b_is_positive_definite = true
		if info > 0 && jobz == .EIGENVECTORS {
			result.num_failures = int(info)
			result.failed_indices = ifail[:result.num_failures]
			result.all_converged = false
		} else {
			result.all_converged = true
		}
	} else {
		result.b_is_positive_definite = false
	}

	work_size = len(work)
	return
}

sygvx :: proc {
	dsygvx,
	ssygvx,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Solve generalized eigenvalue problem A*x = λ*B*x
solve_generalized_eigenvalue_problem :: proc(
	a: Matrix($T),
	b: Matrix($T),
	itype := GeneralizedProblemType.AX_LBX,
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	b_positive_definite: bool,
	info: Info,
) {
	n := a.rows

	// Make copies since matrices are modified
	a_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&a_copy, a)
	b_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&b_copy, b)

	jobz := compute_vectors ? JobzOption.EIGENVECTORS : JobzOption.NO_VECTORS

	when T == f64 {
		result, info_val, _ := dsygv(itype, jobz, uplo, n, a_copy, b_copy, allocator = allocator)
		return result.eigenvalues, result.eigenvectors, result.b_is_positive_definite, info_val
	} else when T == f32 {
		result, info_val, _ := ssygv(itype, jobz, uplo, n, a_copy, b_copy, allocator = allocator)
		return result.eigenvalues, result.eigenvectors, result.b_is_positive_definite, info_val
	} else {
		#panic("Unsupported type for generalized eigenvalue problem")
	}
}

// Fast generalized eigenvalue problem using divide-and-conquer
fast_generalized_eigenvalue_problem :: proc(
	a: Matrix($T),
	b: Matrix($T),
	itype := GeneralizedProblemType.AX_LBX,
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	b_positive_definite: bool,
	info: Info,
) {
	n := a.rows

	// Make copies since matrices are modified
	a_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&a_copy, a)
	b_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&b_copy, b)

	jobz := compute_vectors ? JobzOption.EIGENVECTORS : JobzOption.NO_VECTORS

	when T == f64 {
		result, info_val, _, _ := dsygvd(
			itype,
			jobz,
			uplo,
			n,
			a_copy,
			b_copy,
			allocator = allocator,
		)
		return result.eigenvalues, result.eigenvectors, result.b_is_positive_definite, info_val
	} else when T == f32 {
		result, info_val, _, _ := ssygvd(
			itype,
			jobz,
			uplo,
			n,
			a_copy,
			b_copy,
			allocator = allocator,
		)
		return result.eigenvalues, result.eigenvectors, result.b_is_positive_definite, info_val
	} else {
		#panic("Unsupported type for fast generalized eigenvalue problem")
	}
}

// Selective generalized eigenvalue problem
selective_generalized_eigenvalue_problem :: proc(
	a: Matrix($T),
	b: Matrix($T),
	itype := GeneralizedProblemType.AX_LBX,
	range := EigenRangeOption.ALL,
	vl: T = 0,
	vu: T = 0,
	il: int = 0,
	iu: int = 0,
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	abstol: T = 0,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	num_found: int,
	b_positive_definite: bool,
	all_converged: bool,
	info: Info,
) {
	n := a.rows

	// Make copies since matrices are modified
	a_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&a_copy, a)
	b_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&b_copy, b)

	jobz := compute_vectors ? JobzOption.EIGENVECTORS : JobzOption.NO_VECTORS

	// Create eigenvector matrix if needed
	z: Matrix(T)
	if compute_vectors {
		z = create_matrix(T, n, n, allocator)
	}

	when T == f64 {
		result, info_val, _ := dsygvx(
			itype,
			jobz,
			range,
			uplo,
			n,
			a_copy,
			b_copy,
			vl,
			vu,
			il,
			iu,
			abstol,
			z = z,
			allocator = allocator,
		)
		return result.eigenvalues,
			result.eigenvectors,
			result.num_found,
			result.b_is_positive_definite,
			result.all_converged,
			info_val
	} else when T == f32 {
		result, info_val, _ := ssygvx(
			itype,
			jobz,
			range,
			uplo,
			n,
			a_copy,
			b_copy,
			vl,
			vu,
			il,
			iu,
			abstol,
			z = z,
			allocator = allocator,
		)
		return result.eigenvalues,
			result.eigenvectors,
			result.num_found,
			result.b_is_positive_definite,
			result.all_converged,
			info_val
	} else {
		#panic("Unsupported type for selective generalized eigenvalue problem")
	}
}
