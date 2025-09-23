package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - QR ALGORITHM
// ============================================================================

// Symmetric eigenvalue result
SymmetricEigenResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
	all_positive:     bool, // True if all eigenvalues > 0
}

// Double precision symmetric eigenvalue solver (QR algorithm)
dsyev :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64), // Input matrix (destroyed, eigenvectors on output if jobz == EIGENVECTORS)
	w: []f64 = nil, // Eigenvalues (size n)
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f64),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsyev_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size = dsyev(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsyev_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[n - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	return
}

// Single precision symmetric eigenvalue solver (QR algorithm)
ssyev :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	w: []f32 = nil,
	work: []f32 = nil,
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f32),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssyev_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size = ssyev(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssyev_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	return
}

// Double precision 2-stage symmetric eigenvalue solver
dsyev_2stage :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	w: []f64 = nil,
	work: []f64 = nil,
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f64),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsyev_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size = dsyev_2stage(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsyev_2stage_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[n - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	return
}

// Single precision 2-stage symmetric eigenvalue solver
ssyev_2stage :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	w: []f32 = nil,
	work: []f32 = nil,
	lwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f32),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssyev_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size = ssyev_2stage(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssyev_2stage_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	return
}

syev :: proc {
	dsyev,
	ssyev,
}
syev_2stage :: proc {
	dsyev_2stage,
	ssyev_2stage,
}

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - DIVIDE AND CONQUER
// ============================================================================

// Double precision symmetric eigenvalue solver (divide-and-conquer)
dsyevd :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	w: []f64 = nil,
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsyevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size, _ = dsyevd(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = dsyevd(jobz, uplo, n, a, w, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dsyevd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[n - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision symmetric eigenvalue solver (divide-and-conquer)
ssyevd :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	w: []f32 = nil,
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssyevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size, _ = ssyevd(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = ssyevd(jobz, uplo, n, a, w, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssyevd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Double precision 2-stage symmetric eigenvalue solver (divide-and-conquer)
dsyevd_2stage :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	w: []f64 = nil,
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.dsyevd_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size, _ = dsyevd_2stage(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = dsyevd_2stage(jobz, uplo, n, a, w, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dsyevd_2stage_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[n - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision 2-stage symmetric eigenvalue solver (divide-and-conquer)
ssyevd_2stage :: proc(
	jobz: JobzOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	w: []f32 = nil,
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SymmetricEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
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

		lapack.ssyevd_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
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
		_, _, work_size, _ = ssyevd_2stage(jobz, uplo, n, a, w, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = ssyevd_2stage(jobz, uplo, n, a, w, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssyevd_2stage_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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

		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

syevd :: proc {
	dsyevd,
	ssyevd,
}
syevd_2stage :: proc {
	dsyevd_2stage,
	ssyevd_2stage,
}

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - MRRR
// ============================================================================

// Selective symmetric eigenvalue result
SelectiveSymmetricEigenResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	num_found:        int, // Number of eigenvalues found
	support:          []Blas_Int, // Support arrays for eigenvectors
	all_converged:    bool, // True if all eigenvectors converged
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
}

// Double precision symmetric eigenvalue solver (MRRR)
dsyevr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	isuppz: []Blas_Int = nil, // Support arrays (size 2*max(1,m))
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (query with liwork=-1)
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SelectiveSymmetricEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

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
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
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

	// Allocate support array if not provided
	max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
	allocated_isuppz := isuppz == nil
	if allocated_isuppz && jobz == .EIGENVECTORS {
		isuppz = make([]Blas_Int, 2 * max(1, max_m), allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dsyevr_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(isuppz) if isuppz != nil else nil,
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
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
		_, _, work_size, _ = dsyevr(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = dsyevr(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dsyevr_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(isuppz) if isuppz != nil else nil,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
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

		if jobz == .EIGENVECTORS && isuppz != nil {
			result.support = isuppz[:2 * result.num_found]
		}

		// Analyze eigenvalues
		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[result.num_found - 1]
		result.all_converged = info == .OK

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[result.num_found - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision symmetric eigenvalue solver (MRRR)
ssyevr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	isuppz: []Blas_Int = nil,
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SelectiveSymmetricEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

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
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
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

	// Allocate support array if not provided
	max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
	allocated_isuppz := isuppz == nil
	if allocated_isuppz && jobz == .EIGENVECTORS {
		isuppz = make([]Blas_Int, 2 * max(1, max_m), allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.ssyevr_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(isuppz) if isuppz != nil else nil,
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
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
		_, _, work_size, _ = ssyevr(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = ssyevr(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssyevr_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(isuppz) if isuppz != nil else nil,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
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

		if jobz == .EIGENVECTORS && isuppz != nil {
			result.support = isuppz[:2 * result.num_found]
		}

		// Analyze eigenvalues
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[result.num_found - 1])
		result.all_converged = info == .OK

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[result.num_found - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Double precision 2-stage symmetric eigenvalue solver (MRRR)
dsyevr_2stage :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f64 = 0,
	w: []f64 = nil,
	z: Matrix(f64) = {},
	isuppz: []Blas_Int = nil,
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SelectiveSymmetricEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

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
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
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

	// Allocate support array if not provided
	max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
	allocated_isuppz := isuppz == nil
	if allocated_isuppz && jobz == .EIGENVECTORS {
		isuppz = make([]Blas_Int, 2 * max(1, max_m), allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dsyevr_2stage_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(isuppz) if isuppz != nil else nil,
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
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
		_, _, work_size, _ = dsyevr_2stage(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = dsyevr_2stage(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dsyevr_2stage_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(isuppz) if isuppz != nil else nil,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
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

		if jobz == .EIGENVECTORS && isuppz != nil {
			result.support = isuppz[:2 * result.num_found]
		}

		// Analyze eigenvalues
		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[result.num_found - 1]
		result.all_converged = info == .OK

		if abs(w[0]) > machine_epsilon(f64) {
			result.condition_number = abs(w[result.num_found - 1] / w[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision 2-stage symmetric eigenvalue solver (MRRR)
ssyevr_2stage :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	isuppz: []Blas_Int = nil,
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: SelectiveSymmetricEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

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
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
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

	// Allocate support array if not provided
	max_m := range == .ALL ? n : (range == .INDEX ? iu - il + 1 : n)
	allocated_isuppz := isuppz == nil
	if allocated_isuppz && jobz == .EIGENVECTORS {
		isuppz = make([]Blas_Int, 2 * max(1, max_m), allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.ssyevr_2stage_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			a.data,
			&lda,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			raw_data(isuppz) if isuppz != nil else nil,
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
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
		_, _, work_size, _ = ssyevr_2stage(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = ssyevr_2stage(
			jobz,
			range,
			uplo,
			n,
			a,
			vl,
			vu,
			il,
			iu,
			abstol,
			w,
			z,
			isuppz,
			lwork = -1,
		)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssyevr_2stage_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(isuppz) if isuppz != nil else nil,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
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

		if jobz == .EIGENVECTORS && isuppz != nil {
			result.support = isuppz[:2 * result.num_found]
		}

		// Analyze eigenvalues
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[result.num_found - 1])
		result.all_converged = info == .OK

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[result.num_found - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

syevr :: proc {
	dsyevr,
	ssyevr,
}
syevr_2stage :: proc {
	dsyevr_2stage,
	ssyevr_2stage,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Simple symmetric eigendecomposition
symmetric_eigendecomposition :: proc(
	a: Matrix($T),
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	n := a.rows

	// Make a copy since the matrix is destroyed
	a_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&a_copy, a)

	jobz := compute_vectors ? JobzOption.EIGENVECTORS : JobzOption.NO_VECTORS

	when T == f64 {
		result, info_val, _ := dsyev(jobz, uplo, n, a_copy, allocator = allocator)
		return result.eigenvalues, result.eigenvectors, info_val
	} else when T == f32 {
		result, info_val, _ := ssyev(jobz, uplo, n, a_copy, allocator = allocator)
		return result.eigenvalues, result.eigenvectors, info_val
	} else {
		#panic("Unsupported type for symmetric eigendecomposition")
	}
}

// Fast symmetric eigendecomposition using divide-and-conquer
fast_symmetric_eigendecomposition :: proc(
	a: Matrix($T),
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	n := a.rows

	// Make a copy since the matrix is destroyed
	a_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&a_copy, a)

	jobz := compute_vectors ? JobzOption.EIGENVECTORS : JobzOption.NO_VECTORS

	when T == f64 {
		result, info_val, _, _ := dsyevd(jobz, uplo, n, a_copy, allocator = allocator)
		return result.eigenvalues, result.eigenvectors, info_val
	} else when T == f32 {
		result, info_val, _, _ := ssyevd(jobz, uplo, n, a_copy, allocator = allocator)
		return result.eigenvalues, result.eigenvectors, info_val
	} else {
		#panic("Unsupported type for fast symmetric eigendecomposition")
	}
}

// Selective symmetric eigendecomposition using MRRR
selective_symmetric_eigendecomposition :: proc(
	a: Matrix($T),
	range := EigenRangeOption.ALL,
	vl: T = 0,
	vu: T = 0,
	il: int = 0,
	iu: int = 0,
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	num_found: int,
	info: Info,
) {
	n := a.rows

	// Make a copy since the matrix may be modified
	a_copy := create_matrix(T, n, n, allocator)
	matrix_copy(&a_copy, a)

	jobz := compute_vectors ? JobzOption.EIGENVECTORS : JobzOption.NO_VECTORS

	// Create eigenvector matrix if needed
	z: Matrix(T)
	if compute_vectors {
		z = create_matrix(T, n, n, allocator)
	}

	when T == f64 {
		result, info_val, _, _ := dsyevr(
			jobz,
			range,
			uplo,
			n,
			a_copy,
			vl,
			vu,
			il,
			iu,
			z = z,
			allocator = allocator,
		)
		return result.eigenvalues, result.eigenvectors, result.num_found, info_val
	} else when T == f32 {
		result, info_val, _, _ := ssyevr(
			jobz,
			range,
			uplo,
			n,
			a_copy,
			vl,
			vu,
			il,
			iu,
			z = z,
			allocator = allocator,
		)
		return result.eigenvalues, result.eigenvectors, result.num_found, info_val
	} else {
		#panic("Unsupported type for selective symmetric eigendecomposition")
	}
}
