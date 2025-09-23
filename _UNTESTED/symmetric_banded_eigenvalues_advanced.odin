package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC BANDED EIGENVALUE - DIVIDE AND CONQUER
// ============================================================================
// Uses divide-and-conquer algorithm for faster computation on large matrices

// Double precision divide-and-conquer banded eigenvalue
dsbevd :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int, // Number of superdiagonals/subdiagonals
	ab: Matrix(f64), // Band matrix (modified on output)
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1, // Workspace size (-1 for query)
	iwork: []Blas_Int = nil, // Integer workspace (query with liwork=-1)
	liwork: int = -1, // Integer workspace size (-1 for query)
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.dsbevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			nil, // w
			nil, // z
			&ldz,
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

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}
	result.eigenvalues = w

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_iwork := iwork == nil

	if allocated_work || allocated_iwork {
		// Query for optimal workspace
		work_query: f64
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.dsbevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			raw_data(w),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_query,
			&iwork_query,
			&liwork_query,
			&info_int,
			1,
			1,
		)

		if allocated_work {
			lwork = int(work_query)
			work = make([]f64, lwork, allocator)
		}
		if allocated_iwork {
			liwork = int(iwork_query)
			iwork = make([]Blas_Int, liwork, allocator)
		}
	}
	defer {
		if allocated_work do delete(work)
		if allocated_iwork do delete(iwork)
	}

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dsbevd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]

		epsilon :=
			machine_epsilon(f64) * max(abs(result.min_eigenvalue), abs(result.max_eigenvalue))

		for eigenval in w {
			if eigenval < -epsilon {
				result.num_negative += 1
			} else if eigenval > epsilon {
				result.num_positive += 1
			} else {
				result.num_zero += 1
			}
		}

		result.all_positive = result.num_negative == 0 && result.num_zero == 0
		result.all_non_negative = result.num_negative == 0

		if abs(result.min_eigenvalue) > epsilon {
			result.condition_number = abs(result.max_eigenvalue / result.min_eigenvalue)
		} else if abs(result.max_eigenvalue) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision divide-and-conquer banded eigenvalue
ssbevd :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f32),
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.ssbevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			nil,
			nil,
			&ldz,
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

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}
	result.eigenvalues = w

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_iwork := iwork == nil

	if allocated_work || allocated_iwork {
		// Query for optimal workspace
		work_query: f32
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.ssbevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			raw_data(w),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_query,
			&iwork_query,
			&liwork_query,
			&info_int,
			1,
			1,
		)

		if allocated_work {
			lwork = int(work_query)
			work = make([]f32, lwork, allocator)
		}
		if allocated_iwork {
			liwork = int(iwork_query)
			iwork = make([]Blas_Int, liwork, allocator)
		}
	}
	defer {
		if allocated_work do delete(work)
		if allocated_iwork do delete(iwork)
	}

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssbevd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])

		epsilon := machine_epsilon(f32) * max(abs(w[0]), abs(w[n - 1]))

		for eigenval in w {
			if eigenval < -epsilon {
				result.num_negative += 1
			} else if eigenval > epsilon {
				result.num_positive += 1
			} else {
				result.num_zero += 1
			}
		}

		result.all_positive = result.num_negative == 0 && result.num_zero == 0
		result.all_non_negative = result.num_negative == 0

		if abs(w[0]) > epsilon {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else if abs(w[n - 1]) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// 2-stage versions with divide-and-conquer
dsbevd_2stage :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f64),
	w: []f64 = nil,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	// Implementation similar to dsbevd but calls dsbevd_2stage_
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.dsbevd_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			nil,
			nil,
			&ldz,
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

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}
	result.eigenvalues = w

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_iwork := iwork == nil

	if allocated_work || allocated_iwork {
		work_query: f64
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.dsbevd_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			raw_data(w),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_query,
			&iwork_query,
			&liwork_query,
			&info_int,
			1,
			1,
		)

		if allocated_work {
			lwork = int(work_query)
			work = make([]f64, lwork, allocator)
		}
		if allocated_iwork {
			liwork = int(iwork_query)
			iwork = make([]Blas_Int, liwork, allocator)
		}
	}
	defer {
		if allocated_work do delete(work)
		if allocated_iwork do delete(iwork)
	}

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dsbevd_2stage_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1]

		epsilon :=
			machine_epsilon(f64) * max(abs(result.min_eigenvalue), abs(result.max_eigenvalue))

		for eigenval in w {
			if eigenval < -epsilon {
				result.num_negative += 1
			} else if eigenval > epsilon {
				result.num_positive += 1
			} else {
				result.num_zero += 1
			}
		}

		result.all_positive = result.num_negative == 0 && result.num_zero == 0
		result.all_non_negative = result.num_negative == 0

		if abs(result.min_eigenvalue) > epsilon {
			result.condition_number = abs(result.max_eigenvalue / result.min_eigenvalue)
		} else if abs(result.max_eigenvalue) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

ssbevd_2stage :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f32),
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	// Similar implementation for single precision
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.ssbevd_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			nil,
			nil,
			&ldz,
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

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}
	result.eigenvalues = w

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	allocated_iwork := iwork == nil

	if allocated_work || allocated_iwork {
		work_query: f32
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.ssbevd_2stage_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			raw_data(w),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_query,
			&iwork_query,
			&liwork_query,
			&info_int,
			1,
			1,
		)

		if allocated_work {
			lwork = int(work_query)
			work = make([]f32, lwork, allocator)
		}
		if allocated_iwork {
			liwork = int(iwork_query)
			iwork = make([]Blas_Int, liwork, allocator)
		}
	}
	defer {
		if allocated_work do delete(work)
		if allocated_iwork do delete(iwork)
	}

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.ssbevd_2stage_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1])

		epsilon := machine_epsilon(f32) * max(abs(w[0]), abs(w[n - 1]))

		for eigenval in w {
			if eigenval < -epsilon {
				result.num_negative += 1
			} else if eigenval > epsilon {
				result.num_positive += 1
			} else {
				result.num_zero += 1
			}
		}

		result.all_positive = result.num_negative == 0 && result.num_zero == 0
		result.all_non_negative = result.num_negative == 0

		if abs(w[0]) > epsilon {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else if abs(w[n - 1]) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Proc groups
sbevd :: proc {
	dsbevd,
	ssbevd,
}
sbevd_2stage :: proc {
	dsbevd_2stage,
	ssbevd_2stage,
}

// ============================================================================
// SYMMETRIC BANDED EIGENVALUE - SELECTIVE COMPUTATION
// ============================================================================
// Computes selected eigenvalues and optionally eigenvectors

// Range option for eigenvalue selection
EigenRangeOption :: enum {
	ALL, // 'A' - All eigenvalues
	VALUE, // 'V' - Eigenvalues in range [vl, vu]
	INDEX, // 'I' - Eigenvalues with indices il to iu
}

// Selective eigenvalue result
SelectiveEigenResult :: struct($T: typeid) {
	eigenvalues:    []T, // Selected eigenvalues
	eigenvectors:   Matrix(T), // Selected eigenvectors
	num_found:      int, // Number of eigenvalues found
	failed_indices: []Blas_Int, // Indices of eigenvectors that failed to converge
	num_failures:   int, // Number of convergence failures
	all_converged:  bool, // True if all eigenvectors converged
}

// Double precision selective banded eigenvalue
dsbevx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f64), // Band matrix (preserved on output)
	q: Matrix(f64) = {}, // Orthogonal matrix from reduction (if computed)
	vl: f64 = 0, // Lower bound for eigenvalues (if range == VALUE)
	vu: f64 = 0, // Upper bound for eigenvalues (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance for convergence
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors
	work: []f64 = nil, // Workspace (size 7*n)
	iwork: []Blas_Int = nil, // Integer workspace (size 5*n)
	allocator := context.allocator,
) -> (
	result: SelectiveEigenResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
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
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	ldq := Blas_Int(max(1, n))
	q_ptr: ^f64 = nil
	if q.data != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
	}

	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol

	m: Blas_Int // Number of eigenvalues found
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Allocate eigenvector storage
	ldz := Blas_Int(max(1, n))
	z_ptr: ^f64 = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 7 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array
	ifail := make([]Blas_Int, n, allocator)

	// Call LAPACK
	lapack.dsbevx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		q_ptr,
		&ldq,
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
	}
	if jobz == .VALUES_VECTORS {
		result.eigenvectors = z
	}

	// Check for convergence failures
	result.failed_indices = ifail
	for i in 0 ..< result.num_found {
		if ifail[i] > 0 {
			result.num_failures += 1
		}
	}
	result.all_converged = result.num_failures == 0

	return
}

// Single precision selective banded eigenvalue
ssbevx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f32),
	q: Matrix(f32) = {},
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SelectiveEigenResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
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
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	ldq := Blas_Int(max(1, n))
	q_ptr: ^f32 = nil
	if q.data != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
	}

	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol

	m: Blas_Int
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Allocate eigenvector storage
	ldz := Blas_Int(max(1, n))
	z_ptr: ^f32 = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 7 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array
	ifail := make([]Blas_Int, n, allocator)

	// Call LAPACK
	lapack.ssbevx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		q_ptr,
		&ldq,
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
	}
	if jobz == .VALUES_VECTORS {
		result.eigenvectors = z
	}

	// Check for convergence failures
	result.failed_indices = ifail
	for i in 0 ..< result.num_found {
		if ifail[i] > 0 {
			result.num_failures += 1
		}
	}
	result.all_converged = result.num_failures == 0

	return
}

// 2-stage selective eigenvalue computation
dsbevx_2stage :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f64),
	q: Matrix(f64) = {},
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f64 = 0,
	w: []f64 = nil,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SelectiveEigenResult(f64),
	info: Info,
	work_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
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
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	ldq := Blas_Int(max(1, n))
	q_ptr: ^f64 = nil
	if q.data != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
	}

	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	lwork_int := Blas_Int(lwork)

	m: Blas_Int
	info_int: Info

	// Workspace query
	if lwork == -1 {
		work_query: f64
		ldz := Blas_Int(1)

		lapack.dsbevx_2stage_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			q_ptr,
			&ldq,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			nil,
			nil,
			&ldz,
			&work_query,
			&lwork_int,
			nil,
			nil,
			&info_int,
			1,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Allocate eigenvector storage
	ldz := Blas_Int(max(1, n))
	z_ptr: ^f64 = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsbevx_2stage_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			q_ptr,
			&ldq,
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
			&lwork_query,
			nil,
			nil,
			&info_int,
			1,
			1,
			1,
		)

		lwork = int(work_query)
		work = make([]f64, lwork, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array
	ifail := make([]Blas_Int, n, allocator)

	// Call LAPACK
	lapack.dsbevx_2stage_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		q_ptr,
		&ldq,
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
	}
	if jobz == .VALUES_VECTORS {
		result.eigenvectors = z
	}

	// Check for convergence failures
	result.failed_indices = ifail
	for i in 0 ..< result.num_found {
		if ifail[i] > 0 {
			result.num_failures += 1
		}
	}
	result.all_converged = result.num_failures == 0

	work_size = len(work)
	return
}

ssbevx_2stage :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f32),
	q: Matrix(f32) = {},
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
	allocator := context.allocator,
) -> (
	result: SelectiveEigenResult(f32),
	info: Info,
	work_size: int,
) {
	// Similar implementation for single precision
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
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
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	ldq := Blas_Int(max(1, n))
	q_ptr: ^f32 = nil
	if q.data != nil {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
	}

	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	lwork_int := Blas_Int(lwork)

	m: Blas_Int
	info_int: Info

	// Workspace query
	if lwork == -1 {
		work_query: f32
		ldz := Blas_Int(1)

		lapack.ssbevx_2stage_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			q_ptr,
			&ldq,
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&abstol_val,
			&m,
			nil,
			nil,
			&ldz,
			&work_query,
			&lwork_int,
			nil,
			nil,
			&info_int,
			1,
			1,
			1,
		)

		work_size = int(work_query)
		return result, Info(info_int), work_size
	}

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Allocate eigenvector storage
	ldz := Blas_Int(max(1, n))
	z_ptr: ^f32 = nil
	max_eigenvectors := n
	if range == .INDEX {
		max_eigenvectors = iu - il + 1
	}
	if jobz == .VALUES_VECTORS {
		assert(z.rows >= n && z.cols >= max_eigenvectors, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
	}

	// Allocate workspace
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssbevx_2stage_(
			jobz_cstring,
			range_cstring,
			uplo_cstring,
			&n_int,
			&kd_int,
			ab.data,
			&ldab,
			q_ptr,
			&ldq,
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
			&lwork_query,
			nil,
			nil,
			&info_int,
			1,
			1,
			1,
		)

		lwork = int(work_query)
		work = make([]f32, lwork, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Allocate failure array
	ifail := make([]Blas_Int, n, allocator)

	// Call LAPACK
	lapack.ssbevx_2stage_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		q_ptr,
		&ldq,
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
	}
	if jobz == .VALUES_VECTORS {
		result.eigenvectors = z
	}

	// Check for convergence failures
	result.failed_indices = ifail
	for i in 0 ..< result.num_found {
		if ifail[i] > 0 {
			result.num_failures += 1
		}
	}
	result.all_converged = result.num_failures == 0

	work_size = len(work)
	return
}

// Proc groups
sbevx :: proc {
	dsbevx,
	ssbevx,
}
sbevx_2stage :: proc {
	dsbevx_2stage,
	ssbevx_2stage,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Compute smallest eigenvalues of banded matrix
compute_smallest_eigenvalues :: proc(
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix($T),
	num_eigenvalues: int,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	ab_copy := matrix_clone(&ab, allocator)
	defer matrix_delete(&ab_copy)

	eigenvectors = create_matrix(T, n, num_eigenvalues, allocator)

	when T == f64 {
		result, info_val := dsbevx(
			.VALUES_VECTORS,
			.INDEX,
			uplo,
			n,
			kd,
			ab_copy,
			il = 1,
			iu = num_eigenvalues,
			z = eigenvectors,
			allocator = allocator,
		)
		defer delete(result.failed_indices)
		return result.eigenvalues, eigenvectors, info_val
	} else when T == f32 {
		result, info_val := ssbevx(
			.VALUES_VECTORS,
			.INDEX,
			uplo,
			n,
			kd,
			ab_copy,
			il = 1,
			iu = num_eigenvalues,
			z = eigenvectors,
			allocator = allocator,
		)
		defer delete(result.failed_indices)
		return result.eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for selective eigenvalue computation")
	}
}

// Compute eigenvalues in specified range
compute_eigenvalues_in_range :: proc(
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix($T),
	lower_bound: T,
	upper_bound: T,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	num_found: int,
	info: Info,
) {
	ab_copy := matrix_clone(&ab, allocator)
	defer matrix_delete(&ab_copy)

	// Allocate maximum possible eigenvectors
	eigenvectors = create_matrix(T, n, n, allocator)

	when T == f64 {
		result, info_val := dsbevx(
			.VALUES_VECTORS,
			.VALUE,
			uplo,
			n,
			kd,
			ab_copy,
			vl = lower_bound,
			vu = upper_bound,
			z = eigenvectors,
			allocator = allocator,
		)
		defer delete(result.failed_indices)
		return result.eigenvalues, eigenvectors, result.num_found, info_val
	} else when T == f32 {
		result, info_val := ssbevx(
			.VALUES_VECTORS,
			.VALUE,
			uplo,
			n,
			kd,
			ab_copy,
			vl = lower_bound,
			vu = upper_bound,
			z = eigenvectors,
			allocator = allocator,
		)
		defer delete(result.failed_indices)
		return result.eigenvalues, eigenvectors, result.num_found, info_val
	} else {
		#panic("Unsupported type for selective eigenvalue computation")
	}
}
