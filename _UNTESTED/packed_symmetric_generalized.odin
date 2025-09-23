package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE PROBLEMS
// ============================================================================
// Solves the generalized eigenvalue problem A*x = λ*B*x where A and B are
// symmetric matrices stored in packed format and B is positive definite

// Generalized packed eigenvalue result
GeneralizedPackedEigenResult :: struct($T: typeid) {
	eigenvalues:            []T, // Computed eigenvalues (sorted)
	eigenvectors:           Matrix(T), // Eigenvector matrix (if requested)
	b_is_positive_definite: bool, // True if B was successfully factored
	all_positive:           bool, // True if all eigenvalues > 0
	min_eigenvalue:         f64, // Smallest eigenvalue
	max_eigenvalue:         f64, // Largest eigenvalue
	condition_number:       f64, // max|λ|/min|λ|
	num_negative:           int, // Number of negative eigenvalues
	num_zero:               int, // Number of (near) zero eigenvalues
	num_positive:           int, // Number of positive eigenvalues
}

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE
// ============================================================================

// Double precision packed generalized eigenvalue
dspgv :: proc(
	itype: GeneralizedProblemType,
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f64, // Packed matrix A (modified on output)
	bp: []f64, // Packed matrix B (modified to Cholesky factor)
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f64 = nil, // Workspace (size 3*n)
	allocator := context.allocator,
) -> (
	result: GeneralizedPackedEigenResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

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
	work_size := 3 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dspgv_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(bp),
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Check if B was positive definite
	result.b_is_positive_definite = info == .OK || info > Info(n)

	// Analyze eigenvalues
	if (info == .OK || info > Info(n)) && n > 0 {
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

		if abs(result.min_eigenvalue) > epsilon {
			result.condition_number = abs(result.max_eigenvalue / result.min_eigenvalue)
		} else if abs(result.max_eigenvalue) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	return
}

// Single precision packed generalized eigenvalue
sspgv :: proc(
	itype: GeneralizedProblemType,
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	bp: []f32,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedPackedEigenResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	info_int: Info

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
	work_size := 3 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.sspgv_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(bp),
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Check if B was positive definite
	result.b_is_positive_definite = info == .OK || info > Info(n)

	// Analyze eigenvalues
	if (info == .OK || info > Info(n)) && n > 0 {
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

		if abs(w[0]) > epsilon {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else if abs(w[n - 1]) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	return
}

spgv :: proc {
	dspgv,
	sspgv,
}

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE - DIVIDE AND CONQUER
// ============================================================================

// Double precision divide-and-conquer packed generalized eigenvalue
dspgvd :: proc(
	itype: GeneralizedProblemType,
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f64,
	bp: []f64,
	w: []f64 = nil,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedPackedEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.dspgvd_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
			raw_data(bp),
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
		// Query for optimal workspace
		work_query: f64
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.dspgvd_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
			raw_data(bp),
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
	lapack.dspgvd_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(bp),
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

	// Check if B was positive definite
	result.b_is_positive_definite = info == .OK || info > Info(n)

	// Analyze eigenvalues
	if (info == .OK || info > Info(n)) && n > 0 {
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

// Single precision divide-and-conquer packed generalized eigenvalue
sspgvd :: proc(
	itype: GeneralizedProblemType,
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	bp: []f32,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedPackedEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	itype_int := Blas_Int(itype)
	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.sspgvd_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
			raw_data(bp),
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

		lapack.sspgvd_(
			&itype_int,
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
			raw_data(bp),
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
	lapack.sspgvd_(
		&itype_int,
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(bp),
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

	// Check if B was positive definite
	result.b_is_positive_definite = info == .OK || info > Info(n)

	// Analyze eigenvalues
	if (info == .OK || info > Info(n)) && n > 0 {
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

spgvd :: proc {
	dspgvd,
	sspgvd,
}

// ============================================================================
// PACKED SYMMETRIC GENERALIZED EIGENVALUE - SELECTIVE
// ============================================================================

// Selective generalized packed result
SelectiveGeneralizedPackedResult :: struct($T: typeid) {
	eigenvalues:            []T,
	eigenvectors:           Matrix(T),
	num_found:              int,
	failed_indices:         []Blas_Int,
	num_failures:           int,
	b_is_positive_definite: bool,
}

// Double precision selective packed generalized eigenvalue
dspgvx :: proc(
	itype: GeneralizedProblemType,
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f64,
	bp: []f64,
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f64 = 0,
	w: []f64 = nil,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SelectiveGeneralizedPackedResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	itype_int := Blas_Int(itype)
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
		work = make([]f64, 8 * n, allocator)
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
	lapack.dspgvx_(
		&itype_int,
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(bp),
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
	result.b_is_positive_definite = info == .OK || info > Info(n)

	return
}

// Single precision selective packed generalized eigenvalue
sspgvx :: proc(
	itype: GeneralizedProblemType,
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	bp: []f32,
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
	result: SelectiveGeneralizedPackedResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	itype_int := Blas_Int(itype)
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
		work = make([]f32, 8 * n, allocator)
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
	lapack.sspgvx_(
		&itype_int,
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(bp),
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
	result.b_is_positive_definite = info == .OK || info > Info(n)

	return
}

spgvx :: proc {
	dspgvx,
	sspgvx,
}

// ============================================================================
// PACKED SYMMETRIC ITERATIVE REFINEMENT
// ============================================================================

// Iterative refinement result
PackedRefinementResult :: struct($T: typeid) {
	forward_errors:     []T, // Forward error bounds for each RHS
	backward_errors:    []T, // Backward error bounds for each RHS
	max_forward_error:  f64, // Maximum forward error
	max_backward_error: f64, // Maximum backward error
	improved_accuracy:  bool, // True if refinement improved solution
}

// Complex single precision packed refinement
csprfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex64, // Original packed matrix
	afp: []complex64, // Factored packed matrix
	ipiv: []Blas_Int, // Pivot indices from factorization
	b: Matrix(complex64), // Right-hand side matrix
	x: Matrix(complex64), // Solution matrix (refined on output)
	work: []complex64 = nil, // Workspace (size 2*n)
	rwork: []f32 = nil, // Real workspace (size n)
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(afp) >= n * (n + 1) / 2, "Factored packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)
	defer {
		delete(ferr)
		delete(berr)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f32, n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.csprfs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)raw_data(ap),
		cast(^lapack.complex)raw_data(afp),
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		cast(^lapack.complex)x.data,
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.complex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, f64(ferr[i]))
		result.max_backward_error = max(result.max_backward_error, f64(berr[i]))
	}

	result.improved_accuracy = result.max_backward_error < 1.0

	return
}

// Double precision packed refinement
dsprfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f64,
	afp: []f64,
	ipiv: []Blas_Int,
	b: Matrix(f64),
	x: Matrix(f64),
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(afp) >= n * (n + 1) / 2, "Factored packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.dsprfs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(afp),
		raw_data(ipiv),
		b.data,
		&ldb,
		x.data,
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, ferr[i])
		result.max_backward_error = max(result.max_backward_error, berr[i])
	}

	result.improved_accuracy = result.max_backward_error < 1.0

	return
}

// Single precision packed refinement
ssprfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f32,
	afp: []f32,
	ipiv: []Blas_Int,
	b: Matrix(f32),
	x: Matrix(f32),
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(afp) >= n * (n + 1) / 2, "Factored packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays
	ferr := make([]f32, nrhs, allocator)
	berr := make([]f32, nrhs, allocator)

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.ssprfs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(afp),
		raw_data(ipiv),
		b.data,
		&ldb,
		x.data,
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, f64(ferr[i]))
		result.max_backward_error = max(result.max_backward_error, f64(berr[i]))
	}

	result.improved_accuracy = result.max_backward_error < 1.0

	return
}

// Complex double precision packed refinement
zsprfs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex128,
	afp: []complex128,
	ipiv: []Blas_Int,
	b: Matrix(complex128),
	x: Matrix(complex128),
	work: []complex128 = nil,
	rwork: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: PackedRefinementResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(afp) >= n * (n + 1) / 2, "Factored packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")
	assert(x.rows >= n && x.cols >= nrhs, "Solution matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	ldx := Blas_Int(x.stride)
	info_int: Info

	// Allocate error arrays
	ferr := make([]f64, nrhs, allocator)
	berr := make([]f64, nrhs, allocator)

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		rwork = make([]f64, n, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	// Call LAPACK
	lapack.zsprfs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)raw_data(ap),
		cast(^lapack.doublecomplex)raw_data(afp),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		cast(^lapack.doublecomplex)x.data,
		&ldx,
		raw_data(ferr),
		raw_data(berr),
		cast(^lapack.doublecomplex)raw_data(work),
		raw_data(rwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.forward_errors = ferr
	result.backward_errors = berr

	for i in 0 ..< nrhs {
		result.max_forward_error = max(result.max_forward_error, ferr[i])
		result.max_backward_error = max(result.max_backward_error, berr[i])
	}

	result.improved_accuracy = result.max_backward_error < 1.0

	return
}

sprfs :: proc {
	csprfs,
	dsprfs,
	ssprfs,
	zsprfs,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Solve generalized eigenvalue problem for packed matrices
solve_packed_generalized :: proc(
	ap: []$T,
	bp: []T,
	n: int,
	itype := GeneralizedProblemType.A_LAMBDA_B,
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	// Make copies since they get modified
	ap_copy := make([]T, len(ap), allocator)
	copy(ap_copy, ap)
	defer delete(ap_copy)

	bp_copy := make([]T, len(bp), allocator)
	copy(bp_copy, bp)
	defer delete(bp_copy)

	jobz := compute_vectors ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY

	if compute_vectors {
		eigenvectors = create_matrix(T, n, n, allocator)
	}

	when T == f64 {
		result, info_val := dspgv(
			itype,
			jobz,
			uplo,
			n,
			ap_copy,
			bp_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else when T == f32 {
		result, info_val := sspgv(
			itype,
			jobz,
			uplo,
			n,
			ap_copy,
			bp_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for packed generalized eigenvalue")
	}
}
