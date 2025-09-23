package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC MATRIX OPERATIONS
// ============================================================================
// Operations on symmetric matrices stored in packed format (triangular storage)
// Only upper or lower triangle is stored in a 1D array of size n*(n+1)/2

// ============================================================================
// CONDITION NUMBER ESTIMATION FOR PACKED SYMMETRIC MATRICES
// ============================================================================

// Condition estimation result
PackedConditionResult :: struct {
	rcond:               f64, // Reciprocal condition number
	condition_number:    f64, // Actual condition number (1/rcond)
	is_singular:         bool, // True if matrix is singular
	is_well_conditioned: bool, // True if condition number < 1e6
	is_ill_conditioned:  bool, // True if condition number > 1e10
}

// Complex single precision packed condition estimation
cspcon :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []complex64, // Packed factored matrix from csptrf
	ipiv: []Blas_Int, // Pivot indices from csptrf
	anorm: f32, // 1-norm of original matrix
	work: []complex64 = nil, // Workspace (size 2*n if nil)
	allocator := context.allocator,
) -> (
	result: PackedConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	anorm_val := anorm
	rcond: f32
	info_int: Info

	// Allocate workspace if not provided
	work_size := 2 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.cspcon_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)raw_data(ap),
		raw_data(ipiv),
		&anorm_val,
		&rcond,
		cast(^lapack.complex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = f64(rcond)
	if rcond > 0 {
		result.condition_number = 1.0 / result.rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < machine_epsilon(f32)
	result.is_well_conditioned = result.condition_number < 1e6
	result.is_ill_conditioned = result.condition_number > 1e10

	return
}

// Double precision packed condition estimation
dspcon :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f64, // Packed factored matrix from dsptrf
	ipiv: []Blas_Int, // Pivot indices from dsptrf
	anorm: f64, // 1-norm of original matrix
	work: []f64 = nil, // Workspace (size 2*n if nil)
	iwork: []Blas_Int = nil, // Integer workspace (size n if nil)
	allocator := context.allocator,
) -> (
	result: PackedConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	anorm_val := anorm
	rcond: f64
	info_int: Info

	// Allocate workspace if not provided
	work_size := 2 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.dspcon_(
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(ipiv),
		&anorm_val,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	if rcond > 0 {
		result.condition_number = 1.0 / rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < machine_epsilon(f64)
	result.is_well_conditioned = result.condition_number < 1e6
	result.is_ill_conditioned = result.condition_number > 1e10

	return
}

// Single precision packed condition estimation
sspcon :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	ipiv: []Blas_Int,
	anorm: f32,
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: PackedConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	anorm_val := anorm
	rcond: f32
	info_int: Info

	// Allocate workspace if not provided
	work_size := 2 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.sspcon_(
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(ipiv),
		&anorm_val,
		&rcond,
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = f64(rcond)
	if rcond > 0 {
		result.condition_number = 1.0 / result.rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < machine_epsilon(f32)
	result.is_well_conditioned = result.condition_number < 1e6
	result.is_ill_conditioned = result.condition_number > 1e10

	return
}

// Complex double precision packed condition estimation
zspcon :: proc(
	uplo: UpLoFlag,
	n: int,
	ap: []complex128,
	ipiv: []Blas_Int,
	anorm: f64,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: PackedConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	anorm_val := anorm
	rcond: f64
	info_int: Info

	// Allocate workspace if not provided
	work_size := 2 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zspcon_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)raw_data(ap),
		raw_data(ipiv),
		&anorm_val,
		&rcond,
		cast(^lapack.doublecomplex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.rcond = rcond
	if rcond > 0 {
		result.condition_number = 1.0 / rcond
	} else {
		result.condition_number = math.INF_F64
	}
	result.is_singular = rcond < machine_epsilon(f64)
	result.is_well_conditioned = result.condition_number < 1e6
	result.is_ill_conditioned = result.condition_number > 1e10

	return
}

spcon :: proc {
	cspcon,
	dspcon,
	sspcon,
	zspcon,
}

// ============================================================================
// PACKED SYMMETRIC EIGENVALUE COMPUTATION
// ============================================================================

// Packed eigenvalue result
PackedEigenResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	all_positive:     bool, // True if all eigenvalues > 0
	all_non_negative: bool, // True if all eigenvalues >= 0
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
	num_negative:     int, // Number of negative eigenvalues
	num_zero:         int, // Number of (near) zero eigenvalues
	num_positive:     int, // Number of positive eigenvalues
}

// Double precision packed symmetric eigenvalue
dspev :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f64, // Packed matrix (modified on output)
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f64 = nil, // Workspace (size 3*n if nil)
	allocator := context.allocator,
) -> (
	result: PackedEigenResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

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
	lapack.dspev_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
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

	return
}

// Single precision packed symmetric eigenvalue
sspev :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: PackedEigenResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

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
	lapack.sspev_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
		raw_data(w),
		z_ptr,
		&ldz,
		raw_data(work),
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

	return
}

spev :: proc {
	dspev,
	sspev,
}

// ============================================================================
// PACKED SYMMETRIC EIGENVALUE - DIVIDE AND CONQUER
// ============================================================================

// Double precision divide-and-conquer packed eigenvalue
dspevd :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f64,
	w: []f64 = nil,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: PackedEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

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

		lapack.dspevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
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

		lapack.dspevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
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
	lapack.dspevd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
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

// Single precision divide-and-conquer packed eigenvalue
sspevd :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: PackedEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

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

		lapack.sspevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
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

		lapack.sspevd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			raw_data(ap),
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
	lapack.sspevd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
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

spevd :: proc {
	dspevd,
	sspevd,
}

// ============================================================================
// PACKED SYMMETRIC SELECTIVE EIGENVALUE
// ============================================================================

// Selective packed eigenvalue result
SelectivePackedResult :: struct($T: typeid) {
	eigenvalues:    []T,
	eigenvectors:   Matrix(T),
	num_found:      int,
	failed_indices: []Blas_Int,
	num_failures:   int,
	all_converged:  bool,
}

// Double precision selective packed eigenvalue
dspevx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f64, // Packed matrix (preserved)
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
	result: SelectivePackedResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

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
	lapack.dspevx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
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

// Single precision selective packed eigenvalue
sspevx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
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
	result: SelectivePackedResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")

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
	lapack.sspevx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		raw_data(ap),
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

spevx :: proc {
	dspevx,
	sspevx,
}

// ============================================================================
// PACKED SYMMETRIC GENERALIZED REDUCTION
// ============================================================================

// Generalized eigenvalue problem type
GeneralizedProblemType :: enum Blas_Int {
	A_LAMBDA_B  = 1, // A*x = λ*B*x
	AB_LAMBDA_I = 2, // A*B*x = λ*x
	BA_LAMBDA_I = 3, // B*A*x = λ*x
}

// Double precision packed generalized reduction
dspgst :: proc(
	itype: GeneralizedProblemType,
	uplo: UpLoFlag,
	n: int,
	ap: []f64, // Packed matrix A (modified to standard form)
	bp: []f64, // Packed Cholesky factor of B
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	itype_int := Blas_Int(itype)
	n_int := Blas_Int(n)
	info_int: Info

	// Call LAPACK
	lapack.dspgst_(&itype_int, uplo_cstring, &n_int, raw_data(ap), raw_data(bp), &info_int, 1)

	return Info(info_int)
}

// Single precision packed generalized reduction
sspgst :: proc(
	itype: GeneralizedProblemType,
	uplo: UpLoFlag,
	n: int,
	ap: []f32,
	bp: []f32,
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array A too small")
	assert(len(bp) >= n * (n + 1) / 2, "Packed array B too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	itype_int := Blas_Int(itype)
	n_int := Blas_Int(n)
	info_int: Info

	// Call LAPACK
	lapack.sspgst_(&itype_int, uplo_cstring, &n_int, raw_data(ap), raw_data(bp), &info_int, 1)

	return Info(info_int)
}

spgst :: proc {
	dspgst,
	sspgst,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Convert matrix to packed format
matrix_to_packed :: proc(
	mat: Matrix($T),
	uplo: UpLoFlag = .Lower,
	allocator := context.allocator,
) -> []T {
	n := mat.rows
	assert(mat.rows == mat.cols, "Matrix must be square")

	packed_size := n * (n + 1) / 2
	packed := make([]T, packed_size, allocator)

	idx := 0
	if uplo == .Upper {
		for j in 0 ..< n {
			for i in 0 ..= j {
				packed[idx] = matrix_get(&mat, i, j)
				idx += 1
			}
		}
	} else {
		for j in 0 ..< n {
			for i in j ..< n {
				packed[idx] = matrix_get(&mat, i, j)
				idx += 1
			}
		}
	}

	return packed
}

// Convert packed format to matrix
packed_to_matrix :: proc(
	packed: []$T,
	n: int,
	uplo: UpLoFlag = .Lower,
	allocator := context.allocator,
) -> Matrix(T) {
	assert(len(packed) >= n * (n + 1) / 2, "Packed array too small")

	mat := create_matrix(T, n, n, allocator)

	idx := 0
	if uplo == .Upper {
		for j in 0 ..< n {
			for i in 0 ..= j {
				matrix_set(&mat, i, j, packed[idx])
				if i != j {
					matrix_set(&mat, j, i, packed[idx]) // Symmetric
				}
				idx += 1
			}
		}
	} else {
		for j in 0 ..< n {
			for i in j ..< n {
				matrix_set(&mat, i, j, packed[idx])
				if i != j {
					matrix_set(&mat, j, i, packed[idx]) // Symmetric
				}
				idx += 1
			}
		}
	}

	return mat
}

// Compute eigenvalues of packed symmetric matrix
compute_packed_eigenvalues :: proc(
	ap: []$T,
	n: int,
	uplo: UpLoFlag = .Lower,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	info: Info,
) {
	// Make a copy since ap gets modified
	ap_copy := make([]T, len(ap), allocator)
	copy(ap_copy, ap)
	defer delete(ap_copy)

	when T == f64 {
		result, info_val := dspev(.VALUES_ONLY, uplo, n, ap_copy, allocator = allocator)
		return result.eigenvalues, info_val
	} else when T == f32 {
		result, info_val := sspev(.VALUES_ONLY, uplo, n, ap_copy, allocator = allocator)
		return result.eigenvalues, info_val
	} else {
		#panic("Unsupported type for packed eigenvalue computation")
	}
}

// Check if packed symmetric matrix is positive definite
is_packed_positive_definite :: proc(
	ap: []$T,
	n: int,
	uplo: UpLoFlag = .Lower,
	tolerance := 0.0,
	allocator := context.allocator,
) -> (
	is_pd: bool,
	min_eigenval: f64,
	condition: f64,
) {
	eigenvalues, info := compute_packed_eigenvalues(ap, n, uplo, allocator)
	defer delete(eigenvalues)

	if info != .OK || n == 0 {
		return false, 0, math.INF_F64
	}

	min_eigenval = f64(eigenvalues[0])
	max_eigenval := f64(eigenvalues[n - 1])

	tol := tolerance
	if tol == 0 {
		tol = f64(machine_epsilon(T)) * max(abs(min_eigenval), abs(max_eigenval))
	}

	is_pd = min_eigenval > tol

	if min_eigenval > tol {
		condition = max_eigenval / min_eigenval
	} else {
		condition = math.INF_F64
	}

	return
}
