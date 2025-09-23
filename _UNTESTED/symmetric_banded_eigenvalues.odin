package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC BANDED EIGENVALUE COMPUTATION
// ============================================================================
// Computes all eigenvalues and optionally eigenvectors of a real symmetric
// band matrix using the divide and conquer algorithm

// Job option for eigenvalue computation
EigenJobOption :: enum {
	VALUES_ONLY, // 'N' - Compute eigenvalues only
	VALUES_VECTORS, // 'V' - Compute eigenvalues and eigenvectors
}

// Symmetric banded eigenvalue result
BandedEigenResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	all_positive:     bool, // True if all eigenvalues > 0
	all_non_negative: bool, // True if all eigenvalues >= 0
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ| for non-zero eigenvalues
	num_negative:     int, // Number of negative eigenvalues
	num_zero:         int, // Number of (near) zero eigenvalues
	num_positive:     int, // Number of positive eigenvalues
}

// Double precision symmetric banded eigenvalue computation
dsbev :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int, // Number of superdiagonals/subdiagonals
	ab: Matrix(f64), // Band matrix (modified on output)
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f64 = nil, // Workspace (size 3*n-2 if nil)
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f64),
	info: Info,
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
	work_size := 3 * n - 2
	if work_size < 1 do work_size = 1
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsbev_(
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
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1] // Eigenvalues are sorted in ascending order

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

		// Compute condition number
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

// Single precision symmetric banded eigenvalue computation
ssbev :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int, // Number of superdiagonals/subdiagonals
	ab: Matrix(f32), // Band matrix (modified on output)
	w: []f32 = nil, // Eigenvalues (size n)
	z: Matrix(f32) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f32 = nil, // Workspace (size 3*n-2 if nil)
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f32),
	info: Info,
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
	work_size := 3 * n - 2
	if work_size < 1 do work_size = 1
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssbev_(
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
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1]) // Eigenvalues are sorted in ascending order

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

		// Compute condition number
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

// ============================================================================
// 2-STAGE SYMMETRIC BANDED EIGENVALUE COMPUTATION
// ============================================================================
// Uses a 2-stage algorithm for improved performance on large matrices

// Double precision 2-stage symmetric banded eigenvalue computation
dsbev_2stage :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int, // Number of superdiagonals/subdiagonals
	ab: Matrix(f64), // Band matrix (modified on output)
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1, // Workspace size (-1 for query)
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f64),
	info: Info,
	work_size: int,
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
	info_int: Info

	// Workspace query
	if lwork == -1 {
		work_query: f64
		ldz := Blas_Int(1)

		lapack.dsbev_2stage_(
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
			&info_int,
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
	if allocated_work {
		// Query for optimal workspace
		work_query: f64
		lwork_query := Blas_Int(-1)

		lapack.dsbev_2stage_(
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
			&info_int,
			1,
			1,
		)

		lwork = int(work_query)
		work = make([]f64, lwork, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsbev_2stage_(
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
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = w[0]
		result.max_eigenvalue = w[n - 1] // Eigenvalues are sorted in ascending order

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

		// Compute condition number
		if abs(result.min_eigenvalue) > epsilon {
			result.condition_number = abs(result.max_eigenvalue / result.min_eigenvalue)
		} else if abs(result.max_eigenvalue) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	work_size = len(work)
	return
}

// Single precision 2-stage symmetric banded eigenvalue computation
ssbev_2stage :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	kd: int, // Number of superdiagonals/subdiagonals
	ab: Matrix(f32), // Band matrix (modified on output)
	w: []f32 = nil, // Eigenvalues (size n)
	z: Matrix(f32) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f32 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1, // Workspace size (-1 for query)
	allocator := context.allocator,
) -> (
	result: BandedEigenResult(f32),
	info: Info,
	work_size: int,
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
	info_int: Info

	// Workspace query
	if lwork == -1 {
		work_query: f32
		ldz := Blas_Int(1)

		lapack.ssbev_2stage_(
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
			&info_int,
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
	if allocated_work {
		// Query for optimal workspace
		work_query: f32
		lwork_query := Blas_Int(-1)

		lapack.ssbev_2stage_(
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
			&info_int,
			1,
			1,
		)

		lwork = int(work_query)
		work = make([]f32, lwork, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssbev_2stage_(
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
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.min_eigenvalue = f64(w[0])
		result.max_eigenvalue = f64(w[n - 1]) // Eigenvalues are sorted in ascending order

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

		// Compute condition number
		if abs(w[0]) > epsilon {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else if abs(w[n - 1]) > epsilon {
			result.condition_number = math.INF_F64
		} else {
			result.condition_number = 1.0
		}
	}

	work_size = len(work)
	return
}

// Proc groups
sbev :: proc {
	dsbev,
	ssbev,
}
sbev_2stage :: proc {
	dsbev_2stage,
	ssbev_2stage,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Compute eigenvalues only for symmetric banded matrix
compute_banded_eigenvalues :: proc(
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix($T),
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	info: Info,
) {
	// Make a copy of the band matrix since it gets modified
	ab_copy := matrix_clone(&ab, allocator)
	defer matrix_delete(&ab_copy)

	when T == f64 {
		result, info_val := dsbev(.VALUES_ONLY, uplo, n, kd, ab_copy, allocator = allocator)
		return result.eigenvalues, info_val
	} else when T == f32 {
		result, info_val := ssbev(.VALUES_ONLY, uplo, n, kd, ab_copy, allocator = allocator)
		return result.eigenvalues, info_val
	} else {
		#panic("Unsupported type for banded eigenvalue computation")
	}
}

// Compute eigenvalues and eigenvectors for symmetric banded matrix
compute_banded_eigen_decomposition :: proc(
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix($T),
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	// Make a copy of the band matrix since it gets modified
	ab_copy := matrix_clone(&ab, allocator)
	defer matrix_delete(&ab_copy)

	// Allocate eigenvector matrix
	eigenvectors = create_matrix(T, n, n, allocator)

	when T == f64 {
		result, info_val := dsbev(
			.VALUES_VECTORS,
			uplo,
			n,
			kd,
			ab_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else when T == f32 {
		result, info_val := ssbev(
			.VALUES_VECTORS,
			uplo,
			n,
			kd,
			ab_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for banded eigenvalue computation")
	}
}

// Check if symmetric banded matrix is positive definite via eigenvalues
is_banded_positive_definite :: proc(
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix($T),
	tolerance := 0.0,
	allocator := context.allocator,
) -> (
	is_pd: bool,
	min_eigenval: f64,
	condition: f64,
) {
	eigenvalues, info := compute_banded_eigenvalues(uplo, n, kd, ab, allocator)
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

// Analyze eigenvalue distribution of symmetric banded matrix
analyze_banded_spectrum :: proc(
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix($T),
	allocator := context.allocator,
) -> (
	analysis: SpectrumAnalysis,
) {
	result, info := sbev(
		.VALUES_ONLY,
		uplo,
		n,
		kd,
		matrix_clone(&ab, allocator),
		allocator = allocator,
	)
	defer {
		delete(result.eigenvalues)
		matrix_delete(&result.eigenvectors)
	}

	if info != .OK || n == 0 {
		analysis.success = false
		return
	}

	analysis.success = true
	analysis.num_eigenvalues = n
	analysis.min_eigenvalue = result.min_eigenvalue
	analysis.max_eigenvalue = result.max_eigenvalue
	analysis.condition_number = result.condition_number
	analysis.num_negative = result.num_negative
	analysis.num_zero = result.num_zero
	analysis.num_positive = result.num_positive

	// Compute spectral gap (smallest gap between consecutive eigenvalues)
	if n > 1 {
		analysis.spectral_gap = math.INF_F64
		for i in 1 ..< n {
			gap := f64(result.eigenvalues[i] - result.eigenvalues[i - 1])
			analysis.spectral_gap = min(analysis.spectral_gap, gap)
		}
	}

	// Determine matrix properties
	analysis.is_positive_definite = result.all_positive
	analysis.is_positive_semidefinite = result.all_non_negative
	analysis.is_indefinite = result.num_negative > 0 && result.num_positive > 0
	analysis.is_singular = result.num_zero > 0

	// Compute spectral radius
	analysis.spectral_radius = max(abs(result.min_eigenvalue), abs(result.max_eigenvalue))

	return
}

// Spectrum analysis structure
SpectrumAnalysis :: struct {
	success:                  bool,
	num_eigenvalues:          int,
	min_eigenvalue:           f64,
	max_eigenvalue:           f64,
	spectral_radius:          f64,
	condition_number:         f64,
	spectral_gap:             f64,
	num_negative:             int,
	num_zero:                 int,
	num_positive:             int,
	is_positive_definite:     bool,
	is_positive_semidefinite: bool,
	is_indefinite:            bool,
	is_singular:              bool,
}

// Compare standard vs 2-stage algorithms for banded eigenvalue computation
compare_banded_eigen_algorithms :: proc(
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix($T),
	compute_vectors := false,
	allocator := context.allocator,
) -> (
	comparison: AlgorithmComparison,
) {
	jobz := compute_vectors ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY

	// Standard algorithm
	ab_std := matrix_clone(&ab, allocator)
	defer matrix_delete(&ab_std)

	z_std: Matrix(T)
	if compute_vectors {
		z_std = create_matrix(T, n, n, allocator)
		defer matrix_delete(&z_std)
	}

	when T == f64 {
		result_std, info_std := dsbev(jobz, uplo, n, kd, ab_std, z = z_std, allocator = allocator)
		defer delete(result_std.eigenvalues)
		comparison.standard_success = info_std == .OK

		// 2-stage algorithm
		ab_2stage := matrix_clone(&ab, allocator)
		defer matrix_delete(&ab_2stage)

		z_2stage: Matrix(T)
		if compute_vectors {
			z_2stage = create_matrix(T, n, n, allocator)
			defer matrix_delete(&z_2stage)
		}

		result_2stage, info_2stage, work_size := dsbev_2stage(
			jobz,
			uplo,
			n,
			kd,
			ab_2stage,
			z = z_2stage,
			allocator = allocator,
		)
		defer delete(result_2stage.eigenvalues)
		comparison.two_stage_success = info_2stage == .OK
		comparison.two_stage_work_size = work_size

		// Compare eigenvalues if both succeeded
		if comparison.standard_success && comparison.two_stage_success {
			max_diff := 0.0
			for i in 0 ..< n {
				diff := abs(result_std.eigenvalues[i] - result_2stage.eigenvalues[i])
				max_diff = max(max_diff, diff)
			}
			comparison.max_eigenvalue_difference = max_diff
			comparison.results_match = max_diff < 1e-10
		}
	} else when T == f32 {
		result_std, info_std := ssbev(jobz, uplo, n, kd, ab_std, z = z_std, allocator = allocator)
		defer delete(result_std.eigenvalues)
		comparison.standard_success = info_std == .OK

		// 2-stage algorithm
		ab_2stage := matrix_clone(&ab, allocator)
		defer matrix_delete(&ab_2stage)

		z_2stage: Matrix(T)
		if compute_vectors {
			z_2stage = create_matrix(T, n, n, allocator)
			defer matrix_delete(&z_2stage)
		}

		result_2stage, info_2stage, work_size := ssbev_2stage(
			jobz,
			uplo,
			n,
			kd,
			ab_2stage,
			z = z_2stage,
			allocator = allocator,
		)
		defer delete(result_2stage.eigenvalues)
		comparison.two_stage_success = info_2stage == .OK
		comparison.two_stage_work_size = work_size

		// Compare eigenvalues if both succeeded
		if comparison.standard_success && comparison.two_stage_success {
			max_diff := f64(0)
			for i in 0 ..< n {
				diff := f64(abs(result_std.eigenvalues[i] - result_2stage.eigenvalues[i]))
				max_diff = max(max_diff, diff)
			}
			comparison.max_eigenvalue_difference = max_diff
			comparison.results_match = max_diff < 1e-6
		}
	}

	// Recommendation based on matrix size
	if n > 1000 {
		comparison.recommendation = .TwoStage // Better for large matrices
	} else {
		comparison.recommendation = .Standard // Simpler for small matrices
	}

	return
}

// Algorithm comparison structure
AlgorithmComparison :: struct {
	standard_success:          bool,
	two_stage_success:         bool,
	two_stage_work_size:       int,
	max_eigenvalue_difference: f64,
	results_match:             bool,
	recommendation:            AlgorithmChoice,
}

AlgorithmChoice :: enum {
	Standard,
	TwoStage,
}
