package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC EIGENVALUE SOLVERS - BISECTION AND INVERSE ITERATION
// ============================================================================

// Bisection eigenvalue result with failure tracking
BisectionSymmetricEigenResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	num_found:        int, // Number of eigenvalues found
	failed_indices:   []Blas_Int, // Indices of failed eigenvectors
	num_failures:     int, // Number of failed convergences
	all_converged:    bool, // True if all eigenvectors converged
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
}

// Double precision symmetric eigenvalue solver (bisection and inverse iteration)
dsyevx :: proc(
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
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (size 5*n)
	ifail: []Blas_Int = nil, // Failed indices (size n)
	allocator := context.allocator,
) -> (
	result: BisectionSymmetricEigenResult(f64),
	info: Info,
	work_size: int,
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

		lapack.dsyevx_(
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
		_, _, work_size = dsyevx(jobz, range, uplo, n, a, vl, vu, il, iu, abstol, w, z, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsyevx_(
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
	if info > 0 && jobz == .EIGENVECTORS {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	work_size = len(work)
	return
}

// Single precision symmetric eigenvalue solver (bisection and inverse iteration)
ssyevx :: proc(
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
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: BisectionSymmetricEigenResult(f32),
	info: Info,
	work_size: int,
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

		lapack.ssyevx_(
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
		_, _, work_size = ssyevx(jobz, range, uplo, n, a, vl, vu, il, iu, abstol, w, z, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssyevx_(
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
	if info > 0 && jobz == .EIGENVECTORS {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	work_size = len(work)
	return
}

// Double precision 2-stage symmetric eigenvalue solver (bisection and inverse iteration)
dsyevx_2stage :: proc(
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
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: BisectionSymmetricEigenResult(f64),
	info: Info,
	work_size: int,
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

		lapack.dsyevx_2stage_(
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
		_, _, work_size = dsyevx_2stage(
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
			lwork = -1,
		)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.dsyevx_2stage_(
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
	if info > 0 && jobz == .EIGENVECTORS {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	work_size = len(work)
	return
}

// Single precision 2-stage symmetric eigenvalue solver (bisection and inverse iteration)
ssyevx_2stage :: proc(
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
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: BisectionSymmetricEigenResult(f32),
	info: Info,
	work_size: int,
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

		lapack.ssyevx_2stage_(
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
		_, _, work_size = ssyevx_2stage(
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
			lwork = -1,
		)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	lwork_int = Blas_Int(len(work))

	// Call LAPACK
	lapack.ssyevx_2stage_(
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
	if info > 0 && jobz == .EIGENVECTORS {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	work_size = len(work)
	return
}

syevx :: proc {
	dsyevx,
	ssyevx,
}
syevx_2stage :: proc {
	dsyevx_2stage,
	ssyevx_2stage,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Bisection method for symmetric eigendecomposition
bisection_symmetric_eigendecomposition :: proc(
	a: Matrix($T),
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
	all_converged: bool,
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
		result, info_val, _ := dsyevx(
			jobz,
			range,
			uplo,
			n,
			a_copy,
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
			result.all_converged,
			info_val
	} else when T == f32 {
		result, info_val, _ := ssyevx(
			jobz,
			range,
			uplo,
			n,
			a_copy,
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
			result.all_converged,
			info_val
	} else {
		#panic("Unsupported type for bisection eigendecomposition")
	}
}

// 2-stage bisection method for symmetric eigendecomposition
bisection_symmetric_eigendecomposition_2stage :: proc(
	a: Matrix($T),
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
	all_converged: bool,
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
		result, info_val, _ := dsyevx_2stage(
			jobz,
			range,
			uplo,
			n,
			a_copy,
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
			result.all_converged,
			info_val
	} else when T == f32 {
		result, info_val, _ := ssyevx_2stage(
			jobz,
			range,
			uplo,
			n,
			a_copy,
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
			result.all_converged,
			info_val
	} else {
		#panic("Unsupported type for 2-stage bisection eigendecomposition")
	}
}

// Check if all eigenvectors converged in bisection method
check_eigenvector_convergence :: proc(
	failed_indices: []Blas_Int,
	n: int,
) -> (
	failed_eigenvectors: []int,
	all_converged: bool,
) {
	if len(failed_indices) == 0 {
		return nil, true
	}

	// Convert blasint indices to regular int
	failed := make([]int, len(failed_indices))
	for i, idx in failed_indices {
		failed[i] = int(idx)
	}

	return failed, false
}

// Find eigenvalues in a specific range using bisection
find_eigenvalues_in_range :: proc(
	a: Matrix($T),
	vl: T,
	vu: T,
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	abstol: T = 0,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	num_found: int,
	info: Info,
) {
	eigenvals, eigenvecs, num, _, info_val := bisection_symmetric_eigendecomposition(
		a,
		range = .VALUE,
		vl = vl,
		vu = vu,
		compute_vectors = compute_vectors,
		uplo = uplo,
		abstol = abstol,
		allocator = allocator,
	)

	return eigenvals, eigenvecs, num, info_val
}

// Find specific eigenvalues by index using bisection
find_eigenvalues_by_index :: proc(
	a: Matrix($T),
	il: int, // 1-based lower index
	iu: int, // 1-based upper index
	compute_vectors := false,
	uplo := UpLoFlag.Lower,
	abstol: T = 0,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	num_found: int,
	info: Info,
) {
	eigenvals, eigenvecs, num, _, info_val := bisection_symmetric_eigendecomposition(
		a,
		range = .INDEX,
		il = il,
		iu = iu,
		compute_vectors = compute_vectors,
		uplo = uplo,
		abstol = abstol,
		allocator = allocator,
	)

	return eigenvals, eigenvecs, num, info_val
}
