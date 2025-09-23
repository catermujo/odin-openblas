package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SIMPLE TRIDIAGONAL EIGENVALUE COMPUTATION
// ============================================================================

// Basic eigenvalue result (STERF - eigenvalues only)
EigenResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
	all_positive:     bool, // True if all eigenvalues > 0
}

// Standard eigenproblem result (STEV, STEVD, STEVR - eigenvalues and eigenvectors)
EigenproblemResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	eigenvectors:     Matrix(T), // Computed eigenvectors (optional)
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
	all_positive:     bool, // True if all eigenvalues > 0
	all_converged:    bool, // True if all eigenvalues converged

	// Algorithm-specific fields
	num_found:        int, // Number of eigenvalues found (STEVR)
	support:          []Blas_Int, // Support arrays (STEVR)
}

// Expert eigenvalue result (STEVX - selective computation with bisection/inverse iteration)
ExpertEigenResult :: struct($T: typeid) {
	eigenvalues:    []T, // Computed eigenvalues
	eigenvectors:   Matrix(T), // Computed eigenvectors (optional)
	num_found:      int, // Number of eigenvalues found
	num_failures:   int, // Number of eigenvectors that failed to converge
	failed_indices: []Blas_Int, // Indices of failed eigenvectors
	all_converged:  bool, // True if all requested eigenvectors converged
}

// Double precision eigenvalues only (Pal-Walker-Kahan variant of QL/QR)
dsterf :: proc(
	n: int,
	d: []f64, // Diagonal (modified to eigenvalues)
	e: []f64, // Off-diagonal (destroyed)
) -> (
	result: EigenResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	info_int: Blas_Int

	// Call LAPACK
	lapack.dsterf_(&n_int, raw_data(d), raw_data(e), &info_int)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.eigenvalues = d
		result.min_eigenvalue = d[0]
		result.max_eigenvalue = d[n - 1]
		result.all_positive = d[0] > 0
		result.all_converged = true

		if abs(d[0]) > machine_epsilon(f64) {
			result.condition_number = abs(d[n - 1] / d[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

// Single precision eigenvalues only
ssterf :: proc(n: int, d: []f32, e: []f32) -> (result: EigenResult(f32), info: Info) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	n_int := Blas_Int(n)
	info_int: Blas_Int

	// Call LAPACK
	lapack.ssterf_(&n_int, raw_data(d), raw_data(e), &info_int)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.eigenvalues = d
		result.min_eigenvalue = f64(d[0])
		result.max_eigenvalue = f64(d[n - 1])
		result.all_positive = d[0] > 0

		if abs(d[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(d[n - 1] / d[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

sterf :: proc {
	dsterf,
	ssterf,
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - SIMPLE DRIVER
// ============================================================================

// Double precision simple driver
dstev :: proc(
	jobz: JobzOption,
	n: int,
	d: []f64, // Diagonal (modified to eigenvalues)
	e: []f64, // Off-diagonal (destroyed)
	z: Matrix(f64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	work: []f64 = nil, // Workspace (size 2*n-2 if jobz == EIGENVECTORS)
	allocator := context.allocator,
) -> (
	result: EigenproblemResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	n_int := Blas_Int(n)
	info_int: Blas_Int

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	work_size := jobz == .EIGENVECTORS ? 2 * n - 2 : 0
	if allocated_work && work_size > 0 {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dstev_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work) if work != nil else nil,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.eigenvalues = d
		result.min_eigenvalue = d[0]
		result.max_eigenvalue = d[n - 1]
		result.all_positive = d[0] > 0
		result.all_converged = true

		if abs(d[0]) > machine_epsilon(f64) {
			result.condition_number = abs(d[n - 1] / d[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

// Single precision simple driver
sstev :: proc(
	jobz: JobzOption,
	n: int,
	d: []f32,
	e: []f32,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: EigenproblemResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	n_int := Blas_Int(n)
	info_int: Blas_Int

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	work_size := jobz == .EIGENVECTORS ? 2 * n - 2 : 0
	if allocated_work && work_size > 0 {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.sstev_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work) if work != nil else nil,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.eigenvalues = d
		result.min_eigenvalue = f64(d[0])
		result.max_eigenvalue = f64(d[n - 1])
		result.all_positive = d[0] > 0

		if abs(d[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(d[n - 1] / d[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

stev :: proc {
	dstev,
	sstev,
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - DIVIDE AND CONQUER DRIVER
// ============================================================================

// Double precision divide-and-conquer driver
dstevd :: proc(
	jobz: JobzOption,
	n: int,
	d: []f64, // Diagonal (modified to eigenvalues)
	e: []f64, // Off-diagonal (destroyed)
	z: Matrix(f64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (query with liwork=-1)
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: EigenproblemResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	n_int := Blas_Int(n)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Blas_Int

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dstevd_(
			jobz_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
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
		_, _, work_size, _ = dstevd(jobz, n, d, e, z, lwork = -1)
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = dstevd(jobz, n, d, e, z, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.dstevd_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.eigenvalues = d
		result.min_eigenvalue = d[0]
		result.max_eigenvalue = d[n - 1]
		result.all_positive = d[0] > 0
		result.all_converged = true

		if abs(d[0]) > machine_epsilon(f64) {
			result.condition_number = abs(d[n - 1] / d[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision divide-and-conquer driver
sstevd :: proc(
	jobz: JobzOption,
	n: int,
	d: []f32,
	e: []f32,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: EigenproblemResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	jobz_char: u8 = jobz == .EIGENVECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)

	n_int := Blas_Int(n)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Blas_Int

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.sstevd_(
			jobz_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
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
		_, _, work_size, _ = sstevd(jobz, n, d, e, z, lwork = -1)
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		// Query for optimal workspace
		_, _, _, iwork_size = sstevd(jobz, n, d, e, z, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.sstevd_(
		jobz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		z_ptr,
		&ldz,
		raw_data(work),
		&lwork_int,
		raw_data(iwork),
		&liwork_int,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		result.eigenvalues = d
		result.min_eigenvalue = f64(d[0])
		result.max_eigenvalue = f64(d[n - 1])
		result.all_positive = d[0] > 0

		if abs(d[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(d[n - 1] / d[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

stevd :: proc {
	dstevd,
	sstevd,
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - MRRR DRIVER
// ============================================================================

// Double precision MRRR driver
dstevr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f64, // Diagonal (modified)
	e: []f64, // Off-diagonal (modified)
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	isuppz: []Blas_Int = nil, // Support arrays (size 2*m)
	work: []f64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (query with liwork=-1)
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: EigenproblemResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

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

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Blas_Int

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
		isuppz = make([]Blas_Int, 2 * max_m, allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dstevr_(
			jobz_cstring,
			range_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
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
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
		return result, Info(info_int), work_size, iwork_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size, _ = dstevr(
			jobz,
			range,
			n,
			d,
			e,
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
		_, _, _, iwork_size = dstevr(
			jobz,
			range,
			n,
			d,
			e,
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
	lapack.dstevr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
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
		result.all_positive = w[0] > 0
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

// Single precision MRRR driver
sstevr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f32,
	e: []f32,
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
	result: EigenproblemResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

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

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Blas_Int

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
		isuppz = make([]Blas_Int, 2 * max_m, allocator)
	}

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.sstevr_(
			jobz_cstring,
			range_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
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
		)

		work_size = int(work_query)
		iwork_size = int(iwork_query)
		return result, Info(info_int), work_size, iwork_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		_, _, work_size, _ = sstevr(
			jobz,
			range,
			n,
			d,
			e,
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
		_, _, _, iwork_size = sstevr(
			jobz,
			range,
			n,
			d,
			e,
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
	lapack.sstevr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
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
		result.all_positive = w[0] > 0
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

stevr :: proc {
	dstevr,
	sstevr,
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE/EIGENVECTOR - BISECTION AND INVERSE ITERATION
// ============================================================================

// Double precision bisection and inverse iteration driver
dstevx :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f64, // Diagonal (preserved)
	e: []f64, // Off-diagonal (preserved)
	vl: f64 = 0, // Lower bound (if range == VALUE)
	vu: f64 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f64 = 0, // Absolute tolerance
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	work: []f64 = nil, // Workspace (size 5*n)
	iwork: []Blas_Int = nil, // Integer workspace (size 5*n)
	ifail: []Blas_Int = nil, // Failed indices (size n)
	allocator := context.allocator,
) -> (
	result: ExpertEigenResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

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

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	info_int: Blas_Int

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

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 5 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, n, allocator)
	}

	// Call LAPACK
	lapack.dstevx_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
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
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]
	}

	// Check for failures
	if info > 0 && jobz == .EIGENVECTORS {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	return
}

// Single precision bisection and inverse iteration driver
sstevx :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f32,
	e: []f32,
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f32 = 0,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: ExpertEigenResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

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

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol
	m: Blas_Int
	info_int: Blas_Int

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

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 5 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 5 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, n, allocator)
	}

	// Call LAPACK
	lapack.sstevx_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
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
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]
	}

	// Check for failures
	if info > 0 && jobz == .EIGENVECTORS {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	return
}

stevx :: proc {
	dstevx,
	sstevx,
}

// ============================================================================
// SYMMETRIC MATRIX CONDITION NUMBER ESTIMATION
// ============================================================================

// Symmetric condition result
SymmetricConditionResult :: struct {
	rcond:               f64, // Reciprocal condition number
	condition_number:    f64, // Condition number (1/rcond)
	is_singular:         bool, // True if matrix is singular
	is_well_conditioned: bool, // True if rcond > 0.1
	is_ill_conditioned:  bool, // True if rcond < machine_epsilon
}

// Complex single precision symmetric condition estimation
csycon :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Factored matrix from csytrf
	ipiv: []Blas_Int, // Pivot indices from csytrf
	anorm: f32, // 1-norm of original matrix
	work: []complex64 = nil, // Workspace (size 2*n)
	allocator := context.allocator,
) -> (
	result: SymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f32
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csycon_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
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
		result.is_singular = true
	}

	result.is_well_conditioned = rcond > 0.1
	result.is_ill_conditioned = rcond < machine_epsilon(f32)

	return
}

// Double precision symmetric condition estimation
dsycon :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	anorm: f64,
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f64
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.dsycon_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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
		result.is_singular = true
	}

	result.is_well_conditioned = rcond > 0.1
	result.is_ill_conditioned = rcond < machine_epsilon(f64)

	return
}

// Single precision symmetric condition estimation
ssycon :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	anorm: f32,
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: SymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f32
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.ssycon_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
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
		result.is_singular = true
	}

	result.is_well_conditioned = rcond > 0.1
	result.is_ill_conditioned = rcond < machine_epsilon(f32)

	return
}

// Complex double precision symmetric condition estimation
zsycon :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	anorm: f64,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: SymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f64
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsycon_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
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
		result.is_singular = true
	}

	result.is_well_conditioned = rcond > 0.1
	result.is_ill_conditioned = rcond < machine_epsilon(f64)

	return
}

sycon :: proc {
	csycon,
	dsycon,
	ssycon,
	zsycon,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Simple eigenvalue computation for tridiagonal matrix
simple_tridiagonal_eigenvalues :: proc(
	d: []$T,
	e: []T,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	info: Info,
) {
	n := len(d)

	// Make copies since they get modified
	d_copy := make([]T, n, allocator)
	copy(d_copy, d)

	e_copy := make([]T, max(n - 1, 0), allocator)
	if n > 1 {
		copy(e_copy, e[:n - 1])
	}
	defer delete(e_copy)

	when T == f64 {
		_, info_val := dsterf(n, d_copy, e_copy)
		return d_copy, info_val
	} else when T == f32 {
		_, info_val := ssterf(n, d_copy, e_copy)
		return d_copy, info_val
	} else {
		#panic("Unsupported type for simple eigenvalue computation")
	}
}

// Estimate condition number of symmetric matrix
estimate_symmetric_condition :: proc(
	a_factored: Matrix($T),
	ipiv: []Blas_Int,
	anorm: f64,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	rcond: f64,
	condition_number: f64,
	is_singular: bool,
	info: Info,
) {
	n := a_factored.rows

	when T == complex64 {
		result, info_val := csycon(uplo, n, a_factored, ipiv, f32(anorm), allocator = allocator)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else when T == complex128 {
		result, info_val := zsycon(uplo, n, a_factored, ipiv, anorm, allocator = allocator)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else when T == f64 {
		result, info_val := dsycon(uplo, n, a_factored, ipiv, anorm, allocator = allocator)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else when T == f32 {
		result, info_val := ssycon(uplo, n, a_factored, ipiv, f32(anorm), allocator = allocator)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else {
		#panic("Unsupported type for condition estimation")
	}
}
