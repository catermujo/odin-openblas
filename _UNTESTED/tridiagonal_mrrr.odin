package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - MRRR ALGORITHM
// ============================================================================
// Multiple Relatively Robust Representations - fastest algorithm for tridiagonal eigenproblems

// Job option for MRRR
JobzOption :: enum {
	NO_VECTORS, // 'N' - Eigenvalues only
	EIGENVECTORS, // 'V' - Eigenvalues and eigenvectors
}

// MRRR result structure
MRRRResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	num_found:        int, // Number of eigenvalues found
	support:          []Blas_Int, // Support arrays for eigenvectors (2*m)
	all_converged:    bool, // True if all eigenvectors converged
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
}

// Complex single precision MRRR
cstegr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f32, // Diagonal elements (modified)
	e: []f32, // Off-diagonal elements (modified)
	vl: f32 = 0, // Lower bound (if range == VALUE)
	vu: f32 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	abstol: f32 = 0, // Absolute tolerance (0 = machine precision)
	w: []f32 = nil, // Eigenvalues (size n)
	z: Matrix(complex64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	isuppz: []Blas_Int = nil, // Support arrays (size 2*m)
	work: []f32 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (query with liwork=-1)
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: MRRRResult(complex64),
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
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^complex64 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = cast(^complex64)z.data
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

		lapack.cstegr_(
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
			cast(^lapack.complex)z_ptr,
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
		_, _, work_size, _ = cstegr(
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
		_, _, _, iwork_size = cstegr(
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
	lapack.cstegr_(
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
		cast(^lapack.complex)z_ptr,
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
		// Convert f32 eigenvalues to complex64
		eigenvals := make([]complex64, result.num_found, allocator)
		for i in 0 ..< result.num_found {
			eigenvals[i] = complex(w[i], 0)
		}
		result.eigenvalues = eigenvals

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

// Double precision MRRR
dstegr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f64,
	e: []f64,
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
	result: MRRRResult(f64),
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

		lapack.dstegr_(
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
		_, _, work_size, _ = dstegr(
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
		_, _, _, iwork_size = dstegr(
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
	lapack.dstegr_(
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

// Single precision MRRR
sstegr :: proc(
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
	result: MRRRResult(f32),
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

		lapack.sstegr_(
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
		_, _, work_size, _ = sstegr(
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
		_, _, _, iwork_size = sstegr(
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
	lapack.sstegr_(
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

// Complex double precision MRRR
zstegr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f64,
	e: []f64,
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	abstol: f64 = 0,
	w: []f64 = nil,
	z: Matrix(complex128) = {},
	isuppz: []Blas_Int = nil,
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: MRRRResult(complex128),
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
	z_ptr: ^complex128 = nil
	if jobz == .EIGENVECTORS {
		assert(z.rows >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = cast(^complex128)z.data
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

		lapack.zstegr_(
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
			cast(^lapack.doublecomplex)z_ptr,
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
		_, _, work_size, _ = zstegr(
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
		_, _, _, iwork_size = zstegr(
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
	lapack.zstegr_(
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
		cast(^lapack.doublecomplex)z_ptr,
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
		// Convert f64 eigenvalues to complex128
		eigenvals := make([]complex128, result.num_found, allocator)
		for i in 0 ..< result.num_found {
			eigenvals[i] = complex(w[i], 0)
		}
		result.eigenvalues = eigenvals

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

stegr :: proc {
	cstegr,
	dstegr,
	sstegr,
	zstegr,
}

// ============================================================================
// TRIDIAGONAL INVERSE ITERATION
// ============================================================================
// Computes eigenvectors for given eigenvalues using inverse iteration

// Inverse iteration result structure
InverseIterationResult :: struct($T: typeid) {
	eigenvectors:   Matrix(T), // Computed eigenvectors
	failed_indices: []Blas_Int, // Indices of failed eigenvectors
	num_failures:   int, // Number of failed convergences
	all_converged:  bool, // True if all eigenvectors converged
}

// Complex single precision inverse iteration
cstein :: proc(
	n: int,
	d: []f32, // Diagonal elements
	e: []f32, // Off-diagonal elements
	m: int, // Number of eigenvalues
	w: []f32, // Eigenvalues
	iblock: []Blas_Int, // Block indices from stebz
	isplit: []Blas_Int, // Split points from stebz
	z: Matrix(complex64), // Eigenvector matrix (output)
	work: []f32 = nil, // Workspace (size 5*n)
	iwork: []Blas_Int = nil, // Integer workspace (size n)
	ifail: []Blas_Int = nil, // Failed indices (size m)
	allocator := context.allocator,
) -> (
	result: InverseIterationResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(m >= 0 && m <= n, "Number of eigenvalues must be between 0 and n")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= m, "Eigenvalue array too small")
	assert(len(iblock) >= m, "Block array too small")
	assert(len(isplit) >= n, "Split array too small")
	assert(z.rows >= n && z.cols >= m, "Eigenvector matrix too small")

	n_int := Blas_Int(n)
	m_int := Blas_Int(m)
	ldz := Blas_Int(z.stride)
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 5 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, m, allocator)
	}

	// Call LAPACK
	lapack.cstein_(
		&n_int,
		raw_data(d),
		raw_data(e),
		&m_int,
		raw_data(w),
		raw_data(iblock),
		raw_data(isplit),
		cast(^lapack.complex)z.data,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
	)

	info = Info(info_int)

	// Fill result
	result.eigenvectors = z

	// Check for failures
	if info > 0 {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	return
}

// Double precision inverse iteration
dstein :: proc(
	n: int,
	d: []f64,
	e: []f64,
	m: int,
	w: []f64,
	iblock: []Blas_Int,
	isplit: []Blas_Int,
	z: Matrix(f64),
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: InverseIterationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(m >= 0 && m <= n, "Number of eigenvalues must be between 0 and n")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= m, "Eigenvalue array too small")
	assert(len(iblock) >= m, "Block array too small")
	assert(len(isplit) >= n, "Split array too small")
	assert(z.rows >= n && z.cols >= m, "Eigenvector matrix too small")

	n_int := Blas_Int(n)
	m_int := Blas_Int(m)
	ldz := Blas_Int(z.stride)
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 5 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, m, allocator)
	}

	// Call LAPACK
	lapack.dstein_(
		&n_int,
		raw_data(d),
		raw_data(e),
		&m_int,
		raw_data(w),
		raw_data(iblock),
		raw_data(isplit),
		z.data,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
	)

	info = Info(info_int)

	// Fill result
	result.eigenvectors = z

	// Check for failures
	if info > 0 {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	return
}

// Single precision inverse iteration
sstein :: proc(
	n: int,
	d: []f32,
	e: []f32,
	m: int,
	w: []f32,
	iblock: []Blas_Int,
	isplit: []Blas_Int,
	z: Matrix(f32),
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: InverseIterationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(m >= 0 && m <= n, "Number of eigenvalues must be between 0 and n")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= m, "Eigenvalue array too small")
	assert(len(iblock) >= m, "Block array too small")
	assert(len(isplit) >= n, "Split array too small")
	assert(z.rows >= n && z.cols >= m, "Eigenvector matrix too small")

	n_int := Blas_Int(n)
	m_int := Blas_Int(m)
	ldz := Blas_Int(z.stride)
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 5 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, m, allocator)
	}

	// Call LAPACK
	lapack.sstein_(
		&n_int,
		raw_data(d),
		raw_data(e),
		&m_int,
		raw_data(w),
		raw_data(iblock),
		raw_data(isplit),
		z.data,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
	)

	info = Info(info_int)

	// Fill result
	result.eigenvectors = z

	// Check for failures
	if info > 0 {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	return
}

// Complex double precision inverse iteration
zstein :: proc(
	n: int,
	d: []f64,
	e: []f64,
	m: int,
	w: []f64,
	iblock: []Blas_Int,
	isplit: []Blas_Int,
	z: Matrix(complex128),
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	ifail: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: InverseIterationResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(m >= 0 && m <= n, "Number of eigenvalues must be between 0 and n")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")
	assert(len(w) >= m, "Eigenvalue array too small")
	assert(len(iblock) >= m, "Block array too small")
	assert(len(isplit) >= n, "Split array too small")
	assert(z.rows >= n && z.cols >= m, "Eigenvector matrix too small")

	n_int := Blas_Int(n)
	m_int := Blas_Int(m)
	ldz := Blas_Int(z.stride)
	info_int: Blas_Int

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 5 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	allocated_ifail := ifail == nil
	if allocated_ifail {
		ifail = make([]Blas_Int, m, allocator)
	}

	// Call LAPACK
	lapack.zstein_(
		&n_int,
		raw_data(d),
		raw_data(e),
		&m_int,
		raw_data(w),
		raw_data(iblock),
		raw_data(isplit),
		cast(^lapack.doublecomplex)z.data,
		&ldz,
		raw_data(work),
		raw_data(iwork),
		raw_data(ifail),
		&info_int,
	)

	info = Info(info_int)

	// Fill result
	result.eigenvectors = z

	// Check for failures
	if info > 0 {
		result.num_failures = int(info)
		result.failed_indices = ifail[:result.num_failures]
		result.all_converged = false
	} else {
		result.all_converged = true
	}

	return
}

stein :: proc {
	cstein,
	dstein,
	sstein,
	zstein,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Fast tridiagonal eigendecomposition using MRRR
fast_tridiagonal_eigendecomposition :: proc(
	d: []$T,
	e: []T,
	compute_vectors := false,
	range := EigenRangeOption.ALL,
	vl: T = 0,
	vu: T = 0,
	il: int = 0,
	iu: int = 0,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	n := len(d)

	// Make copies since they get modified
	d_copy := make([]T, n, allocator)
	copy(d_copy, d)
	defer delete(d_copy)

	e_copy := make([]T, max(n - 1, 0), allocator)
	if n > 1 {
		copy(e_copy, e[:n - 1])
	}
	defer delete(e_copy)

	jobz := compute_vectors ? JobzOption.EIGENVECTORS : JobzOption.NO_VECTORS

	if compute_vectors {
		eigenvectors = create_matrix(T, n, n, allocator)
	}

	when T == f64 {
		result, info_val, _, _ := dstegr(
			jobz,
			range,
			n,
			d_copy,
			e_copy,
			vl,
			vu,
			il,
			iu,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else when T == f32 {
		result, info_val, _, _ := sstegr(
			jobz,
			range,
			n,
			d_copy,
			e_copy,
			vl,
			vu,
			il,
			iu,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for MRRR eigendecomposition")
	}
}

// Compute eigenvectors for given eigenvalues
compute_eigenvectors_from_eigenvalues :: proc(
	d: []$T,
	e: []T,
	eigenvalues: []T,
	iblock: []Blas_Int,
	isplit: []Blas_Int,
	allocator := context.allocator,
) -> (
	eigenvectors: Matrix(T),
	all_converged: bool,
	info: Info,
) {
	n := len(d)
	m := len(eigenvalues)

	eigenvectors = create_matrix(T, n, m, allocator)

	when T == f64 {
		result, info_val := dstein(
			n,
			d,
			e,
			m,
			eigenvalues,
			iblock,
			isplit,
			eigenvectors,
			allocator = allocator,
		)
		return eigenvectors, result.all_converged, info_val
	} else when T == f32 {
		result, info_val := sstein(
			n,
			d,
			e,
			m,
			eigenvalues,
			iblock,
			isplit,
			eigenvectors,
			allocator = allocator,
		)
		return eigenvectors, result.all_converged, info_val
	} else when T == complex64 {
		// Extract real parts of eigenvalues
		real_eigenvalues := make([]f32, m, allocator)
		defer delete(real_eigenvalues)
		for i in 0 ..< m {
			real_eigenvalues[i] = real(eigenvalues[i])
		}

		// d and e are real for complex eigenvector computation
		d_real := make([]f32, n, allocator)
		e_real := make([]f32, max(n - 1, 0), allocator)
		defer delete(d_real)
		defer delete(e_real)

		result, info_val := cstein(
			n,
			d_real,
			e_real,
			m,
			real_eigenvalues,
			iblock,
			isplit,
			eigenvectors,
			allocator = allocator,
		)
		return eigenvectors, result.all_converged, info_val
	} else when T == complex128 {
		// Extract real parts of eigenvalues
		real_eigenvalues := make([]f64, m, allocator)
		defer delete(real_eigenvalues)
		for i in 0 ..< m {
			real_eigenvalues[i] = real(eigenvalues[i])
		}

		result, info_val := zstein(
			n,
			d,
			e,
			m,
			real_eigenvalues,
			iblock,
			isplit,
			eigenvectors,
			allocator = allocator,
		)
		return eigenvectors, result.all_converged, info_val
	} else {
		#panic("Unsupported type for inverse iteration")
	}
}
