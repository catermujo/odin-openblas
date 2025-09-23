package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - EXTENDED MRRR WITH TRYRAC
// ============================================================================
// Extended version of MRRR with additional control over algorithm selection

// Extended MRRR result structure
ExtendedMRRRResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	num_found:        int, // Number of eigenvalues found
	num_columns_z:    int, // Number of columns in Z actually used
	support:          []Blas_Int, // Support arrays for eigenvectors (2*m)
	all_converged:    bool, // True if all eigenvectors converged
	used_tryrac:      bool, // True if tryrac algorithm was used
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
}

// Complex single precision extended MRRR
cstemr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f32, // Diagonal elements (modified)
	e: []f32, // Off-diagonal elements (modified)
	vl: f32 = 0, // Lower bound (if range == VALUE)
	vu: f32 = 0, // Upper bound (if range == VALUE)
	il: int = 0, // Lower index (if range == INDEX, 1-based)
	iu: int = 0, // Upper index (if range == INDEX, 1-based)
	nzc: int = 0, // Number of eigenvectors to compute (0 = automatic)
	tryrac: bool = true, // Try to achieve high relative accuracy
	w: []f32 = nil, // Eigenvalues (size n)
	z: Matrix(complex64) = {}, // Eigenvectors (if jobz == EIGENVECTORS)
	isuppz: []Blas_Int = nil, // Support arrays (size 2*m)
	work: []f32 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (query with liwork=-1)
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: ExtendedMRRRResult(complex64),
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
	m: Blas_Int
	nzc_int := Blas_Int(nzc)
	tryrac_int := Blas_Int(tryrac ? 1 : 0)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^complex64 = nil
	if jobz == .EIGENVECTORS {
		max_cols := nzc > 0 ? nzc : n
		assert(z.rows >= n && z.cols >= max_cols, "Eigenvector matrix too small")
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

		lapack.cstemr_(
			jobz_cstring,
			range_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&m,
			raw_data(w),
			cast(^lapack.complex)z_ptr,
			&ldz,
			&nzc_int,
			raw_data(isuppz) if isuppz != nil else nil,
			&tryrac_int,
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
		_, _, work_size, _ = cstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
		_, _, _, iwork_size = cstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
	lapack.cstemr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&m,
		raw_data(w),
		cast(^lapack.complex)z_ptr,
		&ldz,
		&nzc_int,
		raw_data(isuppz) if isuppz != nil else nil,
		&tryrac_int,
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
	result.num_columns_z = int(nzc_int)
	result.used_tryrac = tryrac

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

// Double precision extended MRRR
dstemr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f64,
	e: []f64,
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	nzc: int = 0,
	tryrac: bool = true,
	w: []f64 = nil,
	z: Matrix(f64) = {},
	isuppz: []Blas_Int = nil,
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: ExtendedMRRRResult(f64),
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
	m: Blas_Int
	nzc_int := Blas_Int(nzc)
	tryrac_int := Blas_Int(tryrac ? 1 : 0)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if jobz == .EIGENVECTORS {
		max_cols := nzc > 0 ? nzc : n
		assert(z.rows >= n && z.cols >= max_cols, "Eigenvector matrix too small")
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

		lapack.dstemr_(
			jobz_cstring,
			range_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			&nzc_int,
			raw_data(isuppz) if isuppz != nil else nil,
			&tryrac_int,
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
		_, _, work_size, _ = dstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
		_, _, _, iwork_size = dstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
	lapack.dstemr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		&nzc_int,
		raw_data(isuppz) if isuppz != nil else nil,
		&tryrac_int,
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
	result.num_columns_z = int(nzc_int)
	result.used_tryrac = tryrac

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

// Single precision extended MRRR
sstemr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f32,
	e: []f32,
	vl: f32 = 0,
	vu: f32 = 0,
	il: int = 0,
	iu: int = 0,
	nzc: int = 0,
	tryrac: bool = true,
	w: []f32 = nil,
	z: Matrix(f32) = {},
	isuppz: []Blas_Int = nil,
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: ExtendedMRRRResult(f32),
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
	m: Blas_Int
	nzc_int := Blas_Int(nzc)
	tryrac_int := Blas_Int(tryrac ? 1 : 0)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if jobz == .EIGENVECTORS {
		max_cols := nzc > 0 ? nzc : n
		assert(z.rows >= n && z.cols >= max_cols, "Eigenvector matrix too small")
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

		lapack.sstemr_(
			jobz_cstring,
			range_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&m,
			raw_data(w),
			z_ptr,
			&ldz,
			&nzc_int,
			raw_data(isuppz) if isuppz != nil else nil,
			&tryrac_int,
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
		_, _, work_size, _ = sstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
		_, _, _, iwork_size = sstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
	lapack.sstemr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&m,
		raw_data(w),
		z_ptr,
		&ldz,
		&nzc_int,
		raw_data(isuppz) if isuppz != nil else nil,
		&tryrac_int,
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
	result.num_columns_z = int(nzc_int)
	result.used_tryrac = tryrac

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

// Complex double precision extended MRRR
zstemr :: proc(
	jobz: JobzOption,
	range: EigenRangeOption,
	n: int,
	d: []f64,
	e: []f64,
	vl: f64 = 0,
	vu: f64 = 0,
	il: int = 0,
	iu: int = 0,
	nzc: int = 0,
	tryrac: bool = true,
	w: []f64 = nil,
	z: Matrix(complex128) = {},
	isuppz: []Blas_Int = nil,
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: ExtendedMRRRResult(complex128),
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
	m: Blas_Int
	nzc_int := Blas_Int(nzc)
	tryrac_int := Blas_Int(tryrac ? 1 : 0)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^complex128 = nil
	if jobz == .EIGENVECTORS {
		max_cols := nzc > 0 ? nzc : n
		assert(z.rows >= n && z.cols >= max_cols, "Eigenvector matrix too small")
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

		lapack.zstemr_(
			jobz_cstring,
			range_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			&vl_val,
			&vu_val,
			&il_int,
			&iu_int,
			&m,
			raw_data(w),
			cast(^lapack.doublecomplex)z_ptr,
			&ldz,
			&nzc_int,
			raw_data(isuppz) if isuppz != nil else nil,
			&tryrac_int,
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
		_, _, work_size, _ = zstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
		_, _, _, iwork_size = zstemr(
			jobz,
			range,
			n,
			d,
			e,
			vl,
			vu,
			il,
			iu,
			nzc,
			tryrac,
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
	lapack.zstemr_(
		jobz_cstring,
		range_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&m,
		raw_data(w),
		cast(^lapack.doublecomplex)z_ptr,
		&ldz,
		&nzc_int,
		raw_data(isuppz) if isuppz != nil else nil,
		&tryrac_int,
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
	result.num_columns_z = int(nzc_int)
	result.used_tryrac = tryrac

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

stemr :: proc {
	cstemr,
	dstemr,
	sstemr,
	zstemr,
}

// ============================================================================
// TRIDIAGONAL QR ITERATION
// ============================================================================
// Classic QR algorithm for tridiagonal eigenproblems

// QR iteration result structure
QRIterationResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	all_positive:     bool, // True if all eigenvalues > 0
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
}

// Complex single precision QR iteration
csteqr :: proc(
	compz: CompzOption,
	n: int,
	d: []f32, // Diagonal (modified to eigenvalues)
	e: []f32, // Off-diagonal (destroyed)
	z: Matrix(complex64) = {}, // Eigenvectors (if compz != NO_VECTORS)
	work: []f32 = nil, // Workspace (size 2*n-2 if compz != NO_VECTORS)
	allocator := context.allocator,
) -> (
	result: QRIterationResult(complex64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_char: u8
	switch compz {
	case .NO_VECTORS:
		compz_char = 'N'
	case .TRIDIAGONAL:
		compz_char = 'I'
	case .ORIGINAL:
		compz_char = 'V'
	}
	compz_cstring := cstring(&compz_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^complex64 = nil
	if compz != .NO_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = cast(^complex64)z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	work_size := compz == .NO_VECTORS ? 0 : 2 * n - 2
	if allocated_work && work_size > 0 {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csteqr_(
		compz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		cast(^lapack.complex)z_ptr,
		&ldz,
		raw_data(work) if work != nil else nil,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		// Convert f32 eigenvalues to complex64 result
		eigenvals := make([]complex64, n, allocator)
		for i in 0 ..< n {
			eigenvals[i] = complex(d[i], 0)
		}
		result.eigenvalues = eigenvals

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

// Double precision QR iteration
dsteqr :: proc(
	compz: CompzOption,
	n: int,
	d: []f64,
	e: []f64,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: QRIterationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_char: u8
	switch compz {
	case .NO_VECTORS:
		compz_char = 'N'
	case .TRIDIAGONAL:
		compz_char = 'I'
	case .ORIGINAL:
		compz_char = 'V'
	}
	compz_cstring := cstring(&compz_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f64 = nil
	if compz != .NO_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	work_size := compz == .NO_VECTORS ? 0 : 2 * n - 2
	if allocated_work && work_size > 0 {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsteqr_(
		compz_cstring,
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

		if abs(d[0]) > machine_epsilon(f64) {
			result.condition_number = abs(d[n - 1] / d[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

// Single precision QR iteration
ssteqr :: proc(
	compz: CompzOption,
	n: int,
	d: []f32,
	e: []f32,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: QRIterationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_char: u8
	switch compz {
	case .NO_VECTORS:
		compz_char = 'N'
	case .TRIDIAGONAL:
		compz_char = 'I'
	case .ORIGINAL:
		compz_char = 'V'
	}
	compz_cstring := cstring(&compz_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^f32 = nil
	if compz != .NO_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	work_size := compz == .NO_VECTORS ? 0 : 2 * n - 2
	if allocated_work && work_size > 0 {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssteqr_(
		compz_cstring,
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

// Complex double precision QR iteration
zsteqr :: proc(
	compz: CompzOption,
	n: int,
	d: []f64,
	e: []f64,
	z: Matrix(complex128) = {},
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: QRIterationResult(complex128),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

	compz_char: u8
	switch compz {
	case .NO_VECTORS:
		compz_char = 'N'
	case .TRIDIAGONAL:
		compz_char = 'I'
	case .ORIGINAL:
		compz_char = 'V'
	}
	compz_cstring := cstring(&compz_char)

	n_int := Blas_Int(n)
	info_int: Info

	// Handle eigenvectors
	ldz := Blas_Int(1)
	z_ptr: ^complex128 = nil
	if compz != .NO_VECTORS {
		assert(z.rows >= n && z.cols >= n, "Eigenvector matrix too small")
		ldz = Blas_Int(z.stride)
		z_ptr = cast(^complex128)z.data
		result.eigenvectors = z
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	work_size := compz == .NO_VECTORS ? 0 : 2 * n - 2
	if allocated_work && work_size > 0 {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsteqr_(
		compz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		cast(^lapack.doublecomplex)z_ptr,
		&ldz,
		raw_data(work) if work != nil else nil,
		&info_int,
		1,
	)

	info = Info(info_int)

	// Analyze eigenvalues
	if info == .OK && n > 0 {
		// Convert f64 eigenvalues to complex128 result
		eigenvals := make([]complex128, n, allocator)
		for i in 0 ..< n {
			eigenvals[i] = complex(d[i], 0)
		}
		result.eigenvalues = eigenvals

		result.min_eigenvalue = d[0]
		result.max_eigenvalue = d[n - 1]
		result.all_positive = d[0] > 0

		if abs(d[0]) > machine_epsilon(f64) {
			result.condition_number = abs(d[n - 1] / d[0])
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

steqr :: proc {
	csteqr,
	dsteqr,
	ssteqr,
	zsteqr,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Extended MRRR with high accuracy control
high_accuracy_tridiagonal_eigen :: proc(
	d: []$T,
	e: []T,
	compute_vectors := false,
	range := EigenRangeOption.ALL,
	vl: T = 0,
	vu: T = 0,
	il: int = 0,
	iu: int = 0,
	tryrac := true, // Try for high relative accuracy
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	used_tryrac: bool,
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
		result, info_val, _, _ := dstemr(
			jobz,
			range,
			n,
			d_copy,
			e_copy,
			vl,
			vu,
			il,
			iu,
			tryrac = tryrac,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, result.used_tryrac, info_val
	} else when T == f32 {
		result, info_val, _, _ := sstemr(
			jobz,
			range,
			n,
			d_copy,
			e_copy,
			vl,
			vu,
			il,
			iu,
			tryrac = tryrac,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, result.used_tryrac, info_val
	} else {
		#panic("Unsupported type for extended MRRR")
	}
}

// Classic QR algorithm for tridiagonal eigendecomposition
qr_tridiagonal_eigendecomposition :: proc(
	d: []$T,
	e: []T,
	compute_vectors := false,
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

	e_copy := make([]T, max(n - 1, 0), allocator)
	if n > 1 {
		copy(e_copy, e[:n - 1])
	}
	defer delete(e_copy)

	compz := compute_vectors ? CompzOption.TRIDIAGONAL : CompzOption.NO_VECTORS

	if compute_vectors {
		eigenvectors = create_matrix(T, n, n, allocator)
		// Initialize to identity for TRIDIAGONAL option
		for i in 0 ..< n {
			matrix_set(&eigenvectors, i, i, T(1))
		}
	}

	when T == f64 {
		result, info_val := dsteqr(
			compz,
			n,
			d_copy,
			e_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		eigenvalues = d_copy // d_copy now contains eigenvalues
		return eigenvalues, eigenvectors, info_val
	} else when T == f32 {
		result, info_val := ssteqr(
			compz,
			n,
			d_copy,
			e_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		eigenvalues = d_copy
		return eigenvalues, eigenvectors, info_val
	} else when T == complex64 {
		// For complex types, need real diagonal/off-diagonal
		d_real := make([]f32, n, allocator)
		e_real := make([]f32, max(n - 1, 0), allocator)
		defer delete(d_real)
		defer delete(e_real)

		result, info_val := csteqr(
			compz,
			n,
			d_real,
			e_real,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else when T == complex128 {
		// For complex types, need real diagonal/off-diagonal
		d_real := make([]f64, n, allocator)
		e_real := make([]f64, max(n - 1, 0), allocator)
		defer delete(d_real)
		defer delete(e_real)

		result, info_val := zsteqr(
			compz,
			n,
			d_real,
			e_real,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for QR iteration")
	}
}
