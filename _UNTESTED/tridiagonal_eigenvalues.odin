package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// PACKED SYMMETRIC SOLVE WITH FACTORIZATION
// ============================================================================
// Solves A*X = B using the factorization from sptrf

// Complex single precision packed solve with factorization
csptrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex64, // Factored packed matrix from csptrf
	ipiv: []Blas_Int, // Pivot indices from csptrf
	b: Matrix(complex64), // Right-hand side (modified to solution)
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.csptrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.complex)raw_data(ap),
		raw_data(ipiv),
		cast(^lapack.complex)b.data,
		&ldb,
		&info_int,
		1,
	)

	return Info(info_int)
}

// Double precision packed solve with factorization
dsptrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f64,
	ipiv: []Blas_Int,
	b: Matrix(f64),
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.dsptrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	return Info(info_int)
}

// Single precision packed solve with factorization
ssptrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []f32,
	ipiv: []Blas_Int,
	b: Matrix(f32),
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.ssptrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		raw_data(ap),
		raw_data(ipiv),
		b.data,
		&ldb,
		&info_int,
		1,
	)

	return Info(info_int)
}

// Complex double precision packed solve with factorization
zsptrs :: proc(
	uplo: UpLoFlag,
	n: int,
	nrhs: int,
	ap: []complex128,
	ipiv: []Blas_Int,
	b: Matrix(complex128),
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(nrhs >= 0, "Number of right-hand sides must be non-negative")
	assert(len(ap) >= n * (n + 1) / 2, "Packed array too small")
	assert(len(ipiv) >= n, "Pivot array too small")
	assert(b.rows >= n && b.cols >= nrhs, "RHS matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	nrhs_int := Blas_Int(nrhs)
	ldb := Blas_Int(b.stride)
	info_int: Info

	// Call LAPACK
	lapack.zsptrs_(
		uplo_cstring,
		&n_int,
		&nrhs_int,
		cast(^lapack.doublecomplex)raw_data(ap),
		raw_data(ipiv),
		cast(^lapack.doublecomplex)b.data,
		&ldb,
		&info_int,
		1,
	)

	return Info(info_int)
}

sptrs :: proc {
	csptrs,
	dsptrs,
	ssptrs,
	zsptrs,
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - BISECTION
// ============================================================================
// Computes selected eigenvalues of a tridiagonal matrix using bisection

// Eigenvalue ordering option
EigenvalueOrder :: enum {
	BLOCKS, // 'B' - Eigenvalues ordered by blocks
	ENTIRE, // 'E' - Entire matrix eigenvalues in order
}

// Bisection result for tridiagonal eigenvalues
BisectionResult :: struct($T: typeid) {
	eigenvalues:   []T, // Computed eigenvalues
	num_found:     int, // Number of eigenvalues found
	num_splits:    int, // Number of diagonal blocks
	block_indices: []Blas_Int, // Block index for each eigenvalue
	split_points:  []Blas_Int, // Split points in the matrix
}

// Double precision tridiagonal eigenvalue bisection
dstebz :: proc(
	range: EigenRangeOption,
	order: EigenvalueOrder,
	n: int,
	vl: f64, // Lower bound (if range == VALUE)
	vu: f64, // Upper bound (if range == VALUE)
	il: int, // Lower index (if range == INDEX, 1-based)
	iu: int, // Upper index (if range == INDEX, 1-based)
	abstol: f64, // Absolute tolerance
	d: []f64, // Diagonal elements
	e: []f64, // Off-diagonal elements
	w: []f64 = nil, // Eigenvalues (size n)
	work: []f64 = nil, // Workspace (size 4*n)
	iwork: []Blas_Int = nil, // Integer workspace (size 3*n)
	allocator := context.allocator,
) -> (
	result: BisectionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

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

	order_char: u8 = order == .ENTIRE ? 'E' : 'B'
	order_cstring := cstring(&order_char)

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol

	m: Blas_Int
	nsplit: Blas_Int
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f64, n, allocator)
	}

	// Allocate block and split arrays
	iblock := make([]Blas_Int, n, allocator)
	isplit := make([]Blas_Int, n, allocator)

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 4 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 3 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.dstebz_(
		range_cstring,
		order_cstring,
		&n_int,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		raw_data(d),
		raw_data(e),
		&m,
		&nsplit,
		raw_data(w),
		raw_data(iblock),
		raw_data(isplit),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	result.num_splits = int(nsplit)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]
		result.block_indices = iblock[:result.num_found]
	}
	if result.num_splits > 0 {
		result.split_points = isplit[:result.num_splits]
	}

	return
}

// Single precision tridiagonal eigenvalue bisection
sstebz :: proc(
	range: EigenRangeOption,
	order: EigenvalueOrder,
	n: int,
	vl: f32,
	vu: f32,
	il: int,
	iu: int,
	abstol: f32,
	d: []f32,
	e: []f32,
	w: []f32 = nil,
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: BisectionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(len(d) >= n, "Diagonal array too small")
	assert(len(e) >= n - 1 || n <= 1, "Off-diagonal array too small")

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

	order_char: u8 = order == .ENTIRE ? 'E' : 'B'
	order_cstring := cstring(&order_char)

	n_int := Blas_Int(n)
	vl_val := vl
	vu_val := vu
	il_int := Blas_Int(il)
	iu_int := Blas_Int(iu)
	abstol_val := abstol

	m: Blas_Int
	nsplit: Blas_Int
	info_int: Info

	// Allocate eigenvalues if not provided
	allocated_w := w == nil
	if allocated_w {
		w = make([]f32, n, allocator)
	}

	// Allocate block and split arrays
	iblock := make([]Blas_Int, n, allocator)
	isplit := make([]Blas_Int, n, allocator)

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 4 * n, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		iwork = make([]Blas_Int, 3 * n, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	// Call LAPACK
	lapack.sstebz_(
		range_cstring,
		order_cstring,
		&n_int,
		&vl_val,
		&vu_val,
		&il_int,
		&iu_int,
		&abstol_val,
		raw_data(d),
		raw_data(e),
		&m,
		&nsplit,
		raw_data(w),
		raw_data(iblock),
		raw_data(isplit),
		raw_data(work),
		raw_data(iwork),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.num_found = int(m)
	result.num_splits = int(nsplit)
	if result.num_found > 0 {
		result.eigenvalues = w[:result.num_found]
		result.block_indices = iblock[:result.num_found]
	}
	if result.num_splits > 0 {
		result.split_points = isplit[:result.num_splits]
	}

	return
}

stebz :: proc {
	dstebz,
	sstebz,
}

// ============================================================================
// TRIDIAGONAL EIGENVALUE COMPUTATION - DIVIDE AND CONQUER
// ============================================================================
// Computes all eigenvalues and optionally eigenvectors using divide-and-conquer

// Eigenvector computation option
CompzOption :: enum {
	NO_VECTORS, // 'N' - No eigenvectors
	TRIDIAGONAL, // 'I' - Eigenvectors of tridiagonal matrix
	ORIGINAL, // 'V' - Eigenvectors of original matrix (update Z)
}

// Divide-and-conquer result for tridiagonal eigenvalues
DivideConquerResult :: struct($T: typeid) {
	eigenvalues:      []T, // Computed eigenvalues (sorted)
	eigenvectors:     Matrix(T), // Eigenvector matrix (if requested)
	all_positive:     bool, // True if all eigenvalues > 0
	min_eigenvalue:   f64, // Smallest eigenvalue
	max_eigenvalue:   f64, // Largest eigenvalue
	condition_number: f64, // max|λ|/min|λ|
}

// Complex single precision divide-and-conquer
cstedc :: proc(
	compz: CompzOption,
	n: int,
	d: []f32, // Diagonal (modified to eigenvalues)
	e: []f32, // Off-diagonal (destroyed)
	z: Matrix(complex64) = {}, // Eigenvectors (if compz != NO_VECTORS)
	work: []complex64 = nil, // Workspace (query with lwork=-1)
	lwork: int = -1,
	rwork: []f32 = nil, // Real workspace (query with lrwork=-1)
	lrwork: int = -1,
	iwork: []Blas_Int = nil, // Integer workspace (query with liwork=-1)
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: DivideConquerResult(complex64),
	info: Info,
	work_size: int,
	rwork_size: int,
	iwork_size: int,
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
	lwork_int := Blas_Int(lwork)
	lrwork_int := Blas_Int(lrwork)
	liwork_int := Blas_Int(liwork)
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

	// Workspace query
	if lwork == -1 || lrwork == -1 || liwork == -1 {
		work_query: complex64
		rwork_query: f32
		iwork_query: Blas_Int

		lapack.cstedc_(
			compz_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			cast(^lapack.complex)z_ptr,
			&ldz,
			cast(^lapack.complex)&work_query,
			&lwork_int,
			&rwork_query,
			&lrwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
		)

		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
		return result, Info(info_int), work_size, rwork_size, iwork_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		// Query for optimal workspace
		work_size, rwork_size, iwork_size = cstedc(compz, n, d, e, z, lwork = -1)
		work = make([]complex64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		if lrwork == -1 {
			_, _, rwork_size, _ = cstedc(compz, n, d, e, z, lwork = -1)
		} else {
			rwork_size = lrwork
		}
		rwork = make([]f32, rwork_size, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		if liwork == -1 {
			_, _, _, iwork_size = cstedc(compz, n, d, e, z, lwork = -1)
		} else {
			iwork_size = liwork
		}
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	lrwork_int = Blas_Int(len(rwork))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.cstedc_(
		compz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		cast(^lapack.complex)z_ptr,
		&ldz,
		cast(^lapack.complex)raw_data(work),
		&lwork_int,
		raw_data(rwork),
		&lrwork_int,
		raw_data(iwork),
		&liwork_int,
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

	work_size = len(work)
	rwork_size = len(rwork)
	iwork_size = len(iwork)
	return
}

// Double precision divide-and-conquer
dstedc :: proc(
	compz: CompzOption,
	n: int,
	d: []f64,
	e: []f64,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: DivideConquerResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
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
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
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

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int

		lapack.dstedc_(
			compz_cstring,
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
	allocated_iwork := iwork == nil

	if allocated_work || allocated_iwork {
		// Query for optimal workspace
		work_query: f64
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.dstedc_(
			compz_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_query,
			&iwork_query,
			&liwork_query,
			&info_int,
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
	lapack.dstedc_(
		compz_cstring,
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

// Single precision divide-and-conquer
sstedc :: proc(
	compz: CompzOption,
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
	result: DivideConquerResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
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
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
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

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int

		lapack.sstedc_(
			compz_cstring,
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
	allocated_iwork := iwork == nil

	if allocated_work || allocated_iwork {
		// Query for optimal workspace
		work_query: f32
		iwork_query: Blas_Int
		lwork_query := Blas_Int(-1)
		liwork_query := Blas_Int(-1)

		lapack.sstedc_(
			compz_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			z_ptr,
			&ldz,
			&work_query,
			&lwork_query,
			&iwork_query,
			&liwork_query,
			&info_int,
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
	lapack.sstedc_(
		compz_cstring,
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

// Complex double precision divide-and-conquer
zstedc :: proc(
	compz: CompzOption,
	n: int,
	d: []f64,
	e: []f64,
	z: Matrix(complex128) = {},
	work: []complex128 = nil,
	lwork: int = -1,
	rwork: []f64 = nil,
	lrwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: DivideConquerResult(complex128),
	info: Info,
	work_size: int,
	rwork_size: int,
	iwork_size: int,
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
	lwork_int := Blas_Int(lwork)
	lrwork_int := Blas_Int(lrwork)
	liwork_int := Blas_Int(liwork)
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

	// Workspace query
	if lwork == -1 || lrwork == -1 || liwork == -1 {
		work_query: complex128
		rwork_query: f64
		iwork_query: Blas_Int

		lapack.zstedc_(
			compz_cstring,
			&n_int,
			raw_data(d),
			raw_data(e),
			cast(^lapack.doublecomplex)z_ptr,
			&ldz,
			cast(^lapack.doublecomplex)&work_query,
			&lwork_int,
			&rwork_query,
			&lrwork_int,
			&iwork_query,
			&liwork_int,
			&info_int,
			1,
		)

		work_size = int(real(work_query))
		rwork_size = int(rwork_query)
		iwork_size = int(iwork_query)
		return result, Info(info_int), work_size, rwork_size, iwork_size
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work_size, _, _ = zstedc(compz, n, d, e, z, lwork = -1)
		work = make([]complex128, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	allocated_rwork := rwork == nil
	if allocated_rwork {
		_, rwork_size, _ = zstedc(compz, n, d, e, z, lwork = -1)
		rwork = make([]f64, rwork_size, allocator)
	}
	defer if allocated_rwork do delete(rwork)

	allocated_iwork := iwork == nil
	if allocated_iwork {
		_, _, iwork_size = zstedc(compz, n, d, e, z, lwork = -1)
		iwork = make([]Blas_Int, iwork_size, allocator)
	}
	defer if allocated_iwork do delete(iwork)

	lwork_int = Blas_Int(len(work))
	lrwork_int = Blas_Int(len(rwork))
	liwork_int = Blas_Int(len(iwork))

	// Call LAPACK
	lapack.zstedc_(
		compz_cstring,
		&n_int,
		raw_data(d),
		raw_data(e),
		cast(^lapack.doublecomplex)z_ptr,
		&ldz,
		cast(^lapack.doublecomplex)raw_data(work),
		&lwork_int,
		raw_data(rwork),
		&lrwork_int,
		raw_data(iwork),
		&liwork_int,
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

	work_size = len(work)
	rwork_size = len(rwork)
	iwork_size = len(iwork)
	return
}

stedc :: proc {
	cstedc,
	dstedc,
	sstedc,
	zstedc,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Solve packed symmetric system using factorization
solve_packed_with_factorization :: proc(
	ap: []$T,
	ipiv: []Blas_Int,
	b: Matrix(T),
	uplo := UpLoFlag.Lower,
) -> (
	solution: Matrix(T),
	info: Info,
) {
	n := b.rows
	nrhs := b.cols

	// Solution overwrites b
	solution = b

	when T == complex64 {
		info = csptrs(uplo, n, nrhs, ap, ipiv, solution)
	} else when T == complex128 {
		info = zsptrs(uplo, n, nrhs, ap, ipiv, solution)
	} else when T == f64 {
		info = dsptrs(uplo, n, nrhs, ap, ipiv, solution)
	} else when T == f32 {
		info = ssptrs(uplo, n, nrhs, ap, ipiv, solution)
	} else {
		#panic("Unsupported type for packed solve")
	}

	return
}

// Find selected eigenvalues of tridiagonal matrix
find_tridiagonal_eigenvalues :: proc(
	d: []$T,
	e: []T,
	range := EigenRangeOption.ALL,
	vl: T = 0,
	vu: T = 0,
	il: int = 0,
	iu: int = 0,
	abstol: T = 0,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	num_found: int,
	info: Info,
) {
	n := len(d)

	// Make copies since they might be modified
	d_copy := make([]T, n, allocator)
	copy(d_copy, d)
	defer delete(d_copy)

	e_copy := make([]T, max(n - 1, 0), allocator)
	if n > 1 {
		copy(e_copy, e[:n - 1])
	}
	defer delete(e_copy)

	when T == f64 {
		result, info_val := dstebz(
			range,
			.ENTIRE,
			n,
			vl,
			vu,
			il,
			iu,
			abstol,
			d_copy,
			e_copy,
			allocator = allocator,
		)
		defer {
			delete(result.block_indices)
			delete(result.split_points)
		}
		return result.eigenvalues, result.num_found, info_val
	} else when T == f32 {
		result, info_val := sstebz(
			range,
			.ENTIRE,
			n,
			vl,
			vu,
			il,
			iu,
			abstol,
			d_copy,
			e_copy,
			allocator = allocator,
		)
		defer {
			delete(result.block_indices)
			delete(result.split_points)
		}
		return result.eigenvalues, result.num_found, info_val
	} else {
		#panic("Unsupported type for tridiagonal eigenvalue bisection")
	}
}

// Compute all eigenvalues and eigenvectors of tridiagonal matrix
tridiagonal_eigen_decomposition :: proc(
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
		result, info_val, _, _ := dstedc(
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
		result, info_val, _, _ := sstedc(
			compz,
			n,
			d_copy,
			e_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		eigenvalues = d_copy
		return eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for tridiagonal divide-and-conquer")
	}
}
