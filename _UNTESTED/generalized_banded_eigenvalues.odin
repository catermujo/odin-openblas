package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// GENERALIZED SYMMETRIC BANDED EIGENVALUE PROBLEMS
// ============================================================================
// Solves the generalized eigenvalue problem A*x = λ*B*x where A and B are
// symmetric banded matrices and B is positive definite

// Vector computation option for reduction
VectorOption :: enum {
	NO_VECTORS, // 'N' - No vectors computed
	FORM_VECTORS, // 'V' - Form transformation matrix
}

// ============================================================================
// SYMMETRIC BANDED GENERALIZED EIGENVALUE REDUCTION
// ============================================================================
// Reduces a real symmetric-definite banded generalized eigenproblem
// A*x = λ*B*x to standard form C*y = λ*y

// Double precision banded generalized reduction
dsbgst :: proc(
	vect: VectorOption,
	uplo: UpLoFlag,
	n: int,
	ka: int, // Number of superdiagonals of A
	kb: int, // Number of superdiagonals of B
	ab: Matrix(f64), // Band matrix A (modified on output)
	bb: Matrix(f64), // Band matrix B (must contain Cholesky factor)
	x: Matrix(f64) = {}, // Transformation matrix (if vect == FORM_VECTORS)
	work: []f64 = nil, // Workspace (size 2*n)
	allocator := context.allocator,
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	vect_char: u8 = vect == .FORM_VECTORS ? 'V' : 'N'
	vect_cstring := cstring(&vect_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
	info_int: Info

	// Handle transformation matrix
	ldx := Blas_Int(1)
	x_ptr: ^f64 = nil
	if vect == .FORM_VECTORS {
		assert(x.rows >= n && x.cols >= n, "Transformation matrix too small")
		ldx = Blas_Int(x.stride)
		x_ptr = x.data
	}

	// Allocate workspace if not provided
	work_size := 2 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsbgst_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
		x_ptr,
		&ldx,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	return Info(info_int)
}

// Single precision banded generalized reduction
ssbgst :: proc(
	vect: VectorOption,
	uplo: UpLoFlag,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f32),
	bb: Matrix(f32),
	x: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	vect_char: u8 = vect == .FORM_VECTORS ? 'V' : 'N'
	vect_cstring := cstring(&vect_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
	info_int: Info

	// Handle transformation matrix
	ldx := Blas_Int(1)
	x_ptr: ^f32 = nil
	if vect == .FORM_VECTORS {
		assert(x.rows >= n && x.cols >= n, "Transformation matrix too small")
		ldx = Blas_Int(x.stride)
		x_ptr = x.data
	}

	// Allocate workspace if not provided
	work_size := 2 * n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssbgst_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
		x_ptr,
		&ldx,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	return Info(info_int)
}

sbgst :: proc {
	dsbgst,
	ssbgst,
}

// ============================================================================
// SYMMETRIC BANDED GENERALIZED EIGENVALUE COMPUTATION
// ============================================================================

// Generalized eigenvalue result
GeneralizedBandedEigenResult :: struct($T: typeid) {
	eigenvalues:            []T, // Computed eigenvalues (sorted)
	eigenvectors:           Matrix(T), // Eigenvector matrix (if requested)
	b_is_positive_definite: bool, // True if B was successfully factored
	all_positive:           bool, // True if all eigenvalues > 0
	min_eigenvalue:         f64, // Smallest eigenvalue
	max_eigenvalue:         f64, // Largest eigenvalue
	condition_number:       f64, // max|λ|/min|λ|
}

// Double precision banded generalized eigenvalue
dsbgv :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ka: int, // Number of superdiagonals of A
	kb: int, // Number of superdiagonals of B
	ab: Matrix(f64), // Band matrix A (modified on output)
	bb: Matrix(f64), // Band matrix B (modified to Cholesky factor)
	w: []f64 = nil, // Eigenvalues (size n)
	z: Matrix(f64) = {}, // Eigenvectors (n x n if jobz == VALUES_VECTORS)
	work: []f64 = nil, // Workspace (size 3*n)
	allocator := context.allocator,
) -> (
	result: GeneralizedBandedEigenResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
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
	lapack.dsbgv_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
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
		result.all_positive = w[0] > 0

		if abs(result.min_eigenvalue) > machine_epsilon(f64) {
			result.condition_number = abs(result.max_eigenvalue / result.min_eigenvalue)
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

// Single precision banded generalized eigenvalue
ssbgv :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f32),
	bb: Matrix(f32),
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: GeneralizedBandedEigenResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
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
	lapack.ssbgv_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
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
		result.all_positive = w[0] > 0

		if abs(w[0]) > machine_epsilon(f32) {
			result.condition_number = f64(abs(w[n - 1] / w[0]))
		} else {
			result.condition_number = math.INF_F64
		}
	}

	return
}

sbgv :: proc {
	dsbgv,
	ssbgv,
}

// ============================================================================
// DIVIDE-AND-CONQUER GENERALIZED EIGENVALUE
// ============================================================================

// Double precision divide-and-conquer generalized eigenvalue
dsbgvd :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f64),
	bb: Matrix(f64),
	w: []f64 = nil,
	z: Matrix(f64) = {},
	work: []f64 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedBandedEigenResult(f64),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f64
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.dsbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			ab.data,
			&ldab,
			bb.data,
			&ldbb,
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

		lapack.dsbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			ab.data,
			&ldab,
			bb.data,
			&ldbb,
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
	lapack.dsbgvd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
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
		result.all_positive = w[0] > 0

		if abs(result.min_eigenvalue) > machine_epsilon(f64) {
			result.condition_number = abs(result.max_eigenvalue / result.min_eigenvalue)
		} else {
			result.condition_number = math.INF_F64
		}
	}

	work_size = len(work)
	iwork_size = len(iwork)
	return
}

// Single precision divide-and-conquer generalized eigenvalue
ssbgvd :: proc(
	jobz: EigenJobOption,
	uplo: UpLoFlag,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f32),
	bb: Matrix(f32),
	w: []f32 = nil,
	z: Matrix(f32) = {},
	work: []f32 = nil,
	lwork: int = -1,
	iwork: []Blas_Int = nil,
	liwork: int = -1,
	allocator := context.allocator,
) -> (
	result: GeneralizedBandedEigenResult(f32),
	info: Info,
	work_size: int,
	iwork_size: int,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

	jobz_char: u8 = jobz == .VALUES_VECTORS ? 'V' : 'N'
	jobz_cstring := cstring(&jobz_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
	lwork_int := Blas_Int(lwork)
	liwork_int := Blas_Int(liwork)
	info_int: Info

	// Workspace query
	if lwork == -1 || liwork == -1 {
		work_query: f32
		iwork_query: Blas_Int
		ldz := Blas_Int(1)

		lapack.ssbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			ab.data,
			&ldab,
			bb.data,
			&ldbb,
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

		lapack.ssbgvd_(
			jobz_cstring,
			uplo_cstring,
			&n_int,
			&ka_int,
			&kb_int,
			ab.data,
			&ldab,
			bb.data,
			&ldbb,
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
	lapack.ssbgvd_(
		jobz_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
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

sbgvd :: proc {
	dsbgvd,
	ssbgvd,
}

// ============================================================================
// SELECTIVE GENERALIZED EIGENVALUE
// ============================================================================

// Selective generalized eigenvalue result
SelectiveGeneralizedResult :: struct($T: typeid) {
	eigenvalues:            []T,
	eigenvectors:           Matrix(T),
	num_found:              int,
	failed_indices:         []Blas_Int,
	b_is_positive_definite: bool,
}

// Double precision selective generalized eigenvalue
dsbgvx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f64),
	bb: Matrix(f64),
	q: Matrix(f64) = {},
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
	result: SelectiveGeneralizedResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

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
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
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
	lapack.dsbgvx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
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
	result.failed_indices = ifail
	result.b_is_positive_definite = info == .OK || info > Info(n)

	return
}

// Single precision selective generalized eigenvalue
ssbgvx :: proc(
	jobz: EigenJobOption,
	range: EigenRangeOption,
	uplo: UpLoFlag,
	n: int,
	ka: int,
	kb: int,
	ab: Matrix(f32),
	bb: Matrix(f32),
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
	result: SelectiveGeneralizedResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(ka >= 0 && kb >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= ka + 1 && ab.cols >= n, "A band matrix storage too small")
	assert(bb.rows >= kb + 1 && bb.cols >= n, "B band matrix storage too small")

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
	ka_int := Blas_Int(ka)
	kb_int := Blas_Int(kb)
	ldab := Blas_Int(ab.stride)
	ldbb := Blas_Int(bb.stride)
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
	lapack.ssbgvx_(
		jobz_cstring,
		range_cstring,
		uplo_cstring,
		&n_int,
		&ka_int,
		&kb_int,
		ab.data,
		&ldab,
		bb.data,
		&ldbb,
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
	result.failed_indices = ifail
	result.b_is_positive_definite = info == .OK || info > Info(n)

	return
}

sbgvx :: proc {
	dsbgvx,
	ssbgvx,
}

// ============================================================================
// SYMMETRIC BAND TO TRIDIAGONAL REDUCTION
// ============================================================================

// Tridiagonal reduction result
TridiagonalReductionResult :: struct($T: typeid) {
	diagonal:     []T, // Diagonal elements of tridiagonal matrix
	off_diagonal: []T, // Off-diagonal elements
	q_matrix:     Matrix(T), // Orthogonal transformation matrix (if requested)
}

// Double precision band to tridiagonal reduction
dsbtrd :: proc(
	vect: VectorOption,
	uplo: UpLoFlag,
	n: int,
	kd: int, // Number of superdiagonals/subdiagonals
	ab: Matrix(f64), // Band matrix (modified on output)
	d: []f64 = nil, // Diagonal of tridiagonal (size n)
	e: []f64 = nil, // Off-diagonal of tridiagonal (size n-1)
	q: Matrix(f64) = {}, // Orthogonal matrix (if vect == FORM_VECTORS)
	work: []f64 = nil, // Workspace (size n)
	allocator := context.allocator,
) -> (
	result: TridiagonalReductionResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	vect_char: u8 = vect == .FORM_VECTORS ? 'V' : 'N'
	vect_cstring := cstring(&vect_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	info_int: Info

	// Allocate diagonal if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f64, n, allocator)
	}
	result.diagonal = d

	// Allocate off-diagonal if not provided
	allocated_e := e == nil
	if allocated_e && n > 0 {
		e = make([]f64, n - 1, allocator)
	}
	result.off_diagonal = e

	// Handle Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^f64 = nil
	if vect == .FORM_VECTORS {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
		result.q_matrix = q
	}

	// Allocate workspace if not provided
	work_size := n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsbtrd_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(d),
		raw_data(e),
		q_ptr,
		&ldq,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	return result, Info(info_int)
}

// Single precision band to tridiagonal reduction
ssbtrd :: proc(
	vect: VectorOption,
	uplo: UpLoFlag,
	n: int,
	kd: int,
	ab: Matrix(f32),
	d: []f32 = nil,
	e: []f32 = nil,
	q: Matrix(f32) = {},
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: TridiagonalReductionResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(kd >= 0, "Number of diagonals must be non-negative")
	assert(ab.rows >= kd + 1 && ab.cols >= n, "Band matrix storage too small")

	vect_char: u8 = vect == .FORM_VECTORS ? 'V' : 'N'
	vect_cstring := cstring(&vect_char)
	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	kd_int := Blas_Int(kd)
	ldab := Blas_Int(ab.stride)
	info_int: Info

	// Allocate diagonal if not provided
	allocated_d := d == nil
	if allocated_d {
		d = make([]f32, n, allocator)
	}
	result.diagonal = d

	// Allocate off-diagonal if not provided
	allocated_e := e == nil
	if allocated_e && n > 0 {
		e = make([]f32, n - 1, allocator)
	}
	result.off_diagonal = e

	// Handle Q matrix
	ldq := Blas_Int(1)
	q_ptr: ^f32 = nil
	if vect == .FORM_VECTORS {
		assert(q.rows >= n && q.cols >= n, "Q matrix too small")
		ldq = Blas_Int(q.stride)
		q_ptr = q.data
		result.q_matrix = q
	}

	// Allocate workspace if not provided
	work_size := n
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, work_size, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssbtrd_(
		vect_cstring,
		uplo_cstring,
		&n_int,
		&kd_int,
		ab.data,
		&ldab,
		raw_data(d),
		raw_data(e),
		q_ptr,
		&ldq,
		raw_data(work),
		&info_int,
		1,
		1,
	)

	return result, Info(info_int)
}

sbtrd :: proc {
	dsbtrd,
	ssbtrd,
}

// ============================================================================
// SYMMETRIC RANK-K UPDATE IN RFP FORMAT
// ============================================================================

// RFP (Rectangular Full Packed) format transpose options
RFPTranspose :: enum {
	NORMAL, // 'N' - Normal form
	TRANSPOSE, // 'T' - Transpose form
	CONJUGATE, // 'C' - Conjugate transpose (complex only)
}

// Double precision symmetric rank-k update in RFP format
dsfrk :: proc(
	transr: RFPTranspose,
	uplo: UpLoFlag,
	trans: TransposeFlag,
	n: int,
	k: int,
	alpha: f64,
	a: Matrix(f64),
	beta: f64,
	c: []f64, // RFP format array
) {
	assert(n >= 0 && k >= 0, "Dimensions must be non-negative")

	transr_char: u8
	switch transr {
	case .NORMAL:
		transr_char = 'N'
	case .TRANSPOSE:
		transr_char = 'T'
	case .CONJUGATE:
		transr_char = 'C'
	}
	transr_cstring := cstring(&transr_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	trans_char: u8
	switch trans {
	case .NoTrans:
		trans_char = 'N'
	case .Trans:
		trans_char = 'T'
	case .ConjTrans:
		trans_char = 'C'
	}
	trans_cstring := cstring(&trans_char)

	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	alpha_val := alpha
	beta_val := beta
	lda := Blas_Int(a.stride)

	// Call LAPACK
	lapack.dsfrk_(
		transr_cstring,
		uplo_cstring,
		trans_cstring,
		&n_int,
		&k_int,
		&alpha_val,
		a.data,
		&lda,
		&beta_val,
		raw_data(c),
		1,
		1,
		1,
	)
}

// Single precision symmetric rank-k update in RFP format
ssfrk :: proc(
	transr: RFPTranspose,
	uplo: UpLoFlag,
	trans: TransposeFlag,
	n: int,
	k: int,
	alpha: f32,
	a: Matrix(f32),
	beta: f32,
	c: []f32,
) {
	assert(n >= 0 && k >= 0, "Dimensions must be non-negative")

	transr_char: u8
	switch transr {
	case .NORMAL:
		transr_char = 'N'
	case .TRANSPOSE:
		transr_char = 'T'
	case .CONJUGATE:
		transr_char = 'C'
	}
	transr_cstring := cstring(&transr_char)

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	trans_char: u8
	switch trans {
	case .NoTrans:
		trans_char = 'N'
	case .Trans:
		trans_char = 'T'
	case .ConjTrans:
		trans_char = 'C'
	}
	trans_cstring := cstring(&trans_char)

	n_int := Blas_Int(n)
	k_int := Blas_Int(k)
	alpha_val := alpha
	beta_val := beta
	lda := Blas_Int(a.stride)

	// Call LAPACK
	lapack.ssfrk_(
		transr_cstring,
		uplo_cstring,
		trans_cstring,
		&n_int,
		&k_int,
		&alpha_val,
		a.data,
		&lda,
		&beta_val,
		raw_data(c),
		1,
		1,
		1,
	)
}

sfrk :: proc {
	dsfrk,
	ssfrk,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Solve generalized eigenvalue problem for banded matrices
solve_generalized_banded :: proc(
	a: Matrix($T),
	b: Matrix(T),
	ka: int,
	kb: int,
	compute_vectors := false,
	allocator := context.allocator,
) -> (
	eigenvalues: []T,
	eigenvectors: Matrix(T),
	info: Info,
) {
	n := a.cols
	jobz := compute_vectors ? EigenJobOption.VALUES_VECTORS : EigenJobOption.VALUES_ONLY

	// Make copies since they get modified
	a_copy := matrix_clone(&a, allocator)
	defer matrix_delete(&a_copy)
	b_copy := matrix_clone(&b, allocator)
	defer matrix_delete(&b_copy)

	if compute_vectors {
		eigenvectors = create_matrix(T, n, n, allocator)
	}

	when T == f64 {
		result, info_val := dsbgv(
			jobz,
			.Lower,
			n,
			ka,
			kb,
			a_copy,
			b_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else when T == f32 {
		result, info_val := ssbgv(
			jobz,
			.Lower,
			n,
			ka,
			kb,
			a_copy,
			b_copy,
			z = eigenvectors,
			allocator = allocator,
		)
		return result.eigenvalues, eigenvectors, info_val
	} else {
		#panic("Unsupported type for generalized banded eigenvalue")
	}
}
