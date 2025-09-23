package openblas

import lapack "./f77"
import "core:c"
import "core:math"
import "core:mem"
import "core:slice"

// ============================================================================
// SYMMETRIC MATRIX CONDITION ESTIMATION WITH E FACTOR
// ============================================================================
// Uses the factorization with block diagonal matrix E from sytrf_rk

// Enhanced symmetric condition result
EnhancedSymmetricConditionResult :: struct {
	rcond:               f64, // Reciprocal condition number
	condition_number:    f64, // Condition number (1/rcond)
	is_singular:         bool, // True if matrix is singular
	is_well_conditioned: bool, // True if rcond > 0.1
	is_ill_conditioned:  bool, // True if rcond < machine_epsilon
}

// Complex single precision symmetric condition with E factor
csycon_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64), // Factored matrix from csytrf_rk
	e: []complex64, // Block diagonal matrix E from csytrf_rk
	ipiv: []Blas_Int, // Pivot indices from csytrf_rk
	anorm: f32, // 1-norm of original matrix
	work: []complex64 = nil, // Workspace (size 2*n)
	allocator := context.allocator,
) -> (
	result: EnhancedSymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f32
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csycon_3_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		cast(^lapack.complex)raw_data(e),
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

// Double precision symmetric condition with E factor
dsycon_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	e: []f64,
	ipiv: []Blas_Int,
	anorm: f64,
	work: []f64 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: EnhancedSymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f64
	info_int: Info

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
	lapack.dsycon_3_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(e),
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

// Single precision symmetric condition with E factor
ssycon_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	e: []f32,
	ipiv: []Blas_Int,
	anorm: f32,
	work: []f32 = nil,
	iwork: []Blas_Int = nil,
	allocator := context.allocator,
) -> (
	result: EnhancedSymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f32
	info_int: Info

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
	lapack.ssycon_3_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(e),
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

// Complex double precision symmetric condition with E factor
zsycon_3 :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	e: []complex128,
	ipiv: []Blas_Int,
	anorm: f64,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: EnhancedSymmetricConditionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(e) >= n, "E array too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	anorm_val := anorm
	rcond: f64
	info_int: Info

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, 2 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsycon_3_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		cast(^lapack.doublecomplex)raw_data(e),
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

sycon_3 :: proc {
	csycon_3,
	dsycon_3,
	ssycon_3,
	zsycon_3,
}

// ============================================================================
// SYMMETRIC MATRIX CONVERSION
// ============================================================================
// Convert between different storage formats for symmetric matrices

// Conversion direction
ConversionWay :: enum {
	CONVERT, // 'C' - Convert from standard to split format
	REVERT, // 'R' - Revert from split format to standard
}

// Conversion result
ConversionResult :: struct {
	conversion_complete: bool, // True if conversion was successful
	format_changed:      bool, // True if format actually changed
}

// Complex single precision symmetric conversion
csyconv :: proc(
	uplo: UpLoFlag,
	way: ConversionWay,
	n: int,
	a: Matrix(complex64), // Matrix to convert
	ipiv: []Blas_Int, // Pivot indices
	e: []complex64 = nil, // Block diagonal matrix E (size n)
	allocator := context.allocator,
) -> (
	result: ConversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	way_char: u8 = way == .CONVERT ? 'C' : 'R'
	way_cstring := cstring(&way_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate E if not provided
	allocated_e := e == nil
	if allocated_e {
		e = make([]complex64, n, allocator)
	}
	defer if allocated_e do delete(e)

	// Call LAPACK
	lapack.csyconv_(
		uplo_cstring,
		way_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.complex)raw_data(e),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.conversion_complete = info == .OK
	result.format_changed = true // Conversion always changes format if successful

	return
}

// Double precision symmetric conversion
dsyconv :: proc(
	uplo: UpLoFlag,
	way: ConversionWay,
	n: int,
	a: Matrix(f64),
	ipiv: []Blas_Int,
	e: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: ConversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	way_char: u8 = way == .CONVERT ? 'C' : 'R'
	way_cstring := cstring(&way_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate E if not provided
	allocated_e := e == nil
	if allocated_e {
		e = make([]f64, n, allocator)
	}
	defer if allocated_e do delete(e)

	// Call LAPACK
	lapack.dsyconv_(
		uplo_cstring,
		way_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(e),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.conversion_complete = info == .OK
	result.format_changed = true

	return
}

// Single precision symmetric conversion
ssyconv :: proc(
	uplo: UpLoFlag,
	way: ConversionWay,
	n: int,
	a: Matrix(f32),
	ipiv: []Blas_Int,
	e: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: ConversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	way_char: u8 = way == .CONVERT ? 'C' : 'R'
	way_cstring := cstring(&way_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate E if not provided
	allocated_e := e == nil
	if allocated_e {
		e = make([]f32, n, allocator)
	}
	defer if allocated_e do delete(e)

	// Call LAPACK
	lapack.ssyconv_(
		uplo_cstring,
		way_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(ipiv),
		raw_data(e),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.conversion_complete = info == .OK
	result.format_changed = true

	return
}

// Complex double precision symmetric conversion
zsyconv :: proc(
	uplo: UpLoFlag,
	way: ConversionWay,
	n: int,
	a: Matrix(complex128),
	ipiv: []Blas_Int,
	e: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: ConversionResult,
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")
	assert(len(ipiv) >= n, "Pivot array too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	way_char: u8 = way == .CONVERT ? 'C' : 'R'
	way_cstring := cstring(&way_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	info_int: Info

	// Allocate E if not provided
	allocated_e := e == nil
	if allocated_e {
		e = make([]complex128, n, allocator)
	}
	defer if allocated_e do delete(e)

	// Call LAPACK
	lapack.zsyconv_(
		uplo_cstring,
		way_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(ipiv),
		cast(^lapack.doublecomplex)raw_data(e),
		&info_int,
		1,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.conversion_complete = info == .OK
	result.format_changed = true

	return
}

syconv :: proc {
	csyconv,
	dsyconv,
	ssyconv,
	zsyconv,
}

// ============================================================================
// SYMMETRIC MATRIX EQUILIBRATION
// ============================================================================
// Compute scaling factors to equilibrate a symmetric matrix

// Equilibration result
EquilibrationResult :: struct($T: typeid) {
	scale_factors:  []T, // Scaling factors for equilibration
	scond:          f64, // Ratio of smallest to largest scale factor
	amax:           f64, // Absolute maximum element after equilibration
	is_well_scaled: bool, // True if scond > 0.1 (matrix is well-scaled)
	needs_scaling:  bool, // True if scond < 0.1 (scaling recommended)
	is_singular:    bool, // True if any scale factor is zero
}

// Complex single precision symmetric equilibration
csyequb :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex64),
	s: []f32 = nil, // Scale factors (size n)
	work: []complex64 = nil, // Workspace (size 3*n)
	allocator := context.allocator,
) -> (
	result: EquilibrationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	scond: f32
	amax: f32
	info_int: Info

	// Allocate scale factors if not provided
	allocated_s := s == nil
	if allocated_s {
		s = make([]f32, n, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex64, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.csyequb_(
		uplo_cstring,
		&n_int,
		cast(^lapack.complex)a.data,
		&lda,
		raw_data(s),
		&scond,
		&amax,
		cast(^lapack.complex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.scale_factors = s
	result.scond = f64(scond)
	result.amax = f64(amax)
	result.is_well_scaled = scond > 0.1
	result.needs_scaling = scond < 0.1 && scond > 0

	// Check for singular scaling (any scale factor is zero)
	for i in 0 ..< n {
		if s[i] == 0 {
			result.is_singular = true
			break
		}
	}

	return
}

// Double precision symmetric equilibration
dsyequb :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f64),
	s: []f64 = nil,
	work: []f64 = nil,
	allocator := context.allocator,
) -> (
	result: EquilibrationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	scond: f64
	amax: f64
	info_int: Info

	// Allocate scale factors if not provided
	allocated_s := s == nil
	if allocated_s {
		s = make([]f64, n, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f64, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.dsyequb_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(s),
		&scond,
		&amax,
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.scale_factors = s
	result.scond = scond
	result.amax = amax
	result.is_well_scaled = scond > 0.1
	result.needs_scaling = scond < 0.1 && scond > 0

	// Check for singular scaling
	for i in 0 ..< n {
		if s[i] == 0 {
			result.is_singular = true
			break
		}
	}

	return
}

// Single precision symmetric equilibration
ssyequb :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(f32),
	s: []f32 = nil,
	work: []f32 = nil,
	allocator := context.allocator,
) -> (
	result: EquilibrationResult(f32),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	scond: f32
	amax: f32
	info_int: Info

	// Allocate scale factors if not provided
	allocated_s := s == nil
	if allocated_s {
		s = make([]f32, n, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]f32, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.ssyequb_(
		uplo_cstring,
		&n_int,
		a.data,
		&lda,
		raw_data(s),
		&scond,
		&amax,
		raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.scale_factors = s
	result.scond = f64(scond)
	result.amax = f64(amax)
	result.is_well_scaled = scond > 0.1
	result.needs_scaling = scond < 0.1 && scond > 0

	// Check for singular scaling
	for i in 0 ..< n {
		if s[i] == 0 {
			result.is_singular = true
			break
		}
	}

	return
}

// Complex double precision symmetric equilibration
zsyequb :: proc(
	uplo: UpLoFlag,
	n: int,
	a: Matrix(complex128),
	s: []f64 = nil,
	work: []complex128 = nil,
	allocator := context.allocator,
) -> (
	result: EquilibrationResult(f64),
	info: Info,
) {
	assert(n >= 0, "Matrix dimension must be non-negative")
	assert(a.rows >= n && a.cols >= n, "Matrix too small")

	uplo_char: u8 = uplo == .Upper ? 'U' : 'L'
	uplo_cstring := cstring(&uplo_char)

	n_int := Blas_Int(n)
	lda := Blas_Int(a.stride)
	scond: f64
	amax: f64
	info_int: Info

	// Allocate scale factors if not provided
	allocated_s := s == nil
	if allocated_s {
		s = make([]f64, n, allocator)
	}

	// Allocate workspace if not provided
	allocated_work := work == nil
	if allocated_work {
		work = make([]complex128, 3 * n, allocator)
	}
	defer if allocated_work do delete(work)

	// Call LAPACK
	lapack.zsyequb_(
		uplo_cstring,
		&n_int,
		cast(^lapack.doublecomplex)a.data,
		&lda,
		raw_data(s),
		&scond,
		&amax,
		cast(^lapack.doublecomplex)raw_data(work),
		&info_int,
		1,
	)

	info = Info(info_int)

	// Fill result
	result.scale_factors = s
	result.scond = scond
	result.amax = amax
	result.is_well_scaled = scond > 0.1
	result.needs_scaling = scond < 0.1 && scond > 0

	// Check for singular scaling
	for i in 0 ..< n {
		if s[i] == 0 {
			result.is_singular = true
			break
		}
	}

	return
}

syequb :: proc {
	csyequb,
	dsyequb,
	ssyequb,
	zsyequb,
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

// Estimate condition number with enhanced factorization
estimate_condition_enhanced :: proc(
	a_factored: Matrix($T),
	e: []T,
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
		// Need to handle real/complex E array conversion
		e_complex := make([]complex64, len(e), allocator)
		defer delete(e_complex)
		for i, v in e {
			when T == complex64 {
				e_complex[i] = v
			} else {
				e_complex[i] = complex(f32(v), 0)
			}
		}
		result, info_val := csycon_3(
			uplo,
			n,
			a_factored,
			e_complex,
			ipiv,
			f32(anorm),
			allocator = allocator,
		)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else when T == complex128 {
		e_complex := make([]complex128, len(e), allocator)
		defer delete(e_complex)
		for i, v in e {
			when T == complex128 {
				e_complex[i] = v
			} else {
				e_complex[i] = complex(f64(v), 0)
			}
		}
		result, info_val := zsycon_3(
			uplo,
			n,
			a_factored,
			e_complex,
			ipiv,
			anorm,
			allocator = allocator,
		)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else when T == f64 {
		result, info_val := dsycon_3(uplo, n, a_factored, e, ipiv, anorm, allocator = allocator)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else when T == f32 {
		result, info_val := ssycon_3(
			uplo,
			n,
			a_factored,
			e,
			ipiv,
			f32(anorm),
			allocator = allocator,
		)
		return result.rcond, result.condition_number, result.is_singular, info_val
	} else {
		#panic("Unsupported type for enhanced condition estimation")
	}
}

// Convert symmetric matrix storage format
convert_symmetric_format :: proc(
	a: Matrix($T),
	ipiv: []Blas_Int,
	way: ConversionWay,
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	e: []T,
	conversion_complete: bool,
	info: Info,
) {
	n := a.rows
	e = make([]T, n, allocator)

	when T == complex64 {
		result, info_val := csyconv(uplo, way, n, a, ipiv, e, allocator)
		return e, result.conversion_complete, info_val
	} else when T == complex128 {
		result, info_val := zsyconv(uplo, way, n, a, ipiv, e, allocator)
		return e, result.conversion_complete, info_val
	} else when T == f64 {
		result, info_val := dsyconv(uplo, way, n, a, ipiv, e, allocator)
		return e, result.conversion_complete, info_val
	} else when T == f32 {
		result, info_val := ssyconv(uplo, way, n, a, ipiv, e, allocator)
		return e, result.conversion_complete, info_val
	} else {
		#panic("Unsupported type for symmetric conversion")
	}
}

// Equilibrate symmetric matrix
equilibrate_symmetric :: proc(
	a: Matrix($T),
	uplo := UpLoFlag.Lower,
	allocator := context.allocator,
) -> (
	scale_factors: []$S,
	scond: f64,
	needs_scaling: bool,
	info: Info,
) {
	n := a.rows

	when T == complex64 {
		result, info_val := csyequb(uplo, n, a, allocator = allocator)
		return result.scale_factors, result.scond, result.needs_scaling, info_val
	} else when T == complex128 {
		result, info_val := zsyequb(uplo, n, a, allocator = allocator)
		return result.scale_factors, result.scond, result.needs_scaling, info_val
	} else when T == f64 {
		result, info_val := dsyequb(uplo, n, a, allocator = allocator)
		return result.scale_factors, result.scond, result.needs_scaling, info_val
	} else when T == f32 {
		result, info_val := ssyequb(uplo, n, a, allocator = allocator)
		return result.scale_factors, result.scond, result.needs_scaling, info_val
	} else {
		#panic("Unsupported type for equilibration")
	}
}

// Apply equilibration scaling to symmetric matrix
apply_symmetric_scaling :: proc(a: Matrix($T), s: []$S, uplo := UpLoFlag.Lower) {
	n := a.rows
	assert(len(s) >= n, "Scale factor array too small")

	if uplo == .Upper {
		for i in 0 ..< n {
			for j in i ..< n {
				val := matrix_get(a, i, j)
				when T == complex64 || T == complex128 {
					scaled := val * complex(S(s[i] * s[j]), 0)
				} else {
					scaled := val * T(s[i] * s[j])
				}
				matrix_set(&a, i, j, scaled)
			}
		}
	} else {
		for i in 0 ..< n {
			for j in 0 ..= i {
				val := matrix_get(a, i, j)
				when T == complex64 || T == complex128 {
					scaled := val * complex(S(s[i] * s[j]), 0)
				} else {
					scaled := val * T(s[i] * s[j])
				}
				matrix_set(&a, i, j, scaled)
			}
		}
	}
}
