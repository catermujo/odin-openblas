package openblas

// ===================================================================================
// ENUMS FOR ORTHOGONAL TRANSFORMATIONS
// ===================================================================================
QUERY_WORKSPACE: Blas_Int : -1
// Side for orthogonal transformation
OrthogonalSide :: enum {
	Left, // 'L' - Multiply on the left (premultiply)
	Right, // 'R' - Multiply on the right (postmultiply)
	Both, // 'C'/'T' - Multiply on both sides
}

orthogonal_side_to_cstring :: proc(side: OrthogonalSide) -> cstring {
	switch side {
	case .Left:
		return "L"
	case .Right:
		return "R"
	case .Both:
		return "C"
	}
	unreachable()
}

// Initialization mode for random orthogonal matrix
OrthogonalInit :: enum {
	Identity, // 'I' - Initialize to identity before applying transformation
	None, // 'N' - No initialization, apply to existing matrix
}

orthogonal_init_to_cstring :: proc(init: OrthogonalInit) -> cstring {
	switch init {
	case .Identity:
		return "I"
	case .None:
		return "N"
	}
	unreachable()
}

// Householder reflector side application
ReflectorSide :: enum {
	Left, // Apply from the left
	Right, // Apply from the right
}
side_to_cstring :: proc(side: ReflectorSide) -> cstring {
	switch side {
	case .Left:
		return "L"
	case .Right:
		return "R"
	}
	unreachable()
}

// Householder reflector transpose operation
ReflectorTranspose :: enum {
	None, // No transpose
	Transpose, // Transpose
	Conjugate, // Conjugate transpose (for complex matrices)
}
trans_to_cstring :: proc(trans: ReflectorTranspose) -> cstring {
	switch trans {
	case .None:
		return "N"
	case .Transpose:
		return "T"
	case .Conjugate:
		return "C"
	}
	unreachable()
}

// Householder reflector direction
ReflectorDirection :: enum {
	Forward, // Forward direction
	Backward, // Backward direction
}
direct_to_cstring :: proc(direct: ReflectorDirection) -> cstring {
	switch direct {
	case .Forward:
		return "F"
	case .Backward:
		return "B"
	}
	unreachable()
}

// Householder reflector storage
ReflectorStorage :: enum {
	ColumnWise, // Store reflectors column-wise
	RowWise, // Store reflectors row-wise
}
storev_to_cstring :: proc(storev: ReflectorStorage) -> cstring {
	switch storev {
	case .ColumnWise:
		return "C"
	case .RowWise:
		return "R"
	}
	unreachable()
}

// Random distribution types
RandomDistribution :: enum Blas_Int {
	Uniform          = 1, // Uniform distribution on (0,1)
	UniformMinus1To1 = 2, // Uniform distribution on (-1,1)
	Normal           = 3, // Normal distribution
	ComplexUniform   = 5, // Complex uniform distribution on unit circle
}

distribution_to_cstring :: proc(dist: RandomDistribution) -> cstring {
	switch dist {
	case .Uniform:
		return "U" // Uniform(0,1) - IDIST=1
	case .UniformMinus1To1:
		return "S" // Uniform(-1,1) - IDIST=2 (S for Symmetric)
	case .Normal:
		return "N" // Normal(0,1) - IDIST=3
	case .ComplexUniform:
		// IDIST=5 - Complex uniform on unit circle
		// Only supported by DLARNV/CLARNV/ZLARNV (not DLATMS/DLATMT)
		// There is no character code for this distribution
		return "" // Return empty string to indicate not supported for character-based functions
	}
	unreachable()
}

// Equilibration type for matrix scaling
EquilibrationRequest :: enum {
	None, // No equilibration applied
	Row, // Row equilibration applied
	Column, // Column equilibration applied
	Both, // Both row and column equilibration applied
}

// Convert LAPACK equilibration character to enum
equilibration_request_from_char :: proc(c: byte) -> EquilibrationRequest {
	switch c {
	case 'N':
		return .None
	case 'R':
		return .Row
	case 'C':
		return .Column
	case 'B':
		return .Both
	}
	panic("Unexpected char for Equilibriation")
}

// Convert enum to LAPACK equilibration character string
equilibration_request_to_cstring :: proc(e: EquilibrationRequest) -> cstring {
	switch e {
	case .None:
		return "N"
	case .Row:
		return "R"
	case .Column:
		return "C"
	case .Both:
		return "B"
	}
	unreachable()
}

// Convert enum to LAPACK equilibration character byte
equilibration_request_to_char :: proc(e: EquilibrationRequest) -> byte {
	switch e {
	case .None:
		return 'N'
	case .Row:
		return 'R'
	case .Column:
		return 'C'
	case .Both:
		return 'B'
	}
	unreachable()
}


EquilibrationState :: enum {
	None, // "N" - No equilibration
	Applied, // "Y" - Equilibration was applied
}

// Convert equilibration state to LAPACK character
_equilibration_state_to_char :: proc(equed: EquilibrationState) -> cstring {
	switch equed {
	case .None:
		return "N"
	case .Applied:
		return "Y"
	}
	unreachable()
}

TransposeMode :: enum {
	None, // No transpose
	Transpose, // Transpose
	ConjugateTranspose, // Conjugate transpose (Hermitian)
}
transpose_from_char :: proc(c: byte) -> TransposeMode {
	switch c {
	case 'N':
		return .None
	case 'T':
		return .Transpose
	case 'C':
		return .ConjugateTranspose
	}
	unreachable()
}

transpose_mode_to_cstring :: proc(t: TransposeMode) -> cstring {
	switch t {
	case .None:
		return "N"
	case .Transpose:
		return "T"
	case .ConjugateTranspose:
		return "C"
	}
	unreachable()
}


// Matrix initialization region enumeration
MatrixRegion :: enum {
	Full, // "A" - Full matrix
	Upper, // "U" - Upper triangular part
	Lower, // "L" - Lower triangular part
}

// Convert matrix region to LAPACK character
matrix_region_to_cstring :: proc(region: MatrixRegion) -> cstring {
	switch region {
	case .Full:
		return "A"
	case .Upper:
		return "U"
	case .Lower:
		return "L"
	}
	unreachable()
}

SortDirection :: enum {
	Increasing, // "I" - Sort in increasing order
	Decreasing, // "D" - Sort in decreasing order
}

// Convert sort direction to LAPACK character
sort_direction_to_cstring :: proc(direction: SortDirection) -> cstring {
	switch direction {
	case .Increasing:
		return "I"
	case .Decreasing:
		return "D"
	}
	unreachable()
}

// Factorization option
FactorizationOption :: enum {
	Equilibrate, // "E" - Equilibrate, then factor
	NoFactorization, // "N" - Matrix already factored
	Factor, // "F" - Factor the matrix
}

// Convert factorization option to LAPACK character
_factorization_to_char :: proc(fact: FactorizationOption) -> cstring {
	switch fact {
	case .Equilibrate:
		return "E"
	case .NoFactorization:
		return "N"
	case .Factor:
		return "F"
	}
	unreachable()
}

// ===================================================================================
// SYMMETRY TYPES
// ===================================================================================

// Matrix symmetry types for test matrix generation
MatrixSymmetry :: enum {
	None, // "N" - No symmetry (general matrix)
	Positive_Definite, // "P" - Symmetric positive definite
	Symmetric, // "S" - Symmetric
	Hermitian, // "H" - Hermitian (complex matrices)
	Hermitian_Pos_Def, // "R" - Hermitian positive definite
}

// Convert symmetry type to LAPACK character
_symmetry_to_cstring :: proc(sym: MatrixSymmetry) -> cstring {
	switch sym {
	case .None:
		return "N"
	case .Positive_Definite:
		return "P"
	case .Symmetric:
		return "S"
	case .Hermitian:
		return "H"
	case .Hermitian_Pos_Def:
		return "R"
	}
	unreachable()
}


// ===================================================================================
// PACKING TYPES
// ===================================================================================

// Matrix packing/storage types
MatrixPacking :: enum {
	No_Packing, // "N" - No packing (full storage)
	Upper_Packed, // "U" - Upper triangular packed
	Lower_Packed, // "L" - Lower triangular packed
	Banded, // "B" - Band storage
	Rectangular, // "Q" - Rectangular band
	Zero_Band, // "Z" - Zero off-diagonal bands
}

// Convert packing type to LAPACK character
_packing_to_cstring :: proc(pack: MatrixPacking) -> cstring {
	switch pack {
	case .No_Packing:
		return "N"
	case .Upper_Packed:
		return "U"
	case .Lower_Packed:
		return "L"
	case .Banded:
		return "B"
	case .Rectangular:
		return "Q"
	case .Zero_Band:
		return "Z"
	}
	unreachable()
}

// RFP (Rectangular Full Packed) format transpose options
RFPTranspose :: enum {
	NORMAL, // 'N' - Normal form
	TRANSPOSE, // 'T' - Transpose form
	CONJUGATE, // 'C' - Conjugate transpose (complex only)
}

rfp_transpose_to_cstring :: proc(trans: RFPTranspose) -> cstring {
	switch trans {
	case .NORMAL:
		return "N"
	case .TRANSPOSE:
		return "T"
	case .CONJUGATE:
		return "C"
	}
	unreachable()
}
// ===================================================================================
// MACHINE PARAMETERS
// ===================================================================================

// Machine parameter types that can be queried from LAPACK
MachineParameter :: enum {
	Epsilon, // 'E' - Relative machine precision
	SafeMinimum, // 'S' - Safe minimum (smallest normalized positive number)
	Base, // 'B' - Base of the machine (radix)
	Precision, // 'P' - Precision (eps*base)
	MantissaDigits, // 'N' - Number of mantissa digits in base
	Rounding, // 'R' - 1.0 when rounding occurs in addition, 0.0 otherwise
	MinExponent, // 'M' - Minimum exponent before underflow
	Underflow, // 'U' - Underflow threshold
	MaxExponent, // 'L' - Largest exponent before overflow
	Overflow, // 'O' - Overflow threshold
}

// Convert machine parameter to LAPACK character
machine_parameter_to_cstring :: proc(param: MachineParameter) -> cstring {
	switch param {
	case .Epsilon:
		return "E"
	case .SafeMinimum:
		return "S"
	case .Base:
		return "B"
	case .Precision:
		return "P"
	case .MantissaDigits:
		return "N"
	case .Rounding:
		return "R"
	case .MinExponent:
		return "M"
	case .Underflow:
		return "U"
	case .MaxExponent:
		return "L"
	case .Overflow:
		return "O"
	}
	unreachable()
}


machine_parameter :: proc($T: typeid, param: MachineParameter) -> f64 {
	param_char := machine_parameter_to_cstring(param)

	when T == f32 || T == complex64 {
		// For single precision and complex single
		return f64(lapack.slamch_(param_char))
	} else when T == f64 || T == complex128 {
		// For double precision and complex double
		return lapack.dlamch_(param_char)
	} else {
		// Default to double precision
		return lapack.dlamch_(param_char)
	}
}


MatrixNorm :: enum {
	OneNorm, // 1-norm (maximum column sum)
	InfinityNorm, // infinity-norm (maximum row sum)
	FrobeniusNorm, // Frobenius norm (sqrt of sum of squares)
	MaxNorm, // max norm (largest absolute value)
}
norm_to_cstring :: proc(norm: MatrixNorm) -> cstring {
	switch norm {
	case .OneNorm:
		return "1"
	case .InfinityNorm:
		return "I"
	case .FrobeniusNorm:
		return "F"
	case .MaxNorm:
		return "M"
	}
	unreachable()
}

// Eigenvalue Params:

// Vector computation option for reduction
VectorOption :: enum {
	NO_VECTORS, // 'N' - No vectors computed
	FORM_VECTORS, // 'V' - Form transformation matrix
}

vector_option_to_cstring :: proc(opt: VectorOption) -> cstring {
	switch opt {
	case .NO_VECTORS:
		return "N"
	case .FORM_VECTORS:
		return "V"
	}
	unreachable()
}

// Job option for eigenvalue computation
EigenJobOption :: enum {
	VALUES_ONLY, // 'N' - Compute eigenvalues only
	VALUES_VECTORS, // 'V' - Compute eigenvalues and eigenvectors
}

eigen_job_to_cstring :: proc(job: EigenJobOption) -> cstring {
	switch job {
	case .VALUES_ONLY:
		return "N"
	case .VALUES_VECTORS:
		return "V"
	}
	unreachable()
}

// Range option for eigenvalue/singular value selection
EigenRangeOption :: enum {
	ALL, // 'A' - All eigenvalues/singular values
	VALUE, // 'V' - Eigenvalues/singular values in range [vl, vu]
	INDEX, // 'I' - Eigenvalues/singular values with indices il to iu
}

// Alias for SVD functions (same selection mechanism)
SVDRangeOption :: EigenRangeOption

eigen_range_to_cstring :: proc(range: EigenRangeOption) -> cstring {
	switch range {
	case .ALL:
		return "A"
	case .VALUE:
		return "V"
	case .INDEX:
		return "I"
	}
	unreachable()
}

// Alias for SVD (same conversion)
svd_range_to_cstring :: eigen_range_to_cstring


// ===================================================================================
// MATRIX SCALING TYPE (for DLASCL/SLASCL/CLASCL/ZLASCL)
// ===================================================================================

// Matrix scaling type enumeration
MatrixScalingType :: enum {
	General, // "G" - General matrix
	Lower, // "L" - Lower triangular/trapezoidal
	Upper, // "U" - Upper triangular/trapezoidal
	Hessenberg, // "H" - Upper Hessenberg
	LowerBanded, // "B" - Lower banded triangular
	UpperBanded, // "Q" - Upper banded triangular
	ZeroBanded, // "Z" - Band matrix with KL=KU=0
}

// Convert scaling type to LAPACK character
scaling_type_to_cstring :: proc(scaling_type: MatrixScalingType) -> cstring {
	switch scaling_type {
	case .General:
		return "G"
	case .Lower:
		return "L"
	case .Upper:
		return "U"
	case .Hessenberg:
		return "H"
	case .LowerBanded:
		return "B"
	case .UpperBanded:
		return "Q"
	case .ZeroBanded:
		return "Z"
	}
	unreachable()
}
