package openblas

import "base:builtin"
import "core:math"
import "core:slice"

// Triangle specification
// ===================================================================================
// SHARED PROPERTY ENUMS
// ===================================================================================

// Triangle specification
UpLo :: enum u8 {
	Upper = 'U', // Upper triangle
	Lower = 'L', // Lower triangle
}

// Diagonal type for triangular matrices
Diag :: enum u8 {
	NonUnit = 'N', // Diagonal elements are stored
	Unit    = 'U', // Diagonal elements are assumed to be 1
}

// Side for operations like triangular solve
Side :: enum u8 {
	Left  = 'L', // Operation on the left
	Right = 'R', // Operation on the right
}

// Transpose operations
TransposeState :: enum u8 {
	NoTrans   = 'N', // No transpose
	Trans     = 'T', // Transpose
	ConjTrans = 'C', // Conjugate transpose (Hermitian transpose)
}

// ===================================================================================
// SPECIALIZED BANDED MATRIX TYPES
// ===================================================================================

// Triangular banded matrix (TB format for LAPACK)
// Used for triangular banded systems (e.g., tbtrs, tbcon)
TriBand :: struct($T: typeid) where is_float(T) || is_complex(T) {
	data: []T,           // Band storage array [ldab × n]
	n:    Blas_Int,      // Matrix dimension (n×n)
	k:    Blas_Int,      // Number of super/sub-diagonals
	ldab: Blas_Int,      // Leading dimension (>= k+1)
	uplo: UpLo,          // Upper or lower triangular
	diag: Diag,          // Unit or non-unit diagonal
}

// Symmetric banded matrix (SB format for LAPACK)
// Used for symmetric banded systems (e.g., sbgv, sbev)
SymBand :: struct($T: typeid) where is_float(T) {
	data: []T,           // Band storage array [ldab × n]
	n:    Blas_Int,      // Matrix dimension (n×n)
	kd:   Blas_Int,      // Number of super/sub-diagonals
	ldab: Blas_Int,      // Leading dimension (>= kd+1)
	uplo: UpLo,          // Upper or lower triangle stored
}

// Hermitian banded matrix (HB format for LAPACK)
// Used for Hermitian banded systems (e.g., hbgv, hbev)
HermBand :: struct($T: typeid) where is_complex(T) {
	data: []T,           // Band storage array [ldab × n]
	n:    Blas_Int,      // Matrix dimension (n×n)
	kd:   Blas_Int,      // Number of super/sub-diagonals
	ldab: Blas_Int,      // Leading dimension (>= kd+1)
	uplo: UpLo,          // Upper or lower triangle stored
}
