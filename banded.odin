package openblas

import "base:builtin"
import "core:mem"

// ===================================================================================
// BANDED MATRIX TYPE DEFINITIONS AND BASIC UTILITIES
//
// This file provides the foundation for all banded matrix operations in OpenBLAS/LAPACK.
// Banded matrices are efficient for storing matrices with non-zero elements concentrated
// around the main diagonal.
//
// Banded Matrix Formats:
// - GB: General banded (general rectangular banded matrices)
// - SB: Symmetric banded (real symmetric banded matrices)
// - HB: Hermitian banded (complex Hermitian banded matrices)
// - PB: Positive definite banded (Cholesky-factorizable banded matrices)
// - TB: Triangular banded (upper/lower triangular banded matrices)
//
// Storage Format:
// - Band storage format stores only the diagonals within the band
// - For general banded: stores kl subdiagonals + main diagonal + ku superdiagonals
// - For symmetric/Hermitian: stores only upper or lower triangle bands
// - Leading dimension ldab ≥ kl + ku + 1 for general, ldab ≥ kd + 1 for symmetric
// ===================================================================================

// ===================================================================================
// BANDED MATRIX TYPE DEFINITION
// ===================================================================================

// Banded matrix storage - efficient for sparse banded systems
BandedMatrix :: struct($T: typeid) {
	data:      []T, // Band storage array, size ldab × cols
	rows:      Blas_Int, // Number of rows
	cols:      Blas_Int, // Number of columns
	kl:        Blas_Int, // Number of subdiagonals (lower bandwidth)
	ku:        Blas_Int, // Number of superdiagonals (upper bandwidth)
	ldab:      Blas_Int, // Leading dimension of band storage (>= kl + ku + 1)

	// For symmetric banded matrices
	symmetric: bool, // True if symmetric/Hermitian (only need ku or kl)
}

// ===================================================================================
// BANDED MATRIX CREATION
// ===================================================================================

// Create a general banded matrix
band_make :: proc(rows, cols, kl, ku: int, $T: typeid, allocator := context.allocator) -> BandedMatrix(T) {
	ldab := kl + ku + 1
	return BandedMatrix(T){rows = Blas_Int(rows), cols = Blas_Int(cols), kl = Blas_Int(kl), ku = Blas_Int(ku), ldab = Blas_Int(ldab), data = make([]T, ldab * cols, allocator)}
}

// Create symmetric banded matrix (only sets upper or lower band)
band_make_symmetric :: proc(n, k: int, uplo: UpLo, $T: typeid, allocator := context.allocator) -> BandedMatrix(T) {
	ldab := k + 1 // For symmetric, only need k+1 bands
	bm := BandedMatrix(T) {
		rows      = Blas_Int(n),
		cols      = Blas_Int(n),
		symmetric = true,
		ldab      = Blas_Int(ldab),
		data      = make([]T, ldab * n, allocator),
	}

	if uplo == .Upper {
		bm.ku = Blas_Int(k)
		bm.kl = 0
	} else {
		bm.kl = Blas_Int(k)
		bm.ku = 0
	}

	return bm
}

// Create a triangular banded matrix
band_make_triangular :: proc(
	n: int,
	k: int, // bandwidth
	uplo: UpLo,
	diag: Diag,
	$T: typeid,
	allocator := context.allocator,
) -> TriBand(T) where is_float(T) || is_complex(T) {
	ldab := k + 1
	data_size := ldab * n

	return TriBand(T){data = make([]T, data_size, allocator), n = Blas_Int(n), k = Blas_Int(k), ldab = Blas_Int(ldab), uplo = uplo, diag = diag}
}

// ===================================================================================
// BANDED MATRIX INDEXING AND ACCESS
// ===================================================================================

// Access element (i,j) in banded storage
// Band storage format (column-major):
// AB(ku+1+i-j, j) = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl)
band_index :: proc(bm: ^BandedMatrix($T), i, j: int) -> (idx: int, stored: bool) {
	assert(i >= 0 && i < int(bm.rows) && j >= 0 && j < int(bm.cols), "Index out of bounds")

	// Check if element is within the band
	if i > j + int(bm.kl) || i < j - int(bm.ku) {
		return -1, false
	}

	// Column-major band storage
	idx = int(bm.ku) + i - j + j * int(bm.ldab)
	return idx, true
}

// Get element (i,j) from banded matrix
band_get :: proc(bm: ^BandedMatrix($T), i, j: int) -> (value: T, stored: bool) {
	idx, is_stored := band_index(bm, i, j)
	if !is_stored {
		return T{}, false
	}
	return bm.data[idx], true
}

// Set element (i,j) in banded matrix
band_set :: proc(bm: ^BandedMatrix($T), i, j: int, value: T) -> bool {
	idx, is_stored := band_index(bm, i, j)
	if !is_stored {
		return false
	}
	bm.data[idx] = value
	return true
}

// ===================================================================================
// TRIANGULAR BANDED MATRIX INDEXING
// ===================================================================================

// Access element (i,j) in triangular banded storage
triband_index :: proc(tb: ^TriBand($T), i, j: int) -> (idx: int, stored: bool) where is_float(T) || is_complex(T) {
	assert(i >= 0 && i < int(tb.n) && j >= 0 && j < int(tb.n), "Index out of bounds")

	// Check if element is within the triangular band
	if tb.uplo == .Upper {
		if i > j || i < j - int(tb.k) {
			return -1, false
		}
		// Upper triangular: AB[k + i - j, j] = A[i,j]
		idx = int(tb.k) + i - j + j * int(tb.ldab)
	} else {
		if i < j || i > j + int(tb.k) {
			return -1, false
		}
		// Lower triangular: AB[i - j, j] = A[i,j]
		idx = i - j + j * int(tb.ldab)
	}
	return idx, true
}

// Get element (i,j) from triangular banded matrix
triband_get :: proc(tb: ^TriBand($T), i, j: int) -> (value: T, stored: bool) where is_float(T) || is_complex(T) {
	idx, is_stored := triband_index(tb, i, j)
	if !is_stored {
		return T{}, false
	}
	return tb.data[idx], true
}

// Set element (i,j) in triangular banded matrix
triband_set :: proc(tb: ^TriBand($T), i, j: int, value: T) -> bool where is_float(T) || is_complex(T) {
	idx, is_stored := triband_index(tb, i, j)
	if !is_stored {
		return false
	}
	tb.data[idx] = value
	return true
}

// ===================================================================================
// SYMMETRIC BANDED MATRIX INDEXING
// ===================================================================================

// Access element (i,j) in symmetric banded storage
symband_index :: proc(sb: ^SymBand($T), i, j: int) -> (idx: int, stored: bool) where is_float(T) {
	assert(i >= 0 && i < int(sb.n) && j >= 0 && j < int(sb.n), "Index out of bounds")

	// For symmetric, only one triangle is stored
	if sb.uplo == .Upper {
		if i > j || i < j - int(sb.kd) {
			return -1, false
		}
		// Upper triangle: AB[kd + i - j, j] = A[i,j]
		idx = int(sb.kd) + i - j + j * int(sb.ldab)
	} else {
		if i < j || i > j + int(sb.kd) {
			return -1, false
		}
		// Lower triangle: AB[i - j, j] = A[i,j]
		idx = i - j + j * int(sb.ldab)
	}
	return idx, true
}

// Get element (i,j) from symmetric banded matrix
symband_get :: proc(sb: ^SymBand($T), i, j: int) -> (value: T, stored: bool) where is_float(T) {
	// Check stored triangle first
	idx, is_stored := symband_index(sb, i, j)
	if is_stored {
		return sb.data[idx], true
	}
	// Try opposite triangle (symmetric property)
	idx, is_stored = symband_index(sb, j, i)
	if is_stored {
		return sb.data[idx], true
	}
	return T{}, false
}

// Set element (i,j) in symmetric banded matrix
symband_set :: proc(sb: ^SymBand($T), i, j: int, value: T) -> bool where is_float(T) {
	idx, is_stored := symband_index(sb, i, j)
	if !is_stored {
		return false
	}
	sb.data[idx] = value
	return true
}

// ===================================================================================
// HERMITIAN BANDED MATRIX INDEXING
// ===================================================================================

// Access element (i,j) in Hermitian banded storage
hermband_index :: proc(hb: ^HermBand($T), i, j: int) -> (idx: int, stored: bool) where is_complex(T) {
	assert(i >= 0 && i < int(hb.n) && j >= 0 && j < int(hb.n), "Index out of bounds")

	// For Hermitian, only one triangle is stored
	if hb.uplo == .Upper {
		if i > j || i < j - int(hb.kd) {
			return -1, false
		}
		// Upper triangle: AB[kd + i - j, j] = A[i,j]
		idx = int(hb.kd) + i - j + j * int(hb.ldab)
	} else {
		if i < j || i > j + int(hb.kd) {
			return -1, false
		}
		// Lower triangle: AB[i - j, j] = A[i,j]
		idx = i - j + j * int(hb.ldab)
	}
	return idx, true
}

// Get element (i,j) from Hermitian banded matrix
hermband_get :: proc(hb: ^HermBand($T), i, j: int) -> (value: T, stored: bool) where is_complex(T) {
	// Check stored triangle first
	idx, is_stored := hermband_index(hb, i, j)
	if is_stored {
		return hb.data[idx], true
	}
	// Try opposite triangle (Hermitian property - conjugate transpose)
	idx, is_stored = hermband_index(hb, j, i)
	if is_stored {
		return conj(hb.data[idx]), true
	}
	return T{}, false
}

// Set element (i,j) in Hermitian banded matrix
hermband_set :: proc(hb: ^HermBand($T), i, j: int, value: T) -> bool where is_complex(T) {
	idx, is_stored := hermband_index(hb, i, j)
	if !is_stored {
		return false
	}
	hb.data[idx] = value
	return true
}

// ===================================================================================
// BANDED MATRIX PROPERTIES AND VALIDATION
// ===================================================================================

// Get banded storage parameters from matrix
band_get_params :: proc(bm: ^BandedMatrix($T)) -> (kl, ku, ldab: int) {
	return int(bm.kl), int(bm.ku), int(bm.ldab)
}

// Check if matrix has valid banded structure
band_validate :: proc(bm: ^BandedMatrix($T)) -> bool {
	if bm.ldab < bm.kl + bm.ku + 1 {
		return false
	}
	if len(bm.data) < int(bm.ldab) * int(bm.cols) {
		return false
	}
	return true
}

// Get the bandwidth of a banded matrix
band_get_bandwidth :: proc(bm: ^BandedMatrix($T)) -> (lower: int, upper: int) {
	return int(bm.kl), int(bm.ku) // TODO: flag to compute the bandwidth?? have in auxillery iirc
}

// ===================================================================================
// MEMORY MANAGEMENT
// ===================================================================================

// Delete banded matrix data
band_delete :: proc(bm: ^BandedMatrix($T), allocator := context.allocator) {
	delete(bm.data, allocator)
	bm.data = nil
}

// Delete triangular banded matrix data
triband_delete :: proc(tb: ^TriBand($T), allocator := context.allocator) where is_float(T) || is_complex(T) {
	delete(tb.data, allocator)
	tb.data = nil
}

// Delete symmetric banded matrix data
symband_delete :: proc(sb: ^SymBand($T), allocator := context.allocator) where is_float(T) {
	delete(sb.data, allocator)
	sb.data = nil
}

// Delete Hermitian banded matrix data
hermband_delete :: proc(hb: ^HermBand($T), allocator := context.allocator) where is_complex(T) {
	delete(hb.data, allocator)
	hb.data = nil
}

// Clone a banded matrix
band_clone :: proc(bm: ^BandedMatrix($T), allocator := context.allocator) -> BandedMatrix(T) {
	clone := BandedMatrix(T) {
		rows      = bm.rows,
		cols      = bm.cols,
		kl        = bm.kl,
		ku        = bm.ku,
		ldab      = bm.ldab,
		symmetric = bm.symmetric,
		data      = make([]T, len(bm.data), allocator),
	}
	copy(clone.data, bm.data)
	return clone
}
