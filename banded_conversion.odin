package openblas

import "base:builtin"
import "core:mem"

// ===================================================================================
// BANDED MATRIX CONVERSION FUNCTIONS
//
// This file provides conversion functions for BandedMatrix types:
// - Dense Matrix ↔ BandedMatrix (format conversions)
// - BandedMatrix internal conversions (general ↔ symmetric)
// - LAPACK format parameter extraction
// - Matrix analysis and bandwidth detection
//
// Note: BandedMatrix is now a standalone type, not a Matrix variant.
// For dense matrix operations, use the standard Matrix type.
// ===================================================================================

// ===================================================================================
// DENSE TO BANDED CONVERSION
// ===================================================================================

// ===================================================================================
// DENSE TO BANDED CONVERSION
// ===================================================================================

// Convert a dense matrix to banded format
// Only elements within the specified band are copied
band_from_dns :: proc(source: ^Matrix($T), kl, ku: int, allocator := context.allocator) -> BandedMatrix(T) where is_float(T) || is_complex(T) {
	assert(source.format == .General, "Source matrix must be in general format")

	dest := band_make(int(source.rows), int(source.cols), kl, ku, T, allocator)

	// FIXME: LAPACK COPY??

	// Copy elements within the band
	for j in 0 ..< int(source.cols) {
		start_i := max(0, j - ku)
		end_i := min(int(source.rows), j + kl + 1)

		for i in start_i ..< end_i {
			// Get element from source matrix (column-major)
			src_idx := i + j * int(source.ld)
			value := source.data[src_idx]

			band_set(&dest, i, j, value)
		}
	}

	return dest
}

// Convert a dense symmetric/Hermitian matrix to symmetric banded format
band_from_dns_symmetric :: proc(source: ^Matrix($T), kd: int, uplo: UpLo, allocator := context.allocator) -> BandedMatrix(T) where is_float(T) || is_complex(T) {
	assert(source.format == .Symmetric || source.format == .Hermitian, "Source matrix must be symmetric or Hermitian")
	assert(source.rows == source.cols, "Source matrix must be square")

	dest := band_make_symmetric(int(source.rows), kd, uplo, T, allocator)

	n := int(source.rows)
	ldab := int(dest.ldab)

	if uplo == .Upper {
		// Store upper triangle in band format
		for j in 0 ..< n {
			start_i := max(0, j - kd)
			for i in start_i ..= j {
				// Source element (column-major)
				src_idx := i + j * int(source.ld)
				value := source.data[src_idx]

				// Band storage for upper: AB[kd + i - j + j * ldab] = A[i,j]
				band_idx := kd + i - j + j * ldab
				dest.data[band_idx] = value
			}
		}
	} else {
		// Store lower triangle in band format
		for j in 0 ..< n {
			end_i := min(n, j + kd + 1)
			for i in j ..< end_i {
				// Source element (column-major)
				src_idx := i + j * int(source.ld)
				value := source.data[src_idx]

				// Band storage for lower: AB[i - j + j * ldab] = A[i,j]
				band_idx := i - j + j * ldab
				dest.data[band_idx] = value
			}
		}
	}

	return dest
}

// ===================================================================================
// BANDED TO DENSE CONVERSION
// ===================================================================================

// Convert banded matrix back to dense format
// Zero elements outside the band
dns_from_band :: proc(source: ^BandedMatrix($T), allocator := context.allocator) -> Matrix(T) where is_float(T) || is_complex(T) {
	rows := int(source.rows)
	cols := int(source.cols)

	dest := Matrix(T) {
		data   = make([]T, rows * cols, allocator),
		rows   = source.rows,
		cols   = source.cols,
		ld     = source.rows,
		format = .General,
	}

	// TODO: explicit zero or rely on ZII??

	kl := int(source.kl)
	ku := int(source.ku)
	ldab := int(source.ldab)

	for j in 0 ..< cols {
		start_i := max(0, j - ku)
		end_i := min(rows, j + kl + 1)

		for i in start_i ..< end_i {
			// Get from band storage
			band_idx := ku + i - j + j * ldab
			value := source.data[band_idx]

			// Store in general format (column-major)
			dest.data[i + j * rows] = value
		}
	}

	return dest
}

// Convert symmetric banded matrix to full dense symmetric matrix
dns_from_band_symmetric :: proc(source: ^BandedMatrix($T), uplo: UpLo, allocator := context.allocator) -> Matrix(T) where is_float(T) || is_complex(T) {
	assert(source.symmetric, "Source matrix must be symmetric")
	assert(source.rows == source.cols, "Source matrix must be square")

	n := int(source.rows)
	dest := Matrix(T) {
		data   = make([]T, n * n, allocator),
		rows   = source.rows,
		cols   = source.cols,
		ld     = source.rows,
		format = .Symmetric,
		uplo   = uplo,
	}

	// TODO: explicit zero or rely on ZII??

	kd := int(source.ku) // For symmetric, ku = kd or kl = kd
	if source.kl > 0 {
		kd = int(source.kl)
	}
	ldab := int(source.ldab)

	if uplo == .Upper {
		// Reconstruct from upper triangle band storage
		for j in 0 ..< n {
			start_i := max(0, j - kd)
			for i in start_i ..= j {
				// Get from band storage
				band_idx := kd + i - j + j * ldab
				value := source.data[band_idx]

				// Store in full format (both triangles)
				dest.data[i + j * n] = value
				if i != j {
					when is_complex(T) {
						dest.data[j + i * n] = conj(value) // Hermitian
					} else {
						dest.data[j + i * n] = value // Symmetric
					}
				}
			}
		}
	} else {
		// Reconstruct from lower triangle band storage
		for j in 0 ..< n {
			end_i := min(n, j + kd + 1)
			for i in j ..< end_i {
				// Get from band storage
				band_idx := i - j + j * ldab
				value := source.data[band_idx]

				// Store in full format (both triangles)
				dest.data[i + j * n] = value
				if i != j {
					when is_complex(T) {
						dest.data[j + i * n] = conj(value) // Hermitian
					} else {
						dest.data[j + i * n] = value // Symmetric
					}
				}
			}
		}
	}

	return dest
}

// ===================================================================================
// MATRIX PROPERTIES AND VALIDATION
// ===================================================================================

// Check if a dense matrix has banded structure with given bandwidth
band_is_banded :: proc(mat: ^Matrix($T), kl, ku: int, tolerance: f64 = 1e-12) -> bool {
	if mat.format != .General {
		return false
	}

	rows := int(mat.rows)
	cols := int(mat.cols)

	// Check elements outside the band
	for j in 0 ..< cols {
		for i in 0 ..< rows {
			if i > j + kl || i < j - ku {
				// Element should be zero
				src_idx := i + j * int(mat.ld)
				value := mat.data[src_idx]

				if abs(value) > T(tolerance) {
					return false
				}
			}
		}
	}

	return true
}

// Get effective bandwidth of a dense matrix
dns_get_bandwidth :: proc(mat: ^Matrix($T), tolerance: f64 = 1e-12) -> (kl, ku: int) {
	assert(mat.format == .General, "Matrix must be in general format")

	rows := int(mat.rows)
	cols := int(mat.cols)
	kl_max, ku_max := 0, 0

	for j in 0 ..< cols {
		for i in 0 ..< rows {
			src_idx := i + j * int(mat.ld)
			value := mat.data[src_idx]

			is_nonzero := abs(value) > T(tolerance)

			if is_nonzero {
				if i > j {
					kl_max = max(kl_max, i - j)
				} else if i < j {
					ku_max = max(ku_max, j - i)
				}
			}
		}
	}

	return kl_max, ku_max
}

// ===================================================================================
// LAPACK-SPECIFIC FORMAT CONVERSIONS
// ===================================================================================

// Convert BandedMatrix to LAPACK AB format (for GB routines)
// The matrix is already in the correct format, just extract parameters
band_to_lapack_gb :: proc(bm: ^BandedMatrix($T)) -> (ab: []T, m, n, kl, ku, ldab: Blas_Int) {
	return bm.data, bm.rows, bm.cols, bm.kl, bm.ku, bm.ldab
}

// Convert BandedMatrix to SymBand (for symmetric banded routines)
band_to_symband :: proc(bm: ^BandedMatrix($T), allocator := context.allocator) -> SymBand(T) where is_float(T) {
	assert(bm.symmetric, "Matrix must be symmetric for SB format")
	assert(bm.rows == bm.cols, "Matrix must be square for SB format")

	uplo: UpLo
	kd: Blas_Int

	if bm.ku > 0 {
		uplo = .Upper
		kd = bm.ku
	} else {
		uplo = .Lower
		kd = bm.kl
	}

	return SymBand(T){data = bm.data, n = bm.rows, kd = kd, ldab = bm.ldab, uplo = uplo}
}

// Convert BandedMatrix to HermBand (for Hermitian banded routines)
band_to_hermband :: proc(bm: ^BandedMatrix($T), allocator := context.allocator) -> HermBand(T) where is_complex(T) {
	assert(bm.symmetric, "Matrix must be Hermitian for HB format")
	assert(bm.rows == bm.cols, "Matrix must be square for HB format")

	uplo: UpLo
	kd: Blas_Int

	if bm.ku > 0 {
		uplo = .Upper
		kd = bm.ku
	} else {
		uplo = .Lower
		kd = bm.kl
	}

	return HermBand(T){data = bm.data, n = bm.rows, kd = kd, ldab = bm.ldab, uplo = uplo}
}

// Convert BandedMatrix to TriBand (for triangular banded routines)
band_to_triband :: proc(bm: ^BandedMatrix($T), diag: Diag = .NonUnit) -> TriBand(T) where is_float(T) || is_complex(T) {
	assert(bm.rows == bm.cols, "Matrix must be square for TB format")

	uplo: UpLo
	k: Blas_Int

	if bm.ku > 0 {
		uplo = .Upper
		k = bm.ku
	} else {
		uplo = .Lower
		k = bm.kl
	}

	return TriBand(T){data = bm.data, n = bm.rows, k = k, ldab = bm.ldab, uplo = uplo, diag = diag}
}

// ===================================================================================
// COPY AND UTILITY FUNCTIONS
// ===================================================================================

// Copy data from one banded matrix to another (must have compatible dimensions)
band_copy :: proc(dest: ^BandedMatrix($T), src: ^BandedMatrix(T)) {
	assert(dest.rows == src.rows && dest.cols == src.cols, "Matrices must have same dimensions")
	assert(dest.kl >= src.kl && dest.ku >= src.ku, "Destination must have at least as much bandwidth")
	assert(len(dest.data) >= len(src.data), "Destination data array too small")

	// Simple case: same bandwidth
	if dest.kl == src.kl && dest.ku == src.ku && dest.ldab == src.ldab {
		copy(dest.data, src.data)
		return
	}

	// FIXME: copy routines??

	// General case: copy element by element
	for j in 0 ..< int(src.cols) {
		for i in max(0, j - int(src.ku)) ..< min(int(src.rows), j + int(src.kl) + 1) {
			if value, stored := band_get(src, i, j); stored {
				band_set(dest, i, j, value)
			}
		}
	}
}

// Zero out a banded matrix
band_zero :: proc(bm: ^BandedMatrix($T)) {
	mem.zero(bm.data, len(bm.data))
}
