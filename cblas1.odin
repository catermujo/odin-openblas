package openblas

import blas "./c"
import "base:builtin"
import "base:intrinsics"
import "core:fmt"
import "core:mem"
import "core:slice"

is_float :: intrinsics.type_is_float
is_complex :: intrinsics.type_is_complex

// ==== BLAS Level 1: Vector operations ====

// ===================================================================================
//  DOT PRODUCTS
// ===================================================================================

// Computes the dot product of two vectors: result = sum(x[i] * y[i])
// For complex vectors, this is the unconjugated dot product.
// Supported types: f32, f64, complex64, complex128
v_dot :: proc(x, y: ^Vector($T)) -> T where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		return blas.cblas_sdot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == f64 {
		return blas.cblas_ddot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex64 {
		result := blas.cblas_cdotu(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
		return transmute(complex64)result
	} else when T == complex128 {
		result := blas.cblas_zdotu(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
		return transmute(complex128)result
	}
}

// Computes the conjugate dot product: result = sum(conj(x[i]) * y[i])
// For real vectors, this is identical to v_dot.
// For complex vectors, the first vector is conjugated before multiplication.
// Supported types: f32, f64, complex64, complex128
v_dot_conj :: proc(x, y: ^Vector($T)) -> T where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		return blas.cblas_sdot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == f64 {
		return blas.cblas_ddot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex64 {
		result := blas.cblas_cdotc(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
		return transmute(complex64)result
	} else when T == complex128 {
		result := blas.cblas_zdotc(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
		return transmute(complex128)result
	}
}

// Computes scaled dot product with extended precision accumulation: result = alpha + sum(x[i] * y[i])
// Uses extended precision internally to reduce rounding errors in large sums.
// Only available for f32 vectors.
// Useful for iterative algorithms where error accumulation is a concern.
v_dot_extended :: proc(x, y: ^Vector(f32), alpha: f32) -> f32 {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	return blas.cblas_sdsdot(n, alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
}

// Computes dot product of f32 vectors with f64 accumulation: result = sum(x[i] * y[i])
// Accumulates in double precision for improved accuracy without converting input vectors.
// Useful when input data is f32 but higher precision results are needed.
v_dot_mixed :: proc(x, y: ^Vector(f32)) -> f64 {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	return blas.cblas_dsdot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
}

// Computes dot product and stores result via pointer.
// For complex types, uses the _sub variants for ABI compatibility with Fortran.
// For real types, computes normally and stores.
// Supported types: f32, f64, complex64, complex128
v_dot_sub :: proc(x, y: ^Vector($T), result: ^T) where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == complex64 {
		blas.cblas_cdotu_sub(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, result)
	} else when T == complex128 {
		blas.cblas_zdotu_sub(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, result)
	} else {
		// For real types, compute normally and store
		result^ = v_dot(x, y)
	}
}

// Computes conjugate dot product and stores result via pointer.
// For complex types, uses the _sub variants for ABI compatibility with Fortran.
// For real types, computes normally and stores.
// Supported types: f32, f64, complex64, complex128
v_dot_conj_sub :: proc(x, y: ^Vector($T), result: ^T) where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == complex64 {
		blas.cblas_cdotc_sub(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, result)
	} else when T == complex128 {
		blas.cblas_zdotc_sub(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, result)
	} else {
		// For real types, compute normally and store
		result^ = v_dot_conj(x, y)
	}
}

// ===================================================================================
// VECTOR SUMS
// ===================================================================================

// Computes the sum of absolute values: result = sum(|x[i]|)
// For complex numbers, computes sum(|real(x[i])| + |imag(x[i])|)
// Returns real type even for complex input.
// Useful for computing L1 norm or checking for convergence.
// Supported types: f32, f64, complex64, complex128
v_abs_sum :: proc(x: ^Vector($T)) -> (result: T) where is_float(T) || is_complex(T) {
	n := i64(x.size)
	incx := i64(x.incr)

	when T == f32 {
		return blas.cblas_sasum(n, vector_data_ptr(x), incx)
	} else when T == f64 {
		return blas.cblas_dasum(n, vector_data_ptr(x), incx)
	} else when T == complex64 {
		return blas.cblas_scasum(n, vector_data_ptr(x), incx)
	} else when T == complex128 {
		return blas.cblas_dzasum(n, vector_data_ptr(x), incx)
	}
}

// Computes the sum of vector elements: result = sum(x[i])
// Note: This is an OpenBLAS extension, not in standard BLAS.
// For real types, returns the sum as the same type.
// For complex types, returns a real sum (OpenBLAS specific behavior).
// Supported types: f32, f64
v_sum :: proc(x: ^Vector($T)) -> T where is_float(T) || is_complex(T) {
	n := i64(x.size)
	incx := i64(x.incr)

	when T == f32 {
		return blas.cblas_ssum(n, vector_data_ptr(x), incx)
	} else when T == f64 {
		return blas.cblas_dsum(n, vector_data_ptr(x), incx)
	} else when T == complex64 {
		return blas.cblas_scsum(n, vector_data_ptr(x), incx)
	} else when T == complex128 {
		return blas.cblas_dzsum(n, vector_data_ptr(x), incx)
	}
}

// ===================================================================================
// VECTOR NORMS
// ===================================================================================

// Computes the Euclidean norm (2-norm) of a vector: result = sqrt(sum(x[i]^2))
// Returns the magnitude/length of the vector.
// For complex vectors, computes sqrt(sum(|x[i]|^2)).
// Returns real type even for complex input.
// Supported types: f32, f64, complex64, complex128
v_norm2 :: proc {
	v_norm2_f32_c64,
	v_norm2_f64_c128,
}

v_norm2_f32_c64 :: proc(x: ^Vector($T)) -> f32 where T == f32 || T == complex64 {
	n := i64(x.size)
	incx := i64(x.incr)
	when T == f32 {
		return blas.cblas_snrm2(n, vector_data_ptr(x), incx)
	} else when T == complex64 {
		return blas.cblas_scnrm2(n, vector_data_ptr(x), incx)
	}
}

v_norm2_f64_c128 :: proc(x: ^Vector($T)) -> f64 where T == f64 || T == complex128 {
	n := i64(x.size)
	incx := i64(x.incr)
	when T == f64 {
		return blas.cblas_dnrm2(n, vector_data_ptr(x), incx)
	} else when T == complex128 {
		return blas.cblas_dznrm2(n, vector_data_ptr(x), incx)

	}
}

// ===================================================================================
// VECTOR ELEMENT FINDING
// ===================================================================================

// Finds the index of the element with maximum absolute value
// For complex vectors, compares |real| + |imag|
// Returns 0-based index (CBLAS returns 0-based, unlike Fortran BLAS)
// Supported types: f32, f64, complex64, complex128
v_idx_abs_max :: proc(x: ^Vector($T)) -> int where is_float(T) || is_complex(T) {
	n := i64(x.size)
	incx := i64(x.incr)

	when T == f32 {
		return int(blas.cblas_isamax(n, vector_data_ptr(x), incx))
	} else when T == f64 {
		return int(blas.cblas_idamax(n, vector_data_ptr(x), incx))
	} else when T == complex64 {
		return int(blas.cblas_icamax(n, vector_data_ptr(x), incx))
	} else when T == complex128 {
		return int(blas.cblas_izamax(n, vector_data_ptr(x), incx))
	}
}

// Finds the index of the element with minimum absolute value
// For complex vectors, compares |real| + |imag|
// Returns 0-based index
// Supported types: f32, f64, complex64, complex128
v_idx_abs_min :: proc(x: ^Vector($T)) -> int where is_float(T) || is_complex(T) {
	n := i64(x.size)
	incx := i64(x.incr)

	when T == f32 {
		return int(blas.cblas_isamin(n, vector_data_ptr(x), incx))
	} else when T == f64 {
		return int(blas.cblas_idamin(n, vector_data_ptr(x), incx))
	} else when T == complex64 {
		return int(blas.cblas_icamin(n, vector_data_ptr(x), incx))
	} else when T == complex128 {
		return int(blas.cblas_izamin(n, vector_data_ptr(x), incx))
	}
}

// Finds the maximum absolute value in the vector
// For complex vectors, compares |real| + |imag|
// Returns the actual maximum absolute value (not index)
// Supported types: f32, f64, complex64, complex128
v_abs_max :: proc {
	v_abs_max_f32_c64,
	v_abs_max_f64_c128,
}

v_abs_max_f32_c64 :: proc(x: ^Vector($T)) -> f32 where T == f32 || T == complex64 {
	n := i64(x.size)
	incx := i64(x.incr)
	when T == f32 {
		return blas.cblas_samax(n, vector_data_ptr(x), incx)
	} else when T == complex64 {
		return blas.cblas_scamax(n, vector_data_ptr(x), incx)
	}
}

v_abs_max_f64_c128 :: proc(x: ^Vector($T)) -> f64 where T == f64 || T == complex128 {
	n := i64(x.size)
	incx := i64(x.incr)
	when T == f64 {
		return blas.cblas_damax(n, vector_data_ptr(x), incx)
	} else when T == complex128 {
		return blas.cblas_dzamax(n, vector_data_ptr(x), incx)
	}
}

// Finds the minimum absolute value in the vector
// For complex vectors, compares |real| + |imag|
// Returns the actual minimum absolute value (not index)
// Supported types: f32, f64, complex64, complex128
v_abs_min :: proc {
	v_abs_min_f32_c64,
	v_abs_min_f64_c128,
}

v_abs_min_f32_c64 :: proc(x: ^Vector($T)) -> f32 where T == f32 || T == complex64 {
	n := i64(x.size)
	incx := i64(x.incr)
	when T == f32 {
		return blas.cblas_samin(n, vector_data_ptr(x), incx)
	} else when T == complex64 {
		return blas.cblas_scamin(n, vector_data_ptr(x), incx)
	}
}

v_abs_min_f64_c128 :: proc(x: ^Vector($T)) -> f64 where T == f64 || T == complex128 {
	n := i64(x.size)
	incx := i64(x.incr)
	when T == f64 {
		return blas.cblas_damin(n, vector_data_ptr(x), incx)
	} else when T == complex128 {
		return blas.cblas_dzamin(n, vector_data_ptr(x), incx)
	}
}

// Finds the index of the element with maximum value (not absolute)
// For complex vectors, behavior is implementation-defined
// Returns 0-based index
// Supported types: f32, f64, complex64, complex128
v_idx_max :: proc(x: ^Vector($T)) -> int where is_float(T) || is_complex(T) {
	n := i64(x.size)
	incx := i64(x.incr)

	when T == f32 {
		return int(blas.cblas_ismax(n, vector_data_ptr(x), incx))
	} else when T == f64 {
		return int(blas.cblas_idmax(n, vector_data_ptr(x), incx))
	} else when T == complex64 {
		return int(blas.cblas_icmax(n, vector_data_ptr(x), incx))
	} else when T == complex128 {
		return int(blas.cblas_izmax(n, vector_data_ptr(x), incx))
	}
}

// Finds the index of the element with minimum value (not absolute)
// For complex vectors, behavior is implementation-defined
// Returns 0-based index
// Supported types: f32, f64, complex64, complex128
v_idx_min :: proc(x: ^Vector($T)) -> int where is_float(T) || is_complex(T) {
	n := i64(x.size)
	incx := i64(x.incr)

	when T == f32 {
		return int(blas.cblas_ismin(n, vector_data_ptr(x), incx))
	} else when T == f64 {
		return int(blas.cblas_idmin(n, vector_data_ptr(x), incx))
	} else when T == complex64 {
		return int(blas.cblas_icmin(n, vector_data_ptr(x), incx))
	} else when T == complex128 {
		return int(blas.cblas_izmin(n, vector_data_ptr(x), incx))
	}
}

// ===================================================================================
// VECTOR AXPY OPERATIONS
// ===================================================================================

// Performs scaled vector addition: y = alpha*x + y
// Adds a scaled vector x to vector y, storing result in y.
// Name comes from "a times x plus y".
// Modifies the y vector in-place.
// Supported types: f32, f64, complex64, complex128
v_axpy :: proc(alpha: $T, x: ^Vector(T), y: ^Vector(T)) where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_saxpy(n, alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == f64 {
		blas.cblas_daxpy(n, alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex64 {
		blas.cblas_caxpy(n, &alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex128 {
		blas.cblas_zaxpy(n, &alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	}
}

// Performs scaled vector addition with conjugate: y = alpha*conj(x) + y
// Similar to v_axpy but conjugates the x vector before scaling.
// For real types, behaves identically to v_axpy.
// For complex types, conjugates x before the operation.
// Modifies the y vector in-place.
// Supported types: f32, f64, complex64, complex128
v_axpy_conj :: proc(alpha: $T, x: ^Vector(T), y: ^Vector(T)) where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		// For real types, conjugate is identity, so same as axpy
		blas.cblas_saxpy(n, alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == f64 {
		// For real types, conjugate is identity, so same as axpy
		blas.cblas_daxpy(n, alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex64 {
		blas.cblas_caxpyc(n, &alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex128 {
		blas.cblas_zaxpyc(n, &alpha, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	}
}

// Performs extended scaled vector addition: y = alpha*x + beta*y
// This generalizes axpy by also scaling y with beta.
// Supported types: f32, f64, complex64, complex128
v_axpby :: proc(
	alpha: $T,
	x: ^Vector(T),
	beta: T,
	y: ^Vector(T),
) where is_float(T) ||
	is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_saxpby(n, alpha, vector_data_ptr(x), incx, beta, vector_data_ptr(y), incy)
	} else when T == f64 {
		blas.cblas_daxpby(n, alpha, vector_data_ptr(x), incx, beta, vector_data_ptr(y), incy)
	} else when T == complex64 {
		alpha, beta := alpha, beta
		blas.cblas_caxpby(n, &alpha, vector_data_ptr(x), incx, &beta, vector_data_ptr(y), incy)
	} else when T == complex128 {
		alpha, beta := alpha, beta
		blas.cblas_zaxpby(n, &alpha, vector_data_ptr(x), incx, &beta, vector_data_ptr(y), incy)
	}
}

// ===================================================================================
// VECTOR COPY OPERATIONS
// ===================================================================================

// Copies vector x to vector y: y = x
// Both vectors must have the same length.
// The operation is: y[i] = x[i] for all i
// Supported types: f32, f64, complex64, complex128
v_copy :: proc(x: ^Vector($T), y: ^Vector(T)) where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_scopy(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == f64 {
		blas.cblas_dcopy(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex64 {
		blas.cblas_ccopy(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex128 {
		blas.cblas_zcopy(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	}
}

// ===================================================================================
// VECTOR SWAP OPERATIONS
// ===================================================================================

// Swaps the contents of vectors x and y: x <-> y
// Both vectors must have the same length.
// After the operation: original x becomes y, original y becomes x
// Both vectors are modified in-place.
// Supported types: f32, f64, complex64, complex128
v_swap :: proc(x: ^Vector($T), y: ^Vector(T)) where is_float(T) || is_complex(T) {
	assert(x.size == y.size, "Vector dimensions must match")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_sswap(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == f64 {
		blas.cblas_dswap(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex64 {
		blas.cblas_cswap(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	} else when T == complex128 {
		blas.cblas_zswap(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy)
	}
}

// ===================================================================================
// VECTOR ROTATION OPERATIONS
// ===================================================================================

// Applies a plane rotation to vectors x and y
// Performs: x' = c*x + s*y, y' = c*y - s*x
// Where c = cos(theta) and s = sin(theta) for some angle theta
// Both vectors are modified in-place.
// Supported types: f32, f64, complex64, complex128
v_rot :: proc {
	v_rot_f32_c64,
	v_rot_f64_c128,
}

v_rot_f32_c64 :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	c: f32,
	s: f32,
) where T == f32 ||
	T == complex64 {
	assert(x.size == y.size, "Vector dimensions must match")
	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)
	when T == f32 {
		blas.cblas_srot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, c, s)
	} else when T == complex64 {
		blas.cblas_csrot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, c, s)
	}
}

v_rot_f64_c128 :: proc(
	x: ^Vector($T),
	y: ^Vector(T),
	c: f64,
	s: f64,
) where T == f64 ||
	T == complex128 {
	assert(x.size == y.size, "Vector dimensions must match")
	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)
	when T == f64 {
		blas.cblas_drot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, c, s)
	} else when T == complex128 {
		blas.cblas_zdrot(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, c, s)

	}
}

// Generates the parameters for a Givens rotation
// Given values a and b, computes c, s, and r such that:
// [c  s] [a] = [r]
// [-s c] [b]   [0]
// This zeros out the b component and puts the magnitude in r.
// Used to construct Givens rotations for QR decomposition.
v_rotg :: proc {
	v_rotg_f32_c64,
	v_rotg_f64_c128,
}

v_rotg_f32_c64 :: proc(a: ^$T, b: ^T) -> (c: f32, s: T) where T == f32 || T == complex64 {
	when T == f32 {
		c_val: f32
		s_val: f32
		blas.cblas_srotg(a, b, &c_val, &s_val)
		return c_val, s_val
	} else when T == complex64 {
		c_val: f32
		s_val: complex64
		blas.cblas_crotg(a, b, &c_val, &s_val)
		return c_val, s_val
	}
}

v_rotg_f64_c128 :: proc(a: ^$T, b: ^T) -> (c: f64, s: T) where T == f64 || T == complex128 {
	when T == f64 {
		c_val: f64
		s_val: f64
		blas.cblas_drotg(a, b, &c_val, &s_val)
		return c_val, s_val
	} else when T == complex128 {
		c_val: f64
		s_val: complex128
		blas.cblas_zrotg(a, b, &c_val, &s_val)
		return c_val, s_val
	}
}

// Applies a modified Givens rotation to vectors x and y
// Uses a parameter array P[5] that defines the rotation
// More numerically stable than regular Givens for some cases
// Both vectors are modified in-place.
// Only available for real types (f32, f64)
v_rotm :: proc(x: ^Vector($T), y: ^Vector(T), P: []T) where is_float(T) {
	assert(x.size == y.size, "Vector dimensions must match")
	assert(len(P) >= 5, "Parameter array P must have at least 5 elements")

	n := i64(x.size)
	incx := i64(x.incr)
	incy := i64(y.incr)

	when T == f32 {
		blas.cblas_srotm(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, raw_data(P))
	} else when T == f64 {
		blas.cblas_drotm(n, vector_data_ptr(x), incx, vector_data_ptr(y), incy, raw_data(P))
	} else {
		#panic("Unsupported type for rotm")
	}
}

// Generates parameters for a modified Givens rotation
// Given d1, d2, b1, b2, computes the parameter array P[5]
// More numerically stable construction than regular Givens
// Returns the parameter array for use with v_rotm
// Only available for real types (f32, f64)
v_rotmg :: proc(
	d1: ^$T,
	d2: ^T,
	b1: ^T,
	b2: T,
	allocator := context.allocator,
) -> []T where is_float(T) {
	P := make([]T, 5, allocator)

	when T == f32 {
		blas.cblas_srotmg(d1, d2, b1, b2, raw_data(P))
	} else when T == f64 {
		blas.cblas_drotmg(d1, d2, b1, b2, raw_data(P))
	}

	return P
}

// ===================================================================================
// VECTOR SCALING OPERATIONS
// ===================================================================================

// Scales a vector by a scalar: x = alpha * x
// Multiplies every element of the vector by the scalar alpha.
// The vector is modified in-place.
// Supported types: f32, f64, complex64, complex128
v_scale :: proc(alpha: $T, x: ^Vector(T)) where is_float(T) || is_complex(T) {
	n := i64(x.size)
	incx := i64(x.incr)

	when T == f32 {
		blas.cblas_sscal(n, alpha, vector_data_ptr(x), incx)
	} else when T == f64 {
		blas.cblas_dscal(n, alpha, vector_data_ptr(x), incx)
	} else when T == complex64 {
		blas.cblas_cscal(n, &alpha, vector_data_ptr(x), incx)
	} else when T == complex128 {
		blas.cblas_zscal(n, &alpha, vector_data_ptr(x), incx)
	}
}

// Scales a complex vector by a real scalar: x = alpha * x
// Multiplies every element of a complex vector by a real scalar.
// This is more efficient than using complex multiplication when the scalar is real.
// The vector is modified in-place.
// Supported types: complex64 (scaled by f32), complex128 (scaled by f64)

v_scale_real :: proc {
	v_scale_real_f32_c64,
	v_scale_real_f64_c128,
}

v_scale_real_f32_c64 :: proc(alpha: f32, x: ^Vector(complex64)) {
	n := i64(x.size)
	incx := i64(x.incr)
	blas.cblas_csscal(n, alpha, vector_data_ptr(x), incx)
}

v_scale_real_f64_c128 :: proc(alpha: f64, x: ^Vector(complex128)) {
	n := i64(x.size)
	incx := i64(x.incr)
	blas.cblas_zdscal(n, alpha, vector_data_ptr(x), incx)
}
