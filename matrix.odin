package openblas

import blas "./c"
import lapack "./f77"
import "base:builtin"
import "base:intrinsics"
import "core:fmt"
import "core:mem"
import "core:slice"

// Re-Decls:
Info :: lapack.Info
Blas_Int :: lapack.Blas_Int

// Vector type for BLAS Level 1 and LAPACK operations
Vector :: struct($T: typeid) {
	data:   []T,
	size:   Blas_Int, // Number of elements
	incr:   Blas_Int, // Stride/increment between elements (usually 1)
	offset: Blas_Int, // Starting offset in data array
}

MatrixFormat :: enum {
	General, // GE routines - general dense matrices
	Banded, // GB routines - banded matrices
	Symmetric, // SY routines - symmetric matrices (real)
	Hermitian, // HE routines - Hermitian matrices (complex)
	Packed, // SP/HP/PP routines - packed triangular storage
	Triangular, // TR routines - triangular matrices
	Diagonal, // DI - diagonal matrices
	Tridiagonal, // GT routines - tridiagonal matrices
}

Matrix :: struct($T: typeid) {
	data:       []T,
	rows, cols: Blas_Int,
	ld:         Blas_Int, // Leading dimension (for column-major storage)
	format:     MatrixFormat,
	storage:    struct #raw_union {
		banded:      struct {
			kl, ku: Blas_Int, // Lower/upper bandwidth
			ldab:   Blas_Int, // Leading dimension of band storage
		},
		symmetric:   struct {
			uplo: cstring, // "U" or "L" for LAPACK compatibility
		},
		hermitian:   struct {
			uplo: cstring, // "U" or "L" for LAPACK compatibility
		},
		triangular:  struct {
			uplo:  cstring, // "U" or "L" for upper/lower
			diag:  cstring, // "U" (unit) or "N" (non-unit) diagonal
			trans: cstring, // "N" (none), "T" (transpose), "C" (conjugate transpose)
		},
		packed:      struct {
			uplo: cstring, // "U" or "L" for upper/lower triangular
			n:    Blas_Int, // Matrix dimension for packed storage
		},
		diagonal:    struct {
			// Diagonal matrices store only diagonal elements
			n: Blas_Int, // Matrix dimension
		},
		tridiagonal: struct {
			// Tridiagonal matrices: three diagonals
			dl_offset: Blas_Int, // Offset to lower diagonal in data array
			d_offset:  Blas_Int, // Offset to main diagonal in data array
			du_offset: Blas_Int, // Offset to upper diagonal in data array
		},
	},
}

data_ptr :: proc {
	vector_data_ptr,
	matrix_data_ptr,
}


// ===================================================================================
// VECTOR CREATION AND MANAGEMENT
// ===================================================================================

// Create a new vector with uninitialized data
make_vector :: proc(
	$T: typeid,
	size: int,
	incr: int = 1,
	allocator := context.allocator,
) -> Vector(T) {
	actual_size := size * (incr if incr >= 0 else -incr) // Account for stride
	return Vector(T) {
		data = builtin.make([]T, actual_size, allocator),
		size = Blas_Int(size),
		incr = Blas_Int(incr),
		offset = 0,
	}
}

// Create a vector filled with zeros
vector_zeros :: proc($T: typeid, size: int, allocator := context.allocator) -> Vector(T) {
	v := make_vector(T, size, 1, allocator)
	// Data is already zero-initialized by make
	return v
}

// Create a vector filled with ones
vector_ones :: proc($T: typeid, size: int, allocator := context.allocator) -> Vector(T) {
	v := make_vector(T, size, 1, allocator)
	for i in 0 ..< size {
		v.data[i] = T(1)
	}
	return v
}

// Create a vector from a slice
vector_from_slice :: proc(
	$T: typeid,
	slice: []T,
	incr: int = 1,
	allocator := context.allocator,
) -> Vector(T) {
	v := make_vector(T, len(slice), incr, allocator)
	if incr == 1 {
		copy(v.data, slice)
	} else {
		for i in 0 ..< len(slice) {
			v.data[i * abs(incr)] = slice[i]
		}
	}
	return v
}

// Get element at index (respects stride)
vector_get :: proc(v: ^Vector($T), index: int) -> T {
	return v.data[v.offset + Blas_Int(index) * v.incr]
}

// Set element at index (respects stride)
vector_set :: proc(v: ^Vector($T), index: int, value: T) {
	v.data[v.offset + Blas_Int(index) * v.incr] = value
}

// Get pointer to first element (for BLAS/LAPACK calls)
vector_data_ptr :: proc(v: ^Vector($T)) -> ^T {
	return &v.data[v.offset]
}

// Create a subvector (view into existing vector)
vector_subvector :: proc(v: ^Vector($T), start, length: int) -> Vector(T) {
	return Vector(T) {
		data = v.data,
		size = Blas_Int(length),
		incr = v.incr,
		offset = v.offset + Blas_Int(start) * v.incr,
	}
}

// Delete vector
delete_vector :: proc(v: ^Vector($T)) {
	builtin.delete(v.data)
}

// ===================================================================================
// MATRIX CREATION AND MANAGEMENT
// ===================================================================================

// Create a new matrix with uninitialized data
make_matrix :: proc(
	$T: typeid,
	rows, cols: int,
	format := MatrixFormat.General,
	allocator := context.allocator,
) -> Matrix(T) {
	size := rows * cols
	return Matrix(T) {
		data   = builtin.make([]T, size, allocator),
		rows   = Blas_Int(rows),
		cols   = Blas_Int(cols),
		ld     = Blas_Int(rows), // Column-major: leading dimension is number of rows
		format = format,
	}
}

// Create a matrix filled with zeros
matrix_zeros :: proc($T: typeid, rows, cols: int, allocator := context.allocator) -> Matrix(T) {
	m := make_matrix(T, rows, cols, MatrixFormat.General, allocator)
	// Data is already zero-initialized by make
	return m
}

// Create a matrix filled with ones
matrix_ones :: proc($T: typeid, rows, cols: int, allocator := context.allocator) -> Matrix(T) {
	m := make_matrix(T, rows, cols, MatrixFormat.General, allocator)
	for &elem in m.data {
		elem = T(1)
	}
	return m
}

// Create an identity matrix
matrix_eye :: proc($T: typeid, n: int, allocator := context.allocator) -> Matrix(T) {
	m := matrix_zeros(T, n, n, allocator)
	// Column-major: diagonal elements at i + i*n
	for i in 0 ..< n {
		m.data[i + i * n] = T(1)
	}
	return m
}

// Create a matrix from a 2D slice (row-major input, stored as column-major)
matrix_from_slice :: proc($T: typeid, slice: [][]T, allocator := context.allocator) -> Matrix(T) {
	rows := len(slice)
	cols := len(slice[0]) if rows > 0 else 0
	m := make_matrix(T, rows, cols, MatrixFormat.General, allocator)

	// Convert row-major input to column-major storage
	for i in 0 ..< rows {
		for j in 0 ..< cols {
			m.data[i + j * rows] = slice[i][j]
		}
	}
	return m
}

// Delete matrix
delete_matrix :: proc(m: ^Matrix($T)) {
	builtin.delete(m.data)
}

// Get element at row, col (column-major storage)
matrix_get :: proc(m: ^Matrix($T), row, col: int) -> T {
	return m.data[Blas_Int(row) + Blas_Int(col) * m.ld]
}

// Set element at row, col (column-major storage)
matrix_set :: proc(m: ^Matrix($T), row, col: int, value: T) {
	m.data[Blas_Int(row) + Blas_Int(col) * m.ld] = value
}

// Get pointer to first element (for BLAS/LAPACK calls)
matrix_data_ptr :: proc(m: ^Matrix($T)) -> ^T {
	return &m.data[0] if len(m.data) > 0 else nil
}

// Get pointer to element at row, col
matrix_element_ptr :: proc(m: ^Matrix($T), row, col: int) -> ^T {
	return &m.data[row + col * m.ld]
}

// Print vector (for debugging)
print_vector :: proc(v: ^Vector($T), label := "") {
	if label != "" {
		fmt.printf("%s:\n", label)
	}
	fmt.printf("Vector(%v): [", v.size)
	for i in 0 ..< v.size {
		if i > 0 do fmt.printf(", ")
		fmt.printf("%v", vector_get(v, i))
	}
	fmt.println("]")
}

// Print matrix (for debugging)
print_matrix :: proc(m: ^Matrix($T), label := "") {
	if label != "" {
		fmt.printf("%s:\n", label)
	}
	fmt.printf("Matrix(%v x %v):\n", m.rows, m.cols)
	for i in 0 ..< m.rows {
		fmt.print("[")
		for j in 0 ..< m.cols {
			if j > 0 do fmt.print(", ")
			fmt.printf("%6.2f", m.data[i + j * m.rows]) // Column-major indexing
		}
		fmt.println("]")
	}
}

// Create a submatrix (view into existing matrix)
matrix_submatrix :: proc(m: ^Matrix($T), row_start, col_start, rows, cols: int) -> Matrix(T) {
	return Matrix(T) {
		data    = m.data[row_start + col_start * int(m.ld):],
		rows    = Blas_Int(rows),
		cols    = Blas_Int(cols),
		ld      = m.ld, // Keep the same leading dimension
		format  = m.format,
		storage = m.storage,
	}
}


// Create a symmetric matrix
make_symmetric_matrix :: proc(
	$T: typeid,
	n: int,
	uplo: cstring = "U",
	allocator := context.allocator,
) -> Matrix(T) {
	m := make_matrix(T, n, n, MatrixFormat.Symmetric, allocator)
	m.storage.symmetric.uplo = uplo
	return m
}

// Create a triangular matrix
make_triangular_matrix :: proc(
	$T: typeid,
	n: int,
	uplo: cstring = "U",
	diag: cstring = "N",
	allocator := context.allocator,
) -> Matrix(T) {
	m := make_matrix(T, n, n, MatrixFormat.Triangular, allocator)
	m.storage.triangular.uplo = uplo
	m.storage.triangular.diag = diag
	m.storage.triangular.trans = "N"
	return m
}

// Create a banded matrix
make_banded_matrix :: proc(
	$T: typeid,
	rows, cols: int,
	kl, ku: int, // Lower and upper bandwidth
	allocator := context.allocator,
) -> Matrix(T) {
	// Band storage requires (kl + ku + 1) rows for column-major format
	ldab := kl + ku + 1
	size := ldab * cols
	return Matrix(T) {
		data = builtin.make([]T, size, allocator),
		rows = rows,
		cols = cols,
		ld = rows,
		format = MatrixFormat.Banded,
		storage = {banded = {kl = kl, ku = ku, ldab = ldab}},
	}
}

// Create a packed triangular matrix (compact storage)
make_packed_matrix :: proc(
	$T: typeid,
	n: int,
	uplo: cstring = "U",
	allocator := context.allocator,
) -> Matrix(T) {
	// Packed storage uses n*(n+1)/2 elements
	size := n * (n + 1) / 2
	return Matrix(T) {
		data = builtin.make([]T, size, allocator),
		rows = n,
		cols = n,
		ld = 0, // Not used for packed storage
		format = MatrixFormat.Packed,
		storage = {packed = {uplo = uplo, n = n}},
	}
}

// ===================================================================================
// ERROR HANDLING
// ===================================================================================

// Error handler for BLAS routines
// Reports errors in BLAS function calls with detailed information.
// p: Error code/parameter position
// rout: Name of the BLAS routine that encountered the error
// form: Format string for error message
// args: Additional arguments for the format string
get_error :: proc(p: blas.blasint, rout: cstring, form: cstring, args: ..any) {
	// blas.cblas_xerbla(p, rout, form, ..args) // FIXME: cast to #c_varags
}
