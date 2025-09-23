package openblas_c

import "core:c"

_ :: c

foreign import lib "../../vendor/linalg/windows-x64/lib/openblas64.lib"

// LAPACKE_EXAMPLE_AUX_ ::

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	print_matrix_rowmajor :: proc(desc: cstring, m: c.int, n: c.int, mat: ^f64, ldm: c.int) ---
	print_matrix_colmajor :: proc(desc: cstring, m: c.int, n: c.int, mat: ^f64, ldm: c.int) ---
	print_vector :: proc(desc: cstring, n: c.int, vec: ^c.int) ---
}
