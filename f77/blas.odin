package f77

import "core:c"

_ :: c

when ODIN_OS == .Windows {
	foreign import lib "../../vendor/linalg/windows-x64/lib/openblas64.lib"
} else when ODIN_OS == .Linux {
	// Use ILP64 version of OpenBLAS (64-bit integers)
	foreign import lib "system:openblas64"
}

// F77BLAS_H ::

/*Set the threading backend to a custom callback.*/
openblas_dojob_callback :: proc "c" (_: c.int, _: rawptr, _: c.int)

openblas_threads_callback :: proc "c" (
	_: c.int,
	_: openblas_dojob_callback,
	_: c.int,
	_: c.size_t,
	_: rawptr,
	_: c.int,
)

@(default_calling_convention = "c", link_prefix = "")
foreign lib {
	xerbla_ :: proc(_: cstring, info: ^blasint, _: blasint) -> c.int ---
	openblas_set_num_threads_ :: proc(_: ^c.int) ---
	sdot_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	sdsdot_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dsdot_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> f64 ---
	ddot_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qdot_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	sbdot_ :: proc(_: ^blasint, _: ^bfloat16, _: ^blasint, _: ^bfloat16, _: ^blasint) -> f32 ---
	sbstobf16_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^bfloat16, _: ^blasint) ---
	sbdtobf16_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^bfloat16, _: ^blasint) ---
	sbf16tos_ :: proc(_: ^blasint, _: ^bfloat16, _: ^blasint, _: ^f32, _: ^blasint) ---
	dbf16tod_ :: proc(_: ^blasint, _: ^bfloat16, _: ^blasint, _: ^f64, _: ^blasint) ---
	cdotu_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> openblas_complex_float ---
	cdotc_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> openblas_complex_float ---
	zdotu_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> openblas_complex_double ---
	zdotc_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> openblas_complex_double ---
	xdotu_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> openblas_complex_xdouble ---
	xdotc_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> openblas_complex_xdouble ---
	saxpy_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	daxpy_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qaxpy_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	caxpy_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zaxpy_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xaxpy_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	caxpyc_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zaxpyc_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xaxpyc_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	scopy_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dcopy_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qcopy_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ccopy_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zcopy_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xcopy_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	sswap_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dswap_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qswap_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	cswap_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zswap_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xswap_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	sasum_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	scasum_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dasum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qasum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	dzasum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qxasum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	ssum_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	scsum_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dsum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qsum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	dzsum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qxsum_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	isamax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	idamax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	iqamax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	icamax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	izamax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	ixamax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	ismax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	idmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	iqmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	icmax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	izmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	ixmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	isamin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	idamin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	iqamin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	icamin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	izamin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	ixamin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	ismin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	idmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	iqmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	icmin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> blasint ---
	izmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	ixmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> blasint ---
	samax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	damax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qamax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	scamax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dzamax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qxamax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	samin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	damin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qamin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	scamin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dzamin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qxamin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	smax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	scmax_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dzmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qxmax_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	smin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	scmin_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dzmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qxmin_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	sscal_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dscal_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qscal_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cscal_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zscal_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xscal_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	csscal_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zdscal_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xqscal_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	snrm2_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	scnrm2_ :: proc(_: ^blasint, _: ^f32, _: ^blasint) -> f32 ---
	dnrm2_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qnrm2_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	dznrm2_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	qxnrm2_ :: proc(_: ^blasint, _: ^f64, _: ^blasint) -> f64 ---
	srot_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32) ---
	drot_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64) ---
	qrot_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64) ---
	csrot_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32) ---
	zdrot_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64) ---
	xqrot_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64) ---
	srotg_ :: proc(_: ^f32, _: ^f32, _: ^f32, _: ^f32) ---
	drotg_ :: proc(_: ^f64, _: ^f64, _: ^f64, _: ^f64) ---
	qrotg_ :: proc(_: ^f64, _: ^f64, _: ^f64, _: ^f64) ---
	crotg_ :: proc(_: ^f32, _: ^f32, _: ^f32, _: ^f32) ---
	zrotg_ :: proc(_: ^f64, _: ^f64, _: ^f64, _: ^f64) ---
	xrotg_ :: proc(_: ^f64, _: ^f64, _: ^f64, _: ^f64) ---
	srotmg_ :: proc(_: ^f32, _: ^f32, _: ^f32, _: ^f32, _: ^f32) ---
	drotmg_ :: proc(_: ^f64, _: ^f64, _: ^f64, _: ^f64, _: ^f64) ---
	srotm_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32) ---
	drotm_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---
	qrotm_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---

	/* Level 2 routines */
	sger_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dger_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qger_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	cgeru_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	cgerc_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zgeru_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	zgerc_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xgeru_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xgerc_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	sbgemv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^bfloat16, _: ^blasint, _: ^bfloat16, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	sgemv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dgemv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qgemv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cgemv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zgemv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xgemv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	strsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dtrsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qtrsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ctrsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	ztrsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xtrsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	strmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dtrmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qtrmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ctrmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	ztrmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xtrmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	stpsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dtpsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qtpsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	ctpsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	ztpsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xtpsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	stpmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dtpmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qtpmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	ctpmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	ztpmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xtpmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	stbmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dtbmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qtbmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ctbmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	ztbmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xtbmv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	stbsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dtbsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qtbsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ctbsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	ztbsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xtbsv_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ssymv_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dsymv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qsymv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	csymv_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zsymv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xsymv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	sspmv_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dspmv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qspmv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cspmv_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zspmv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xspmv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	ssyr_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dsyr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qsyr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	csyr_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zsyr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xsyr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ssyr2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dsyr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qsyr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	csyr2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zsyr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xsyr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	sspr_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32) ---
	dspr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64) ---
	qspr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64) ---
	cspr_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32) ---
	zspr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64) ---
	xspr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64) ---
	sspr2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32) ---
	dspr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---
	qspr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---
	cspr2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32) ---
	zspr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---
	xspr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---
	cher_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zher_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xher_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	chpr_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32) ---
	zhpr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64) ---
	xhpr_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64) ---
	cher2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zher2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xher2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	chpr2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32) ---
	zhpr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---
	xhpr2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64) ---
	chemv_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zhemv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xhemv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	chpmv_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zhpmv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xhpmv_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	snorm_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	dnorm_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	cnorm_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	znorm_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	sgbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dgbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qgbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cgbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zgbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xgbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	ssbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dsbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qsbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	csbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zsbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xsbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	chbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zhbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xhbmv_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---

	/* Level 3 routines */
	sbgemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^bfloat16, _: ^blasint, _: ^bfloat16, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	sgemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dgemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qgemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cgemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zgemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xgemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cgemm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zgemm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xgemm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	sgemmt_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dgemmt_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cgemmt_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zgemmt_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	sge2mm_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) -> c.int ---
	dge2mm_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---
	cge2mm_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) -> c.int ---
	zge2mm_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---
	strsm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dtrsm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qtrsm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ctrsm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	ztrsm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xtrsm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	strmm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	dtrmm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	qtrmm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ctrmm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	ztrmm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	xtrmm_ :: proc(_: cstring, _: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	ssymm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dsymm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qsymm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	csymm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zsymm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xsymm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	csymm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zsymm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xsymm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	ssyrk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dsyrk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qsyrk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	csyrk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zsyrk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xsyrk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	ssyr2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dsyr2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	qsyr2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	csyr2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zsyr2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xsyr2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	chemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zhemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xhemm_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	chemm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zhemm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xhemm3m_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cherk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zherk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xherk_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cher2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zher2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	xher2k_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cher2m_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) -> c.int ---
	zher2m_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---
	xher2m_ :: proc(_: cstring, _: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---
	sgemt_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	dgemt_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	cgemt_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	zgemt_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	sgema_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	dgema_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	cgema_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	zgema_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	sgems_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	dgems_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	cgems_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) -> c.int ---
	zgems_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) -> c.int ---
	sgemc_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) -> c.int ---
	dgemc_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---
	qgemc_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---
	cgemc_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) -> c.int ---
	zgemc_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---
	xgemc_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) -> c.int ---

	/* Lapack routines */
	sgetf2_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	dgetf2_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^Info) -> c.int ---
	qgetf2_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	cgetf2_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	zgetf2_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	xgetf2_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	sgetrf_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	dgetrf_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^Info) -> c.int ---
	qgetrf_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	cgetrf_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	zgetrf_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	xgetrf_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	slaswp_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	dlaswp_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	qlaswp_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	claswp_ :: proc(_: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	zlaswp_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	xlaswp_ :: proc(_: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint, _: ^blasint) -> c.int ---
	sgetrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dgetrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^Info) -> c.int ---
	qgetrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	cgetrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zgetrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xgetrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	sgesv_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dgesv_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qgesv_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	cgesv_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zgesv_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xgesv_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	spotf2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dpotf2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qpotf2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	cpotf2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zpotf2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xpotf2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	spotrf_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dpotrf_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qpotrf_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	cpotrf_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zpotrf_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xpotrf_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	spotri_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dpotri_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qpotri_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	cpotri_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zpotri_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xpotri_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	spotrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dpotrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qpotrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	cpotrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zpotrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xpotrs_ :: proc(_: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	slauu2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dlauu2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qlauu2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	clauu2_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zlauu2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xlauu2_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	slauum_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dlauum_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qlauum_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	clauum_ :: proc(_: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	zlauum_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xlauum_ :: proc(_: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	strti2_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dtrti2_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qtrti2_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	ctrti2_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	ztrti2_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xtrti2_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	strtri_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	dtrtri_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	qtrtri_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	ctrtri_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f32, _: ^blasint, _: ^blasint) -> c.int ---
	ztrtri_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	xtrtri_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^f64, _: ^blasint, _: ^blasint) -> c.int ---
	slamch_ :: proc(_: cstring) -> f32 ---
	dlamch_ :: proc(_: cstring) -> f64 ---
	qlamch_ :: proc(_: cstring) -> f64 ---
	slamc3_ :: proc(_: ^f32, _: ^f32) -> f32 ---
	dlamc3_ :: proc(_: ^f64, _: ^f64) -> f64 ---
	qlamc3_ :: proc(_: ^f64, _: ^f64) -> f64 ---

	/* BLAS extensions */
	saxpby_ :: proc(_: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	daxpby_ :: proc(_: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	caxpby_ :: proc(_: ^blasint, _: rawptr, _: ^f32, _: ^blasint, _: rawptr, _: ^f32, _: ^blasint) ---
	zaxpby_ :: proc(_: ^blasint, _: rawptr, _: ^f64, _: ^blasint, _: rawptr, _: ^f64, _: ^blasint) ---
	somatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	domatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	comatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^blasint) ---
	zomatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^blasint) ---
	simatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^blasint) ---
	dimatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^blasint) ---
	cimatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^blasint) ---
	zimatcopy_ :: proc(_: cstring, _: cstring, _: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^blasint) ---
	sgeadd_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	dgeadd_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
	cgeadd_ :: proc(_: ^blasint, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint, _: ^f32, _: ^f32, _: ^blasint) ---
	zgeadd_ :: proc(_: ^blasint, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint, _: ^f64, _: ^f64, _: ^blasint) ---
}
