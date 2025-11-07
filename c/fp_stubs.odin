package openblas_c
// Floating point classification stubs for Windows OpenBLAS compatibility
// OpenBLAS built with MinGW expects these Microsoft-specific functions

import "core:c"
import "core:math"

when ODIN_OS == .Windows {
	// These are Microsoft floating point classification functions that
	// OpenBLAS expects when built with MinGW. We provide stubs that
	// delegate to Odin's math functions.

	@(export, link_name = "_dclass")
	_dclass :: proc "c" (x: f64) -> c.int {
		// Return values based on fpclassify standard
		// 2 = normal, 1 = zero, 0 = denormal, -1 = infinite, -2 = nan
		if math.is_nan(x) {return -2}
		if math.is_inf(x, 0) {return -1}
		if x == 0.0 {return 1}
		// For simplicity, treat all other values as normal
		return 2
	}

	@(export, link_name = "_fdclass")
	_fdclass :: proc "c" (x: f32) -> c.int {
		if math.is_nan(f64(x)) {return -2}
		if math.is_inf(f64(x), 0) {return -1}
		if x == 0.0 {return 1}
		return 2
	}
}
