//! # volk-rs
//!
//! Rust bindings for the volk library.
//!
//! <https://github.com/ThomasHabets/volk-rs>
//!
//! [VOLK][volk] is the Vector-Optimized Library of Kernels. It is a library
//! that contains kernels of hand-written SIMD code for different mathematical
//! operations. Since each SIMD architecture can be very different and no
//! compiler has yet come along to handle vectorization properly or highly
//! efficiently, VOLK approaches the problem differently.
//!
//! [volk]: https://github.com/gnuradio/volk
use num_complex::Complex;
use paste::paste;

pub(crate) mod ffi {
    use super::Complex;
    use libc::{c_float, c_uint};

    #[link(name = "volk")]
    unsafe extern "C" {
        /// Take square root of 32bit floats.
        pub static volk_32f_sqrt_32f:
            extern "C" fn(out: *mut c_float, inp: *const c_float, len: c_uint);

        /// Multiply two vectors of complex numbers, pairwise.
        pub static volk_32fc_x2_multiply_32fc: extern "C" fn(
            out: *mut Complex<f32>,
            in0: *const Complex<f32>,
            in1: *const Complex<f32>,
            len: c_uint,
        );
        pub unsafe static volk_32fc_s32f_atan2_32f:
            extern "C" fn(out: *mut f32, in0: *const Complex<f32>, scale: f32, len: c_uint);
        pub unsafe static volk_32f_atan_32f:
            extern "C" fn(out: *mut f32, in0: *const f32, len: c_uint);

        pub unsafe static volk_32fc_x2_multiply_conjugate_32fc: extern "C" fn(
            out: *mut Complex<f32>,
            in0: *const Complex<f32>,
            in1: *const Complex<f32>,
            len: c_uint,
        );

        /// Get the machine alignment in bytes.
        #[must_use]
        pub fn volk_get_alignment() -> libc::size_t;

        // fn volk_malloc(size: usize, alignment: usize) -> *mut core::ffi::c_void;
        // fn volk_free(ptr: *mut core::ffi::c_void);
    }
}

#[derive(Debug)]
pub enum VolkError {
    InvalidArgument,
}

/// Get the machine alignment in bytes.
///
/// # Panics
///
/// Can't happen.
#[must_use]
#[allow(clippy::useless_conversion)]
pub fn volk_get_alignment() -> usize {
    (unsafe { ffi::volk_get_alignment() })
        .try_into()
        .expect("size_t failed to convert to usize")
}

macro_rules! make_funcs {
    (
        $(#[$meta:meta])*
        fn $name:ident($( $arg:ident : $ty:ty ),* $(,)?) $block:block
        checks { $(($a:expr, $b:expr)),* }
    ) => {
        make_funcs! {
            $(#[$meta])*
            fn $name($( $arg: $ty ),*) -> () $block
            checks { $(($a, $b)),* }
        }
    };
    (
        $(#[$meta:meta])*
        fn $name:ident($( $arg:ident : $ty:ty ),* $(,)?) -> $ret:ty $block:block
        checks { $(($a:expr, $b:expr)),* }

) => {
        paste! {
            $(#[$meta])*
            #[doc = concat!("\n\nThis version panics on bounds check failure.")]
            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            pub fn $name($( $arg : $ty ),*) -> $ret {
                $(assert_eq!($a, $b, "Slice lengths do not match");)*
                $(assert!(libc::c_uint::try_from($a).is_ok(), "Slice length does not fit in c_uint");)*
                $(assert!(libc::c_uint::try_from($b).is_ok(), "Slice length does not fit in c_uint");)*
                $block
            }
            $(#[$meta])*
            #[doc = concat!("\n\nThis version returns Err on bounds check failure.\n# Errors\nSlice lengths don't match, or don't fit in `libc::c_uint`.")]
            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            pub fn [<try_ $name>]($( $arg : $ty ),*) -> Result<$ret, VolkError> {
                $(if $a != $b {
                    return Err(VolkError::InvalidArgument);
                })*
                $(if libc::c_uint::try_from($a).is_err() {
                    return Err(VolkError::InvalidArgument);
                })*
                $(if libc::c_uint::try_from($b).is_err() {
                    return Err(VolkError::InvalidArgument);
                })*
                Ok($block)
            }
            $(#[$meta])*
            #[doc = concat!("\n\nThis unsafe version does NO bounds checks.\n\n# Safety\nCaller must ensure slice lengths are equal and fit in `libc::c_uint`.")]
            #[inline]
            #[allow(clippy::cast_possible_truncation)]
            pub unsafe fn [<$name  _unchecked>]($( $arg : $ty ),*) -> $ret {
                $(debug_assert_eq!($a, $b);)*
                $block
            }
        }
    }
}

make_funcs! {
    /// Take square root of a vector of floats.
    fn volk_32f_sqrt_32f(out: &mut [f32], inp: &[f32]) {
        unsafe { ffi::volk_32f_sqrt_32f(out.as_mut_ptr(), inp.as_ptr(), inp.len() as libc::c_uint) }
    }
    checks {
        (out.len(), inp.len())
    }
}

make_funcs! {
    /// Computes the arctan for each value in a complex vector and applies a
    /// normalization factor.
    fn volk_32fc_s32f_atan2_32f(out: &mut [f32], inp: &[Complex<f32>], scale: f32) {
        unsafe { ffi::volk_32fc_s32f_atan2_32f(out.as_mut_ptr(), inp.as_ptr(), scale, inp.len() as libc::c_uint) }
    }
    checks {
        (out.len(), inp.len())
    }
}

make_funcs! {
    /// Computes the arctan for each value in a complex vector and applies a
    /// normalization factor.
    fn volk_32f_atan_32f(out: &mut [f32], inp: &[f32]) {
        unsafe { ffi::volk_32f_atan_32f(out.as_mut_ptr(), inp.as_ptr(), inp.len() as libc::c_uint) }
    }
    checks {
        (out.len(), inp.len())
    }
}

make_funcs! {
    /// Multiply two complex vectors.
    fn volk_32fc_x2_multiply_32fc(
        out: &mut [Complex<f32>],
        in0: &[Complex<f32>],
        in1: &[Complex<f32>],
    ) {
        let func = unsafe { ffi::volk_32fc_x2_multiply_32fc };
        func(
            out.as_mut_ptr(),
            in0.as_ptr(),
            in1.as_ptr(),
            in0.len() as libc::c_uint,
        );
    }
    checks {
        (in0.len(), in1.len()),
        (out.len(), in0.len())
    }
}

make_funcs! {
    /// Multiplies a complex vector by the conjugate of a second complex vector
    /// and returns the complex result.
    fn volk_32fc_x2_multiply_conjugate_32fc(
        out: &mut [Complex<f32>],
        in0: &[Complex<f32>],
        in1: &[Complex<f32>],
    ) {
        let func = unsafe { ffi::volk_32fc_x2_multiply_conjugate_32fc };
        func(
            out.as_mut_ptr(),
            in0.as_ptr(),
            in1.as_ptr(),
            in0.len() as libc::c_uint,
        );
    }
    checks {
        (in0.len(), in1.len()),
        (out.len(), in0.len())
    }
}

make_funcs! {
    /// Multiply two complex vectors in place.
    ///
    /// Note that this is NOT formally defined to be safe, in volk. But GNU
    /// Radio makes this assumption, so "it should be fine".
    fn volk_32fc_x2_multiply_32fc_inplace(out: &mut [Complex<f32>], in0: &[Complex<f32>]) {
        let func = unsafe { ffi::volk_32fc_x2_multiply_32fc };
        func(
            out.as_mut_ptr(),
            in0.as_ptr(),
            out.as_ptr(),
            in0.len() as libc::c_uint,
        );
    }
    checks {
        (out.len(), in0.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    trait MyAbs {
        fn abs(self) -> f32;
    }
    impl MyAbs for f32 {
        fn abs(self) -> f32 {
            f32::abs(self)
        }
    }
    impl MyAbs for Complex<f32> {
        fn abs(self) -> f32 {
            self.norm()
        }
    }

    fn assert_close<T>(got: &[T], want: &[T])
    where
        T: MyAbs + Copy + PartialEq + std::fmt::Debug + std::fmt::Display,
        T: std::ops::Sub<T>,
        <T as std::ops::Sub>::Output: MyAbs,
    {
        assert_eq!(got.len(), want.len());
        for (n, (g, w)) in got.iter().copied().zip(want.iter().copied()).enumerate() {
            let diff = (g - w).abs();
            assert!(
                diff < 0.001,
                "Comparison failed:\n    got:  {got:?}\n    want: {want:?}\n    diff: {diff:?}\n    at entry {n}: diff between {g} and {w} is {diff}"
            );
        }
    }

    #[test]
    fn test_volk_32f_sqrt_32f() {
        for (input, want) in &[
            (vec![4.0f32], vec![2.0f32]),
            (
                vec![0.0f32, 1.0, 2.0, 4.0],
                vec![0.0, 1.0, std::f32::consts::SQRT_2, 2.0],
            ),
        ] {
            assert_eq!(input.len(), want.len());
            let mut got = vec![0.0f32; want.len()];
            volk_32f_sqrt_32f(&mut got, input);
            assert_close(&got, want);
        }
    }

    #[test]
    #[should_panic(expected = "Slice lengths do not match")]
    fn length_mismatch_panic_1() {
        let input = vec![0.0f32; 10];
        let mut out = vec![0.0f32; 11];
        volk_32f_sqrt_32f(&mut out, &input);
    }

    #[test]
    #[should_panic(expected = "Slice lengths do not match")]
    fn length_mismatch_panic_2() {
        let input = vec![0.0f32; 11];
        let mut out = vec![0.0f32; 10];
        volk_32f_sqrt_32f(&mut out, &input);
    }

    #[test]
    fn length_mismatch_error() {
        let mut a = vec![0.0f32; 10];
        let mut b = vec![0.0f32; 11];
        assert!(try_volk_32f_sqrt_32f(&mut a, &b).is_err());
        assert!(try_volk_32f_sqrt_32f(&mut b, &a).is_err());
    }

    #[test]
    fn test_volk_32f_sqrt_32f_try() {
        for right in [true, false] {
            for (input, want) in &[
                (vec![4.0f32], vec![2.0f32]),
                (
                    vec![0.0f32, 1.0, 2.0, 4.0],
                    vec![0.0, 1.0, std::f32::consts::SQRT_2, 2.0],
                ),
            ] {
                assert_eq!(input.len(), want.len());
                let len = if right { want.len() } else { 123 };
                let mut got = vec![0.0f32; len];
                let rc = try_volk_32f_sqrt_32f(&mut got, input);
                assert_eq!(right, rc.is_ok());
                if rc.is_ok() {
                    assert_close(&got, want);
                }
            }
        }
    }

    #[test]
    fn test_volk_32fc_x2_multiply_32fc() {
        for (in0, in1, want) in &[
            (
                vec![Complex::new(4.0f32, 0.0)],
                vec![Complex::new(2.0f32, 0.0)],
                vec![Complex::new(8.0f32, 0.0)],
            ),
            (
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(-2.0, 0.0),
                    Complex::new(4.0, 0.0),
                ],
                vec![
                    Complex::new(0.1, 0.0),
                    Complex::new(-2.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-1.0, 2.0),
                ],
                vec![
                    Complex::new(0.0, 0.0),
                    Complex::new(-2.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(-4.0, 8.0),
                ],
            ),
        ] {
            assert_eq!(in0.len(), want.len());
            assert_eq!(in1.len(), want.len());
            let mut got = vec![Complex::default(); want.len()];
            volk_32fc_x2_multiply_32fc(&mut got, in0, in1);
            assert_close(&got, want);
        }
    }

    #[test]
    fn alignment() {
        let align = volk_get_alignment();
        let want = 0;
        assert!(align > want, "alignment {align} needs to be > {want}");
    }
}
