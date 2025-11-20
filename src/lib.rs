//! # volk-rs
//! 
//! Rust bindings for the volk library.
//! 
//! https://github.com/ThomasHabets/volk-rs
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
        pub static mut volk_32f_sqrt_32f:
            extern "C" fn(out: *mut c_float, inp: *const c_float, len: c_uint);

        /// Multiply two vectors of complex numbers, pairwise.
        pub static mut volk_32fc_x2_multiply_32fc: extern "C" fn(
            out: *mut Complex<f32>,
            in0: *const Complex<f32>,
            in1: *const Complex<f32>,
            len: c_uint,
        );
        // fn volk_malloc(size: usize, alignment: usize) -> *mut core::ffi::c_void;
        // fn volk_malloc(size: usize, alignment: usize) -> *mut core::ffi::c_void;
        // fn volk_free(ptr: *mut core::ffi::c_void);
    }
}

#[derive(Debug)]
pub enum VolkError {
    InvalidArgument,
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
            pub fn $name($( $arg : $ty ),*) -> $ret {
                $(assert_eq!($a, $b);)*
                $block
            }
            $(#[$meta])*
            #[doc = concat!("\n\nThis version returns Err on bounds check failure.")]
            #[inline]
            pub fn [<try_ $name>]($( $arg : $ty ),*) -> Result<$ret, VolkError> {
                $(if $a != $b {
                    return Err(VolkError::InvalidArgument);
                })*
                Ok($block)
            }
            $(#[$meta])*
            #[doc = concat!("\n\nThis unsafe version does NO bounds checks.")]
            #[inline]
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
        )
    }
    checks {
        (in0.len(), in1.len()),
        (out.len(), in0.len())
    }
}

make_funcs! {
    /// Multiply two complex vectors in place.
    fn volk_32fc_x2_multiply_32fc_inplace(out: &mut [Complex<f32>], in0: &[Complex<f32>]) {
        let func = unsafe { ffi::volk_32fc_x2_multiply_32fc };
        func(
            out.as_mut_ptr(),
            in0.as_ptr(),
            out.as_ptr(),
            in0.len() as libc::c_uint,
        )
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
            (vec![0.0f32, 1.0, 2.0, 4.0], vec![0.0, 1.0, 1.414213, 2.0]),
        ] {
            assert_eq!(input.len(), want.len());
            let mut got = vec![0.0f32; want.len()];
            volk_32f_sqrt_32f(&mut got, input);
            assert_close(&got, want);
        }
    }

    #[test]
    fn test_volk_32f_sqrt_32f_try() {
        for right in [true, false] {
            for (input, want) in &[
                (vec![4.0f32], vec![2.0f32]),
                (vec![0.0f32, 1.0, 2.0, 4.0], vec![0.0, 1.0, 1.414213, 2.0]),
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
}
