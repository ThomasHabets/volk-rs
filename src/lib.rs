use num_complex::Complex;

mod raw {
    use super::Complex;
    use libc::{c_float, c_uint};
    #[link(name = "volk")]
    unsafe extern "C" {
        pub(crate) static mut volk_32f_sqrt_32f:
            Option<extern "C" fn(out: *mut c_float, inp: *const c_float, len: c_uint)>;
        pub(crate) static mut volk_32fc_x2_multiply_32fc_u: Option<
            extern "C" fn(
                out: *mut Complex<f32>,
                in0: *const Complex<f32>,
                in1: *const Complex<f32>,
                len: c_uint,
            ),
        >;
        // fn volk_malloc(size: usize, alignment: usize) -> *mut core::ffi::c_void;
        // fn volk_malloc(size: usize, alignment: usize) -> *mut core::ffi::c_void;
        // fn volk_free(ptr: *mut core::ffi::c_void);
    }
}

// Take square root of 32bit float.
pub fn volk_32f_sqrt_32f(out: &mut [f32], inp: &[f32]) {
    let func = unsafe { raw::volk_32f_sqrt_32f.unwrap() };
    func(out.as_mut_ptr(), inp.as_ptr(), inp.len() as libc::c_uint)
}

// Multiply two vectors.
pub fn volk_32fc_x2_multiply_32fc(
    out: &mut [Complex<f32>],
    in0: &[Complex<f32>],
    in1: &[Complex<f32>],
) {
    assert_eq!(in0.len(), in1.len());
    assert_eq!(out.len(), in0.len());
    let func = unsafe { raw::volk_32fc_x2_multiply_32fc_u.unwrap() };
    func(
        out.as_mut_ptr(),
        in0.as_ptr(),
        in1.as_ptr(),
        in0.len() as libc::c_uint,
    )
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
