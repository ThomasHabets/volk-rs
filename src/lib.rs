use libc::{c_float, c_uint};

#[link(name = "volk")]
unsafe extern "C" {
    fn volk_initialize();
    // void volk_32f_sqr_32f(float* out, const float* in, unsigned int len);
    fn volk_32f_sqrt_32f(out: *mut c_float, inp: *const c_float, len: c_uint);

    // (Optional) aligned allocation helpers from VOLK:
    // void* volk_malloc(size_t size, size_t alignment);
    // void  volk_free(void* ptr);
    fn volk_malloc(size: usize, alignment: usize) -> *mut core::ffi::c_void;
    fn volk_free(ptr: *mut core::ffi::c_void);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqr() {
        //unsafe { volk_initialize(); }
        let input = vec![1.2f32];
        let mut output = vec![0.0f32];
        //unsafe { volk_32f_sqrt_32f(output.as_mut_ptr(), input.as_ptr(), input.len() as libc::c_uint ) };
        unsafe {
            let n = 1;
                    let align = 64usize;
            let in_ptr = volk_malloc(n * core::mem::size_of::<f32>(), align) as *mut c_float;
        let out_ptr = volk_malloc(n * core::mem::size_of::<f32>(), align) as *mut c_float;
        unsafe { volk_32f_sqrt_32f(out_ptr, in_ptr, input.len() as libc::c_uint ) };
        }
        assert_eq!(output, vec![2.4f32]);
    }
}
