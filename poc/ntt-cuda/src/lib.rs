// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use sppark::{NTTInputOutputOrder, NTTDirection, NTTType};

extern "C" {
    fn compute_ntt(
        device_id: usize,
        inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> sppark::Error;

    fn compute_ntt_on_device(
        device_id: usize,
        d_inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> sppark::Error;

    fn ntt_alloc_and_upload(
        device_id: usize,
        d_out: *mut *mut core::ffi::c_void,
        h_in: *const core::ffi::c_void,
        nelems: usize,
    ) -> sppark::Error;

    fn ntt_download_and_free(
        device_id: usize,
        h_out: *mut core::ffi::c_void,
        d_in: *mut core::ffi::c_void,
        nelems: usize,
    ) -> sppark::Error;

    fn ntt_batch_alloc_and_upload(
        device_id: usize,
        d_out: *mut *mut core::ffi::c_void,
        h_in: *const core::ffi::c_void,
        nelems: usize,
        ncols: usize,
    ) -> sppark::Error;

    fn ntt_batch_compute(
        device_id: usize,
        d_inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        ncols: usize,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> sppark::Error;

    fn ntt_batch_download_and_free(
        device_id: usize,
        h_out: *mut core::ffi::c_void,
        d_in: *mut core::ffi::c_void,
        nelems: usize,
        ncols: usize,
    ) -> sppark::Error;
}

/// Compute an in-place NTT on the input data.
#[allow(non_snake_case)]
pub fn NTT<T>(device_id: usize, inout: &mut [T], order: NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut _,
            len.trailing_zeros(),
            order,
            NTTDirection::Forward,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

/// Compute an in-place iNTT on the input data.
#[allow(non_snake_case)]
pub fn iNTT<T>(device_id: usize, inout: &mut [T], order: NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut _,
            len.trailing_zeros(),
            order,
            NTTDirection::Inverse,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

#[allow(non_snake_case)]
pub fn coset_NTT<T>(
    device_id: usize,
    inout: &mut [T],
    order: NTTInputOutputOrder,
) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut _,
            len.trailing_zeros(),
            order,
            NTTDirection::Forward,
            NTTType::Coset,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

#[allow(non_snake_case)]
pub fn coset_iNTT<T>(
    device_id: usize,
    inout: &mut [T],
    order: NTTInputOutputOrder,
) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut _,
            len.trailing_zeros(),
            order,
            NTTDirection::Inverse,
            NTTType::Coset,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

/// Opaque handle to device-resident NTT data.
pub struct DeviceNTTData {
    device_id: usize,
    ptr: *mut core::ffi::c_void,
    len: usize,
}

unsafe impl Send for DeviceNTTData {}

impl DeviceNTTData {
    /// Upload host data to the GPU, returning a device handle.
    pub fn upload<T>(device_id: usize, data: &[T]) -> Self {
        let len = data.len();
        if (len & (len - 1)) != 0 {
            panic!("data.len() is not power of 2");
        }

        let mut d_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        let err = unsafe {
            ntt_alloc_and_upload(
                device_id,
                &mut d_ptr,
                data.as_ptr() as *const _,
                len,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }

        DeviceNTTData { device_id, ptr: d_ptr, len }
    }

    /// Compute an in-place NTT on device-resident data.
    #[allow(non_snake_case)]
    pub fn NTT(&mut self, order: NTTInputOutputOrder) {
        let err = unsafe {
            compute_ntt_on_device(
                self.device_id,
                self.ptr,
                self.len.trailing_zeros(),
                order,
                NTTDirection::Forward,
                NTTType::Standard,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }

    /// Compute an in-place iNTT on device-resident data.
    #[allow(non_snake_case)]
    pub fn iNTT(&mut self, order: NTTInputOutputOrder) {
        let err = unsafe {
            compute_ntt_on_device(
                self.device_id,
                self.ptr,
                self.len.trailing_zeros(),
                order,
                NTTDirection::Inverse,
                NTTType::Standard,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }

    /// Download device data to host and free GPU memory.
    pub fn download<T>(self, out: &mut [T]) {
        assert_eq!(out.len(), self.len);
        let err = unsafe {
            ntt_download_and_free(
                self.device_id,
                out.as_mut_ptr() as *mut _,
                self.ptr,
                self.len,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        // ptr is now freed, don't drop
        core::mem::forget(self);
    }

    pub fn len(&self) -> usize { self.len }
}

/// Opaque handle to a batch of device-resident NTT columns.
/// All columns have the same length (power of 2) and are stored contiguously.
pub struct DeviceBatchNTTData {
    device_id: usize,
    ptr: *mut core::ffi::c_void,
    col_len: usize,
    ncols: usize,
}

unsafe impl Send for DeviceBatchNTTData {}

impl DeviceBatchNTTData {
    /// Upload a flat buffer of `ncols` contiguous columns to the GPU.
    /// `data.len()` must equal `col_len * ncols`, and `col_len` must be a power of 2.
    pub fn upload<T>(device_id: usize, data: &[T], col_len: usize) -> Self {
        assert!((col_len & (col_len - 1)) == 0, "col_len must be power of 2");
        assert!(data.len() % col_len == 0, "data.len() must be multiple of col_len");
        let ncols = data.len() / col_len;

        let mut d_ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        let err = unsafe {
            ntt_batch_alloc_and_upload(
                device_id,
                &mut d_ptr,
                data.as_ptr() as *const _,
                col_len,
                ncols,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }

        DeviceBatchNTTData { device_id, ptr: d_ptr, col_len, ncols }
    }

    /// Apply forward NTT to all columns.
    #[allow(non_snake_case)]
    pub fn NTT_all(&mut self, order: NTTInputOutputOrder) {
        let err = unsafe {
            ntt_batch_compute(
                self.device_id,
                self.ptr,
                self.col_len.trailing_zeros(),
                self.ncols,
                order,
                NTTDirection::Forward,
                NTTType::Standard,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }

    /// Apply inverse NTT to all columns.
    #[allow(non_snake_case)]
    pub fn iNTT_all(&mut self, order: NTTInputOutputOrder) {
        let err = unsafe {
            ntt_batch_compute(
                self.device_id,
                self.ptr,
                self.col_len.trailing_zeros(),
                self.ncols,
                order,
                NTTDirection::Inverse,
                NTTType::Standard,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
    }

    /// Download all columns to host and free GPU memory.
    pub fn download<T>(self, out: &mut [T]) {
        assert_eq!(out.len(), self.col_len * self.ncols);
        let err = unsafe {
            ntt_batch_download_and_free(
                self.device_id,
                out.as_mut_ptr() as *mut _,
                self.ptr,
                self.col_len,
                self.ncols,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        core::mem::forget(self);
    }

    pub fn col_len(&self) -> usize { self.col_len }
    pub fn ncols(&self) -> usize { self.ncols }
    pub fn as_mut_ptr(&self) -> *mut core::ffi::c_void { self.ptr }
}

// RISC Zero batch API — operates on device-resident data with caller-provided stream.
extern "C" {
    pub fn sppark_batch_NTT(
        stream: *mut core::ffi::c_void,
        d_inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_iNTT(
        stream: *mut core::ffi::c_void,
        d_inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_expand(
        stream: *mut core::ffi::c_void,
        d_out: *mut core::ffi::c_void,
        d_in: *const core::ffi::c_void,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_zk_shift(
        stream: *mut core::ffi::c_void,
        d_inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    pub fn sppark_batch_bit_reverse(
        stream: *mut core::ffi::c_void,
        d_inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> sppark::Error;

    // Stream management
    pub fn sppark_create_stream(
        out: *mut *mut core::ffi::c_void,
    ) -> sppark::Error;

    pub fn sppark_sync_stream(
        stream: *mut core::ffi::c_void,
    ) -> sppark::Error;

    pub fn sppark_destroy_stream(stream: *mut core::ffi::c_void);

    // Device memory management
    pub fn sppark_alloc_gpu(
        d_out: *mut *mut core::ffi::c_void,
        bytes: usize,
    ) -> sppark::Error;

    pub fn sppark_free_gpu(d_ptr: *mut core::ffi::c_void);

    pub fn sppark_htod_on_stream(
        d_dst: *mut core::ffi::c_void,
        h_src: *const core::ffi::c_void,
        bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> sppark::Error;

    pub fn sppark_dtoh_on_stream(
        h_dst: *mut core::ffi::c_void,
        d_src: *const core::ffi::c_void,
        bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> sppark::Error;
}
