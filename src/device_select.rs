// device_select.rs
// -----------------------------------------------------------
// Device selection helper for Candle (CPU vs CUDA)
// -----------------------------------------------------------
// Author  : Marcus Schlieper
// Company : ExpChat.ai
// Date    : 2026-01-04
// License : MIT / Apache-2.0
//
// Description
// Central helper to select the candle Device via a compile time constant.
// It supports CUDA (device 0) with safe fallback to CPU.
//
// History
// - 2026-01-04 Marcus Schlieper: initial version (USE_CUDA switch)
// - 2026-01-04 Marcus Schlieper: fix Result signature (Result<Device>)
// -----------------------------------------------------------
use candle::Device;
pub const USE_CUDA: bool = false;
pub const I_CUDA_DEVICE_INDEX: usize = 0;

pub fn get_default_device() -> candle::Device {
    // History
    // - 2026-01-04 Marcus Schlieper: robust cuda init with cpu fallback
    let mut res_dev;
    if USE_CUDA {
        res_dev = Device::new_cuda(I_CUDA_DEVICE_INDEX)
    } else {
        res_dev = Ok(Device::Cpu)
    }

    match res_dev {
        Ok(v) => v,
        Err(e) => {
           // Keep ASCII only, provide actionable diagnostics
                eprintln!("cuda init failed, index={}", I_CUDA_DEVICE_INDEX);
                eprintln!("cuda init error: {}", e);
                eprintln!("fallback cpu");
            Device::Cpu
        },
    }
}