use std::{
    any::Any,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(all(feature = "cuda-12060", feature = "cuda-13000"))]
compile_error!("Enable exactly one Neo CUDA build feature: cuda-12060 or cuda-13000, not both.");

#[cfg(not(any(feature = "cuda-12060", feature = "cuda-13000")))]
compile_error!("Enable exactly one Neo CUDA build feature: cuda-12060 or cuda-13000.");

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaGraph as CudarcCudaGraph, CudaSlice, CudaStream, DeviceRepr,
        DriverError, LaunchArgs, LaunchConfig, PinnedHostSlice, PushKernelArg, ValidAsZeroBits,
        sys,
    },
    nvrtc::{Ptx, compile_ptx, result as nvrtc_result},
};

include!("runtime/error.rs");
include!("runtime/context_device.rs");
include!("runtime/stream_graph.rs");
include!("runtime/module.rs");
include!("runtime/resource_types.rs");
include!("runtime/structured_uploads.rs");
include!("runtime/instance_uploads.rs");
include!("runtime/visibility_uploads.rs");
include!("runtime/sparse_texture_uploads.rs");
include!("runtime/material_stream_uploads.rs");
include!("runtime/mesh_uploads.rs");
include!("runtime/resource_packing.rs");
include!("runtime/visibility_packing.rs");
include!("runtime/sparse_texture_packing.rs");
include!("runtime/material_stream_packing.rs");
include!("runtime/resource_helpers.rs");
include!("runtime/cuda_prelude.rs");
include!("runtime/compiler_compile.rs");
include!("runtime/compiler_paths.rs");
include!("runtime/compiler_diagnostics.rs");
include!("runtime/compiler_search_path.rs");
include!("runtime/memory_host.rs");
include!("runtime/memory_fence.rs");
include!("runtime/memory_device.rs");
include!("runtime/kernel.rs");
include!("runtime/draw_device.rs");
include!("runtime/draw_streams.rs");
include!("runtime/material_abi.rs");
include!("runtime/material_bindings.rs");
include!("runtime/material_requirements.rs");
include!("runtime/material_kernel.rs");
include!("runtime/draw_policy_core.rs");
include!("runtime/draw_culling.rs");
include!("runtime/draw_policy_config.rs");
include!("runtime/draw_target.rs");
include!("runtime/draw_contract.rs");
include!("runtime/draw_recipes.rs");
include!("runtime/shared_draw_streams.rs");
include!("runtime/shared_frames.rs");
include!("runtime/d3d12_external.rs");
include!("runtime/image.rs");
include!("runtime/tests.rs");
