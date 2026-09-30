# CUDA 完整模型 Profiler

- 独立外部诊断进程；原始无 Profiler 基线保持单独身份。
- NSys：585 次完整 forward，每次 28 层，368610 次 kernel。
- 正式 workload 的测量部分在本时间线中共 258 次 forward。
- device span、kernel time、GPU 活跃区间并集、host elapsed 与硬件计数器分别报告。
- 每种 Profiler 只有一个进程；开关差异不构成显著性或加速结论。
- 项目模型/KV/调度归项目，GEMM 归 cuBLAS；不包含 GPU Serving 或 PagedAttention。

## Kernel 时间线

| Kernel | 次数 | 总时间 ms | 中位 us |
| --- | ---: | ---: | ---: |
| `minillm::cuda::<unnamed>::qk_kernel(minillm::cuda::DeviceTensorView<const unsigned short>, minillm::cuda::KvShape, unsigned long, minillm::cuda::DeviceTensorView<const float>, unsigned long, minillm::cuda::DeviceTensorView<const int>, minillm::cuda::DeviceTensorView<const int>, unsigned long, minillm::cuda::DeviceTensorView<float>, int *)` | 16380 | 7470.363 | 4.576 |
| `ampere_sgemm_128x64_tn` | 45080 | 2991.916 | 63.201 |
| `minillm::cuda::<unnamed>::pv_kernel(minillm::cuda::DeviceTensorView<const unsigned short>, minillm::cuda::KvShape, unsigned long, unsigned long, minillm::cuda::DeviceTensorView<const int>, minillm::cuda::DeviceTensorView<const int>, unsigned long, minillm::cuda::DeviceTensorView<float>, minillm::cuda::DeviceTensorView<float>, int *)` | 16380 | 2133.590 | 14.400 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)5, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>, float>>(T13)` | 27300 | 1331.821 | 51.969 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, float, (bool)0, (bool)1, (bool)1, (bool)0, (int)9, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>, float>>(T13)` | 36400 | 1177.138 | 35.072 |
| `void gemv2T_kernel_val<int, int, float, float, float, float, (int)128, (int)16, (int)4, (int)4, (bool)0, (bool)0, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>, float>>(T13, T6, T6)` | 355 | 954.440 | 2566.119 |
| `minillm::cuda::<unnamed>::softmax_kernel(minillm::cuda::KvShape, unsigned long, minillm::cuda::DeviceTensorView<const int>, minillm::cuda::DeviceTensorView<const int>, unsigned long, minillm::cuda::DeviceTensorView<float>, minillm::cuda::DeviceTensorView<float>, int *)` | 16380 | 681.703 | 2.496 |
| `minillm::cuda::<unnamed>::norm_kernel(minillm::cuda::DeviceTensorView<const float>, minillm::cuda::DeviceTensorView<const float>, minillm::cuda::DeviceTensorView<float>, float)` | 65890 | 317.448 | 3.328 |
| `ampere_sgemm_64x32_sliced1x4_tn` | 2240 | 82.703 | 38.753 |
| `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, float, float, float, (bool)1, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)` | 21980 | 77.041 | 3.648 |
| `minillm::cuda::<unnamed>::rope_kernel(minillm::cuda::DeviceTensorView<float>, minillm::cuda::DeviceTensorView<const int>, minillm::cuda::DeviceTensorView<const float>, int *)` | 32760 | 74.268 | 1.472 |
| `void minillm::cuda::<unnamed>::pointwise_kernel<(bool)0>(minillm::cuda::DeviceTensorView<float>, minillm::cuda::DeviceTensorView<const float>)` | 32760 | 70.656 | 1.088 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)8, (int)9, (bool)0, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>>(cublasGemmSmallNParams<T9, T10, T11, T1>)` | 840 | 68.708 | 78.017 |
| `minillm::cuda::<unnamed>::argmax_kernel(minillm::cuda::DeviceTensorView<const float>, minillm::cuda::DeviceTensorView<int>, int *)` | 370 | 67.331 | 172.866 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)2, (int)2, (bool)1, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>>(cublasGemmSmallNParams<T9, T10, T11, T1>)` | 985 | 52.265 | 36.192 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)1, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>>(cublasGemmSmallNParams<T9, T10, T11, T1>)` | 985 | 51.686 | 36.864 |
| `void minillm::cuda::<unnamed>::pointwise_kernel<(bool)1>(minillm::cuda::DeviceTensorView<float>, minillm::cuda::DeviceTensorView<const float>)` | 16380 | 51.675 | 1.216 |
| `minillm::cuda::<unnamed>::store_kernel(minillm::cuda::DeviceTensorView<unsigned short>, minillm::cuda::KvShape, unsigned long, minillm::cuda::DeviceTensorView<const float>, minillm::cuda::DeviceTensorView<const float>, minillm::cuda::DeviceTensorView<const int>, minillm::cuda::DeviceTensorView<const int>, int *)` | 16380 | 34.163 | 1.856 |
| `minillm::cuda::<unnamed>::finite_kernel(minillm::cuda::DeviceTensorView<const float>, int *)` | 16380 | 26.244 | 1.088 |
| `ampere_sgemm_128x32_tn` | 560 | 21.390 | 35.472 |
| `ampere_sgemm_32x32_sliced1x4_tn` | 280 | 16.663 | 58.321 |
| `void gemmSN_TN_kernel<float, (int)128, (int)16, (int)2, (int)4, (int)4, (int)4, (bool)0, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>>(cublasGemmSmallNParams<T9, T10, T11, T1>)` | 5 | 13.309 | 2594.344 |
| `minillm::cuda::<unnamed>::gather_kernel(minillm::cuda::DeviceTensorView<const float>, minillm::cuda::DeviceTensorView<const int>, minillm::cuda::DeviceTensorView<float>, int *)` | 955 | 1.721 | 1.408 |
| `minillm::cuda::<unnamed>::reset_kernel(int *)` | 585 | 0.522 | 0.896 |

## NCU

- Kernel：`minillm::cuda::<unnamed>::pv_kernel(minillm::cuda::DeviceTensorView<const unsigned short>, minillm::cuda::KvShape, unsigned long, unsigned long, minillm::cuda::DeviceTensorView<const int>, minillm::cuda::DeviceTensorView<const int>, unsigned long, minillm::cuda::DeviceTensorView<float>, minillm::cuda::DeviceTensorView<float>, int *)`。
- 目标为 chunked-prefill-1536 第一次 measured repetition 的最后一个 chunk、最后一层 PV。
- `gpu__time_duration.sum`：971808.0 ns。
- `sm__cycles_active.avg`：2218796.28 cycle。
- `dram__cycles_active.avg`：583408.0 cycle。
- `sm__warps_active.avg.pct_of_peak_sustained_active`：70.96 %。
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`：7.51 %。
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`：77.42 %。
- `profiler__replayer_passes`：8.0 pass。

## Profiler 开关诊断

| 用例 | NSys 相对变化 % | NCU 相对变化 % |
| --- | ---: | ---: |
| prefill-16 | -2.12 | 150.98 |
| prefill-128 | -14.46 | 43.35 |
| chunked-prefill-256 | -5.50 | 15.40 |
| chunked-prefill-1536 | -2.24 | 9.96 |
| decode-16 | 5.45 | 42.81 |
| decode-256 | 0.23 | 32.27 |
| decode-1536 | 0.75 | 12.88 |
| batch-2 | 3.36 | 44.32 |
| batch-4 | 0.03 | 36.05 |
| mixed-16-2 | 9.31 | 23.71 |
| generate-16-32 | 5.59 | 39.07 |
| generate-128-32 | 3.63 | 31.35 |

## 限制

- 未锁频；进程边界遥测不是连续 GPU 频率或功耗曲线。
- 选定 NCU kernel 的回放结果不能代替正常进程，也不能外推其他 shape。
- 微基准的正序/逆序差异没有在这里重做受控实验，不能据模型时间线宣称已消除。
- 关联基线摘要不替代完整模型与微基准归档；完整交付包须同时复核各归档。
