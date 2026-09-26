# 自有 CUDA 模型性能基线

- 状态：`measurement_inconclusive`。
- 主指标为无 profiler 的 host forward 到 token 延迟，包含必要 copy、finite/argmax 和同步。
- 每个配置使用 5 个独立 trial；每进程 2 次 warmup、3 次测量。表中负百分比表示 CUDA 更快。
- A/A 噪声取 CPU 与 CUDA 的较大值；超过 10% 时不可据此宣称加速或没有退化。
- 95% 区间为 5^5 次成对 trial 重采样的 percentile bootstrap；仅 5 个 trial，区间稳定性有限。
- 数值门禁继承同一 Runtime 源码的完整语料验收；本包保留摘要及源码对应，不重复常驻数值 reference。
- 本报告仅覆盖模型层，不代表 microbenchmark、Profiler、HTTP TTFT、GPU Serving 或 PagedAttention。

| CPU 对照 | 用例 | CPU ms | CUDA ms | 配对差异 % | 95% 区间 % | 噪声 % | 结论 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| cpu8 | prefill-16 | 202.179 | 17.026 | -91.58 | [-92.27, -90.68] | 7.57 | faster |
| cpu8 | prefill-128 | 1460.390 | 31.432 | -97.86 | [-97.98, -97.75] | 18.83 | measurement_inconclusive |
| cpu8 | chunked-prefill-256 | 3014.161 | 64.881 | -97.88 | [-98.00, -97.70] | 5.68 | faster |
| cpu8 | chunked-prefill-1536 | 22797.955 | 1055.615 | -95.35 | [-95.68, -95.18] | 6.92 | faster |
| cpu8 | decode-16 | 41.932 | 12.779 | -69.52 | [-73.08, -67.75] | 7.33 | faster |
| cpu8 | decode-256 | 43.171 | 13.221 | -69.89 | [-73.71, -66.10] | 8.32 | faster |
| cpu8 | decode-1536 | 56.697 | 17.342 | -69.60 | [-70.95, -67.74] | 7.12 | faster |
| cpu8 | batch-2 | 60.005 | 13.914 | -76.93 | [-77.34, -75.57] | 7.81 | faster |
| cpu8 | batch-4 | 99.136 | 14.005 | -85.57 | [-87.22, -84.56] | 14.61 | measurement_inconclusive |
| cpu8 | mixed-16-2 | 239.038 | 16.254 | -92.97 | [-93.33, -92.92] | 7.38 | faster |
| cpu8 | generate-16-32 | 1516.971 | 404.045 | -73.03 | [-74.74, -72.04] | 10.72 | measurement_inconclusive |
| cpu8 | generate-128-32 | 2775.244 | 426.563 | -84.39 | [-85.54, -84.19] | 9.26 | faster |
| cpu16 | prefill-16 | 174.721 | 17.466 | -90.00 | [-91.11, -88.92] | 23.06 | measurement_inconclusive |
| cpu16 | prefill-128 | 1117.588 | 33.477 | -97.05 | [-97.47, -96.76] | 18.83 | measurement_inconclusive |
| cpu16 | chunked-prefill-256 | 2135.273 | 68.789 | -96.90 | [-96.96, -96.75] | 9.14 | faster |
| cpu16 | chunked-prefill-1536 | 16327.617 | 1039.755 | -93.55 | [-93.70, -93.33] | 6.92 | faster |
| cpu16 | decode-16 | 57.172 | 12.236 | -77.69 | [-78.66, -77.26] | 7.69 | faster |
| cpu16 | decode-256 | 60.108 | 13.226 | -77.66 | [-79.48, -76.94] | 6.73 | faster |
| cpu16 | decode-1536 | 75.925 | 17.348 | -76.61 | [-77.99, -74.68] | 12.22 | measurement_inconclusive |
| cpu16 | batch-2 | 68.280 | 13.744 | -79.99 | [-80.66, -79.57] | 15.52 | measurement_inconclusive |
| cpu16 | batch-4 | 100.390 | 14.476 | -85.74 | [-86.30, -84.89] | 14.61 | measurement_inconclusive |
| cpu16 | mixed-16-2 | 193.612 | 17.039 | -91.52 | [-91.67, -90.66] | 7.38 | faster |
| cpu16 | generate-16-32 | 1882.356 | 405.618 | -77.89 | [-78.47, -77.36] | 10.72 | measurement_inconclusive |
| cpu16 | generate-128-32 | 2768.409 | 426.802 | -84.03 | [-84.86, -82.99] | 13.19 | measurement_inconclusive |
