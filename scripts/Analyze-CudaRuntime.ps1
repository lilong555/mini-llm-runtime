param([Parameter(Mandatory)][string]$Directory)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$python = if (Get-Command python3 -ErrorAction SilentlyContinue) { 'python3' } else { 'python' }
& $python (Join-Path $PSScriptRoot 'analyze_cuda_benchmark.py') --directory $Directory --write
if ($LASTEXITCODE -ne 0) { throw 'CUDA 模型性能证据复核失败，原有摘要保持不变。' }
