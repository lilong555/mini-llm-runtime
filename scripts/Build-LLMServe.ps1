param(
    [switch]$Cuda,
    [string]$BuildDirectory = '',
    [ValidateRange(1, 64)][int]$Jobs = 6,
    [string]$CudaArchitectures = '89',
    [string]$Target = 'all'
)

$ErrorActionPreference = 'Stop'
$env:VSLANG = '1033'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $root $(if ($Cuda) { 'build\cuda' } else { 'build\cpu' })
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'Visual Studio C++ build tools are required.' }
    Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}
$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
$arguments = @('-S', $root, '-B', $BuildDirectory, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', "-DLLMSERVE_CUDA=$($Cuda.IsPresent.ToString().ToUpperInvariant())")
if ($Cuda) {
    $nvcc = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
    if (-not $nvcc -and $env:CUDA_PATH) { $nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe' }
    if (-not $nvcc -or -not (Test-Path -LiteralPath $nvcc)) { throw 'CUDA Toolkit nvcc was not found.' }
    $arguments += @("-DCMAKE_CUDA_COMPILER=$nvcc", "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures")
}
& $cmake @arguments
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
& $cmake --build $BuildDirectory --parallel $Jobs --target $Target
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
