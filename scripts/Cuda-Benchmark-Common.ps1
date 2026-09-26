function Get-CudaBenchmarkEnvironment {
    $result = [ordered]@{ at_utc = (Get-Date).ToUniversalTime().ToString('o')
        load_average_raw = $null; processor = $null; cpu_frequency_mhz = $null; gpu = $null; unavailable = @() }
    if (Test-Path -LiteralPath '/proc/loadavg') { $result.load_average_raw = (Get-Content -Raw /proc/loadavg).Trim() }
    else { $result.unavailable += 'load_average: /proc/loadavg 不可用' }
    if (Test-Path -LiteralPath '/proc/cpuinfo') {
        $cpuInfo = Get-Content /proc/cpuinfo
        $processor = $cpuInfo | Where-Object { $_ -match '^model name\s*:' } | Select-Object -First 1
        if ($processor) { $result.processor = ($processor -split ':', 2)[1].Trim() }
        $values = @($cpuInfo | Where-Object { $_ -match '^cpu MHz\s*:' } |
            ForEach-Object { [double]::Parse(($_ -split ':', 2)[1].Trim(), [Globalization.CultureInfo]::InvariantCulture) })
        $result.cpu_frequency_mhz = $values
    } else { $result.unavailable += 'cpu_frequency: /proc/cpuinfo 不可用' }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $raw = @(& nvidia-smi '--query-gpu=uuid,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,utilization.gpu,memory.used,memory.free' `
            '--format=csv,noheader,nounits' 2>&1 | ForEach-Object { "$_" })
        if ($LASTEXITCODE -eq 0) {
            $records = @($raw | ConvertFrom-Csv -Header @(
                'uuid', 'driver_version', 'temperature_c', 'sm_clock_mhz', 'memory_clock_mhz', 'power_w',
                'utilization_percent', 'used_mib', 'free_mib'))
            foreach ($record in $records) {
                $record.uuid = $record.uuid.Trim()
                $record.driver_version = $record.driver_version.Trim()
                foreach ($field in @('temperature_c', 'sm_clock_mhz', 'memory_clock_mhz', 'power_w',
                    'utilization_percent', 'used_mib', 'free_mib')) {
                    $number = 0.0
                    if ([double]::TryParse($record.$field, [Globalization.NumberStyles]::Float,
                        [Globalization.CultureInfo]::InvariantCulture, [ref]$number) -and
                        -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)) {
                        $record.$field = $number
                    } else {
                        $result.unavailable += "nvidia-smi $field`: $($record.$field)"
                        $record.$field = $null
                    }
                }
            }
            $result.gpu = [ordered]@{ raw = $raw; records = $records }
        } else { $result.unavailable += 'nvidia-smi 查询失败'; $result.gpu = [ordered]@{ raw = $raw; records = $null } }
    } else { $result.unavailable += 'nvidia-smi 不可用' }
    return $result
}
