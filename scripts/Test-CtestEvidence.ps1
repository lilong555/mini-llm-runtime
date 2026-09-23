param([string]$Directory = '')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $Directory) {
    $Directory = Join-Path (Split-Path -Parent $PSScriptRoot) 'benchmarks/results'
}
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$diagnosticPrefix = (Join-Path $Directory 'diagnostics') + [System.IO.Path]::DirectorySeparatorChar
$comparison = if ($env:OS -eq 'Windows_NT') { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
$allFiles = @(Get-ChildItem -LiteralPath $Directory -Recurse -File -Filter '*.xml')
$files = @($allFiles | Where-Object { -not $_.FullName.StartsWith($diagnosticPrefix, $comparison) })
$diagnostics = @($allFiles | Where-Object { $_.FullName.StartsWith($diagnosticPrefix, $comparison) })
if ($files.Count -eq 0) { throw 'No CTest XML reports were found.' }
$suiteCount = 0
$caseCount = 0
$uncountedSuites = 0
foreach ($file in $files) {
    $document = [System.Xml.XmlDocument]::new()
    $document.XmlResolver = $null
    $document.Load($file.FullName)
    $suite = $document.DocumentElement
    if (-not $suite -or $suite.get_LocalName() -ne 'testsuite' -or
        $suite.GetAttribute('failures') -ne '0' -or
        $suite.GetAttribute('skipped') -ne '0' -or
        $suite.GetAttribute('disabled') -ne '0') {
        throw "Incomplete or failed CTest report: $($file.FullName)"
    }
    $tests = @($suite.SelectNodes('testcase'))
    if ($tests.Count -eq 0 -or $tests.Count -ne [int]$suite.GetAttribute('tests')) {
        throw "CTest suite count does not match: $($file.FullName)"
    }
    foreach ($test in $tests) {
        $body = $test.SelectSingleNode('system-out')
        if ($test.GetAttribute('status') -ne 'run' -or
            $test.SelectSingleNode('failure') -or $test.SelectSingleNode('error') -or
            -not $body) {
            throw "Missing or failed test output: $($file.FullName)"
        }
        $text = $body.get_InnerText()
        $counts = [regex]::Match($text, '(?m)^([1-9][0-9]*)/([1-9][0-9]*) (?:benchmark validation )?tests passed\r?$')
        $legacyBenchmark = $test.GetAttribute('name') -eq 'benchmark-validation' -and
            $text.Trim() -ceq 'benchmark validation fixtures passed'
        if ((-not $counts.Success -and -not $legacyBenchmark) -or
            ($counts.Success -and $counts.Groups[1].Value -ne $counts.Groups[2].Value) -or
            $text.Contains('[FAIL]') -or $text.Contains('output was removed')) {
            throw "Truncated or inconsistent test output: $($file.FullName)"
        }
        ++$suiteCount
        if ($counts.Success) {
            $caseCount += [int]$counts.Groups[1].Value
        } else {
            ++$uncountedSuites
        }
    }
}
[pscustomobject]@{
    Reports = $files.Count
    SuiteExecutions = $suiteCount
    CaseExecutions = $caseCount
    SuitesWithoutCaseCounts = $uncountedSuites
    DiagnosticReports = @($diagnostics | ForEach-Object { $_.FullName })
    Status = if ($uncountedSuites -eq 0) { 'passed' } else { 'passed_with_unreported_case_counts' }
}
