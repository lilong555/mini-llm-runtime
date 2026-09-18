param([string]$Directory = '')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $Directory) {
    $Directory = Join-Path (Split-Path -Parent $PSScriptRoot) 'benchmarks/results'
}
$files = @(Get-ChildItem -LiteralPath $Directory -Recurse -File -Filter '*.xml')
if ($files.Count -eq 0) { throw 'No CTest XML reports were found.' }
$suiteCount = 0
$caseCount = 0
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
        $counts = [regex]::Match($text, '(?m)^([1-9][0-9]*)/([1-9][0-9]*) tests passed\r?$')
        if (-not $counts.Success -or $counts.Groups[1].Value -ne $counts.Groups[2].Value -or
            $text.Contains('[FAIL]') -or $text.Contains('output was removed')) {
            throw "Truncated or inconsistent test output: $($file.FullName)"
        }
        ++$suiteCount
        $caseCount += [int]$counts.Groups[1].Value
    }
}
[pscustomobject]@{
    Reports = $files.Count
    SuiteExecutions = $suiteCount
    CaseExecutions = $caseCount
    Status = 'passed'
}
