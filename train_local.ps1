param(
    [string]$DataRoot = "",
    [string]$OutputPath = "",
    [string]$Config = "",
    [string]$Resume = "",
    [string]$PythonPath = "M:\Anaconda_envs\envs\retinexformer\python.exe",
    [int]$Workers = -1,
    [int]$StopAfter = 0,
    [switch]$Foreground
)
$ErrorActionPreference = "Stop"
$cliArgs = @("-u", (Join-Path $PSScriptRoot "train.py"))
if ($DataRoot) { $cliArgs += @("--data-root", $DataRoot) }
if ($OutputPath) { $cliArgs += @("--output", $OutputPath) }
if ($Config) { $cliArgs += @("--config", $Config) }
if ($Resume) { $cliArgs += @("--resume", $Resume) }
if ($Workers -ge 0) { $cliArgs += @("--workers", [string]$Workers) }
if ($StopAfter -gt 0) { $cliArgs += @("--stop-after", [string]$StopAfter) }
Push-Location $PSScriptRoot
try {
    if ($Foreground) {
        & $PythonPath @cliArgs
        if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
    } else {
        $logDirectory = Join-Path $PSScriptRoot "launch_logs"
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
        $stdout = Join-Path $logDirectory ("train_" + $stamp + ".log")
        $stderr = Join-Path $logDirectory ("train_" + $stamp + ".err.log")
        function Quote-NativeArgument([string]$Value) {
            $escaped = [regex]::Replace($Value, '(\*)"', '$1$1\"')
            $escaped = [regex]::Replace($escaped, '(\+)$', '$1$1')
            return '"' + $escaped + '"'
        }
        $argumentLine = ($cliArgs | ForEach-Object { Quote-NativeArgument $_ }) -join " "
        $process = Start-Process -FilePath $PythonPath -ArgumentList $argumentLine -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        Write-Output ("pid=" + $process.Id)
        Write-Output ("stdout=" + $stdout)
        Write-Output ("stderr=" + $stderr)
    }
} finally { Pop-Location }
