$ErrorActionPreference = 'Stop'
$demoPython = 'M:\Anaconda_envs\envs\retinexformer\python.exe'
$queueDirectory = Join-Path $PSScriptRoot 'runs\ablation_10k_seed100_20260921'
New-Item -ItemType Directory -Path $queueDirectory -Force | Out-Null
$launchStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$queueStdout = Join-Path $queueDirectory "queue_stdout_$launchStamp.log"
$queueStderr = Join-Path $queueDirectory "queue_stderr_$launchStamp.log"
$queueArguments = @('-u', '-X', 'utf8', ('"' + (Join-Path $PSScriptRoot 'run_ablation_queue.py') + '"'), '--queue-dir', ('"' + $queueDirectory + '"'))
$queueProcess = Start-Process -FilePath $demoPython -ArgumentList $queueArguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $queueStdout -RedirectStandardError $queueStderr -PassThru
[pscustomobject]@{ PID = $queueProcess.Id; Queue = $queueDirectory; Stdout = $queueStdout; Stderr = $queueStderr }
