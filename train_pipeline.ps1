param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = "C:\tmp\ChudGPT-venv\Scripts\python.exe"
$logDirectory = Join-Path $projectRoot "reports"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirectory "public_training_$timestamp.log"

if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python environment not found at $pythonPath" }
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
Set-Location -LiteralPath $projectRoot
$env:PYTHONUNBUFFERED = "1"

function Invoke-LoggedStage {
    param([string]$Name, [string[]]$Arguments)
    "[$(Get-Date -Format s)] START: $Name" | Tee-Object -FilePath $logPath -Append
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pythonPath @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    $stageExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorPreference
    if ($stageExitCode -ne 0) { throw "$Name failed with exit code $stageExitCode. See $logPath" }
    "[$(Get-Date -Format s)] COMPLETE: $Name" | Tee-Object -FilePath $logPath -Append
}

try {
    Invoke-LoggedStage "dataset preparation" @("prepare.py")
    Invoke-LoggedStage "base language-model training" @("train_public.py", "--device", $Device)
    Invoke-LoggedStage "response-only conversational fine-tuning" @("finetune_public.py", "--device", $Device)
    Invoke-LoggedStage "checkpoint evaluation" @("evaluate_public.py", "--device", $Device)
    "[$(Get-Date -Format s)] SUCCESS: native API checkpoint is ready at $projectRoot\checkpoints\chat\best.pt" | Tee-Object -FilePath $logPath -Append
}
catch {
    "[$(Get-Date -Format s)] FAILED: $($_.Exception.Message)" | Tee-Object -FilePath $logPath -Append
    exit 1
}
