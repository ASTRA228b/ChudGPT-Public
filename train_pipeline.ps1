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
    Invoke-LoggedStage "dense Public corpus build" @("build_public_data.py")
    Invoke-LoggedStage "balanced alignment build" @("build_alignment_data.py")
    Invoke-LoggedStage "Public v3 corpus alignment" @("fine_tune.py", "--config", "configs/public_v3.yaml", "--device", $Device)
    Invoke-LoggedStage "Public v4 response alignment" @("fine_tune.py", "--config", "configs/public_v4_alignment.yaml", "--device", $Device)
    Invoke-LoggedStage "Public v5 dense training" @("fine_tune.py", "--config", "configs/public_v5_dense.yaml", "--device", $Device)
    Invoke-LoggedStage "Public v6 balanced correction" @("fine_tune.py", "--config", "configs/public_v6_alignment.yaml", "--device", $Device)
    Invoke-LoggedStage "Public v7 AI and identity alignment" @("fine_tune.py", "--config", "configs/public_v7_identity.yaml", "--device", $Device)
    Invoke-LoggedStage "Public v8 model-family alignment" @("fine_tune.py", "--config", "configs/public_v8_family.yaml", "--device", $Device)
    Invoke-LoggedStage "checkpoint evaluation" @("evaluate_public.py", "--device", $Device)
    "[$(Get-Date -Format s)] SUCCESS: native API checkpoint is ready at $projectRoot\checkpoints\public_v8\best.pt" | Tee-Object -FilePath $logPath -Append
}
catch {
    "[$(Get-Date -Format s)] FAILED: $($_.Exception.Message)" | Tee-Object -FilePath $logPath -Append
    exit 1
}
