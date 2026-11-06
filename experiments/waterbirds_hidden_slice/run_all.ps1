param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-Setting {
    param([string]$Name, [string]$Default)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

function Split-Setting {
    param([string]$Value)
    return @($Value -split '\s+' | Where-Object { $_ })
}

function Invoke-Stage {
    param([string]$Script, [string[]]$Arguments)
    Write-Host ""
    Write-Host "### $Script"
    & $script:PythonExe "-u" (Join-Path $script:Here $Script) @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Script failed with exit code $LASTEXITCODE"
    }
}

# Paths and runtime.
$PythonExe = Get-Setting "PYTHON" "python"
$PythonVersion = (& $PythonExe -c "import platform, sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info >= (3, 8) else 2)").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "$PythonExe is Python $PythonVersion; Python 3.8 or newer is required. Set PYTHON to a compatible interpreter."
}
$RootDir = Get-Setting "ROOT_DIR" (Get-Setting "DATA_ROOT" "data/wilds")
$OutDir = Get-Setting "OUT_DIR" "build/results/waterbirds_hidden_slice_v2"
$DataDir = Get-Setting "DATA_DIR" (Get-Setting "PREPARED_DIR" (Join-Path $OutDir "prepared"))
$ImageRoot = Get-Setting "IMAGE_ROOT" (Join-Path $RootDir "waterbirds_v1.0")
$FeatureCache = Get-Setting "FEATURE_CACHE" "build/features/waterbirds_mobilenetv3_small_attested.npz"
$NuisanceCache = Get-Setting "NUISANCE_CACHE" "build/features/waterbirds_clip_vit_base_patch32_nuisance.npz"
$Device = Get-Setting "DEVICE" "auto"
$NumWorkers = Get-Setting "NUM_WORKERS" "8"
$PrepareSeed = Get-Setting "PREPARE_SEED" "17"
$Download = Get-Setting "DOWNLOAD" "0"
$AllowModelDownload = Get-Setting "ALLOW_MODEL_DOWNLOAD" "0"
$ClipRevision = Get-Setting "CLIP_REVISION" ""
$Amp = Get-Setting "AMP" "1"
$PinMemory = Get-Setting "PIN_MEMORY" "1"
$Deterministic = Get-Setting "DETERMINISTIC" "1"

# Frozen scientific protocol.
$SeedsText = Get-Setting "SEEDS" "17 18 19 20 21"
$Folds = Get-Setting "FOLDS" "5"
$GateRestarts = Get-Setting "GATE_RESTARTS" (Get-Setting "RESTARTS" "10")
$MainMass = Get-Setting "MAIN_MASS" "0.10"
$MassesText = Get-Setting "MASSES" "0.05 0.10 0.20"
$MainShrinkage = Get-Setting "MAIN_SHRINKAGE" "0.10"
$ShrinkagesText = Get-Setting "SHRINKAGES" "0.05 0.10 0.20"
$ClipModel = Get-Setting "CLIP_MODEL" "openai/clip-vit-base-patch32"
$Ridge = Get-Setting "RIDGE" "0.001"
$Epsilon = Get-Setting "EPSILON" (Get-Setting "EPS" "1e-8")
$EssFraction = Get-Setting "ESS_FRACTION" (Get-Setting "MIN_ESS_FRACTION" "0.8")
$CapMultiplier = Get-Setting "CAP_MULTIPLIER" (Get-Setting "MAX_WEIGHT_MULTIPLIER" "2.0")
$CalibrationFraction = Get-Setting "CALIBRATION_FRACTION" "0.15"
$HeadSteps = Get-Setting "HEAD_STEPS" "1200"
$HeadLr = Get-Setting "HEAD_LR" "0.01"
$HeadWeightDecay = Get-Setting "HEAD_WEIGHT_DECAY" "0.0001"
$GateSteps = Get-Setting "GATE_STEPS" "1000"
$GateLr = Get-Setting "GATE_LR" "0.03"
$LossLambdasText = Get-Setting "LOSS_LAMBDAS" "0.1 0.3 1.0"
$FisherLambdasText = Get-Setting "FISHER_LAMBDAS" "0.01 0.03 0.1"
$CombinedLossLambdasText = Get-Setting "COMBINED_LOSS_LAMBDAS" $LossLambdasText
$CombinedFisherLambdasText = Get-Setting "COMBINED_FISHER_LAMBDAS" $FisherLambdasText
$AccuracyBudget = Get-Setting "ACCURACY_BUDGET" "0.02"
$BalancedAccuracyBudget = Get-Setting "BALANCED_ACCURACY_BUDGET" "0.02"
$PerClassAccuracyBudget = Get-Setting "PER_CLASS_ACCURACY_BUDGET" "0.02"
$EarlyStoppingPatience = Get-Setting "EARLY_STOPPING_PATIENCE" "12"
$EarlyStoppingEvalEvery = Get-Setting "EARLY_STOPPING_EVAL_EVERY" "25"
$EarlyStoppingMinDelta = Get-Setting "EARLY_STOPPING_MIN_DELTA" "1e-5"
$BootstrapReplicates = Get-Setting "BOOTSTRAP_REPLICATES" (Get-Setting "N_BOOT" "2000")
$BootstrapConfidence = Get-Setting "BOOTSTRAP_CONFIDENCE" "0.95"
$RandomSlices = Get-Setting "RANDOM_SLICES" "200"
$AggregateConfidence = Get-Setting "AGGREGATE_CONFIDENCE" "0.95"
$FigureSeed = Get-Setting "FIGURE_SEED" "17"
$TopK = Get-Setting "TOP_K" "6"
$Resume = Get-Setting "RESUME" "1"

# Preparation batch size.
$EvalBatchSize = Get-Setting "EVAL_BATCH_SIZE" "128"

if ($Smoke) {
    $OutDir = Get-Setting "SMOKE_OUT_DIR" "build/results/waterbirds_hidden_slice_v2_smoke"
    $DataDir = Get-Setting "SMOKE_DATA_DIR" (Join-Path $OutDir "prepared")
    $SeedsText = "17"
    $Folds = "2"
    $GateRestarts = "1"
    $MassesText = "0.10"
    $ShrinkagesText = "0.10"
    $HeadSteps = Get-Setting "SMOKE_HEAD_STEPS" "20"
    $GateSteps = Get-Setting "SMOKE_GATE_STEPS" "20"
    $BootstrapReplicates = Get-Setting "SMOKE_BOOTSTRAP_REPLICATES" "100"
    $RandomSlices = Get-Setting "SMOKE_RANDOM_SLICES" "5"
    Write-Warning "SMOKE MODE checks plumbing only; its scientific outputs are meaningless."
}

$Seeds = Split-Setting $SeedsText
$Masses = Split-Setting $MassesText
$Shrinkages = Split-Setting $ShrinkagesText
$LossLambdas = Split-Setting $LossLambdasText
$FisherLambdas = Split-Setting $FisherLambdasText
$CombinedLossLambdas = Split-Setting $CombinedLossLambdasText
$CombinedFisherLambdas = Split-Setting $CombinedFisherLambdasText
$ResumeArg = if ($Resume -eq "1") { "--resume" } else { "--no-resume" }
$DeterministicArg = if ($Deterministic -eq "1") { "--deterministic" } else { "--no-deterministic" }

Write-Host "=============================================================="
Write-Host " Waterbirds metadata-blind hidden-slice pipeline (corrected v2)"
Write-Host " python        : $PythonExe ($PythonVersion)"
Write-Host " out-dir       : $OutDir"
Write-Host " device        : $Device"
Write-Host " seeds/folds   : $SeedsText / $Folds"
Write-Host " mass          : $MainMass (sweep: $MassesText)"
Write-Host " shrinkage     : $MainShrinkage (sweep: $ShrinkagesText)"
Write-Host " nuisances     : CLIP text (primary), random axis, oracle probe"
Write-Host " gate restarts : $GateRestarts"
Write-Host " deterministic : $Deterministic"
Write-Host "=============================================================="

$stageArgs = @(
    "--root-dir", $RootDir,
    "--out-dir", $DataDir,
    "--feature-cache", $FeatureCache,
    "--nuisance-cache", $NuisanceCache,
    "--clip-model", $ClipModel,
    "--seed", $PrepareSeed,
    "--batch-size", $EvalBatchSize,
    "--device", $Device,
    "--num-workers", $NumWorkers
)
if (-not [string]::IsNullOrWhiteSpace($ClipRevision)) { $stageArgs += @("--clip-revision", $ClipRevision) }
if ($Download -eq "1") { $stageArgs += "--download" }
if ($AllowModelDownload -eq "1") { $stageArgs += "--allow-model-download" }
if ($Amp -eq "1") { $stageArgs += "--amp" } else { $stageArgs += "--no-amp" }
if ($PinMemory -eq "1") { $stageArgs += "--pin-memory" } else { $stageArgs += "--no-pin-memory" }
$stageArgs += $DeterministicArg

$preflightArgs = @(
    "--root-dir", $RootDir,
    "--out-dir", $DataDir,
    "--feature-cache", $FeatureCache,
    "--nuisance-cache", $NuisanceCache,
    "--clip-model", $ClipModel,
    "--device", $Device
)
if (-not [string]::IsNullOrWhiteSpace($ClipRevision)) { $preflightArgs += @("--clip-revision", $ClipRevision) }
if ($AllowModelDownload -eq "1") { $preflightArgs += "--allow-model-download" }
Invoke-Stage "preflight.py" $preflightArgs

Invoke-Stage "prepare_data.py" $stageArgs

$stageArgs = @(
    "--data-dir", $DataDir,
    "--out-dir", $OutDir,
    "--seeds"
) + $Seeds + @(
    "--device", $Device,
    "--calibration-fraction", $CalibrationFraction,
    "--head-steps", $HeadSteps,
    "--head-lr", $HeadLr,
    "--head-weight-decay", $HeadWeightDecay,
    "--folds", $Folds,
    "--gate-steps", $GateSteps,
    "--gate-lr", $GateLr,
    "--gate-restarts", $GateRestarts,
    "--slice-mass", $MainMass,
    "--mass-sensitivity"
) + $Masses + @(
    "--shrinkage", $MainShrinkage,
    "--shrinkage-sensitivity"
) + $Shrinkages + @(
    "--ess-fraction", $EssFraction,
    "--cap-multiplier", $CapMultiplier,
    "--ridge", $Ridge,
    "--epsilon", $Epsilon,
    $DeterministicArg,
    $ResumeArg
)
Invoke-Stage "discover.py" $stageArgs

$stageArgs = @(
    "--data-dir", $DataDir,
    "--out-dir", $OutDir,
    "--seeds"
) + $Seeds + @(
    "--device", $Device,
    "--loss-lambdas"
) + $LossLambdas + @(
    "--fisher-lambdas"
) + $FisherLambdas + @(
    "--combined-loss-lambdas"
) + $CombinedLossLambdas + @(
    "--combined-fisher-lambdas"
) + $CombinedFisherLambdas + @(
    "--accuracy-budget", $AccuracyBudget,
    "--balanced-accuracy-budget", $BalancedAccuracyBudget,
    "--per-class-accuracy-budget", $PerClassAccuracyBudget,
    "--head-steps", $HeadSteps,
    "--head-lr", $HeadLr,
    "--head-weight-decay", $HeadWeightDecay,
    "--early-stopping-patience", $EarlyStoppingPatience,
    "--early-stopping-eval-every", $EarlyStoppingEvalEvery,
    "--early-stopping-min-delta", $EarlyStoppingMinDelta,
    $DeterministicArg,
    $ResumeArg
)
Invoke-Stage "repair.py" $stageArgs

$stageArgs = @(
    "--data-dir", $DataDir,
    "--out-dir", $OutDir,
    "--seeds"
) + $Seeds + @(
    "--bootstrap-replicates", $BootstrapReplicates,
    "--bootstrap-confidence", $BootstrapConfidence,
    "--random-slices", $RandomSlices,
    $ResumeArg
)
Invoke-Stage "certify.py" $stageArgs

# This is the first stage allowed to reveal metadata; oracle controls are produced here.
$stageArgs = @(
    "--data-dir", $DataDir,
    "--out-dir", $OutDir,
    "--seeds"
) + $Seeds + @(
    "--device", $Device,
    "--head-steps", $HeadSteps,
    "--head-lr", $HeadLr,
    "--head-weight-decay", $HeadWeightDecay,
    $DeterministicArg,
    $ResumeArg
)
Invoke-Stage "posthoc.py" $stageArgs

$stageArgs = @(
    "--out-dir", $OutDir,
    "--seeds"
) + $Seeds + @(
    "--confidence", $AggregateConfidence
)
Invoke-Stage "aggregate.py" $stageArgs

$stageArgs = @(
    "--out-dir", $OutDir,
    "--report", (Join-Path $OutDir "waterbirds_hidden_slice_report.md"),
    "--latex", (Join-Path $OutDir "waterbirds_hidden_slice_table.tex"),
    "--rebuttal", (Join-Path $OutDir "rebuttal_w3.txt")
)
Invoke-Stage "make_report.py" $stageArgs

if (-not $Smoke) {
    $stageArgs = @(
        "--data-dir", $DataDir,
        "--out-dir", $OutDir,
        "--image-root", $ImageRoot,
        "--seed", $FigureSeed,
        "--top-k", $TopK
    )
    Invoke-Stage "make_figure.py" $stageArgs
} else {
    Write-Host ""
    Write-Host "### make_figure.py (skipped: incomplete smoke runs cannot emit paper figures)"
}

Write-Host ""
if (-not $Smoke) {
    Write-Host "Done. Review $OutDir/aggregate.json, $OutDir/main_table.csv,"
    Write-Host "$OutDir/waterbirds_hidden_slice_report.md, $OutDir/rebuttal_w3.txt, and the figure files."
} else {
    Write-Host "Smoke check done. Outputs under $OutDir are diagnostic-only; no paper figure was created."
}
