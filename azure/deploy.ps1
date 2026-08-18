<#
.SYNOPSIS
  Deploy MediaGenStudio to the cheapest Azure App Service tier (Linux, Python,
  Free F1 plan by default). Deploys the GUI/API only - no GPU, so local Krea 2/
  ComfyUI generation won't run there; the MiniMax hosted API video backend can
  still work if you set MINIMAX_API_KEY.

.NOTES
  Requires: az CLI logged in (az login) as the account that should own the
  resources. Run from anywhere; paths are resolved relative to this script.
#>
param(
    [string]$ResourceGroup = "mediagenstudio-rg",
    [string]$Location = "eastus",
    [string]$AppServicePlan = "mediagenstudio-plan",
    [string]$WebAppName = "mediagenstudio-$((Get-Random -Minimum 1000 -Maximum 9999))",
    [string]$Sku = "F1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$stagingDir = Join-Path $env:TEMP "mediagenstudio-deploy"
$zipPath = Join-Path $env:TEMP "mediagenstudio-deploy.zip"

Write-Host "== Checking Azure login =="
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    throw "Not logged in. Run 'az login' first, then re-run this script."
}
Write-Host "Logged in as $($account.user.name), subscription $($account.name)"

Write-Host "== Staging deployment package =="
if (Test-Path $stagingDir) { Remove-Item -Recurse -Force $stagingDir }
New-Item -ItemType Directory -Path $stagingDir | Out-Null

foreach ($item in @("server.py", "config.py", "jobs.py", "engines", "static", "config.example.json")) {
    Copy-Item -Path (Join-Path $repoRoot $item) -Destination $stagingDir -Recurse
}
Copy-Item -Path (Join-Path $PSScriptRoot "requirements.txt") -Destination (Join-Path $stagingDir "requirements.txt")

if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $zipPath

Write-Host "== Creating resource group $ResourceGroup in $Location =="
az group create --name $ResourceGroup --location $Location --output none

Write-Host "== Creating App Service plan $AppServicePlan (SKU $Sku, Linux) =="
az appservice plan create --name $AppServicePlan --resource-group $ResourceGroup `
    --location $Location --is-linux --sku $Sku --output none

Write-Host "== Creating Web App $WebAppName =="
az webapp create --name $WebAppName --resource-group $ResourceGroup `
    --plan $AppServicePlan --runtime "PYTHON:3.11" --output none

Write-Host "== Configuring startup command and build settings =="
az webapp config set --name $WebAppName --resource-group $ResourceGroup `
    --startup-file "gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 600 server:app" `
    --output none
az webapp config appsettings set --name $WebAppName --resource-group $ResourceGroup `
    --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000 --output none

Write-Host "== Deploying code (zip deploy) =="
az webapp deploy --name $WebAppName --resource-group $ResourceGroup `
    --src-path $zipPath --type zip --output none

$url = "https://$WebAppName.azurewebsites.net"
Write-Host ""
Write-Host "== Done =="
Write-Host "App URL: $url"
Write-Host "Resource group: $ResourceGroup (delete with: az group delete --name $ResourceGroup --yes --no-wait)"
Write-Host ""
Write-Host "To enable the MiniMax cloud video fallback, set an app setting:"
Write-Host "  az webapp config appsettings set --name $WebAppName --resource-group $ResourceGroup --settings MINIMAX_API_KEY=<your-key>"
