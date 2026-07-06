# Lock Windows power settings for overnight training (run once before sleep).
# Most changes work without admin; unattended-sleep override needs elevation.
#
# From WSL (no admin):  bash scripts/keep_awake.sh lock
# From Windows (admin): Right-click PowerShell -> Run as administrator, then:
#   powershell -ExecutionPolicy Bypass -File C:\...\scripts\win_lock_power.ps1

$ErrorActionPreference = "Continue"

function Invoke-PowerCfg([string[]]$PowerCfgArgs) {
    $exe = "$env:SystemRoot\System32\powercfg.exe"
    & $exe @PowerCfgArgs 2>&1 | Out-String | ForEach-Object { $_.Trim() }
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Output "=== win_lock_power $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') admin=$isAdmin ==="

$schemeOut = Invoke-PowerCfg @("/getactivescheme")
$GUID = [regex]::Match($schemeOut, "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}").Value
if (-not $GUID) {
    Write-Error "Could not parse active power scheme GUID from: $schemeOut"
    exit 1
}
Write-Output "Active scheme: $GUID"

$sets = @(
    @("/change", "standby-timeout-ac", "0"),
    @("/change", "standby-timeout-dc", "0"),
    @("/change", "hibernate-timeout-ac", "0"),
    @("/change", "hibernate-timeout-dc", "0"),
    @("/change", "monitor-timeout-ac", "0"),
    @("/change", "monitor-timeout-dc", "0"),
    @("/SETACVALUEINDEX", $GUID, "SUB_BUTTONS", "LIDACTION", "0"),
    @("/SETDCVALUEINDEX", $GUID, "SUB_BUTTONS", "LIDACTION", "0"),
    @("/SETACVALUEINDEX", $GUID, "SUB_SLEEP", "HYBRIDSLEEP", "0"),
    @("/SETDCVALUEINDEX", $GUID, "SUB_SLEEP", "HYBRIDSLEEP", "0"),
    @("/SETACVALUEINDEX", $GUID, "SUB_BUTTONS", "UIBUTTON_ACTION", "1"),
    @("/SETDCVALUEINDEX", $GUID, "SUB_BUTTONS", "UIBUTTON_ACTION", "1"),
    @("/SETACTIVE", $GUID)
)

foreach ($a in $sets) {
    $out = Invoke-PowerCfg $a
    if ($out) { Write-Output ("  powercfg {0} -> {1}" -f ($a -join " "), $out) }
    else      { Write-Output ("  powercfg {0} -> ok" -f ($a -join " ")) }
}

# Extend Windows Update active hours to cover overnight (8:00 -> 23:59).
try {
    $ux = "HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings"
    if ($isAdmin) {
        Set-ItemProperty -Path $ux -Name ActiveHoursStart -Value 8 -Type DWord
        Set-ItemProperty -Path $ux -Name ActiveHoursEnd   -Value 23 -Type DWord
        Write-Output "  Windows Update active hours set to 08:00-23:00"
    } else {
        $ah = Get-ItemProperty $ux -ErrorAction SilentlyContinue
        Write-Output ("  Windows Update active hours (read-only): {0}:00 - {1}:00 (change needs admin)" -f $ah.ActiveHoursStart, $ah.ActiveHoursEnd)
    }
} catch {
    Write-Output "  Windows Update active hours: skipped ($($_.Exception.Message))"
}

Write-Output ""
Write-Output "=== verification ==="
Invoke-PowerCfg @("/query", $GUID, "SUB_SLEEP", "STANDBYIDLE") | Select-String "当前|Current" | ForEach-Object { Write-Output $_.Line }
Invoke-PowerCfg @("/query", $GUID, "SUB_VIDEO", "VIDEOIDLE")   | Select-String "当前|Current" | ForEach-Object { Write-Output $_.Line }
Write-Output ""
Write-Output "Done. Display may still turn off (harmless); system sleep/hibernate should be blocked."
Write-Output "IMPORTANT: On Modern Standby laptops, keep the lid OPEN or plugged into AC."
Write-Output "Pair with: bash scripts/keep_awake.sh start"
