# Prevent Windows sleep/hibernate while this script runs (no admin required).
# Uses SetThreadExecutionState — tells the OS the system is in use.
# Refresh every 30s so a stray reset cannot leave the machine unprotected overnight.
#
# Start from WSL: bash scripts/keep_awake.sh start
# Stop:           bash scripts/keep_awake.sh stop

param(
    [int]$IntervalSec = 30,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class WinPower {
    public const uint ES_CONTINUOUS       = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED  = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
    public const uint ES_AWAYMODE_REQUIRED = 0x00000040;
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
$KeepAwakeFlags = [uint32]2147483713

function Write-Log([string]$Msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Msg
    Write-Output $line
    if ($LogFile -ne "") {
        Add-Content -Path $LogFile -Value $line -Encoding UTF8
    }
}

Write-Log "keep-awake started (refresh every ${IntervalSec}s, pid=$PID)"

try {
    while ($true) {
        $r = [WinPower]::SetThreadExecutionState($KeepAwakeFlags)
        if ($r -eq 0) {
            Write-Log "WARN: SetThreadExecutionState returned 0"
        }
        Start-Sleep -Seconds $IntervalSec
    }
}
finally {
    # Release continuous flag on exit so normal power policy resumes.
    [void][WinPower]::SetThreadExecutionState([WinPower]::ES_CONTINUOUS)
    Write-Log "keep-awake stopped, power lock released"
}
