$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'FastSpeech 2 - Live Progress'
$taskWork = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\work'))
$taskLog = Get-ChildItem -LiteralPath $taskWork -Filter 'english_run*.stdout.log' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $taskLog) {
    throw 'No English run log was found.'
}
Write-Host 'FastSpeech 2 - English pipeline' -ForegroundColor Cyan
Write-Host 'New progress lines appear automatically.'
Write-Host 'Closing this viewer leaves the training process running.'
Write-Host ''
Get-Content -LiteralPath $taskLog.FullName -Tail 20 -Wait
