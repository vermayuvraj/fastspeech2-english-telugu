$ErrorActionPreference = 'Stop'
$taskPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\work\runtime\Scripts\python.exe'))
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'The prepared local environment is missing. Use the README setup commands.'
}
Write-Host 'First accept access at https://huggingface.co/datasets/ai4bharat/Svarah'
Write-Host 'Paste your token only into the official Hugging Face CLI prompt below.'
& $taskPython (Join-Path $PSScriptRoot 'scripts\sign_in_huggingface.py')
