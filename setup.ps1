param([string]$PythonExecutable = 'python')

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonCommand = Get-Command -Name $PythonExecutable -CommandType Application -ErrorAction Stop | Select-Object -First 1
& $pythonCommand.Source -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Failed to install dependencies.' }
& $pythonCommand.Source -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
