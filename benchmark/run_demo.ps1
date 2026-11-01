param(
    [string]$PythonExe = (Join-Path $env:USERPROFILE 'miniconda3\envs\project-beta\python.exe'),
    [string]$Retrievers = 'bm25,project'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw '请用 -PythonExe 指定已安装 memory-rag-mcp 依赖的 Python。只跑 BM25 时也可以指定普通 Python 并传 -Retrievers bm25。'
}
& $PythonExe (Join-Path $PSScriptRoot 'run_benchmark.py') demo --retrievers $Retrievers
exit $LASTEXITCODE
