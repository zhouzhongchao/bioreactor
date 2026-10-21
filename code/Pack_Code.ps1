$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$uploadRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$uploadDirectory = Join-Path (Split-Path -Parent $uploadRoot) 'data/output/distribution'
New-Item -ItemType Directory -Path $uploadDirectory -Force | Out-Null
$uploadZip = Join-Path $uploadDirectory ('bioreactor_code_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.zip')
$uploadExcludedDirectories = @('.git','.python310','.venv','venv','runtime','node_modules','__pycache__','.pytest_cache','.pio','output','nano25_recordings','logs','model_backups')
function Get-UploadFiles([string]$directory) {
    foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
        if ($item.PSIsContainer) {
            if ($uploadExcludedDirectories -notcontains $item.Name) { Get-UploadFiles $item.FullName }
        } elseif ($item.Name -notmatch '(?i)(\.py[co]$|\.bak$|\.tmp$|\.log$|\.zip$|^\.env($|\.)|^auth\.json$|^credentials\.json$|\.key$|^codex_candidate_.+\.json$|^codex_compensator_candidate_.+\.json$|_(previous|shadow|rejected)\.pt$|_state_replay\.npz$)' -and $item.Name -ne '工作簿1.xlsx') {
            $item
        }
    }
}
$uploadFiles = @(Get-UploadFiles $uploadRoot)
if ($uploadFiles.Count -eq 0) { throw 'No source files found.' }
$uploadArchive = [IO.Compression.ZipFile]::Open($uploadZip, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($item in $uploadFiles) {
        $relative = $item.FullName.Substring($uploadRoot.Length + 1).Replace('\','/')
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($uploadArchive, $item.FullName, 'code/' + $relative, [IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally { $uploadArchive.Dispose() }
Write-Host ('Saved {0} files: {1}' -f $uploadFiles.Count, $uploadZip)
