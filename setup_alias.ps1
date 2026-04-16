# Script para configurar alias global del REPL de Redes Neuronales
# Ejecuta este script como administrador para configurar el alias

$projectPath = "c:\Users\soled\PycharmProjects\PythonProject1"

# Crear función para el alias
$function = @'
function neural-repl {
    python "$projectPath\neural_repl.py"
}
'@

# Agregar al perfil de PowerShell
$profilePath = $PROFILE
if (Test-Path $profilePath) {
    $profileContent = Get-Content $profilePath -Raw
    if ($profileContent -notmatch "function neural-repl") {
        Add-Content -Path $profilePath -Value $function
        Write-Host "Alias 'neural-repl' agregado a tu perfil de PowerShell." -ForegroundColor Green
        Write-Host "Reinicia PowerShell para usar el nuevo alias." -ForegroundColor Yellow
    } else {
        Write-Host "El alias 'neural-repl' ya existe en tu perfil." -ForegroundColor Yellow
    }
} else {
    # Crear perfil si no existe
    New-Item -Path $profilePath -ItemType File -Force
    Add-Content -Path $profilePath -Value $function
    Write-Host "Perfil de PowerShell creado y alias 'neural-repl' agregado." -ForegroundColor Green
    Write-Host "Reinicia PowerShell para usar el nuevo alias." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Después de reiniciar PowerShell, podrás usar 'neural-repl' desde cualquier lugar." -ForegroundColor Cyan
