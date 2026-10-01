# Loads the x64 Visual Studio C++ environment into the current PowerShell window,
# so cmake and nvcc can find cl.exe. Run it with a leading dot from the repository root:
#
#   . .\scripts\dev_shell.ps1
#
# If PowerShell refuses to run scripts, allow local scripts once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with the C++ workload was not found" }

Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
