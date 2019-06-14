$pythonPath = $(Get-Item Env:\PYTHON)[0].Value.split(';')[1];
$pythonExe = "${pythonPath}python.exe";
& $pythonExe -m pip install --user -r "./requirement.txt";
& $pythonExe -m pip install ./pyoctree-0.2.10-cp36-cp36m-win_amd64.whl