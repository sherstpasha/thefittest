@ECHO OFF
setlocal
pushd %~dp0
if "%DOC_VERSION%"=="" set DOC_VERSION=0.3.0
if "%DOC_VERSIONS%"=="" set DOC_VERSIONS=%DOC_VERSION%,0.2.7
if "%PYTHON%"=="" set PYTHON=python
if "%1"=="" goto help
if "%1"=="html" goto html
%PYTHON% -m sphinx -M %1 source _build %SPHINXOPTS%
goto finish
:help
%PYTHON% -m sphinx -M help source _build
goto finish
:html
%PYTHON% build.py %SPHINXOPTS%
:finish
set result=%ERRORLEVEL%
popd
exit /b %result%
