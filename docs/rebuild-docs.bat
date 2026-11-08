@ECHO OFF
setlocal
pushd %~dp0
call make.bat html
set result=%ERRORLEVEL%
popd
exit /b %result%
