@ECHO OFF
REM Script to rebuild documentation and copy to docs root for GitHub Pages

echo Cleaning old build...
call make.bat clean

echo Building documentation...
call make.bat html

echo Copying HTML files to docs root for GitHub Pages...
xcopy /E /Y /I build\html\* .\ >nul

echo Removing unnecessary build artifacts...
rmdir /S /Q build