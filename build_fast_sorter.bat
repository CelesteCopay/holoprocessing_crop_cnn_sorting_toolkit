@echo off
setlocal
cd /d "%~dp0"

echo Building Fast Image Sorter executable...
echo.

python -m pip install -r requirements-fast_sorter.txt
if errorlevel 1 goto :fail

python -m PyInstaller --noconfirm --clean FastImageSorter.spec
if errorlevel 1 goto :fail

echo.
echo Done! Executable is at:
echo   dist\FastImageSorter.exe
echo.
echo Give your friend:
echo   - dist\FastImageSorter.exe
echo   - Their unsorted image folder
echo   - A destination folder with one subfolder per category
echo.
echo First run: Select Input Folder, then Select Destination Folder.
echo Progress is saved as fast_sorter_progress.json next to the .exe
goto :eof

:fail
echo Build failed.
exit /b 1
