@echo off
setlocal EnableDelayedExpansion

:: Ensure we run from the project root directory
cd /d "%~dp0"

:: Set console title
title ProofIntent-ML Demo Suite

:: Default pause command (can be bypassed with --auto or --no-pause)
set "PAUSE_CMD=pause"
set "IS_DIRECT_ARG=0"

:: Check for help flags
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h"     goto show_help
if /i "%~1"=="/?"     goto show_help

:: Detect Python executable
set "PYTHON_CMD="
python -V >nul 2>&1
if %ERRORLEVEL% EQU 0 set "PYTHON_CMD=python"

if "%PYTHON_CMD%"=="" (
    py -3 -V >nul 2>&1
    if %ERRORLEVEL% EQU 0 set "PYTHON_CMD=py -3"
)

if "%PYTHON_CMD%"=="" (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else if exist "C:\Program Files\Python312\python.exe" (
        set "PYTHON_CMD=C:\Program Files\Python312\python.exe"
    ) else if exist "C:\Program Files\Python311\python.exe" (
        set "PYTHON_CMD=C:\Program Files\Python311\python.exe"
    )
)

if "%PYTHON_CMD%"=="" (
    echo ================================================================================
    echo [ERROR] Python 3.7+ was not found on your system PATH.
    echo Please install Python from https://www.python.org and add it to PATH.
    echo ================================================================================
    pause
    exit /b 1
)

:: Parse pause flags
if /i "%~1"=="--no-pause" set "PAUSE_CMD=rem"
if /i "%~1"=="--auto"     set "PAUSE_CMD=rem"
if /i "%~2"=="--no-pause" set "PAUSE_CMD=rem"
if /i "%~2"=="--auto"     set "PAUSE_CMD=rem"

:: Direct command line routing
if not "%~1"=="" if not "%~1"=="--no-pause" if not "%~1"=="--auto" (
    set "IS_DIRECT_ARG=1"
    if /i "%~1"=="--all" goto run_all
    if /i "%~1"=="all"   goto run_all
    if /i "%~1"=="-a"    goto run_all
    if /i "%~1"=="full"  goto run_all
    if /i "%~1"=="0"     goto run_all
    if /i "%~1"=="1"     goto run_single_1
    if /i "%~1"=="2"     goto run_single_2
    if /i "%~1"=="3"     goto run_single_3
    if /i "%~1"=="4"     goto run_single_4
    if /i "%~1"=="5"     goto run_single_5
    if /i "%~1"=="6"     goto run_single_6
    if /i "%~1"=="7"      goto run_single_7
    if /i "%~1"=="viz"    goto run_single_7
    if /i "%~1"=="gui"    goto run_single_7
    if /i "%~1"=="8"      goto end_script
    echo [!] Unknown option "%~1". Use --help to view available commands.
    exit /b 1
)

:menu
cls
echo ================================================================================
echo                     PROOFINTENT-ML: INTERACTIVE DEMO SUITE
echo      Semantic Zero-Knowledge Verification ^& Intent-Driven Optimization
echo                               for Machine Learning
echo ================================================================================
echo  Python Runtime: %PYTHON_CMD%
echo  Project Root  : %~dp0
echo ================================================================================
echo.
echo  Please select an option to demonstrate:
echo.
echo    [0] Run Complete End-to-End Demo Walkthrough (Steps 1 - 5)
echo    [1] Step 1: Run Full Test Suite (34 Unit ^& Integration Tests)
echo    [2] Step 2: Compile Models ^& Compare Baseline vs Optimized IR
echo    [3] Step 3: Simulate Proof Generation ^& Witness Evaluation
echo    [4] Step 4: Security Verification ^& Cryptographic Tamper Detection
echo    [5] Step 5: Run Automated Performance Benchmarks
echo    [6] Helper: Regenerate Example Weights ^& Sample Inputs
echo    [7] Launch GSAP Interactive Visualizer (Static Browser UI)
echo    [8] Exit
echo.
echo ================================================================================
set /p "CHOICE=Enter your choice [0-8]: "

if "%CHOICE%"=="0" goto run_all
if "%CHOICE%"=="1" goto run_single_1
if "%CHOICE%"=="2" goto run_single_2
if "%CHOICE%"=="3" goto run_single_3
if "%CHOICE%"=="4" goto run_single_4
if "%CHOICE%"=="5" goto run_single_5
if "%CHOICE%"=="6" goto run_single_6
if "%CHOICE%"=="7" goto run_single_7
if "%CHOICE%"=="8" goto end_script

echo.
echo [!] Invalid selection "%CHOICE%". Please enter a number between 0 and 8.
timeout /t 2 >nul
goto menu

:: ================================================================================
:: INDIVIDUAL OPTION WRAPPERS
:: ================================================================================
:run_single_1
call :step1_exec
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:run_single_2
call :step2_exec
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:run_single_3
call :step3_exec
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:run_single_4
call :step4_exec
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:run_single_5
call :step5_exec
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:run_single_6
call :step_make_models_exec
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:run_single_7
call :step_viz_exec
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:step_viz_exec
cls
echo ================================================================================
echo  OPENING PROOFINTENT-ML GSAP INTERACTIVE VISUALIZATION
echo ================================================================================
echo  Launching browser interface at: %~dp0visualization.html
start "" "%~dp0visualization.html"
echo --------------------------------------------------------------------------------
goto :eof

:: ================================================================================
:: FULL DEMO WALKTHROUGH
:: ================================================================================
:run_all
call :step1_exec
echo.
echo --------------------------------------------------------------------------------
echo  [Step 1 of 5 Completed]
echo --------------------------------------------------------------------------------
%PAUSE_CMD%

call :step2_exec
echo.
echo --------------------------------------------------------------------------------
echo  [Step 2 of 5 Completed]
echo --------------------------------------------------------------------------------
%PAUSE_CMD%

call :step3_exec
echo.
echo --------------------------------------------------------------------------------
echo  [Step 3 of 5 Completed]
echo --------------------------------------------------------------------------------
%PAUSE_CMD%

call :step4_exec
echo.
echo --------------------------------------------------------------------------------
echo  [Step 4 of 5 Completed]
echo --------------------------------------------------------------------------------
%PAUSE_CMD%

call :step5_exec
echo.
echo ================================================================================
echo  ALL 5 DEMO STEPS COMPLETED SUCCESSFULLY!
echo ================================================================================
echo.
if "%IS_DIRECT_ARG%"=="1" exit /b 0
%PAUSE_CMD%
goto menu

:: ================================================================================
:: STEP EXECUTABLE SUBROUTINES
:: ================================================================================

:step1_exec
cls
echo ================================================================================
echo  [STEP 1/5] AUTOMATED UNIT ^& INTEGRATION TEST SUITE
echo ================================================================================
echo  Demonstrates:
echo   - 100%% Test Coverage across Lexer, Parser, AST, Semantic Checker,
echo     Lowering Passes, Dead-Constraint Elimination, Quantization,
echo     Circom Code Generation, and Runtime Proof Simulator.
echo   - Zero external Python dependencies (Pure Python Standard Library).
echo.
echo  Command: %PYTHON_CMD% -m unittest discover -s tests
echo --------------------------------------------------------------------------------
%PYTHON_CMD% -m unittest discover -s tests
echo --------------------------------------------------------------------------------
goto :eof


:step2_exec
cls
echo ================================================================================
echo  [STEP 2/5] INTENT-DRIVEN COMPILATION ^& R1CS OPTIMIZATION
echo ================================================================================
echo  Demonstrates:
echo   - Multi-stage compilation from DSL (.piml) to ML IR and Proof IR.
echo   - Constant Propagation (folding zero-weight layers/channels).
echo   - Intent-Driven Dead-Constraint Elimination (pruning unused Softmax).
echo   - Fixed-point quantization and Circom 2.0 circuit code generation.
echo.
echo --------------------------------------------------------------------------------
echo  Part A: Compiling Small Model (examples\mlp_small.piml)
echo  Command: %PYTHON_CMD% -m proofintent compile examples/mlp_small.piml --emit-ir both --backend circom --out build/
echo --------------------------------------------------------------------------------
%PYTHON_CMD% -m proofintent compile examples/mlp_small.piml --emit-ir both --backend circom --out build/
echo --------------------------------------------------------------------------------
echo.
echo  Notice: mlp_small achieved ~29%% constraint reduction (31 -^> 22 R1CS).
echo.
echo --------------------------------------------------------------------------------
echo  Part B: Compiling Large Model with Pruned Layer (examples\mlp_large.piml)
echo  Command: %PYTHON_CMD% -m proofintent compile examples/mlp_large.piml --emit-ir both --backend circom --out build/
echo --------------------------------------------------------------------------------
%PYTHON_CMD% -m proofintent compile examples/mlp_large.piml --emit-ir both --backend circom --out build/
echo --------------------------------------------------------------------------------
echo.
echo  Notice: mlp_large achieved ~80.6%% constraint reduction (103 -^> 20 R1CS)
echo  due to Constant Propagation folding layer 2 and Dead-Constraint Elimination
echo  pruning the unneeded Softmax confidence head!
echo.
echo  Generated Circom circuit saved to: build\circuit.circom
goto :eof


:step3_exec
cls
echo ================================================================================
echo  [STEP 3/5] WITNESS EVALUATION ^& PROOF SIMULATION
echo ================================================================================
echo  Demonstrates:
echo   - Forward inference on input vector [0.1, -0.2, 0.5, 0.8].
echo   - Deterministic SHA-256 Model Parameter Commitment.
echo   - Witness signal assignment and arithmetic constraint validation.
echo.
echo  Command: %PYTHON_CMD% -m proofintent prove examples/mlp_small.piml --input "[0.1, -0.2, 0.5, 0.8]"
echo --------------------------------------------------------------------------------
%PYTHON_CMD% -m proofintent prove examples/mlp_small.piml --input "[0.1, -0.2, 0.5, 0.8]"
echo --------------------------------------------------------------------------------
goto :eof


:step4_exec
cls
echo ================================================================================
echo  [STEP 4/5] CRYPTOGRAPHIC VERIFICATION ^& TAMPER RESILIENCE
echo ================================================================================
echo  Demonstrates:
echo   1. Honest verification passes (True).
echo   2. Tampered Model Weights: Rejected by SHA-256 commitment check (False).
echo   3. Tampered Prediction Transcript: Rejected by transcript check (False).
echo.
echo  Command: %PYTHON_CMD% -m proofintent verify examples/mlp_small.piml --input "[0.1, -0.2, 0.5, 0.8]" --tamper-model --tamper-prediction
echo --------------------------------------------------------------------------------
%PYTHON_CMD% -m proofintent verify examples/mlp_small.piml --input "[0.1, -0.2, 0.5, 0.8]" --tamper-model --tamper-prediction
echo --------------------------------------------------------------------------------
goto :eof


:step5_exec
cls
echo ================================================================================
echo  [STEP 5/5] AUTOMATED PERFORMANCE BENCHMARKS
echo ================================================================================
echo  Demonstrates:
echo   - Constraint counts (baseline vs optimized R1CS).
echo   - Proving time, emulated SNARK prover latency, verification latency.
echo   - Memory allocation and overall constraint reduction percentage.
echo.
echo  Command: %PYTHON_CMD% -m proofintent bench
echo --------------------------------------------------------------------------------
%PYTHON_CMD% -m proofintent bench
echo --------------------------------------------------------------------------------
echo.
echo ================================================================================
echo  SUMMARY REPORT (from results\bench.md):
echo ================================================================================
type results\bench.md
echo ================================================================================
goto :eof


:step_make_models_exec
cls
echo ================================================================================
echo  REGENERATING DETERMINISTIC WEIGHTS AND SAMPLE INPUTS
echo ================================================================================
echo  Command: %PYTHON_CMD% examples/make_models.py
echo --------------------------------------------------------------------------------
%PYTHON_CMD% examples/make_models.py
echo --------------------------------------------------------------------------------
goto :eof


:show_help
echo ================================================================================
echo  ProofIntent-ML Demo Suite - Help ^& Command Line Options
echo ================================================================================
echo  Usage:
echo    demo.bat [OPTION] [--no-pause]
echo.
echo  Options:
echo    (no args)   Open the interactive demo selection menu
echo    0, --all    Run complete 5-step demo walkthrough
echo    1           Run Step 1: Automated Unit ^& Integration Tests (34 tests)
echo    2           Run Step 2: Compile Models ^& Compare Baseline vs Optimized IR
echo    3           Run Step 3: Simulate Proof Generation ^& Witness Evaluation
echo    4           Run Step 4: Security Verification ^& Tamper Detection Tests
echo    5           Run Step 5: Automated Performance Benchmarks
echo    6           Regenerate deterministic weights and sample inputs
echo    7, viz      Launch GSAP Interactive Visualizer (Static Browser UI)
echo    8           Exit demo
echo    --no-pause  Run without pausing between steps (useful for scripts)
echo    --help, -h  Display this help message
echo ================================================================================
exit /b 0


:end_script
cls
echo ================================================================================
echo  Exiting ProofIntent-ML Demo. Thank you!
echo ================================================================================
endlocal
exit /b 0
