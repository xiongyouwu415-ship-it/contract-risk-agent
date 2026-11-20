@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ============================================================
rem  合同风险审查多 Agent 系统 —— 一键启动 Web 服务
rem  双击本文件即可；关闭此窗口服务就停止。
rem
rem  默认端口 8011。有 DEEPSEEK_API_KEY 就走真实模型，
rem  没有则自动降级为离线模拟（不产生费用，结论仅供演示）。
rem
rem  想强制模式或换端口：
rem    start.bat --real            强制真实模型
rem    start.bat --mock            强制离线模拟
rem    start.bat --port 8080       换端口
rem ============================================================

set PORT=8011

echo.
echo   合同风险审查多 Agent 系统（默认端口 %PORT%，可用 --port 覆盖）
echo   界面地址： http://127.0.0.1:%PORT%/
echo   接口文档： http://127.0.0.1:%PORT%/docs
echo   按 Ctrl+C 或关闭本窗口即可停止服务。
echo.

python server.py --port %PORT% %*
if errorlevel 1 (
    echo.
    echo [启动失败] 请检查是否已安装依赖：
    echo     pip install -r requirements.txt
    echo.
)
pause
