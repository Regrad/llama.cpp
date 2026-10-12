@echo off
setlocal
if not defined QWEN_SERVER goto :missing
if not defined QWEN_MODEL goto :missing
if not defined QWEN_MTP goto :missing
if not defined QWEN_MMPROJ goto :missing
if not defined QWEN_API_KEY_FILE goto :missing
if not exist "%QWEN_SERVER%" goto :missing
if not exist "%QWEN_MODEL%" goto :missing
if not exist "%QWEN_MTP%" goto :missing
if not exist "%QWEN_MMPROJ%" goto :missing
if not exist "%QWEN_API_KEY_FILE%" goto :missing
if not defined QWEN_MTP_P_MIN set "QWEN_MTP_P_MIN=0.0"
title Qwen Flash Next - Vulkan MTP3
"%QWEN_SERVER%" ^
  --model "%QWEN_MODEL%" ^
  --mmproj "%QWEN_MMPROJ%" ^
  --alias "qwen-flash-next-vulkan-mtp3" ^
  --host 0.0.0.0 --port 1234 --jinja ^
  --ctx-size 262144 --parallel 1 ^
  --n-gpu-layers 999 --device Vulkan0 --split-mode none ^
  --fit off --load-mode none --lazy-mode on ^
  --spec-type draft-mtp --spec-draft-model "%QWEN_MTP%" ^
  --spec-draft-device Vulkan0 --spec-draft-ngl 999 ^
  --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min %QWEN_MTP_P_MIN% ^
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 ^
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 ^
  --cache-ram 0 --no-cache-idle-slots --cache-prompt ^
  --batch-size 2048 --ubatch-size 1024 --threads 8 --threads-batch 8 ^
  --ctx-checkpoints 36 --checkpoint-min-step 8192 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --presence-penalty 0.0 --repeat-penalty 1.0 ^
  --reasoning on --reasoning-format auto --reasoning-effort medium ^
  --reasoning-budget -1 --reasoning-preserve ^
  --image-min-tokens 1024 --metrics ^
  --api-key-file "%QWEN_API_KEY_FILE%"
exit /b %ERRORLEVEL%
:missing
echo Set QWEN_SERVER, QWEN_MODEL, QWEN_MTP, QWEN_MMPROJ and QWEN_API_KEY_FILE to existing local files.
exit /b 1
