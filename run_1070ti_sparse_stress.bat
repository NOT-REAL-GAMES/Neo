@echo off
setlocal
cd /d "%~dp0"

cargo run --no-default-features --features cuda-12060 -p neo-quad-stress-3d --release -- ^
  --width 1920 ^
  --height 1080 ^
  --draw-backend cuda-tiled ^
  --instance-stress-variant macrocell ^
  --instance-materials sparse-texture ^
  --sparse-feedback off ^
  --gpu-preset auto ^
  --kernel-target-fps 0 ^
  --present-target-fps 0 ^
  --no-hot-reload

pause
