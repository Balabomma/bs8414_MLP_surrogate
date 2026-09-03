set -e
for d in models_mlp_70_15_15_grouped models_mlp_70_15_15_grouped_seed1337 models_mlp_70_15_15_grouped_seed2024; do
  for m in 0 1 2 3 4; do
    echo "=== $d member $m ==="
    MLP_SPLIT=70_15_15 PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe shap_attribution.py \
      --model-dir "$d" --member "$m" --only "kernel SHAP" 2>&1 | sed -n '/ADDITIVE-FEATURE/,$p'
  done
done
