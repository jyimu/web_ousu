import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
import sys
import os

# 添加路径以确保能导入 app.py
sys.path.append(os.path.dirname(__file__))

# ✅ 关键修复：先导入模块，不直接导入变量
import app as app

# 加载模型与查找表（这会初始化 PIGMENT_NAMES_EN）
print("正在加载模型...")
app.load_models()
app.load_lookup_table()

# ✅ 现在从 app 模块中获取已初始化的变量
PIGMENT_NAMES_EN = app.PIGMENT_NAMES_EN
print(f"PIGMENT_NAMES_EN = {PIGMENT_NAMES_EN} ({len(PIGMENT_NAMES_EN)} 种)")

# 加载验证数据
print("\n正在加载验证数据...")
df = pd.read_csv('scaler/data/key.csv')
print(f"DataFrame 形状: {df.shape}")

# 确保 PIGMENT_NAMES_EN 不为空
if not PIGMENT_NAMES_EN:
    raise ValueError("PIGMENT_NAMES_EN 为空，请检查模型加载过程")

# 提取真实配比和RGB值
true_recipes = df[PIGMENT_NAMES_EN].values
rgb_values = df[['R', 'G', 'B']].values

print(f"真实配比形状: {true_recipes.shape}")
print(f"RGB值形状: {rgb_values.shape}")

# 存储所有样本的R²
r2_scores = []

print("\n开始计算R²...")
for i, (rgb, true_recipe) in enumerate(zip(rgb_values, true_recipes)):
    print(f"\n--- 样本 {i} ---")
    print(f"RGB: {rgb} -> 真实配比: {true_recipe}")
    
    # 预测配比
    hsl = app.rgb_to_hsl(np.array(rgb, dtype=float))
    predicted_recipe = app.predict_with_inverse_model(rgb, hsl, enable_fallback=True)
    
    print(f"预测配比: {predicted_recipe}")
    
    # 计算R²
    try:
        r2 = r2_score(true_recipe, predicted_recipe)
        r2_scores.append(r2)
        print(f"R² = {r2:.4f}")
    except Exception as e:
        print(f"❌ 计算R²失败: {e}")
        continue

# 输出统计结果
print("\n" + "="*50)
print("=== R² 验证结果 ===")
if r2_scores:
    print(f"有效样本数: {len(r2_scores)}")
    print(f"平均 R²: {np.mean(r2_scores):.4f}")
    print(f"标准差: {np.std(r2_scores):.4f}")
    print(f"最小 R²: {np.min(r2_scores):.4f}")
    print(f"最大 R²: {np.max(r2_scores):.4f}")
else:
    print("没有成功计算任何R²分数")
