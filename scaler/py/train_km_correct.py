# scaler/py/train_km_correct.py - 修复版
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import sys
import os

# 添加utils模块路径（确保可导入）
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.color_utils import rgb_to_ks, load_and_normalize_data

CONFIG = {
    'primary_data': 'scaler/data/key.csv',
    'secondary_data': ['scaler/data/data.csv'],
    'model_save_path': './models/km_correct.pkl',
    'ridge_alpha': 0.0,
    'val_split': 0.2,
}

def train_km() -> tuple[np.ndarray, list, dict]:
    """
    训练KM模型（带验证集评估）
    
    Returns:
        A_matrix: (31, 7) 的转换矩阵
        pigment_names: 颜料名称列表
        metrics: 包含训练/验证R²的字典
    """
    # 加载数据（使用公共函数）
    df = load_and_normalize_data([CONFIG['primary_data']] + CONFIG['secondary_data'])
    
    # 数据准备
    W = df.iloc[:, 3:].values  # (n_samples, 7)
    ks = rgb_to_ks(df[['R','G','B']].values)  # (n_samples, 31)
    
    print(f"配方矩阵 W shape: {W.shape}")
    print(f"K/S矩阵 ks shape: {ks.shape}")
    
    # 划分训练/验证集
    W_train, W_val, ks_train, ks_val = train_test_split(
        W, ks, test_size=CONFIG['val_split'], random_state=42
    )
    
    # Ridge整体求解
    print("\n[训练] 使用Ridge整体求解31个波段...")
    ridge = Ridge(alpha=CONFIG['ridge_alpha'], fit_intercept=False, random_state=42)
    ridge.fit(W_train, ks_train)
    
    # 修复：ridge.coef_ shape为(31, 7)，直接使用
    print(f"ridge.coef_ shape: {ridge.coef_.shape}")
    A_matrix = ridge.coef_  # (31, 7)
    
    # 验证
    ks_pred_train = W_train @ A_matrix.T
    r2_train = r2_score(ks_train.ravel(), ks_pred_train.ravel())
    
    ks_pred_val = W_val @ A_matrix.T
    r2_val = r2_score(ks_val.ravel(), ks_pred_val.ravel())
    
    print(f"训练集 K/S R² = {r2_train:.5f}")
    print(f"验证集 K/S R² = {r2_val:.5f}")
    
    return A_matrix, df.columns[3:].tolist(), {'r2_train': r2_train, 'r2_val': r2_val}

if __name__ == '__main__':
    try:
        A_matrix, pigment_names, metrics = train_km()
        
        model_data = {
            'A_matrix': A_matrix,
            'pigment_names': pigment_names,
            'training_metrics': metrics,
            'config': CONFIG,
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
        joblib.dump(model_data, CONFIG['model_save_path'])
        print(f"\n✅ 模型已保存: {CONFIG['model_save_path']}")
        print(f"   训练集R²: {metrics['r2_train']:.5f}")
        print(f"   验证集R²: {metrics['r2_val']:.5f}")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        sys.exit(1)
