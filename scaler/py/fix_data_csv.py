# fix_data_csv.py - 安全归一化CSV文件
import pandas as pd
import numpy as np
import os
import shutil
from typing import Optional, Tuple

def normalize_csv(filename: str, backup: bool = True, recipe_start_col: int = 3) -> Optional[Tuple[float, float, float, float]]:
    """
    归一化CSV文件的配方列（带备份）
    
    Args:
        filename: CSV文件路径
        backup: 是否创建备份
        recipe_start_col: 配方开始的列索引（0-based）
    
    Returns:
        若成功返回 (原最小行和, 原最大行和, 新最小行和, 新最大行和)
        失败返回 None
    """
    try:
        # 创建备份
        if backup and os.path.exists(filename):
            backup_path = f"{filename}.backup"
            shutil.copy2(filename, backup_path)
            print(f"💾 备份已创建: {backup_path}")
        
        # 读取数据
        df = pd.read_csv(filename)
        
        # 检查列数
        if len(df.columns) <= recipe_start_col:
            raise ValueError(f"CSV至少需要{recipe_start_col+1}列")
        
        # 归一化配方列
        recipe_cols = df.iloc[:, recipe_start_col:]
        original_sum = recipe_cols.sum(axis=1).values
        
        # 避免除零
        row_sums = recipe_cols.sum(axis=1).replace(0, 1)
        df.iloc[:, recipe_start_col:] = recipe_cols.div(row_sums, axis=0).fillna(0)
        
        # 保存
        df.to_csv(filename, index=False)
        
        # 统计信息
        orig_min, orig_max = original_sum.min(), original_sum.max()
        new_min = df.iloc[:, recipe_start_col:].sum(axis=1).min()
        new_max = df.iloc[:, recipe_start_col:].sum(axis=1).max()
        
        print(f"✅ {filename} 已归一化")
        print(f"   原行和范围: [{orig_min:.2f}, {orig_max:.2f}]")
        print(f"   新行和范围: [{new_min:.2f}, {new_max:.2f}]")
        
        return orig_min, orig_max, new_min, new_max
        
    except Exception as e:
        print(f"❌ 处理 {filename} 失败: {e}")
        return None

if __name__ == '__main__':
    # 使用配置文件管理路径
    DATA_FILES = [
        'scaler/data/key.csv',
        'scaler/data/data.csv',
        'scaler/data/text.csv'
    ]
    
    for file_path in DATA_FILES:
        if os.path.exists(file_path):
            normalize_csv(file_path, backup=True)
        else:
            print(f"⚠️  文件不存在: {file_path}")
