# scaler/py/utils/color_utils.py - 颜色转换与数据加载工具（修复版）
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
from typing import List, Union

def rgb_to_reflectance(rgb: np.ndarray, wavelengths: np.ndarray = None, 
                       rgb_wavelengths: np.ndarray = None) -> np.ndarray:
    """
    RGB转反射率曲线（支持批量处理）
    
    Args:
        rgb: RGB值，形状 (n_samples, 3) 或 (3,)
        wavelengths: 目标波长数组，默认 400-700nm, step=10
        rgb_wavelengths: RGB对应的波长，默认 [630, 530, 450]
    
    Returns:
        反射率曲线，形状 (n_samples, len(wavelengths))
    """
    if wavelengths is None:
        wavelengths = np.arange(400, 710, 10)
    if rgb_wavelengths is None:
        rgb_wavelengths = np.array([630, 530, 450])
    
    # 归一化RGB到[0,1]
    rgb_norm = np.clip(rgb / 255.0, 0, 1)
    
    # 确保输入是二维的 (n_samples, 3)
    if rgb_norm.ndim == 1:
        rgb_norm = rgb_norm.reshape(1, -1)
    
    # 对每个样本单独插值（关键修复：fill_value传递标量）
    results = []
    for i in range(rgb_norm.shape[0]):
        # fill_value必须是标量或(2,)的标量元组，不能是数组
        f = interp1d(rgb_wavelengths, rgb_norm[i], kind='linear',
                     bounds_error=False, 
                     fill_value=(float(rgb_norm[i, 0]), float(rgb_norm[i, -1])))
        results.append(f(wavelengths))
    
    return np.array(results)

def reflectance_to_ks(rho: np.ndarray) -> np.ndarray:
    """反射率转K/S值（Kubelka-Munk公式）"""
    rho = np.clip(rho, 1e-6, 1-1e-6)
    return (1 - rho)**2 / (2 * rho)

def rgb_to_ks(rgb: np.ndarray, **kwargs) -> np.ndarray:
    """
    RGB直接转K/S值
    输入: (n_samples, 3) 或 (3,)
    输出: (n_samples, 31)
    """
    rho = rgb_to_reflectance(rgb, **kwargs)
    return reflectance_to_ks(rho)

def load_and_normalize_data(data_files: List[str], recipe_start_col: int = 3) -> pd.DataFrame:
    """
    加载并归一化多个CSV文件
    自动忽略不存在或格式错误的文件
    """
    dfs = []
    for path in data_files:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # 检查列数是否足够
                if len(df.columns) <= recipe_start_col:
                    print(f"⚠️  文件 {path} 列数不足（需要>{recipe_start_col}列），跳过")
                    continue
                # 归一化配方列
                df.iloc[:, recipe_start_col:] = df.iloc[:, recipe_start_col:].div(
                    df.iloc[:, recipe_start_col:].sum(axis=1), axis=0).fillna(0)
                dfs.append(df)
            except Exception as e:
                print(f"⚠️  读取文件 {path} 失败: {e}，跳过")
        else:
            print(f"⚠️  数据文件不存在: {path}")
    
    if not dfs:
        raise FileNotFoundError("未找到任何有效的数据文件")
    
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"✅ 加载数据成功: 总样本数 = {len(df_all)}")
    return df_all
