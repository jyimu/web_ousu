# evaluate_all.py - 统一评估框架（对比所有模型）
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os
from typing import Dict, Callable, Tuple

# 从公共模块导入
from utils.color_utils import rgb_to_ks, load_and_normalize_data
from train_transformer import ColorTransformer
from hybrid_model import HybridColorModel

class ModelEvaluator:
    """统一模型评估器"""
    
    def __init__(self, test_csv: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.df = self._load_data(test_csv)
        self.recipes = torch.tensor(self.df.iloc[:, 3:].values, dtype=torch.float32)
        self.rgbs_true = torch.tensor(self.df[['R','G','B']].values, dtype=torch.float32) / 255.0
        self.ks_true = rgb_to_ks(self.df[['R','G','B']].values)
        
    def _load_data(self, test_csv: str) -> pd.DataFrame:
        """加载并预处理测试数据"""
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"测试文件不存在: {test_csv}")
        
        df = pd.read_csv(test_csv)
        pig_names = [col for col in df.columns if col not in ['R','G','B']]
        df = df[['R','G','B'] + pig_names]
        df.iloc[:, 3:] = df.iloc[:, 3:].div(df.iloc[:, 3:].sum(axis=1), axis=0).fillna(0)
        return df
    
    def evaluate_km(self, model_path: str) -> Dict[str, float]:
        """评估KM模型（正向+反向）"""
        m = joblib.load(model_path)
        A = m['A_matrix']
        
        # 正向预测：配方 -> K/S
        ks_pred_forward = self.df.iloc[:, 3:].values @ A.T
        r2_ks_forward = r2_score(self.ks_true.ravel(), ks_pred_forward.ravel())
        
        # 反向反演：K/S -> 配方 (NNLS)
        from scipy.optimize import nnls
        recipe_pred = np.vstack([nnls(A, ks)[0] for ks in self.ks_true])
        recipe_pred = recipe_pred / recipe_pred.sum(axis=1, keepdims=True)
        r2_recipe = r2_score(self.df.iloc[:, 3:].values.ravel(), recipe_pred.ravel())
        
        return {
            'K/S_Forward_R2': r2_ks_forward,
            'Recipe_Inverse_R2': r2_recipe,
        }
    
    def evaluate_transformer(self, model_path: str) -> Dict[str, float]:
        """评估Transformer模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = ColorTransformer(**checkpoint.get('model_config', {}))
        model.load_state_dict(checkpoint['model_state'])
        model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            pred_rgb, pred_ks = model(self.recipes.to(self.device))
            
            rgb_loss = mean_squared_error(self.rgbs_true.numpy(), pred_rgb.cpu().numpy())
            r2_rgb = r2_score(self.rgbs_true.numpy().ravel(), pred_rgb.cpu().numpy().ravel())
            r2_ks = r2_score(self.ks_true.ravel(), pred_ks.cpu().numpy().ravel())
            
        return {
            'RGB_R2': r2_rgb,
            'K/S_R2': r2_ks,
            'RGB_MSE': rgb_loss
        }
    
    def evaluate_hybrid(self, model_path: str) -> Dict[str, float]:
        """评估混合模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = HybridColorModel(checkpoint['km_A_matrix'])
        model.load_state_dict(checkpoint['model_state'])
        model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            pred_rgb, pred_ks = model(self.recipes.to(self.device))
            
            rgb_loss = mean_squared_error(self.rgbs_true.numpy(), pred_rgb.cpu().numpy())
            r2_rgb = r2_score(self.rgbs_true.numpy().ravel(), pred_rgb.cpu().numpy().ravel())
            r2_ks = r2_score(self.ks_true.ravel(), pred_ks.cpu().numpy().ravel())
            
        return {
            'RGB_R2': r2_rgb,
            'K/S_R2': r2_ks,
            'RGB_MSE': rgb_loss
        }

def run_evaluation():
    """运行所有模型评估"""
    # 测试集配置
    TEST_SETS = {
        'key_test': 'scaler/data/key.csv',
        'data_test': 'scaler/data/data.csv',
    }
    
    # 模型配置 (路径, 评估函数)
    MODELS = {
        'KM (Ridge)': ('./models/km_correct.pkl', 'km'),
        'Transformer': ('./models/transformer_model.pkl', 'transformer'),
        'Hybrid': ('./models/hybrid_model.pkl', 'hybrid'),
    }
    
    results = {}
    
    for test_name, test_path in TEST_SETS.items():
        print(f"\n{'='*50}")
        print(f"测试集: {test_name} ({test_path})")
        print(f"{'='*50}")
        
        if not os.path.exists(test_path):
            print(f"⚠️  文件不存在，跳过")
            continue
        
        try:
            evaluator = ModelEvaluator(test_path)
            
            for model_name, (model_path, model_type) in MODELS.items():
                print(f"\n--- {model_name} ---")
                
                if not os.path.exists(model_path):
                    print(f"⚠️  模型文件不存在: {model_path}")
                    continue
                
                try:
                    if model_type == 'km':
                        metrics = evaluator.evaluate_km(model_path)
                    elif model_type == 'transformer':
                        metrics = evaluator.evaluate_transformer(model_path)
                    elif model_type == 'hybrid':
                        metrics = evaluator.evaluate_hybrid(model_path)
                    else:
                        continue
                    
                    # 打印结果
                    for metric_name, value in metrics.items():
                        print(f"  {metric_name}: {value:.5f}")
                    
                    # 存储结果
                    results[f"{test_name}_{model_name}"] = metrics
                    
                except Exception as e:
                    print(f"  ❌ 评估失败: {e}")
                    
        except Exception as e:
            print(f"❌ 初始化评估器失败: {e}")
    
    # 打印总结
    print(f"\n{'='*50}")
    print("评估总结")
    print(f"{'='*50}")
    for key, metrics in results.items():
        print(f"{key}: {metrics}")

if __name__ == '__main__':
    run_evaluation()
