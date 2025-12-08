# scaler/py/hybrid_model.py - 修复版
import joblib
import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.color_utils import load_and_normalize_data

class HybridColorModel(nn.Module):
    def __init__(self, km_A_matrix: np.ndarray, n_pigments: int = 7, d_model: int = 128):
        super().__init__()
        self.register_buffer('A_matrix', torch.tensor(km_A_matrix, dtype=torch.float32))
        
        self.transformer = nn.Sequential(
            nn.Linear(n_pigments, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model*2), nn.ReLU(),
            nn.Linear(d_model*2, 31)
        )
        
        self.rgb_head = nn.Sequential(
            nn.Linear(31, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid()
        )
        
    def forward(self, recipe: torch.Tensor):
        ks_km = recipe @ self.A_matrix.T
        ks_residual = self.transformer(recipe)
        ks = ks_km + ks_residual
        rgb = self.rgb_head(ks)
        return rgb, ks

def train_hybrid(km_model_path: str, data_files: list, save_path: str = './models/hybrid_model.pkl'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    df = load_and_normalize_data(data_files)
    recipes = torch.tensor(df.iloc[:, 3:].values, dtype=torch.float32)
    rgbs = torch.tensor(df[['R','G','B']].values, dtype=torch.float32) / 255.0
    
    # 加载KM模型
    km_model = joblib.load(km_model_path)
    A_matrix = km_model['A_matrix']
    
    # 模型
    model = HybridColorModel(A_matrix, n_pigments=recipes.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 简单训练（样本少，全数据训练）
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        pred_rgb, _ = model(recipes.to(device))
        loss = criterion(pred_rgb, rgbs.to(device))
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.6f}")
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'km_A_matrix': A_matrix,
        'pigment_names': df.columns[3:].tolist(),
    }, save_path)
    print(f"\n✅ 混合模型已保存: {save_path}")

if __name__ == '__main__':
    DATA_FILES = ['scaler/data/key.csv', 'scaler/data/data.csv', 'scaler/data/text.csv']
    train_hybrid('./models/km_correct.pkl', DATA_FILES)
