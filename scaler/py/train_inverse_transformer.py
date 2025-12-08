# train_inverse_transformer.py - 训练逆向模型（RGB → Recipe）修复版
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Dict, Any

# 添加utils模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.color_utils import load_and_normalize_data

CONFIG = {
    'data_files': ['scaler/data/key.csv', 'scaler/data/data.csv', 'scaler/data/text.csv'],
    'model_save_path': './models/inverse_transformer.pkl',
    'epochs': 500,
    'lr': 1e-3,
    'val_split': 0.2,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'max_patience': 100,
}

class InverseTransformer(nn.Module):
    """逆向Transformer：RGB → Recipe"""
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        
        # RGB编码器
        self.rgb_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 位置编码（扩展31个波段位置）
        self.pos_encoding = nn.Parameter(torch.randn(1, 31, d_model) * 0.1)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Recipe解码器（输出7种颜料）
        self.recipe_decoder = nn.Sequential(
            nn.Linear(d_model * 31, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Sigmoid()  # 输出0-1范围
        )
        
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # RGB编码
        x = self.rgb_encoder(rgb.unsqueeze(1))  # (batch, 1, d_model)
        x = x.expand(-1, 31, -1)  # (batch, 31, d_model)
        x = x + self.pos_encoding
        
        # Transformer编码
        x = self.encoder(x)  # (batch, 31, d_model)
        
        # 展平并解码为配方
        x = x.reshape(x.size(0), -1)  # (batch, 31*d_model)
        recipe = self.recipe_decoder(x)  # (batch, 7)
        return recipe

def train_inverse_model() -> Dict[str, Any]:
    """训练逆向模型（修复版）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据（关键修复：在函数内获取PIGMENT_NAMES）
    df = load_and_normalize_data(CONFIG['data_files'])
    pigment_names = df.columns[3:].tolist()  # ✅ 修复：在函数内定义
    rgbs = torch.tensor(df[['R','G','B']].values, dtype=torch.float32) / 255.0
    recipes = torch.tensor(df.iloc[:, 3:].values, dtype=torch.float32)
    
    print(f"数据加载完成: RGB shape={rgbs.shape}, Recipe shape={recipes.shape}")
    
    # 划分数据集
    dataset = TensorDataset(rgbs, recipes)
    val_size = max(1, int(len(dataset) * CONFIG['val_split']))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=min(4, train_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(4, val_size))
    
    # 模型
    model = InverseTransformer(
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n[训练] 开始训练逆向模型...")
    for epoch in range(CONFIG['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_rgb, batch_recipe in train_loader:
            batch_rgb, batch_recipe = batch_rgb.to(device), batch_recipe.to(device)
            
            optimizer.zero_grad()
            pred_recipe = model(batch_rgb)
            loss = criterion(pred_recipe, batch_recipe)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_rgb, batch_recipe in val_loader:
                pred_recipe = model(batch_rgb.to(device))
                val_loss += criterion(pred_recipe, batch_recipe.to(device)).item()
        
        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        if epoch % 50 == 0 or epoch == CONFIG['epochs']-1:
            print(f"Epoch {epoch:3d}: Train={avg_train:.6f}, Val={avg_val:.6f}")
        
        # 学习率调度
        scheduler.step(avg_val)
        
        # 早停机制
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['max_patience']:
                print(f"早停触发，epoch {epoch}, 最佳Val Loss: {best_val_loss:.6f}")
                break
    
    # 加载最佳状态
    model.load_state_dict(best_state)
    
    # ✅ 修复：使用函数内定义的 pigment_names
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'pigment_names': pigment_names,  # ✅ 使用局部变量
        'model_config': {
            'd_model': CONFIG['d_model'],
            'nhead': CONFIG['nhead'],
            'num_layers': CONFIG['num_layers']
        },
        'config': CONFIG,
        'best_val_loss': best_val_loss,
    }, CONFIG['model_save_path'])
    
    print(f"\n✅ 逆向模型已保存: {CONFIG['model_save_path']}")
    print(f"   最佳验证Loss: {best_val_loss:.6f}")
    
    return {
        'model': model,
        'val_loss': best_val_loss,
        'pigment_names': pigment_names
    }

if __name__ == '__main__':
    try:
        result = train_inverse_model()
        print(f"\n训练完成！可使用模型: {CONFIG['model_save_path']}")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
