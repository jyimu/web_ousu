# scaler/py/train_transformer.py - 修复版
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.color_utils import load_and_normalize_data

CONFIG = {
    'data_files': ['scaler/data/key.csv', 'scaler/data/data.csv', 'scaler/data/text.csv'],
    'model_save_path': './models/transformer_model.pkl',
    'max_epochs': 1000,
    'lr': 1e-3,
    'val_split': 0.2,
    'early_stop_patience': 50,
    'batch_size': 4,
}

class ColorTransformer(nn.Module):
    def __init__(self, n_pigments: int = 7, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        
        self.recipe_embedding = nn.Linear(n_pigments, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 31, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ks_decoder = nn.Linear(d_model, 1)
        
        self.rgb_head = nn.Sequential(
            nn.Linear(31, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 3), nn.Sigmoid()
        )
        
    def forward(self, recipe: torch.Tensor):
        x = self.recipe_embedding(recipe).unsqueeze(1).expand(-1, 31, -1)
        x = x + self.pos_encoding
        x = self.encoder(x)
        ks = self.ks_decoder(x).squeeze(-1)
        rgb = self.rgb_head(ks)
        return rgb, ks

def train_transformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    df = load_and_normalize_data(CONFIG['data_files'])
    recipes = torch.tensor(df.iloc[:, 3:].values, dtype=torch.float32)
    rgbs = torch.tensor(df[['R','G','B']].values, dtype=torch.float32) / 255.0
    
    n_samples = len(df)
    print(f"样本数: {n_samples}")
    
    # 划分数据集
    val_size = max(1, int(n_samples * CONFIG['val_split']))
    indices = torch.randperm(n_samples)
    train_idx, val_idx = indices[:n_samples-val_size], indices[n_samples-val_size:]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(recipes[train_idx], rgbs[train_idx]),
        batch_size=CONFIG['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(recipes[val_idx], rgbs[val_idx]),
        batch_size=CONFIG['batch_size']
    )
    
    # 模型
    model = ColorTransformer(n_pigments=recipes.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n[训练] 开始训练...")
    for epoch in range(CONFIG['max_epochs']):
        # 训练
        model.train()
        train_loss = 0
        for batch_recipes, batch_rgbs in train_loader:
            batch_recipes, batch_rgbs = batch_recipes.to(device), batch_rgbs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_recipes)[0], batch_rgbs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_recipes, batch_rgbs in val_loader:
                pred_rgb, _ = model(batch_recipes.to(device))
                val_loss += criterion(pred_rgb, batch_rgbs.to(device)).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train={avg_train:.6f}, Val={avg_val:.6f}")
        
        # 早停
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"早停触发，epoch {epoch}, 最佳Val Loss: {best_val_loss:.6f}")
                break
    
    model.load_state_dict(best_state)
    
    # 保存
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'pigment_names': df.columns[3:].tolist(),
        'model_config': {'n_pigments': recipes.shape[1], 'd_model': model.d_model},
        'best_val_loss': best_val_loss,
    }, CONFIG['model_save_path'])
    
    print(f"\n✅ Transformer模型已保存: {CONFIG['model_save_path']}")
    return {'val_loss': best_val_loss}

if __name__ == '__main__':
    train_transformer()
