# app.py – 国画配色系统 + 腾讯混元3D
from flask import Flask, request, jsonify, send_from_directory
from functools import wraps
from time import time
from collections import defaultdict
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import os
import sys
from typing import Dict, Any, Tuple
import base64
from PIL import Image
import io

# =============== 腾讯云SDK导入===============
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ai3d.v20250513 import ai3d_client, models

# =============== 配置国画配色utils路径 ===============
sys.path.append(os.path.join(os.path.dirname(__file__), 'scaler/py/utils'))
from scaler.py.utils.color_utils import rgb_to_ks, load_and_normalize_data

data_files = ['scaler/data/key.csv', 'scaler/data/data.csv']

# ==================== 国画配色参数配置 =====================================
# HSL判定阈值
HSL_WHITE_L_THRESHOLD = 0.85       
HSL_BLACK_L_THRESHOLD = 0.12       
GRAY_SAT_THRESHOLD = 0.15          

# 补色开关
ADD_COMPONENT_ENABLE = True        
ADD_COMPONENT_RATIO = 0.08         

# 融合参数
LOOKUP_THRESHOLD = 8.0             
FUSION_TOP_K = 5                   
FUSION_POWER = 2                   
FUSION_CONFIDENCE = 1.0           

# 存在性判断阈值
PRESENCE_THRESHOLD = 0.015         
MIN_RETAIN_COLORS = 2              

# Hybrid验证阈值
HYBRID_ERROR_THRESHOLD = 30.0      

# 中文映射
CHINESE_NAME_MAP = {
    "white": "白色",
    "stoneGreen": "石绿",
    "stoneBlue": "石青", 
    "vineYellow": "藤黄",
    "red": "朱红",
    "ocher": "赭石",
    "black": "黑色"
}

# =============== 混元3D配置===============
TENCENT_SECRET_ID = 
TENCENT_SECRET_KEY =      
TENCENT_REGION = "ap-guangzhou"
# ====================================================================

# =============== 防空 ===============
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MAX_IMAGE_DIMENSION = 2048  # 最大边长
MAX_PROMPT_LENGTH = 500  # 文字描述最大长度
RATE_LIMIT_PER_MIN = 10  # 每分钟最大请求数
# 内存频率限制器
rate_limit_store = defaultdict(list)
# ====================================================================

# 模型路径
KM_IMPROVED_PATH = r'./models/km_correct.pkl'
HYBRID_MODEL_PATH = r'./models/hybrid_model.pkl'
INVERSE_MODEL_PATH = r'./models/inverse_transformer.pkl'

# 全局变量
km_improved_model = None
hybrid_model = None
inverse_model = None
km_A_matrix = None
PIGMENT_NAMES_EN = []
PIGMENT_NAMES_CN = []
WHITE_IDX = -1
BLACK_IDX = -1
rgb_lookup_table = {}

class InverseTransformer(nn.Module):
    """逆向Transformer：RGB → Recipe"""
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, 31, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.recipe_decoder = nn.Sequential(
            nn.Linear(d_model * 31, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 7), nn.Sigmoid()
        )
        
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.rgb_encoder(rgb.unsqueeze(1))
        x = x.expand(-1, 31, -1)
        x = x + self.pos_encoding
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        recipe = self.recipe_decoder(x)
        return recipe

class HybridModel(nn.Module):
    """Hybrid模型：Recipe → RGB + K/S（动态维度）"""
    def __init__(self, km_A_matrix, n_pigments=7, d_model=None):
        super().__init__()
        self.register_buffer('A_matrix', torch.tensor(km_A_matrix, dtype=torch.float32))
        self.n_pigments = n_pigments
        
        if d_model:
            dim1 = d_model
            dim2 = d_model * 2
        else:
            dim1 = 128
            dim2 = 256
        
        self.transformer = nn.Sequential(
            nn.Linear(n_pigments, dim1), nn.ReLU(),
            nn.Linear(dim1, dim2), nn.ReLU(),
            nn.Linear(dim2, 31)
        )
        
        dim_rgb = 64
        self.rgb_head = nn.Sequential(
            nn.Linear(31, dim_rgb), nn.ReLU(), nn.Linear(dim_rgb, 3), nn.Sigmoid()
        )
        
    def forward(self, recipe):
        ks_km = recipe @ self.A_matrix.T
        ks_residual = self.transformer(recipe)
        ks = ks_km + ks_residual
        rgb = self.rgb_head(ks)
        return rgb, ks

def load_models():
    """加载三个模型（动态维度版）"""
    global km_improved_model, hybrid_model, inverse_model, km_A_matrix
    global PIGMENT_NAMES_EN, PIGMENT_NAMES_CN, WHITE_IDX, BLACK_IDX
    
    try:
        if not os.path.exists(KM_IMPROVED_PATH):
            raise FileNotFoundError(f"KM改进模型不存在: {KM_IMPROVED_PATH}")
        
        km_data = joblib.load(KM_IMPROVED_PATH)
        km_A_matrix = km_data['A_matrix']
        
        if 'pigment_names' in km_data:
            PIGMENT_NAMES_EN = km_data['pigment_names']
        else:
            df_temp = load_and_normalize_data(['scaler/data/key.csv'])
            PIGMENT_NAMES_EN = df_temp.columns[3:].tolist()
        
        PIGMENT_NAMES_CN = [CHINESE_NAME_MAP.get(name, name) for name in PIGMENT_NAMES_EN]
        print(f"✅ KM改进模型加载成功: A_MATRIX {km_A_matrix.shape}")
        
        if not os.path.exists(HYBRID_MODEL_PATH):
            raise FileNotFoundError(f"Hybrid模型不存在: {HYBRID_MODEL_PATH}")
        
        with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
            hybrid_checkpoint = torch.load(HYBRID_MODEL_PATH, map_location='cpu', weights_only=False)
        
        state_dict = hybrid_checkpoint['model_state']
        dim1 = state_dict['transformer.0.weight'].shape[0]
        dim2 = state_dict['transformer.2.weight'].shape[0]
        
        print(f"   Hybrid模型维度: {dim1}→{dim2}→31（从checkpoint推断）")
        
        hybrid_model = HybridModel(
            hybrid_checkpoint['km_A_matrix'], 
            n_pigments=len(PIGMENT_NAMES_EN),
            d_model=dim1
        )
        hybrid_model.load_state_dict(state_dict)
        hybrid_model.eval()
        print(f"✅ Hybrid模型加载成功")
        
        if not os.path.exists(INVERSE_MODEL_PATH):
            raise FileNotFoundError(f"逆向模型不存在: {INVERSE_MODEL_PATH}")
        
        inverse_checkpoint = torch.load(INVERSE_MODEL_PATH, map_location='cpu')
        model_config = inverse_checkpoint.get('model_config', {})
        
        inverse_model = InverseTransformer(
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 4)
        )
        inverse_model.load_state_dict(inverse_checkpoint['model_state'])
        inverse_model.eval()
        
        print(f"✅ 逆向Transformer加载成功: {len(PIGMENT_NAMES_EN)} 种颜料")
        print(f"   映射关系: {dict(zip(PIGMENT_NAMES_EN, PIGMENT_NAMES_CN))}")
        
        WHITE_IDX = PIGMENT_NAMES_EN.index('white') if 'white' in PIGMENT_NAMES_EN else -1
        BLACK_IDX = PIGMENT_NAMES_EN.index('black') if 'black' in PIGMENT_NAMES_EN else -1
        
        print(f"   白色索引: {WHITE_IDX}（{PIGMENT_NAMES_CN[WHITE_IDX] if WHITE_IDX>=0 else '无'}）")
        print(f"   黑色索引: {BLACK_IDX}（{PIGMENT_NAMES_CN[BLACK_IDX] if BLACK_IDX>=0 else '无'}）")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

def load_lookup_table():
    """加载RGB查找表"""
    global rgb_lookup_table
    
    try:
        df_all = load_and_normalize_data(data_files)
        for _, row in df_all.iterrows():
            rgb_key = (int(row['R']), int(row['G']), int(row['B']))
            rgb_lookup_table[rgb_key] = row[PIGMENT_NAMES_EN].values.astype(np.float64)
        print(f"✅ RGB查找表构建完成: {len(rgb_lookup_table)} 条记录")
    except Exception as e:
        print(f"⚠️  数据集加载警告: {e}")

def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """RGB转HSL"""
    r, g, b = rgb / 255.0
    max_, min_ = max(r, g, b), min(r, g, b)
    delta = max_ - min_
    l = (max_ + min_) / 2
    
    if delta == 0:
        s, h = 0, 0
    else:
        s = delta / (1 - abs(2 * l - 1))
        if max_ == r: h = 60 * (((g - b) / delta) % 6)
        elif max_ == g: h = 60 * ((b - r) / delta + 2)
        else: h = 60 * ((r - g) / delta + 4)
    
    return np.array([h, s, l])

def ultra_safe_normalize(w: np.ndarray) -> np.ndarray:
    """归一化"""
    w = np.nan_to_num(np.clip(np.asarray(w, dtype=np.float64), 0, 1), 
                      nan=0.0, posinf=1.0, neginf=0.0)
    return w / max(w.sum(), 1e-6)

def rgb_distance(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    return np.linalg.norm(rgb1 - rgb2)

def rgb_fusion_predict(target_rgb: np.ndarray) -> Tuple[Dict[str, float], float]:
    """RGB融合预测（返回中文结果）"""
    target_rgb = np.asarray(target_rgb, dtype=np.int32)
    
    if not rgb_lookup_table:
        return {}, 0.0
    
    dist_list = [(rgb_distance(target_rgb, k), v) for k, v in rgb_lookup_table.items()]
    dist_list.sort(key=lambda x: x[0])
    
    if not dist_list:
        return {}, 0.0
    
    topk = dist_list[:FUSION_TOP_K]
    d_max = topk[-1][0] + 1e-6
    
    weights = np.array([(d_max - d[0]) / d_max ** FUSION_POWER for d in topk])
    weights = weights / (weights.sum() + 1e-6)
    
    recipe = np.average([d[1] for d in topk], axis=0, weights=weights)
    recipe = ultra_safe_normalize(recipe)
    
    result = {
        PIGMENT_NAMES_CN[i]: round(recipe[i] * 100, 1)
        for i in range(len(PIGMENT_NAMES_EN)) if recipe[i] > 0.01
    }
    
    confidence = float(np.max(recipe)) if len(recipe) > 0 else 0.0
    return result, confidence

def get_presence_mask_from_km(rgb: np.ndarray) -> np.ndarray:
    """使用KM改进模型判断颜色存在性"""
    ks = rgb_to_ks(rgb, wavelengths=np.arange(400, 710, 10), rgb_wavelengths=np.array([630, 530, 450]))[0]
    
    w_km, _ = nnls(km_A_matrix, ks)
    w_km_norm = ultra_safe_normalize(w_km)
    
    presence_mask = w_km_norm > PRESENCE_THRESHOLD
    
    if presence_mask.sum() < MIN_RETAIN_COLORS:
        top_k_idx = np.argsort(w_km_norm)[-MIN_RETAIN_COLORS:]
        presence_mask = np.zeros_like(w_km_norm, dtype=bool)
        presence_mask[top_k_idx] = True
    
    existing = [(PIGMENT_NAMES_CN[i], w_km_norm[i]) for i in range(len(w_km_norm)) if presence_mask[i]]
    print(f"   KM存在性判断: 保留 {[f'{name}({val:.3f})' for name, val in existing]}")
    
    removed = [(PIGMENT_NAMES_CN[i], w_km_norm[i]) for i in range(len(w_km_norm)) if not presence_mask[i]]
    if removed:
        print(f"   KM过滤颜色: {[(name, f'{val:.3f}') for name, val in removed]}")
        print("-"*60)
    return presence_mask

def predict_with_km_fallback(rgb: np.ndarray, hsl: np.ndarray) -> np.ndarray:
    """KM模型备用方案"""
    ks = rgb_to_ks(rgb, wavelengths=np.arange(400, 710, 10), rgb_wavelengths=np.array([630, 530, 450]))[0]
    
    w_raw, _ = nnls(km_A_matrix, ks)
    w_norm = ultra_safe_normalize(w_raw)
    
    existing_mask = w_norm > PRESENCE_THRESHOLD
    if existing_mask.sum() < MIN_RETAIN_COLORS:
        top_k_idx = np.argsort(w_norm)[-MIN_RETAIN_COLORS:]
        existing_mask = np.zeros_like(w_norm, dtype=bool)
        existing_mask[top_k_idx] = True
    
    w_filtered = w_norm * existing_mask
    w_norm = ultra_safe_normalize(w_filtered)
    
    if WHITE_IDX >= 0 and BLACK_IDX >= 0:
        if hsl[2] > HSL_WHITE_L_THRESHOLD and hsl[1] < GRAY_SAT_THRESHOLD:
            if w_norm[BLACK_IDX] > 0.01:
                w_norm[BLACK_IDX] = 0.0
                w_norm = ultra_safe_normalize(w_norm)
            if ADD_COMPONENT_ENABLE and w_norm[WHITE_IDX] < 0.01:
                add_white = max(0.05, (hsl[2] - HSL_WHITE_L_THRESHOLD) * ADD_COMPONENT_RATIO)
                w_norm[WHITE_IDX] = add_white
                w_norm = ultra_safe_normalize(w_norm)
        elif hsl[2] < HSL_BLACK_L_THRESHOLD:
            if w_norm[WHITE_IDX] > 0.01:
                w_norm[WHITE_IDX] = 0.0
                w_norm = ultra_safe_normalize(w_norm)
            if ADD_COMPONENT_ENABLE and w_norm[BLACK_IDX] < 0.01:
                add_black = max(0.05, (HSL_BLACK_L_THRESHOLD - hsl[2]) * ADD_COMPONENT_RATIO)
                w_norm[BLACK_IDX] = add_black
                w_norm = ultra_safe_normalize(w_norm)
    
    return w_norm

def predict_with_inverse_model(rgb: np.ndarray, hsl: np.ndarray, 
                               enable_fallback: bool = True) -> np.ndarray:
    """逆向模型预测 + KM存在性判断 + Hybrid验证"""
    try:
        presence_mask = get_presence_mask_from_km(rgb)
        
        with torch.no_grad():
            rgb_tensor = torch.tensor(rgb / 255.0, dtype=torch.float32).unsqueeze(0)
            w_raw = inverse_model(rgb_tensor).squeeze().numpy()
        
        print(f"   原始预测: {[f'{PIGMENT_NAMES_CN[i]}:{w_raw[i]:.3f}' for i in range(len(w_raw))]}")
        
        w_filtered = w_raw * presence_mask
        filtered_info = [(PIGMENT_NAMES_CN[i], w_filtered[i]) for i in range(len(w_filtered)) if w_filtered[i] > 0]
        print(f"   应用KM掩码: {filtered_info}")
        
        if w_filtered.sum() > 0:
            w_norm = ultra_safe_normalize(w_filtered)
        else:
            print(f"⚠️ 过滤后全为零，回退到KM配方")
            return predict_with_km_fallback(rgb, hsl)
        
        if WHITE_IDX >= 0 and BLACK_IDX >= 0:
            is_bright_gray = hsl[2] > HSL_WHITE_L_THRESHOLD and hsl[1] < GRAY_SAT_THRESHOLD
            is_dark = hsl[2] < HSL_BLACK_L_THRESHOLD
            
            if is_bright_gray:
                if w_norm[BLACK_IDX] > 0.01:
                    print(f"⚠️ 高亮灰色(L={hsl[2]:.3f}) 删除冲突色 {PIGMENT_NAMES_CN[BLACK_IDX]} {w_norm[BLACK_IDX]*100:.1f}%")
                    w_norm[BLACK_IDX] = 0.0
                    w_norm = ultra_safe_normalize(w_norm)
                
                if ADD_COMPONENT_ENABLE and w_norm[WHITE_IDX] < 0.01:
                    add_white = max(0.05, (hsl[2] - HSL_WHITE_L_THRESHOLD) * ADD_COMPONENT_RATIO)
                    w_norm[WHITE_IDX] = add_white
                    w_norm = ultra_safe_normalize(w_norm)
                    print(f"✅ 偏白补色 {PIGMENT_NAMES_CN[WHITE_IDX]} {add_white*100:.1f}%")
            
            elif is_dark:
                if w_norm[WHITE_IDX] > 0.01:
                    print(f"⚠️ 低亮(L={hsl[2]:.3f}) 删除冲突色 {PIGMENT_NAMES_CN[WHITE_IDX]} {w_norm[WHITE_IDX]*100:.1f}%")
                    w_norm[WHITE_IDX] = 0.0
                    w_norm = ultra_safe_normalize(w_norm)
                
                if ADD_COMPONENT_ENABLE and w_norm[BLACK_IDX] < 0.01:
                    add_black = max(0.05, (HSL_BLACK_L_THRESHOLD - hsl[2]) * ADD_COMPONENT_RATIO)
                    w_norm[BLACK_IDX] = add_black
                    w_norm = ultra_safe_normalize(w_norm)
                    print(f"✅ 偏黑补色 {PIGMENT_NAMES_CN[BLACK_IDX]} {add_black*100:.1f}%")
        
        print("-"*60)
        print("   进行Hybrid模型验证...")
        with torch.no_grad():
            w_tensor = torch.tensor(w_norm, dtype=torch.float32).unsqueeze(0)
            pred_rgb, _ = hybrid_model(w_tensor)
            pred_rgb_np = (pred_rgb.squeeze().numpy() * 255).astype(np.int32)
        
        error = np.linalg.norm(pred_rgb_np - rgb)
        print(f"   Hybrid验证: 预测RGB={pred_rgb_np.tolist()}, 目标RGB={rgb.astype(int).tolist()}, 误差={error:.2f}")
        
        if error > HYBRID_ERROR_THRESHOLD:
            print(f"❌ Hybrid验证失败: 误差过大({error:.2f}>{HYBRID_ERROR_THRESHOLD:.2f})")
            print(rgb,hsl)
        else:
            print(f"✅ Hybrid验证通过")
        
        return w_norm
        
    except Exception as e:
        print(f"❌ 逆向模型预测失败: {e}")
        if enable_fallback:
            print(f"   回退到KM模型...")
            return predict_with_km_fallback(rgb, hsl)
        else:
            raise

## =============== 防空设计工具函数 ===============
def validate_and_clean_image(image_data: str) -> Tuple[str, Dict[str, Any]]:
    """防空核心：验证图片并返回清理后的数据"""
    if not image_data:
        return "", {"valid": False, "error": "图片数据为空"}
    
    try:
        # 防空：检查Base64数据头
        if not image_data.startswith('data:image/'):
            return "", {"valid": False, "error": "无效的图片格式，必须是data URI格式"}
        
        # 防空：提取实际数据
        try:
            header, data = image_data.split(',', 1)
        except ValueError:
            return "", {"valid": False, "error": "图片数据格式错误"}
        
        # 防空：检查文件类型
        file_type = header.split(';')[0].split('/')[1].lower()
        if f'.{file_type}' not in ALLOWED_IMAGE_TYPES:
            return "", {"valid": False, "error": f"不支持的图片类型: {file_type}，仅支持{ALLOWED_IMAGE_TYPES}"}
        
        # 防空：解码并检查大小
        try:
            img_bytes = base64.b64decode(data)
        except Exception:
            return "", {"valid": False, "error": "Base64解码失败"}
        
        if len(img_bytes) > MAX_IMAGE_SIZE:
            return "", {"valid": False, "error": f"图片大小超过限制（{MAX_IMAGE_SIZE / 1024 / 1024}MB）"}
        
        # 防空：检查图片真实尺寸和格式
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                width, height = img.size
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    return "", {"valid": False, "error": f"图片尺寸过大: {width}x{height}（最大支持{MAX_IMAGE_DIMENSION}px）"}
                
                # 防空：转换为RGB模式（避免透明度问题）
                if img.mode in ('RGBA', 'LA', 'P', 'CMYK'):
                    img = img.convert('RGB')
                    # 重新转换为Base64
                    buffered = io.BytesIO()
                    img.save(buffered, format='JPEG', quality=95)
                    clean_data = base64.b64encode(buffered.getvalue()).decode()
                    return f"data:image/jpeg;base64,{clean_data}", {"valid": True}
        except Exception as e:
            return "", {"valid": False, "error": f"图片解析失败: {str(e)}"}
        
        return image_data, {"valid": True}
        
    except Exception as e:
        return "", {"valid": False, "error": f"图片处理失败: {str(e)}"}

def cleanup_temp_data(request_data: Dict[str, Any]):
    """防空：清理临时数据防止内存泄漏"""
    # 删除request中的大对象引用
    if 'image' in request_data and len(request_data['image']) > 10000:
        request_data['image'] = '[IMAGE_DATA_CLEARED]'

def rate_limit(max_per_minute=RATE_LIMIT_PER_MIN):
    """防空：频率限制装饰器"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr
            now = time()
            
            # 防空：清理过期记录
            rate_limit_store[client_ip] = [
                t for t in rate_limit_store[client_ip] 
                if now - t < 60
            ]
            
            # 防空：检查频率并返回标准化错误码
            if len(rate_limit_store[client_ip]) >= max_per_minute:
                return jsonify({
                    "success": False,
                    "error": f"请求过于频繁，每分钟最多{max_per_minute}次",
                    "code": "RATE_LIMITED"
                }), 429
            
            rate_limit_store[client_ip].append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator

# =============== Flask路由 ===============
app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """国画配色预测接口"""
    try:
        data = request.get_json()
        if not data or 'R' not in data or 'G' not in data or 'B' not in data:
            return jsonify({"error": "缺少RGB参数"}), 400
        
        rgb = np.array([data['R'], data['G'], data['B']], dtype=np.float64)
        hsl = rgb_to_hsl(rgb)
        enable_fallback = data.get('enableFallback', True)
        
        print("\n" + "="*60)
        print("🎯 新预测请求")
        print(f"   RGB: {rgb.astype(int).tolist()}  HSL: H={hsl[0]:.1f}° S={hsl[1]:.3f} L={hsl[2]:.3f}")
        print(f"hsl: {hsl}")
        
        if hsl[2] > HSL_WHITE_L_THRESHOLD and hsl[1] < GRAY_SAT_THRESHOLD:
            print("🪄 判定为纯白")
            white_name = PIGMENT_NAMES_CN[WHITE_IDX] if WHITE_IDX >= 0 else "白色"
            return jsonify({white_name: 100.0})
        
        if hsl[2] < HSL_BLACK_L_THRESHOLD:
            print("🪄 判定为纯黑")
            black_name = PIGMENT_NAMES_CN[BLACK_IDX] if BLACK_IDX >= 0 else "黑色"
            return jsonify({black_name: 100.0})
        
        fusion_result, confidence = rgb_fusion_predict(rgb)
        print(f"融合方案:{fusion_result}")
        print(f"[RGB融合] 置信度={confidence:.3f}, 阈值={FUSION_CONFIDENCE}")
        
        if confidence >= FUSION_CONFIDENCE:
            print(f"✅ 采用融合配方: {fusion_result}")
            return jsonify(fusion_result)
        
        rgb_key = tuple(rgb.astype(int))
        if rgb_key in rgb_lookup_table:
            print(f"✅ 数据集精确命中")
            w_norm = ultra_safe_normalize(rgb_lookup_table[rgb_key])
        else:
            print(f"❌ 未命中数据集，启动KM+Hybrid流程...")
            w_norm = predict_with_inverse_model(rgb, hsl, enable_fallback)
        
        result = {
            PIGMENT_NAMES_CN[i]: round(w_norm[i] * 100, 1)
            for i in range(len(PIGMENT_NAMES_EN)) if w_norm[i] > 0.005
        }
        
        print(f"📤 最终配方: {result}")
        print("="*60)
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"预测失败: {str(e)}"}), 500

# =============== 混元3D接口===============
def create_ai3d_client():
    """创建腾讯云3D客户端"""
    cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
    client_profile = ClientProfile()
    client_profile.httpProfile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
    return ai3d_client.Ai3dClient(cred, TENCENT_REGION, client_profile)

@app.route('/api/hunyuan3d/submit', methods=['POST'])
@rate_limit(max_per_minute=RATE_LIMIT_PER_MIN)  # 防空：频率限制
def hunyuan3d_submit():
    """提交混元3D专业版任务（防空强化版）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "请求数据不能为空",
                "code": "EMPTY_REQUEST"
            }), 400
        
        # 防空：提取参数（允许空字符串）
        prompt = data.get("prompt", "").strip()
        image_data = data.get("image", "").strip()
        
        # 防空：至少提供一个输入
        if not prompt and not image_data:
            return jsonify({
                "success": False,
                "error": "请提供文字描述或图片", 
                "code": "NO_INPUT_PROVIDED"
            }), 400
        
        # 防空：文字长度限制
        if len(prompt) > MAX_PROMPT_LENGTH:
            return jsonify({
                "success": False,
                "error": f"文字描述过长（最大{MAX_PROMPT_LENGTH}字）", 
                "code": "PROMPT_TOO_LONG"
            }), 400
        
        # 防空：图片验证（如果有）
        clean_image = ""
        input_type = "text_only"  # 记录输入类型用于监控
        if image_data:
            clean_image, validation = validate_and_clean_image(image_data)
            if not validation["valid"]:
                return jsonify({
                    "success": False,
                    "error": validation["error"], 
                    "code": "INVALID_IMAGE"
                }), 400
            input_type = "image_only" if not prompt else "text+image"
        
        # 防空：无文字时的日志记录
        if not prompt and clean_image:
            print(f"⚠️ 防空警告: 收到纯图片请求，建议补充文字描述以提升效果")
        
        client = create_ai3d_client()
        req = models.SubmitHunyuanTo3DProJobRequest()
        
        # ✅ v20250513专业版参数
        params = {}
        if prompt:
            params["Prompt"] = prompt
        
        # 防空：确保图片数据有效才添加
        if clean_image:
            params["Image"] = clean_image
        
        # 防空：清理请求数据
        cleanup_temp_data(data)
        
        req.from_json_string(str(params).replace("'", '"'))
        resp = client.SubmitHunyuanTo3DProJob(req)
        
        return jsonify({
            "success": True,
            "jobId": resp.JobId,
            "requestId": resp.RequestId,
            "message": "任务提交成功",
            "inputType": input_type  # 防空：返回输入类型供前端参考
        })

    except Exception as e:
        error_msg = f"提交失败: {str(e)}"
        print(f"❌ 防空捕获异常: {error_msg}")
        
        # 防空：异常分类
        error_code = "SUBMIT_ERROR"
        if "NoPermission" in str(e):
            error_code = "AUTH_FAILED"
        elif "LimitExceeded" in str(e):
            error_code = "RATE_LIMIT"
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "code": error_code,
            "suggestion": "请检查图片格式和大小，或稍后重试"
        }), 500
        
    finally:
        # 防空：强制清理
        if 'data' in locals():
            cleanup_temp_data(data)

@app.route('/api/hunyuan3d/query/<job_id>')
def hunyuan3d_query(job_id):
    """查询混元3D专业版任务状态"""
    try:
        if not job_id:
            return jsonify({"error": "JobId不能为空", "code": "INVALID_JOB_ID"}), 400

        client = create_ai3d_client()
        req = models.QueryHunyuanTo3DProJobRequest()
        req.JobId = job_id
        
        resp = client.QueryHunyuanTo3DProJob(req)
        
        result_files = []
        if resp.ResultFile3Ds:
            for file in resp.ResultFile3Ds:
                result_files.append({
                    "type": file.Type,
                    "url": file.Url,
                    "previewUrl": file.PreviewImageUrl
                })
        
        return jsonify({
            "success": True,
            "status": resp.Status,
            "errorCode": getattr(resp, "ErrorCode", ""),
            "errorMessage": getattr(resp, "ErrorMessage", ""),
            "resultFiles": result_files,
            "createTime": getattr(resp, "CreateTime", ""),
            "updateTime": getattr(resp, "UpdateTime", "")
        })

    except Exception as e:
        print(f"❌ 查询任务失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"查询失败: {str(e)}",
            "code": "QUERY_ERROR"
        }), 500

@app.route('/hunyuan3d')
def hunyuan3d_page():
    """混元3D创作页面"""
    return send_from_directory('.', 'hunyuan3d.html')

# =============== 主程序入口 ===============
if __name__ == '__main__':
    # 初始化
    print("="*60)
    print("🎨 国画配色系统 + 🤖 腾讯混元3D专业版（防空加强版）")
    print("="*60)
    
    # 检查密钥配置
    if TENCENT_SECRET_ID and TENCENT_SECRET_KEY:
        print("✅ 混元3D密钥已配置")
    else:
        print("⚠️ 混元3D密钥未配置")
        print("   请在代码第68-69行设置")
    
    # 防空：显示配置
    print("\n" + "="*60)
    print("  防空配置")
    print("="*60)
    print(f"   图片大小限制: {MAX_IMAGE_SIZE / 1024 / 1024}MB")
    print(f"   图片尺寸限制: {MAX_IMAGE_DIMENSION}px")
    print(f"   允许类型: {ALLOWED_IMAGE_TYPES}")
    print(f"   频率限制: {RATE_LIMIT_PER_MIN}次/分钟")
    print("="*60)
    
    load_models()
    load_lookup_table()
    
    print("\n" + "="*60)
    print("  服务状态")
    print("="*60)
    print(f"   ✅ KM存在性判断: {KM_IMPROVED_PATH}")
    print(f"   ✅ Hybrid验证: {HYBRID_MODEL_PATH}")
    print(f"   ✅ 逆向Transformer: {INVERSE_MODEL_PATH}")
    print(f"   存在性阈值: {PRESENCE_THRESHOLD}  | 最少保留: {MIN_RETAIN_COLORS}种")
    print(f"   Hybrid误差阈值: {HYBRID_ERROR_THRESHOLD}")
    print("="*60)
    print("✅ 服务启动成功！")
    print("   国画配色: http://localhost:5000/photo_looker.html")
    print("   AI 3D创作: http://localhost:5000/hunyuan3d")
    print("="*60 + "\n")
    
    app.run(debug=False, port=5000, host='0.0.0.0')
