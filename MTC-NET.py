#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
import re
import pandas as pd
import numpy as np
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# -----------------------------
# 1. 核心路径配置
# -----------------------------
# 输入文件夹：存放 12 个特征 Excel 表
input_dir = r"C:\Users\HP\Desktop\投稿文件\data\Data input"
# 数组总文件夹：包含 12 个工况文件夹
array_base_dir = r"C:\Users\HP\Desktop\投稿文件\data\data input2"

# -----------------------------
# 2. 增强型正则匹配与加载函数
# -----------------------------
def get_condition_keys(filename):
    """提取温度(含负数)和倍率，例如 -5℃-0.5C -> (-5.0, 0.5)"""
    # 匹配负号和数字: (-?\d+)
    temp_match = re.search(r'(-?\d+)\s*[℃°度]', filename)
    clean_name = filename.replace('℃', '').replace('°C', '')
    rate_match = re.search(r'(\d+\.?\d*)\s*[Cc]', clean_name)
    
    temp = float(temp_match.group(1)) if temp_match else None
    rate = float(rate_match.group(1)) if rate_match else None
    return temp, rate

def load_stress_field(file_path):
    """智能读取应力场 (41x27)"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, index_col=None, header=None)
        else:
            df = pd.read_excel(file_path, index_col=None, header=None)
        
        # 自动探测数据区域：如果是带行列索引的，取最后 41x27
        if df.shape[0] >= 41 and df.shape[1] >= 27:
            return df.iloc[-41:, -27:].values.astype(np.float32)
    except:
        pass
    return None

# ==========================================
# 模块一：扫描并精准匹配路径 (包含 Arrays 搜索)
# ==========================================
print(f"🕵️‍♂️ 正在执行多级路径匹配，目标：{array_base_dir}...")

if not os.path.exists(input_dir) or not os.path.exists(array_base_dir):
    raise ValueError(f"❌ 路径不存在，请检查：\n输入库: {input_dir}\n数据库: {array_base_dir}")

# 扫描特征文件
input_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
input_dict = {}
for f in input_files:
    key = get_condition_keys(f)
    if key[0] is not None and key[1] is not None:
        input_dict[key] = os.path.join(input_dir, f)

# 扫描工况文件夹并进入子目录 Arrays
valid_tasks = []
total_files_to_read = 0

all_sub_folders = [f for f in os.listdir(array_base_dir) if os.path.isdir(os.path.join(array_base_dir, f))]

for folder_name in all_sub_folders:
    key = get_condition_keys(folder_name)
    
    # 检查特征库里有没有对应的工况
    if key not in input_dict:
        continue
    
    # 核心修改：进入子文件夹搜索 Arrays 文件夹
    work_path = os.path.join(array_base_dir, folder_name, "Arrays")
    
    if not os.path.exists(work_path):
        print(f"⚠️ 跳过: {folder_name} 中未找到 'Arrays' 文件夹")
        continue

    # 在 Arrays 文件夹内搜寻 TimeStep 文件
    ts_files = [f for f in os.listdir(work_path) if 'TimeStep_' in f and f.endswith(('.xlsx', '.csv'))]
    
    if ts_files:
        # 排序确保时序正确
        ts_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        valid_tasks.append({
            'key': key,
            'feature_file': input_dict[key],
            'data_path': work_path,
            'files': ts_files
        })
        total_files_to_read += len(ts_files)

if total_files_to_read == 0:
    raise ValueError("❌ 匹配失败：请检查文件夹名称是否包含温度/倍率，且内部是否有 Arrays 文件夹及 TimeStep 数据。")

print(f"✅ 成功匹配 {len(valid_tasks)} 个工况，共计 {total_files_to_read} 个场数据文件。\n")

# ==========================================
# 模块二：核心读取逻辑 (保持单进度条)
# ==========================================
all_X_features = []
all_stress_fields = []

with tqdm(total=total_files_to_read, desc="🚀 数据全量装载中", colour='blue', ncols=100) as pbar:
    for task in valid_tasks:
        key = task['key']
        feature_file = task['feature_file']
        data_path = task['data_path']
        files = task['files']
        
        # 1. 读取输入特征 (跳过表头，提取指定列)
        try:
            df_feat = pd.read_excel(feature_file)
            # 这里的 iloc[1:, ...] 是为了跳过第一行表头后的数据起始行，请根据实际表结构微调
            X_data = df_feat.iloc[1:, [1, 11, 16, 17, 18, 19]].values.astype(np.float32)
        except Exception as e:
            tqdm.write(f"❌ 特征读取失败: {key[0]}℃-{key[1]}C | 错误: {e}")
            pbar.update(len(files))
            continue
            
        # 2. 循环读取 Arrays 里的场文件
        cond_fields = []
        for f_name in files:
            sf = load_stress_field(os.path.join(data_path, f_name))
            if sf is not None:
                cond_fields.append(sf)
            pbar.update(1)
            
        if not cond_fields:
            continue
            
        # 3. 截断对齐 (特征行数 vs 场文件个数)
        min_len = min(len(X_data), len(cond_fields))
        all_X_features.append(X_data[:min_len])
        all_stress_fields.append(np.array(cond_fields[:min_len]))
        
        tqdm.write(f"🌟 完成装载: {key[0]}℃ | {key[1]}C | 对齐点数: {min_len}")

# ==========================================
# 模块三：数据合并
# ==========================================ru
if not all_X_features:
    raise ValueError("🚨 没有任何数据被加载，请检查特征表的行列索引是否正确。")

X_final = np.vstack(all_X_features)
Y_final = np.concatenate(all_stress_fields, axis=0)

print("\n" + "="*50)
print(f"🎯 数据集准备就绪！")
print(f"输入特征 (X): {X_final.shape}")
print(f"输出场数据 (Y): {Y_final.shape}")
print("="*50)


# In[2]:


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. 精准指定盲测集工况 (Multi-Condition Blind Test)
# -----------------------------
# 定义你指定的 3 个盲测工况关键词
target_test_conditions = [(35.0, 1.5), (10.0, 1.5), (-5.0, 0.5)]

test_idx = []
train_val_idx = []

print("🔍 正在根据目标工况进行索引匹配...")

# 遍历加载阶段生成的任务列表，进行归类
for i, task in enumerate(valid_tasks):
    curr_key = task['key'] # 获取该索引对应的 (温度, 倍率)
    if curr_key in target_test_conditions:
        test_idx.append(i)
    else:
        train_val_idx.append(i)

# 安全检查
if not test_idx:
    raise ValueError("🚨 匹配失败！未能找到指定的测试工况，请确认温度和倍率数值是否准确。")

print(f"✅ 划分完成！")
print(f"   ➤ 独立盲测工况索引: {test_idx} (包含: {[valid_tasks[i]['key'] for i in test_idx]})")
print(f"   ➤ 训练池工况索引: {train_val_idx} (共 {len(train_val_idx)} 个工况)")

def get_subset(indices, X_list, Y_list):
    X_sub = np.vstack([X_list[i] for i in indices])
    Y_sub = np.concatenate([Y_list[i] for i in indices], axis=0)
    return X_sub, Y_sub

# 获取训练池和测试集的原始数据
X_train_val_raw, y_train_val_raw = get_subset(train_val_idx, all_X_features, all_stress_fields)
X_test_raw, y_test_raw = get_subset(test_idx, all_X_features, all_stress_fields)

# 展平场数据为 1107 维 (41x27)
y_train_val_flat = y_train_val_raw.reshape(len(y_train_val_raw), -1)
y_test_flat = y_test_raw.reshape(len(y_test_raw), -1)

# -----------------------------
# 2. 从训练池中切出 10% 作为验证集
# -----------------------------
# 使用 shuffle=False 保持各个工况内部的时序连续性
X_train_raw, X_val_raw, y_train_flat, y_val_flat = train_test_split(
    X_train_val_raw, y_train_val_flat, test_size=0.10, shuffle=False
)

# -----------------------------
# 3. 严格归一化 (Scaler 仅学习训练集分布)
# -----------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_norm = scaler_X.fit_transform(X_train_raw)
y_train_norm = scaler_y.fit_transform(y_train_flat)

X_val_norm = scaler_X.transform(X_val_raw)
y_val_norm = scaler_y.transform(y_val_flat)

X_test_norm = scaler_X.transform(X_test_raw)
y_test_norm = scaler_y.transform(y_test_flat)

# -----------------------------
# 4. 转换为 Tensor 并创建加载器
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor_split(X, y):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return X_t[:, :3], X_t[:, 3:], y_t

P_train, Aux_train, y_train_t = to_tensor_split(X_train_norm, y_train_norm)
P_val, Aux_val, y_val_t = to_tensor_split(X_val_norm, y_val_norm)
P_test, Aux_test, y_test_t = to_tensor_split(X_test_norm, y_test_norm)

train_dataset = TensorDataset(P_train, Aux_train, y_train_t)
val_dataset = TensorDataset(P_val, Aux_val, y_val_t)
test_dataset = TensorDataset(P_test, Aux_test, y_test_t)

# 建议适当增大 batch_size 以加快训练，或保持 16/32
batch_size = 16  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n" + "="*50)
print(f"📦 MTC-Net 数据流准备完毕！")
print(f"   ➤ 训练集 (Train): {len(train_dataset)} 样本")
print(f"   ➤ 验证集 (Val):   {len(val_dataset)} 样本")
print(f"   ➤ 盲测集 (Test):  {len(test_dataset)} 样本 (3个代表性工况)")
print("="*50)


# In[3]:


import torch
import os

# 1. 明确路径
save_dir = r"C:\Users\HP\Desktop\MTC-Net_StressField_Results"
save_path = os.path.join(save_dir, "processed_tensors.pt")

# 2. ✨ 核心补丁：检查并创建文件夹
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"📁 检测到文件夹不存在，已自动创建: {save_dir}")

# 3. 执行保存
print("💾 正在将处理好的 Tensor 数据保存到硬盘...")
torch.save({
    'P_train': P_train, 'Aux_train': Aux_train, 'y_train_t': y_train_t,
    'P_val': P_val, 'Aux_val': Aux_val, 'y_val_t': y_val_t,
    'P_test': P_test, 'Aux_test': Aux_test, 'y_test_t': y_test_t,
    'scaler_y': scaler_y 
}, save_path)

print(f"✅ 保存成功！文件位于: {save_path}")


# In[7]:


import torch
import os
from torch.utils.data import TensorDataset, DataLoader

# 1. 指定文件路径 (必须和刚才保存的一致)
load_path = r"C:\Users\HP\Desktop\MTC-Net_StressField_Results\processed_tensors.pt"

print("🔄 正在从硬盘执行【一键回归】...")

if os.path.exists(load_path):
    # ✨ 核心：读取数据
    # weights_only=False 是为了兼容 scaler 等非 Tensor 对象
    checkpoint = torch.load(load_path, weights_only=False)
    
    # 2. 还原所有 Tensor 变量
    P_train, Aux_train, y_train_t = checkpoint['P_train'], checkpoint['Aux_train'], checkpoint['y_train_t']
    P_val, Aux_val, y_val_t = checkpoint['P_val'], checkpoint['Aux_val'], checkpoint['y_val_t']
    P_test, Aux_test, y_test_t = checkpoint['P_test'], checkpoint['Aux_test'], checkpoint['y_test_t']
    
    # 3. 还原归一化工具 (这非常重要，否则后面没法反归一化出真实应力值)
    scaler_y = checkpoint['scaler_y']
    
    # 4. 重新打包为 DataLoader (训练引擎直接吃这个)
    batch_size = 16  # 你可以根据显存大小调整，通常 16 或 32
    train_loader = DataLoader(TensorDataset(P_train, Aux_train, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(P_val, Aux_val, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(P_test, Aux_test, y_test_t), batch_size=batch_size, shuffle=False)

    print("\n" + "="*50)
    print("✅ 数据已完美回归！")
    print(f"   ➤ 恢复样本总数: {len(P_train) + len(P_val) + len(P_test)}")
    print(f"   ➤ 盲测工况已锁定，Scaler 已就绪")
    print("="*50)
else:
    print("🚨 找不到备份文件，请确认路径是否正确或是否已经跑过保存步骤！")


# In[8]:


from torch.utils.data import TensorDataset

train_dataset = TensorDataset(P_train, Aux_train, y_train_t)
val_dataset   = TensorDataset(P_val, Aux_val, y_val_t)
test_dataset  = TensorDataset(P_test, Aux_test, y_test_t)


# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import time

# -----------------------------
# 1. 定义核心组件
# -----------------------------
class CausalDilatedConv1d(nn.Module):
    """因果扩张卷积：提取应力迟滞特征"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    def forward(self, x):
        return F.relu(self.conv(x))[:, :, :-self.conv.padding[0]]

class MTCNet(nn.Module):
    """Multi-modal Temporal-spatial Coupling Network (MTC-Net)"""
    def __init__(self, d_model=128, grid_h=41, grid_w=27, nhead=4, num_layers=2):
        super().__init__()
        self.HW = grid_h * grid_w
        
        # --- (a) 时域输入映射 ---
        self.p_proj = nn.Linear(3, d_model)
        self.aux_proj = nn.Linear(3, d_model)
        
        # --- (b) TCN 时域编码 ---
        self.p_tcn = nn.Sequential(
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=1),
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=2)
        )
        self.aux_tcn = nn.Sequential(
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=1),
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=2)
        )
        
        # --- (c) 跨模态物理对齐 (Cross-Attention) ---
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # --- (d) 空间拓扑解码与场生成 ---
        coords = self._create_mesh_grid(grid_h, grid_w)
        self.register_buffer("coords", coords)
        self.spatial_pe = nn.Sequential(
            nn.Linear(2, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*2)
        self.spatial_transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 输出为单点应力值
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

    def _create_mesh_grid(self, H, W):
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        ), dim=-1).view(-1, 2)
        return coords

    def forward(self, p_sparse, aux):
        B = p_sparse.size(0)
        
        p_feat = self.p_proj(p_sparse).unsqueeze(-1)   
        aux_feat = self.aux_proj(aux).unsqueeze(-1) 
        
        p_encoded = self.p_tcn(p_feat).transpose(1, 2)     
        aux_encoded = self.aux_tcn(aux_feat).transpose(1, 2) 
        
        attn_out, _ = self.cross_attn(query=p_encoded, key=aux_encoded, value=aux_encoded)
        fused_context = self.layer_norm(p_encoded + attn_out) 
        
        grid_tokens = self.spatial_pe(self.coords).unsqueeze(0).repeat(B, 1, 1) 
        st_tokens = grid_tokens + fused_context
        decoded_field = self.spatial_transformer(st_tokens) 
        
        out_flat = self.output_head(decoded_field).squeeze(-1) 
        return out_flat

# -----------------------------
# 2. 实例化模型与张量连通性自检 (带进度条)
# -----------------------------
print("🛠️ 开始构建并测试 MTC-Net 核心架构...")

build_steps = [
    "初始化时域编码器 (TCN)", 
    "构建跨模态注意力层 (Cross-Attention)", 
    "注入 2D 空间位置编码", 
    "搭建 Transformer 拓扑解码器", 
    "执行张量连通性自检 (Forward Pass)"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建装配进度条
with tqdm(total=len(build_steps), desc="🧠 网络装配进度", colour='magenta', ncols=90) as pbar:
    
    # 模拟装配过程的视觉反馈
    for i in range(4):
        pbar.set_postfix_str(build_steps[i])
        time.sleep(0.2) # 仅用于视觉展示装配过程
        pbar.update(1)
        
    model = MTCNet().to(device)
    
    # 第 5 步：真正的连通性测试
    pbar.set_postfix_str(build_steps[4])
    dummy_p = torch.randn(2, 3).to(device)
    dummy_aux = torch.randn(2, 3).to(device)
    
    # 阻断梯度计算以加速测试
    with torch.no_grad():
        out = model(dummy_p, dummy_aux)
        
    time.sleep(0.2)
    pbar.update(1)

print("\n" + "="*50)
print(f"🌟 MTC-Net 初始化完美完成！")
print(f"   ➤ 模型驻留设备: {device}")
print(f"   ➤ 总训练参数量: {sum(p.numel() for p in model.parameters()):,} 个")
print(f"✅ 前向传播自检成功！输出形状: {list(out.shape)} (预期为 [Batch, 1107])")
print("="*50)


# In[10]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

output_dir = r"C:\Users\HP\Desktop\MTC-Net_输出结果"
os.makedirs(output_dir, exist_ok=True)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

safe_batch_size = 32  

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=safe_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=safe_batch_size, shuffle=False)

# -----------------------------
# 1. 训练超参数配置
# -----------------------------
num_epochs = 10       
learning_rate = 5e-5

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 🛡️ 防爆修改 1：关闭 AMP 混合精度！Transformer 对精度很敏感，FP16 容易炸出 NaN
use_amp = False  

# -----------------------------
# 2. 防爆版引擎启动
# -----------------------------
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

print(f"\n🛡️ 挂载防爆盾！安全引擎启动！当前 Batch Size = {safe_batch_size}")

epoch_pbar = tqdm(range(num_epochs), desc="📈 总体训练进度", colour='green', ncols=100)

for epoch in epoch_pbar:
    
    model.train()
    train_loss = 0.0
    
    train_inner_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, colour='yellow', ncols=85, mininterval=1.0)
    
    for p_b, aux_b, y_b in train_inner_pbar:
        p_b, aux_b, y_b = p_b.to(device), aux_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        
        # 纯 32 位精度前向传播，稳如老狗
        outputs = model(p_b, aux_b)
        loss = criterion(outputs, y_b)
        
        # 🛡️ 防爆修改 2：实时拦截 NaN 病毒！
        if torch.isnan(loss):
            raise ValueError(f"🚨 警报！在 Epoch {epoch+1} 检测到 Loss 变为 NaN！训练被迫紧急停止。")
            
        loss.backward()
        
        # 🛡️ 防爆修改 3：收紧梯度裁剪 (把 1.0 降到 0.5，严防死守梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    scheduler.step()
    
    # [2] 验证阶段
    is_best = False
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p_b, aux_b, y_b in val_loader:
                p_b, aux_b, y_b = p_b.to(device), aux_b.to(device), y_b.to(device)
                outputs = model(p_b, aux_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        if epoch != 0 and len(val_losses) > 0:
            val_losses.extend([val_losses[-1]] * (4 if epoch != num_epochs -1 else 0))
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(output_dir, 'MTC_Net_Best_Weights.pth'))
            is_best = True
            
        status_str = f"Val MSE: {avg_val_loss:.5f} | Best: {best_val_loss:.5f}"
        if is_best:
            status_str += " ⭐"
        epoch_pbar.set_postfix_str(status_str)
        
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write(f"Epoch [{epoch+1:03d}/{num_epochs}] ┃ Train MSE: {avg_train_loss:.5f} ┃ Val MSE: {avg_val_loss:.5f} ┃ LR: {current_lr:.2e}")
    else:
        val_losses.append(val_losses[-1] if len(val_losses) > 0 else 0)

print("\n" + "="*50)
print(f"🎉 训练完美结束！无 NaN 感染！")
print(f"🏆 锁定的最佳验证集 Loss: {best_val_loss:.6f}")
print("="*50)

# -----------------------------
# 3. 渲染顶刊级 Loss 收敛曲线
# -----------------------------
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2.5, alpha=0.9)
ax.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2.5, alpha=0.9)
ax.set_title('MTC-Net Convergence Curve', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6, which='both')
ax.legend(loc='upper right', framealpha=0.9, fontsize=11, edgecolor='black')

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
curve_path = os.path.join(output_dir, 'Loss_Convergence_Curve.png')
plt.savefig(curve_path, bbox_inches='tight', facecolor='white')
plt.close()


# In[11]:


# -----------------------------
# 4. 离线持久化存储 (保存全量进度)
# -----------------------------
tensor_save_path = os.path.join(output_dir, 'Processed_Data_Bundle.pt')
print(f"\n💾 正在打包全量 Tensor 数据和 Scaler，准备转存至硬盘...")

try:
    data_bundle = {
        'P_train': P_train, 'Aux_train': Aux_train, 'y_train_t': y_train_t,
        'P_val': P_val, 'Aux_val': Aux_val, 'y_val_t': y_val_t,
        'P_test': P_test, 'Aux_test': Aux_test, 'y_test_t': y_test_t,
        'scaler_y': scaler_y  # 核心：反归一化必备！
    }
    torch.save(data_bundle, tensor_save_path, _use_new_zipfile_serialization=True)
    
    print("="*50)
    print(f"✅ 进度转存彻底成功！")
    print(f"📍 核心数据包: {tensor_save_path}")
    print(f"📍 最佳权重包: {os.path.join(output_dir, 'MTC_Net_Best_Weights.pth')}")
    print("💡 下次打开电脑，直接运行【一键恢复】脚本即可直接出图，无需再跑前面的代码！")
    print("="*50)
except Exception as e:
    print(f"❌ 保存失败，请检查: {e}")


# In[15]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心路径配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = r"C:\Users\HP\Desktop\MTC-Net_输出结果"

bundle_path = os.path.join(output_dir, 'Processed_Data_Bundle.pt')
weight_path = os.path.join(output_dir, 'MTC_Net_Best_Weights.pth')

# ==========================================
# 2. 声明 MTC-Net 核心架构
# ==========================================
class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    def forward(self, x):
        return F.relu(self.conv(x))[:, :, :-self.conv.padding[0]]

class MTCNet(nn.Module):
    def __init__(self, d_model=128, grid_h=41, grid_w=27, nhead=4, num_layers=2):
        super().__init__()
        self.HW = grid_h * grid_w
        self.p_proj = nn.Linear(3, d_model)
        self.aux_proj = nn.Linear(3, d_model)
        
        self.p_tcn = nn.Sequential(
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=1),
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=2)
        )
        self.aux_tcn = nn.Sequential(
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=1),
            CausalDilatedConv1d(d_model, d_model, kernel_size=3, dilation=2)
        )
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
        coords = self._create_mesh_grid(grid_h, grid_w)
        self.register_buffer("coords", coords)
        self.spatial_pe = nn.Sequential(
            nn.Linear(2, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model)
        )
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model*2)
        self.spatial_transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

    def _create_mesh_grid(self, H, W):
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        ), dim=-1).view(-1, 2)
        return coords

    def forward(self, p_sparse, aux):
        B = p_sparse.size(0)
        p_feat = self.p_proj(p_sparse).unsqueeze(-1)   
        aux_feat = self.aux_proj(aux).unsqueeze(-1) 
        
        p_encoded = self.p_tcn(p_feat).transpose(1, 2)     
        aux_encoded = self.aux_tcn(aux_feat).transpose(1, 2) 
        
        attn_out, _ = self.cross_attn(query=p_encoded, key=aux_encoded, value=aux_encoded)
        fused_context = self.layer_norm(p_encoded + attn_out) 
        
        grid_tokens = self.spatial_pe(self.coords).unsqueeze(0).repeat(B, 1, 1) 
        st_tokens = grid_tokens + fused_context
        decoded_field = self.spatial_transformer(st_tokens) 
        
        out_flat = self.output_head(decoded_field).squeeze(-1) 
        return out_flat

# ==========================================
# 3. 瞬间读取数据与模型权重
# ==========================================
print("🔄 正在从硬盘执行【纯净版一键恢复】...")
bundle = torch.load(bundle_path, weights_only=False)
P_test = bundle['P_test']
Aux_test = bundle['Aux_test']
y_test_t = bundle['y_test_t']
scaler_y = bundle['scaler_y']

model = MTCNet().to(device)
model.load_state_dict(torch.load(weight_path, weights_only=True))
model.eval()

# ==========================================
# ==========================================
# 4. 执行盲测推理 (防爆显存分批版)
# ==========================================
print("⚡ 正在执行盲测推演 (采用分批推理防爆显存)...")

outputs_list = []
batch_size = 32  # 每次只算 32 个时刻，4GB 显卡绝对安全

with torch.no_grad():
    # 对 Aux_test 进行全局 NaN 清洗
    Aux_test = torch.nan_to_num(Aux_test, nan=0.0) 
    
    # 循环切片，分批送入显卡
    for i in range(0, len(P_test), batch_size):
        # 取出一个 batch
        p_batch = P_test[i:i+batch_size].to(device)
        aux_batch = Aux_test[i:i+batch_size].to(device)
        
        # 前向传播算结果
        out_batch = model(p_batch, aux_batch)
        
        # 立刻把结果转回 CPU，释放宝贵的 GPU 显存
        outputs_list.append(out_batch.cpu())

# 把所有批次的结果拼接成一个完整的大张量
outputs = torch.cat(outputs_list, dim=0)

# 反归一化为真实的应力值 (kPa)
test_pred = scaler_y.inverse_transform(outputs.numpy())
test_true = scaler_y.inverse_transform(y_test_t.numpy())

print("="*50)
print("✅ 推理圆满成功！显存安全着陆。")
print("="*50)


# In[8]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from tqdm.auto import tqdm
import os
import torch

# ---------------------------------------------------------
# 1. 路径初始化与模型载入
# ---------------------------------------------------------
print("🔬 阶段一：执行盲测集推理与数据清洗...")

# 设置二级子文件夹
array_dir = os.path.join(output_dir, "Excel_Arrays")
img_dir = os.path.join(output_dir, "Stress_Fields")
os.makedirs(array_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# 载入权重
if best_model_state is not None:
    model.load_state_dict(best_model_state)
else:
    model.load_state_dict(torch.load(os.path.join(output_dir, 'MTC_Net_Best_Weights.pth'), weights_only=True))

model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for p_b, aux_b, y_b in tqdm(test_loader, desc="🧠 盲测推理", colour='cyan', ncols=90):
        p_b, aux_b = p_b.to(device), aux_b.to(device)
        
        # 🛡️ 内存补丁：实时清洗 Aux 特征中的 NaN
        aux_b = torch.nan_to_num(aux_b, nan=0.0)
        
        outputs = model(p_b, aux_b)
        test_preds.append(outputs.cpu())
        test_targets.append(y_b.cpu())

test_pred = torch.cat(test_preds, dim=0).numpy()
test_target = torch.cat(test_targets, dim=0).numpy()

# 自检与清洗
test_target = np.nan_to_num(test_target, nan=0.0)
if np.isnan(test_pred).any():
    raise ValueError("🚨 错误：预测结果仍含有 NaN，请检查模型权重状态！")

# 反归一化
test_pred_denorm = scaler_y.inverse_transform(test_pred)
test_target_denorm = scaler_y.inverse_transform(test_target)

# 计算指标
test_mae = mean_absolute_error(test_target_denorm, test_pred_denorm)
test_rmse = np.sqrt(mean_squared_error(test_target_denorm, test_pred_denorm))
test_r2 = r2_score(test_target_denorm, test_pred_denorm)

print("\n" + "="*50)
print(f"🏆 盲测性能: MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")
print(f"📂 输出目录: {output_dir}")
print("="*50 + "\n")

# ---------------------------------------------------------
# 2. 500 DPI 高清云图渲染与 Excel 导出
# ---------------------------------------------------------
print("🎨 阶段二：正在生成 500 DPI 高清对比图并导出数据矩阵...")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

total_test_samples = len(test_pred_denorm)
sample_num = min(50, total_test_samples)
sample_indices = np.linspace(0, total_test_samples - 1, sample_num, dtype=int)

# 统一色标，确保对比严谨
global_vmin = min(test_pred_denorm.min(), test_target_denorm.min())
global_vmax = max(test_pred_denorm.max(), test_target_denorm.max())

for i, idx in enumerate(tqdm(sample_indices, desc="🖼️ 渲染中", colour='magenta', ncols=90)):
    # 场数据重构 (41x27)
    pred_field = test_pred_denorm[idx].reshape(41, 27)
    true_field = test_target_denorm[idx].reshape(41, 27)
    
    # 翻转以对齐物理坐标轴
    pred_field = np.flipud(pred_field)
    true_field = np.flipud(true_field)
    error_field = true_field - pred_field
    abs_error_field = np.abs(error_field)
    
    # --- A. 导出 Excel (无索引无表头纯数据) ---
    pd.DataFrame(pred_field).to_excel(os.path.join(array_dir, f'Pred_Field_{i+1:02d}_Step{idx}.xlsx'), index=False, header=False)
    pd.DataFrame(true_field).to_excel(os.path.join(array_dir, f'True_Field_{i+1:02d}_Step{idx}.xlsx'), index=False, header=False)
    pd.DataFrame(error_field).to_excel(os.path.join(array_dir, f'Error_Field_{i+1:02d}_Step{idx}.xlsx'), index=False, header=False)
    
    # --- B. 500 DPI 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5)) # 此时 figsize 只是比例
    
    # 真值
    im0 = axes[0].imshow(true_field, cmap='RdBu_r', vmin=global_vmin, vmax=global_vmax, extent=[0, 130, 0, 200], interpolation='bilinear')
    axes[0].set_title('Real Data (Ground Truth)', fontsize=16, fontweight='bold', pad=12)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label('Stress (kPa)', fontsize=12)
    
    # MTC-Net 重构
    im1 = axes[1].imshow(pred_field, cmap='RdBu_r', vmin=global_vmin, vmax=global_vmax, extent=[0, 130, 0, 200], interpolation='bilinear')
    axes[1].set_title('MTC-Net Estimated', fontsize=16, fontweight='bold', pad=12)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label('Stress (kPa)', fontsize=12)
    
    # 绝对误差 (保持经典的红色系)
    im2 = axes[2].imshow(abs_error_field, cmap='Reds', extent=[0, 130, 0, 200], interpolation='bilinear')
    axes[2].set_title('Absolute Error Map', fontsize=16, fontweight='bold', pad=12)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label('Error (kPa)', fontsize=12)
    
    # 美化坐标轴与加边框
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.add_patch(patches.Rectangle((0, 0), 130, 200, linewidth=1.5, edgecolor='black', facecolor='none'))
        
    plt.tight_layout()
    
    # 保存图片：关键点 dpi=500
    save_name = f'Field_Comparison_{i+1:02d}_Step{idx}.png'
    plt.savefig(os.path.join(img_dir, save_name), dpi=500, bbox_inches='tight', facecolor='white')
    
    plt.close(fig) # 及时关闭图形释放内存

print("\n" + "="*50)
print(f"🎉 处理完成！请检查文件夹：\n1. 数据矩阵: {array_dir}\n2. 高清图片(500DPI): {img_dir}")
print("="*50)

