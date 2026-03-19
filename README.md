# Biology-guided Multi-view Semantic- and Relation-aware Heterogeneous Graph Neural Network for Drug-Disease Association Prediction

## 1. 当前目录下保留的核心文件

- `run.py`：主流程入口
- `pipeline_utils.py`：数据处理、相似矩阵构建、异构图构建、参考资产并入
- `msrhgnn_model.py`：模型主体
- `train_model.py`：训练入口
- `requirements.txt`：依赖列表

参考目录中真正需要的资产已经迁移到主目录：

- `data/reference_graph_builder/processed.zip`
- `data/reference_graph_builder/final_hetero_data_raw.pt`

所以当前项目已经可以独立运行。

---

## 2. 环境要求

建议：

- Python 3.10 或 3.11
- pip 最新版

---

## 3. 环境安装命令

### 3.1 创建虚拟环境

#### Windows PowerShell
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.2 升级 pip

```bash
python -m pip install --upgrade pip
```

### 3.3 安装 PyTorch

#### CPU 版本
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 如果你有 CUDA
请按你本机 CUDA 版本去装对应的 PyTorch。  

### 3.4 安装项目依赖

```bash
pip install -r requirements.txt
```

### 3.5 单独确保这两个包装好

因为主流程要读取：

- `data/reference_graph_builder/final_hetero_data_raw.pt`

所以必须确保：

```bash
pip install torch-geometric
```

如果你要启用 BioBERT 文本编码，而不是 `tfidf_svd` fallback，再安装：

```bash
pip install transformers
```

### 3.6 一套完整环境安装命令

#### Windows PowerShell
```bash
cd E:\2023\校创2025\模型
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install torch-geometric
pip install transformers
```

---

## 4. 主流程做什么

`run.py` 会完成这些事：

1. 读取当前目录下原始数据
2. 构建 drug / disease / gene 特征
3. 构建 3 个 drug similarity view
4. 构建 3 个 disease similarity view
5. 构建 drug-target-gene / disease-gene 等异构关系
6. 自动并入 `data/reference_graph_builder/processed.zip`
7. 自动尝试并入 `data/reference_graph_builder/final_hetero_data_raw.pt`
8. 生成最终图：
   - `data/final/final_graph_data.pt`

---

## 5. 当前模型结构

`msrhgnn_model.py` 已按你给的要求实现。

### Step 1：多源相似矩阵

#### Disease 3 views
- `disease__disimnet_o__disease`
- `disease__disimnet_h__disease`
- `disease__disimnet_g__disease`

#### Drug 3 views
- `drug__drugsim_morgan__drug`
- `drug__drugsim_gip__drug`
- `drug__drugsim_drsie__drug`

### Step 2：自适应融合 + Graph Transformer

每个视图：

- 先过 `WeightedGINLayer`
- 再过 `MultiPoolAdaptiveFusion`
  - GAP
  - GMP
  - GSP
- 再过 `SparseGraphTransformerLayer`

### Step 3：异构视图

#### Low-order
- drug-target-gene
- disease-gene
- train drug-disease

#### High-order
- Drug-Protein-Disease
- Disease-Protein-Disease
- Disease-Protein-Protein-Disease

### Step 4：Gate + Concat + Bilinear

- gate 融合局部/全局异构特征
- similarity view 与 heterogeneous view 拼接
- bilinear decoder 输出 drug-disease score

---

## 6. 如何运行

### 6.1 先构图

推荐先跑：

```bash
python run.py --skip-gene-network
```

说明：

- `--skip-gene-network`：跳过从超大 STRING 原始文件重新构边
- 即使跳过，主流程仍会自动读取：
  - `data/reference_graph_builder/processed.zip`
  - `data/reference_graph_builder/final_hetero_data_raw.pt`

### 6.2 查看构图报告

重点看：

- `data/reports/validation_report.json`
- `data/reports/reference_processed_report.json`
- `data/reports/reference_pt_report.json`
- `data/reports/run_summary.json`

### 6.3 训练

#### 冒烟测试
```bash
python train_model.py --graph data/final/final_graph_data.pt --epochs 1 --eval-every 1 --out-dir artifacts/smoke_test
```

#### 正式训练
```bash
python train_model.py --graph data/final/final_graph_data.pt --epochs 100 --hidden-dim 128 --lr 1e-3 --dropout 0.2 --out-dir artifacts/train_run
```

---

## 7. 输出文件

### 构图输出

- `data/processed/`
- `data/final/final_graph_data.pt`
- `data/reports/validation_report.json`
- `data/reports/reference_processed_report.json`
- `data/reports/reference_pt_report.json`
- `data/reports/run_summary.json`

### 训练输出

- `artifacts/train_run/model.pt`
- `artifacts/train_run/summary.json`
- `artifacts/train_run/history.json`

---

## 8. 当前已知边界

1. **OMIM 原始文本缺失**
   - 当前用 CTD disease text / definition 替代

2. **`string_edges.csv` 当前为空**
   - 所以目前不会得到有效的 `gene_network_edges.csv`

3. **如果没装 `torch_geometric`**
   - 主流程无法真正读取 `final_hetero_data_raw.pt`

---

## 9. 推荐直接执行的完整命令

```bash
cd E:\2023\校创2025\模型
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install torch-geometric
pip install transformers
python run.py --skip-gene-network
python train_model.py --graph data/final/final_graph_data.pt --epochs 100 --hidden-dim 128 --lr 1e-3 --dropout 0.2 --out-dir artifacts/train_run
```

---

## 10. 最终结论

现在这个项目目录已经是一个独立可运行版本：

- 不再依赖已删除的参考目录
- 运行时要用的参考资产已经迁移到：
  - `data/reference_graph_builder/`

你现在只需要：

1. 配环境
2. 装依赖
3. 跑 `run.py`
4. 跑 `train_model.py`

即可。
