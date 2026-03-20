# Biology-guided Multi-view Semantic- and Relation-aware Heterogeneous Graph Neural Network for Drug-Disease Association Prediction

## 1. 项目文件

主流程只依赖以下文件：

- `run.py`：数据处理与最终图构建入口
- `pipeline_utils.py`：数据处理、相似度构建、图构建工具函数
- `msrhgnn_model.py`：MSRHGNN 风格主模型
- `train_model.py`：训练与评估脚本
- `predict_candidates.py`：给指定疾病输出候选药物 Top-K
- `requirements.txt`：依赖列表
- `data/`：数据、处理结果、报告
- `artifacts/`：训练结果与预测结果
- `others/`：历史文件，不参与主流程

已并入主流程的参考资产：

- `data/reference_graph_builder/processed.zip`
- `data/reference_graph_builder/final_hetero_data_raw.pt`

---

## 2. 环境要求

建议（我使用的）：

- Pytorch 2.0.0 ；
- python 3.8；
- ubuntu 20.04;
- cuda 11.8;

---

## 3. 安装步骤

### 3.1 创建虚拟环境

按上面配置

### 3.2 安装项目依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 4. 数据构建

### 4.1 构建主流程数据

如果当前没有 STRING / HumanNet 原始网络文件，先运行：

```bash
python run.py --skip-gene-network
```

这一步会完成：

1. DrugBank 药物解析
2. drug / disease / gene 特征构建
3. 3 个 drug similarity views 构建
4. 3 个 disease similarity views 构建
5. drug-target-gene / disease-gene 关系构建
6. 参考 `processed.zip` / `final_hetero_data_raw.pt` 中可用内容合并
7. 最终异构图导出

核心输出：

- `data/final/final_graph_data.pt`

### 4.2 如果要补齐完整高阶视图

需要额外提供：

- `data/processed/gene_network_edges.csv`

格式至少包含三列：

- `gene1`
- `gene2`
- `weight`

只要这个文件存在并且非空，主流程会自动并入：

- `gene__interacts__gene`
- Disease-Protein-Protein-Disease 高阶路径

如果没有该文件，代码可以运行，但高阶视图是**部分实现**。

---

## 5. 模型结构

`msrhgnn_model.py` 当前实现包括：

### 5.1 相似性视图

#### Disease 3 views

- `disease__disimnet_o__disease`
- `disease__disimnet_h__disease`
- `disease__disimnet_g__disease`

#### Drug 3 views

- `drug__drugsim_morgan__drug`
- `drug__drugsim_gip__drug`
- `drug__drugsim_drsie__drug`

### 5.2 视图编码

- `WeightedGINLayer`
- `MultiPoolAdaptiveFusion`
  - GAP
  - GMP
  - GSP
- `SparseGraphTransformerLayer`

### 5.3 异构关系建模

#### Low-order

- drug-target-gene
- disease-gene
- train-only drug-disease

#### High-order

- Drug-Protein-Disease
- Disease-Protein-Disease
- Disease-Protein-Protein-Disease

### 5.4 输出层

- gate fusion
- concat
- bilinear decoder

### 5.5 已修复的问题

训练阶段已修复标签泄露：

- `DrugSim_GIP`
- `DrugSim_DRSIE`
- `DiSimNet_G` 的 `shared_drug_jaccard_fallback`

前两个关系会在训练时基于 **train split** 动态重建，而不是直接用全量正样本构图。
当缺失 `disease_gene_edges.csv` 时，`DiSimNet_G` 现在会安全禁用，不再回退到基于全量 `drug_disease_edges.csv` 的相似图。

---

## 6. 训练

### 6.1 冒烟测试

```bash
python train_model.py --graph data/final/final_graph_data.pt --epochs 1 --eval-every 1 --out-dir artifacts/smoke_test
```

### 6.2 正常训练

```bash
python train_model.py --graph data/final/final_graph_data.pt --epochs 100 --hidden-dim 128 --lr 1e-3 --dropout 0.2 --out-dir artifacts/train_run
```

训练输出：

- `artifacts/train_run/model.pt`
- `artifacts/train_run/summary.json`
- `artifacts/train_run/history.json`

---

## 7. 候选药物预测

新增脚本：`predict_candidates.py`

作用：

- 加载训练好的模型
- 复用训练时的 train-only GIP / DRSIE 重建逻辑
- 重建 metapath 关系
- 针对指定疾病输出 Top-K 候选药物
- 默认排除已知 drug-disease 关联，只保留新候选药物

### 7.1 示例：输出 Alzheimer Disease 的前 10 个候选药物

```bash
python predict_candidates.py --disease "Alzheimer Disease" --top-k 10 --out artifacts/predictions/alz_top10.csv
```

### 7.2 指定模型文件

```bash
python predict_candidates.py --model artifacts/train_run/model.pt --disease "Alzheimer Disease" --top-k 10 --out artifacts/predictions/alz_top10.csv
```

### 7.3 按 DiseaseID 查询

```bash
python predict_candidates.py --disease MESH:D000544 --top-k 10 --out artifacts/predictions/alz_top10.csv
```

### 7.4 输出列

输出 CSV 包含：

- `rank`
- `drugbank_id`
- `drug_name`
- `disease_id`
- `disease_name`
- `score`
- `logit`
- `num_target_genes`
- `target_genes`

---

## 8. 结果文件

### 8.1 数据与图

- `data/processed/`
- `data/final/final_graph_data.pt`
- `data/reports/validation_report.json`
- `data/reports/reference_processed_report.json`
- `data/reports/reference_pt_report.json`
- `data/reports/run_summary.json`

### 8.2 训练结果

- `artifacts/train_run/model.pt`
- `artifacts/train_run/summary.json`
- `artifacts/train_run/history.json`

### 8.3 候选药物预测结果

- `artifacts/predictions/*.csv`

---

## 9. 当前边界

1. OMIM 当前由 CTD 文本替代，能跑通，但语义来源不是完全等价。
2. 现有 `processed.zip` 与 `final_hetero_data_raw.pt` 不包含 `gene__interacts__gene`。
3. 因此，仅靠现有 pt / zip 资产，**不能自动补齐完整 PPI 高阶分支**。
4. 当前代码已经把完整高阶逻辑写好；真正缺的是可用的 gene-gene / PPI 数据。
5. 没有 `gene_network_edges.csv` 时，模型仍可训练，但属于**部分高阶视图版本**。

---

## 10. 如何检查当前状态

重点看两个文件：

- `data/reports/validation_report.json`
- `artifacts/train_run/summary.json`

现在会明确给出：

- high-order 是否 complete
- 缺的是不是 `gene__interacts__gene`
- 当前激活了哪些 metapath relations

---

## 11. 从环境到预测的一条完整流程

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install torch-geometric
python run.py --skip-gene-network
python train_model.py --graph data/final/final_graph_data.pt --epochs 100 --hidden-dim 128 --lr 1e-3 --dropout 0.2 --out-dir artifacts/train_run
python predict_candidates.py --model artifacts/train_run/model.pt --disease "Alzheimer Disease" --top-k 10 --out artifacts/predictions/alz_top10.csv
```
