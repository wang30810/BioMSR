import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/hy-tmp/else/data/data"
PT_PATH = "/hy-tmp/else/processed/final_hetero_data_strict.pt"
K_NEIGHBORS = 5

def rescue():
    print("🚑 开始终极抢救 G 视图网络 (sim_g)...")
    
    # 1. 提取所有关联的纯净 Gene IDs
    print("  -> 加载 CTD 疾病基因关联...")
    # 【核心修复】强制覆写列名，绝不给 Pandas 报错的机会！
    cols_g = ['GeneSymbol', 'GeneID', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceChemicalName', 'InferenceScore', 'OmimIDs', 'PubMedIDs']
    df_dg = pd.read_csv(os.path.join(DATA_DIR, "CTD_curated_genes_diseases.csv.gz"), compression='gzip', comment='#', header=None, names=cols_g, low_memory=False)
        
    df_dg = df_dg[df_dg['DirectEvidence'].str.contains('marker|mechanism', na=False, case=False)].dropna(subset=['GeneID'])
    # 彻底洗净：去除 .0 和空格
    df_dg['GeneID_clean'] = df_dg['GeneID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    dis_genes = df_dg.groupby('DiseaseID')['GeneID_clean'].apply(set).to_dict()

    # 2. 提取极其纯净的 STRING 映射 (绝招：只取纯数字的 alias)
    print("  -> 加载 STRING 别名表 (使用纯数字过滤绝招)...")
    df_alias = pd.read_csv(os.path.join(DATA_DIR, "9606.protein.aliases.v12.0.txt.gz"), sep='\t', compression='gzip', header=None, comment='#', names=['string_id', 'alias', 'source'])
    # Entrez ID 全部是纯数字！这招完美避开 source 命名不规范的问题
    mask = df_alias['string_id'].str.startswith('9606.') & df_alias['alias'].astype(str).str.isnumeric()
    ensp_to_gene = dict(zip(df_alias.loc[mask, 'string_id'], df_alias.loc[mask, 'alias'].astype(str)))
    del df_alias
    print(f"  ✅ 提取到纯数字基因映射: {len(ensp_to_gene)} 个")

    # 3. 提取高质量边
    print("  -> 加载 STRING 互作网络...")
    df_links = pd.read_csv(os.path.join(DATA_DIR, "9606.protein.links.v12.0.txt.gz"), sep=' ', compression='gzip', engine='c')
    df_links = df_links[df_links['combined_score'] >= 700].copy()
    df_links['gene1'] = df_links['protein1'].map(ensp_to_gene)
    df_links['gene2'] = df_links['protein2'].map(ensp_to_gene)
    df_links = df_links.dropna(subset=['gene1', 'gene2'])
    print(f"  ✅ 映射成功的有效互作边: {len(df_links)} 条")

    g_sim = defaultdict(dict)
    # 用 zip 加速读取，拒绝卡顿
    for g1, g2, score in zip(df_links['gene1'].values, df_links['gene2'].values, df_links['combined_score'].values):
        s = score / 1000.0
        g_sim[g1][g2] = g_sim[g2][g1] = s

    # 4. 对齐疾病顺序
    data = torch.load(PT_PATH, weights_only=False)
    cols = ['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings']
    df_dis = pd.read_csv(os.path.join(DATA_DIR, "CTD_diseases.csv"), sep=',', quoting=1, dtype=str, comment='#', header=None, names=cols)
    df_dis.fillna('', inplace=True)
    DISEASE_KEYWORDS = ["Alzheimer", "Parkinson", "Huntington", "ALS", "amyotrophic lateral sclerosis", "schizophrenia", "depression", "bipolar", "dementia", "Lewy body", "multiple sclerosis", "epilepsy", "autism", "ADHD", "anxiety", "OCD", "panic disorder", "PTSD", "major depressive", "manic"]
    mask_dis = df_dis['DiseaseName'].str.contains('|'.join(DISEASE_KEYWORDS), case=False, na=False)
    disease_ids = df_dis[mask_dis]['DiseaseID'].tolist()
    
    n = len(disease_ids)
    sim_gene = np.zeros((n, n))
    
    # 5. 计算相似度 (修复相同基因相互不认识的 Bug)
    print("  -> 重织疾病 G 视图相似度...")
    for i, d1 in enumerate(tqdm(disease_ids)):
        g1s = dis_genes.get(d1, set())
        for j, d2 in enumerate(disease_ids):
            if i == j: sim_gene[i, j] = 1.0; continue
            if i > j: continue
            
            g2s = dis_genes.get(d2, set())
            if not g1s or not g2s: continue
            
            # 【关键修复】g == h 时强制给 1.0 相似度
            sum1 = sum([max([1.0 if g == h else g_sim.get(g, {}).get(h, 0.0) for h in g2s] + [0]) for g in g1s])
            sum2 = sum([max([1.0 if g == h else g_sim.get(h, {}).get(g, 0.0) for g in g1s] + [0]) for h in g2s])
            
            sim_gene[i, j] = sim_gene[j, i] = (sum1 + sum2) / (len(g1s) + len(g2s))

    print(f"  📊 稀疏化前，存在基因联系的疾病对数量: {(sim_gene > 0).sum()} / {n*n}")

    # 稀疏化
    for i in range(n):
        idx = np.argsort(sim_gene[i])[-K_NEIGHBORS-1:-1]
        mask_idx = np.zeros(n, dtype=bool); mask_idx[idx] = True
        sim_gene[i, ~mask_idx] = 0.0

    final_edges = (sim_gene > 0).sum()
    print(f"  📊 稀疏化后，成功保留边数量: {final_edges}")

    # 6. 保存回 PyG
    g_idx = np.nonzero(sim_gene)
    data['disease', 'sim_g', 'disease'].edge_index = torch.tensor(np.stack(g_idx), dtype=torch.long)
    data['disease', 'sim_g', 'disease'].edge_attr = torch.tensor(sim_gene[g_idx], dtype=torch.float).unsqueeze(1)
    
    torch.save(data, PT_PATH)
    print("\n🎉 抢救成功！G视图已完美融入图对象！")

if __name__ == "__main__":
    rescue()