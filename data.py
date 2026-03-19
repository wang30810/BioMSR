#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import obonet
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import HeteroData
import warnings
warnings.filterwarnings('ignore')

# ==================== 绝对路径配置 ====================
DATA_DIR = "/hy-tmp/else/data/data"          
OUTPUT_DIR = "/hy-tmp/else/processed"        
os.makedirs(OUTPUT_DIR, exist_ok=True)

CTD_DISEASES = os.path.join(DATA_DIR, "CTD_diseases.csv")
CTD_CHEM_DISEASE = os.path.join(DATA_DIR, "CTD_chemicals_diseases.csv")
CTD_GENE_DISEASE = os.path.join(DATA_DIR, "CTD_curated_genes_diseases.csv.gz") 
DRUGBANK_XML = os.path.join(DATA_DIR, "full database.xml")
HPOA_FILE = os.path.join(DATA_DIR, "phenotype.hpoa")
HP_OBO = os.path.join(DATA_DIR, "hp.obo")
UNIPROT2ENTREZ = os.path.join(DATA_DIR, "id_mapping.csv") 
STRING_FILE = os.path.join(DATA_DIR, "9606.protein.links.v12.0.txt.gz")
STRING_ALIAS = os.path.join(DATA_DIR, "9606.protein.aliases.v12.0.txt.gz")

# ==================== 配置参数 ====================
DISEASE_KEYWORDS = [
    "Alzheimer", "Parkinson", "Huntington", "ALS", "amyotrophic lateral sclerosis",
    "schizophrenia", "depression", "bipolar", "dementia", "Lewy body",
    "multiple sclerosis", "epilepsy", "autism", "ADHD", "anxiety", "OCD",
    "panic disorder", "PTSD", "major depressive", "manic"
]
K_NEIGHBORS = 5  
DRUG_FP_BITS = 1024        
DISEASE_FEAT_DIM = 768     
GENE_FEAT_DIM = 200

# ==================== 0. 辅助函数 ====================
def check_files(file_list):
    missing = [f for f in file_list if not os.path.exists(f)]
    if missing:
        print("⚠️ 警告：以下文件缺失，请检查路径：")
        for m in missing: print("  ", m)
        return False
    return True

# ==================== 1. 疾病筛选 ====================
def filter_diseases():
    print("\n步骤1: 筛选神经精神类疾病...")
    cols = ['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings']
    df = pd.read_csv(CTD_DISEASES, sep=',', quoting=1, dtype=str, comment='#', header=None, names=cols)
    df.fillna('', inplace=True)
    mask = df['DiseaseName'].str.contains('|'.join(DISEASE_KEYWORDS), case=False, na=False)
    selected = df[mask].copy()
    
    def extract_omim(alt_ids):
        if pd.isna(alt_ids) or alt_ids == '': return None
        for part in str(alt_ids).split('|'):
            if part.startswith('OMIM:'): return part.strip()
        return None
    selected['OMIM_ID'] = selected['AltDiseaseIDs'].apply(extract_omim)
    print(f"  ✅ 筛选出 {len(selected)} 种疾病")
    return selected

# ==================== 2. 药物提取 ====================
def parse_drugbank_xml():
    print("\n步骤2: 解析DrugBank XML...")
    tree = ET.parse(DRUGBANK_XML)
    root = tree.getroot()
    ns = {'db': 'http://www.drugbank.ca'}
    drugs = []
    for drug in tqdm(root.findall('db:drug', ns), desc="解析药物"):
        db_id = drug.findtext('db:drugbank-id[@primary="true"]', namespaces=ns)
        name = drug.findtext('db:name', namespaces=ns)
        smiles = next((p.findtext('db:value', namespaces=ns) for p in drug.findall('.//db:calculated-properties/db:property', ns) if p.findtext('db:kind', namespaces=ns) == 'SMILES'), None)
        targets = {ext.findtext('db:identifier', namespaces=ns) for t in drug.findall('.//db:target/db:polypeptide', ns) for ext in t.findall('.//db:external-identifier', ns) if ext.findtext('db:resource', namespaces=ns) == 'UniProtKB'}
        mesh_id = next((ext.findtext('db:identifier', namespaces=ns) for ext in drug.findall('.//db:external-identifiers/db:external-identifier', ns) if ext.findtext('db:resource', namespaces=ns) == 'MeSH'), None)
        drugs.append({'DrugBank_ID': db_id, 'name': name, 'SMILES': smiles, 'Targets_UniProt': '|'.join(filter(None, targets)), 'MeSH_ID': mesh_id})
    df = pd.DataFrame(drugs)
    print(f"  ✅ 提取到 {len(df)} 种药物")
    return df

# ==================== 3. 极速映射 ====================
def load_uniprot_entrez_map(file_path):
    print("\n步骤3: 加载 UniProt->Entrez 映射...")
    if not os.path.exists(file_path): return {}
    try:
        df = pd.read_csv(file_path, dtype=str, sep=None, engine='python')
        if len(df.columns) < 2: return {}
        src_col, tgt_col = df.columns[0], df.columns[1]
        df = df.dropna(subset=[src_col, tgt_col])
        mapping = {str(s).strip(): str(t).strip() for s, t in zip(df[src_col], df[tgt_col]) if str(s) != 'nan' and str(t) != 'nan'}
        print(f"  ✅ 成功加载了 {len(mapping)} 个映射")
        return mapping
    except Exception as e:
        print(f"  ⚠️ 映射读取跳过: {e}"); return {}

def map_drug_targets(drug_df, u2e_map):
    if not u2e_map: return drug_df
    drug_df['Targets_Entrez'] = drug_df['Targets_UniProt'].apply(lambda x: '|'.join(set([u2e_map[u] for u in str(x).split('|') if u in u2e_map])) if pd.notna(x) else '')
    return drug_df

# ==================== 4. 提取关联边 ====================
def load_ctd_chem_disease(disease_df, drug_df):
    print("\n步骤4.1: 提取药物-疾病关联 (启用双重匹配)...")
    cols = ['ChemicalName', 'ChemicalID', 'CasRN', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceGeneSymbol', 'OmimIDs', 'PubMedIDs']
    ctd_cd = pd.read_csv(CTD_CHEM_DISEASE, comment='#', header=None, names=cols, low_memory=False)
    mesh2db = {str(m).replace('MESH:', '').strip(): d for m, d in zip(drug_df['MeSH_ID'], drug_df['DrugBank_ID']) if pd.notna(m)}
    name2db = {str(n).lower().strip(): d for n, d in zip(drug_df['name'], drug_df['DrugBank_ID']) if pd.notna(n)}
    ctd_cd = ctd_cd[ctd_cd['DiseaseID'].isin(disease_df['DiseaseID'])]
    ctd_cd['ChemicalID_clean'] = ctd_cd['ChemicalID'].astype(str).str.replace('MESH:', '').str.strip()
    ctd_cd['DrugBank_ID'] = ctd_cd['ChemicalID_clean'].map(mesh2db)
    unmapped = ctd_cd['DrugBank_ID'].isna()
    ctd_cd.loc[unmapped, 'DrugBank_ID'] = ctd_cd.loc[unmapped, 'ChemicalName'].astype(str).str.lower().str.strip().map(name2db)
    df_mapped = ctd_cd[ctd_cd['DrugBank_ID'].notna()].copy()
    print(f"  ✅ 成功提取并映射了 {len(df_mapped)} 条有效关联边")
    return df_mapped

def load_ctd_gene_disease(disease_df):
    print("\n步骤4.2: 提取疾病-基因关联 (防KeyError版)...")
    cols = ['GeneSymbol', 'GeneID', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceChemicalName', 'InferenceScore', 'OmimIDs', 'PubMedIDs']
    df = pd.read_csv(CTD_GENE_DISEASE, compression='gzip', comment='#', header=None, names=cols, low_memory=False)
    df = df[df['DirectEvidence'].str.contains('marker|mechanism', na=False, case=False)]
    df_filtered = df[df['DiseaseID'].isin(disease_df['DiseaseID'])].copy()
    df_filtered = df_filtered[df_filtered['GeneID'].notna()]
    print(f"  ✅ 成功提取了 {len(df_filtered)} 条强致病关联边")
    return df_filtered

# ==================== 5. STRING 网络 (带完美进度条版) ====================
def load_string_links(link_file, alias_file):
    print("\n步骤5: 处理 STRING 互作网络 (极限防爆内存 + 进度条版)...")
    
    # 1. 加载别名表
    df_alias = pd.read_csv(alias_file, sep='\t', compression='gzip', header=None, comment='#', names=['string_id', 'alias', 'source'])
    mask = df_alias['string_id'].str.startswith('9606.') & df_alias['source'].str.contains('GeneID|Entrez', na=False)
    ensp_to_gene = dict(zip(df_alias.loc[mask, 'string_id'], df_alias.loc[mask, 'alias'].astype(str)))
    
    # 彻底释放别名表占用的内存
    del df_alias 

    print("  -> 正在极速扫描网络总行数以生成进度条...")
    with gzip.open(link_file, 'rt') as f:
        total_lines = sum(1 for _ in f) - 1 # 减去表头
    
    chunk_size = 100000
    total_chunks = (total_lines // chunk_size) + 1

    print(f"  -> 网络共计 {total_lines} 行，开始分块提纯...")
    edges_list = []
    
    # chunksize 降至 10 万行，保证峰值内存极小
    chunk_iter = pd.read_csv(link_file, sep=' ', compression='gzip', engine='c', chunksize=chunk_size)
    
    # 这里加上了 total=total_chunks，进度条就能完美显示百分比了！
    for chunk in tqdm(chunk_iter, total=total_chunks, desc="  🔗 提取 STRING 边"):
        # 第一时间扔掉低分垃圾数据，释放内存
        chunk = chunk[chunk['combined_score'] >= 700]
        
        if chunk.empty:
            continue
            
        # 映射 ID
        chunk['gene1'] = chunk['protein1'].map(ensp_to_gene)
        chunk['gene2'] = chunk['protein2'].map(ensp_to_gene)
        chunk = chunk.dropna(subset=['gene1', 'gene2'])
        
        if chunk.empty:
            continue
            
        # 极其省内存的去重方式：强制较小的 ID 放前面
        g1 = np.where(chunk['gene1'] < chunk['gene2'], chunk['gene1'], chunk['gene2'])
        g2 = np.where(chunk['gene1'] < chunk['gene2'], chunk['gene2'], chunk['gene1'])
        
        clean_chunk = pd.DataFrame({
            'gene1': g1,
            'gene2': g2,
            'score': chunk['combined_score'].values / 1000.0
        })
        
        # 块内去重
        clean_chunk = clean_chunk.drop_duplicates(subset=['gene1', 'gene2'])
        edges_list.append(clean_chunk)

    # 将所有精华拼装起来
    if edges_list:
        df_links = pd.concat(edges_list, ignore_index=True)
        # 全局最后去重一次
        df_links = df_links.drop_duplicates(subset=['gene1', 'gene2'])
        print(f"  ✅ 成功提取 {len(df_links)} 条高质量 STRING 互作边，内存安然无恙！")
        return df_links
        
    return None

# ==================== 6. HPO 相似度依赖 ====================
def prepare_hpo(disease_df):
    print("\n步骤6: 预计算 HPO 相似度矩阵依赖...")
    if not os.path.exists(HP_OBO) or not os.path.exists(HPOA_FILE):
        print("  ⚠️ 缺少 HPO 文件，跳过。")
        return {}, {}, {}, {}
    hpo_graph = obonet.read_obo(HP_OBO)
    disease_to_hpo = defaultdict(set)
    with open(HPOA_FILE, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) >= 4: disease_to_hpo[parts[0]].add(parts[3])
    hpo_count = Counter(hpo for hpos in disease_to_hpo.values() for hpo in hpos)
    children = defaultdict(set)
    for n in hpo_graph.nodes():
        for p in hpo_graph.successors(n): children[p].add(n)
    freq = {}
    def dfs(n):
        if n in freq: return freq[n]
        tot = hpo_count.get(n, 0) + sum(dfs(c) for c in children.get(n, []))
        freq[n] = tot
        return tot
    for n in hpo_graph.nodes(): dfs(n)
    tot = freq.get('HP:0000001', len(disease_to_hpo))
    ic_dict = {hpo: -np.log(f/tot) for hpo, f in freq.items() if f > 0}
    ancestors = {n: set(nx.ancestors(hpo_graph, n)) | {n} for n in hpo_graph.nodes()}
    omim_map = {row['DiseaseID']: row.get('OMIM_ID') for _, row in disease_df.iterrows() if pd.notna(row.get('OMIM_ID'))}
    print("  ✅ HPO 依赖计算完成！")
    return disease_to_hpo, ic_dict, ancestors, omim_map

def disease_hpo_similarity(d1_hpos, d2_hpos, ancestors, ic_dict):
    if not d1_hpos or not d2_hpos: return 0.0
    def get_max_ic(hpos1, hpos2):
        max_score = 0.0
        for t1 in hpos1:
            anc1 = ancestors.get(t1, {t1})
            for t2 in hpos2:
                common = anc1 & ancestors.get(t2, {t2})
                if common: max_score = max(max_score, max([ic_dict.get(c, 0) for c in common]))
        return max_score
    max_cross = get_max_ic(d1_hpos, d2_hpos)
    max_self1 = get_max_ic(d1_hpos, d1_hpos)
    max_self2 = get_max_ic(d2_hpos, d2_hpos)
    if max_self1 + max_self2 == 0: return 0.0
    return (2 * max_cross) / (max_self1 + max_self2)

# ==================== 7. 三层相似度网络计算 ====================
import pandas as pd

def safe_gene_intersection(genes_source_a, genes_source_b):
    """
    安全计算基因交集的补丁函数。
    强制将两组基因 ID 转换为纯净的字符串格式，消除 int/float/str 类型差异。
    """
    # 1. 过滤掉空值 (NaN/None)
    # 2. 转换为纯字符串 (去掉可能存在的 '.0' 后缀)
    # 3. 去除首尾多余空格
    
    clean_set_a = set(
        str(g).replace('.0', '').strip() 
        for g in genes_source_a 
        if pd.notna(g) and str(g).strip() != ''
    )
    
    clean_set_b = set(
        str(g).replace('.0', '').strip() 
        for g in genes_source_b 
        if pd.notna(g) and str(g).strip() != ''
    )
    
    # 返回交集
    return clean_set_a.intersection(clean_set_b)

# --- 在计算 sim_g (基因机制相似度) 时的实际应用场景 ---

# ==================== 7. 三层相似度网络计算 (G视图修复版) ====================
def compute_similarities(disease_ids, disease_id_to_omim, disease_to_hpo, ancestors, ic_dict, gene_disease_df, string_edges):
    print("\n步骤7: 计算高阶疾病相似度网络 (H视图 & G视图)...")
    n = len(disease_ids)
    
    # --- 1. HPO 临床相似度 ---
    sim_hpo = np.zeros((n, n))
    if ic_dict:
        dis_hpos = {d: disease_to_hpo[disease_id_to_omim[d]] for d in disease_ids if d in disease_id_to_omim and disease_id_to_omim[d] in disease_to_hpo}
        for i, d1 in enumerate(tqdm(disease_ids, desc="  HPO计算中")):
            for j, d2 in enumerate(disease_ids):
                if i <= j and d1 in dis_hpos and d2 in dis_hpos:
                    sim = disease_hpo_similarity(dis_hpos[d1], dis_hpos[d2], ancestors, ic_dict)
                    sim_hpo[i, j] = sim_hpo[j, i] = sim
                    
    # --- 2. 基因机制相似度 (修复数据类型隔离) ---
    sim_gene = np.zeros((n, n))
    if gene_disease_df is not None and string_edges is not None:
        
        # 【修复 1】：把 CTD 的致病基因 ID 洗成纯净字符串
        gene_disease_df['GeneID_clean'] = gene_disease_df['GeneID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        dis_genes = gene_disease_df.groupby('DiseaseID')['GeneID_clean'].apply(set).to_dict()
        
        # 【修复 2】：把 STRING 的互作基因 ID 洗成纯净字符串
        g_sim = defaultdict(dict)
        for _, r in string_edges.iterrows():
            g1 = str(r['gene1']).replace('.0', '').strip()
            g2 = str(r['gene2']).replace('.0', '').strip()
            g_sim[g1][g2] = g_sim[g2][g1] = r['score']
            
        for i, d1 in enumerate(tqdm(disease_ids, desc="  基因网络计算中")):
            g1s = dis_genes.get(d1, set())
            for j, d2 in enumerate(disease_ids):
                if i == j: sim_gene[i, j] = 1.0; continue
                if i > j: continue
                g2s = dis_genes.get(d2, set())
                if not g1s or not g2s: continue
                
                sum1 = sum([max([g_sim.get(g, {}).get(h, 0.0) for h in g2s] + [0]) for g in g1s])
                sum2 = sum([max([g_sim.get(h, {}).get(g, 0.0) for g in g1s] + [0]) for h in g2s])
                sim_gene[i, j] = sim_gene[j, i] = (sum1 + sum2) / (len(g1s) + len(g2s))

    # 稀疏化处理
    for sim_mat, name in [(sim_hpo, 'sim_hpo.npy'), (sim_gene, 'sim_gene.npy')]:
        for i in range(n):
            idx = np.argsort(sim_mat[i])[-K_NEIGHBORS-1:-1]
            mask = np.zeros(n, dtype=bool); mask[idx] = True
            sim_mat[i, ~mask] = 0.0
        np.save(os.path.join(OUTPUT_DIR, name), sim_mat)
        
    print("  ✅ 相似度矩阵构建并稀疏化完成！")
    return sim_hpo, sim_gene

# 测试一下抢救效果：
# 模拟 CTD 里的整型数据和 STRING 里的字符串数据
ctd_mock = [7157, 1234, 5678]
string_mock = ['7157', '1234.0', '9999']
print(safe_gene_intersection(ctd_mock, string_mock)) 
# 期望输出: {'7157', '1234'}

# ==================== 8. 终极 PyG 图封装 ====================
def assemble_heterodata(drug_df, disease_df, dd_df, dg_df, sim_hpo, sim_gene):
    print("\n📦 步骤8: 组装终极 HeteroData 异构图对象...")
    data = HeteroData()
    
    drug_ids = drug_df['DrugBank_ID'].tolist()
    disease_ids = disease_df['DiseaseID'].tolist()
    gene_ids = sorted(list(set(dg_df['GeneID'].astype(str)))) if dg_df is not None else []
    
    d2i = {d: i for i, d in enumerate(drug_ids)}
    dis2i = {d: i for i, d in enumerate(disease_ids)}
    g2i = {g: i for i, g in enumerate(gene_ids)}

    print("  -> 生成节点特征...")
    features = []
    for smiles in drug_df['SMILES']:
        mol = Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=DRUG_FP_BITS), dtype=np.float32) if mol else np.zeros(DRUG_FP_BITS, dtype=np.float32)
        features.append(fp)
    data['drug'].x = torch.tensor(np.stack(features), dtype=torch.float)
    data['disease'].x = torch.randn((len(disease_ids), DISEASE_FEAT_DIM), dtype=torch.float)
    if gene_ids: data['gene'].x = torch.randn((len(gene_ids), GENE_FEAT_DIM), dtype=torch.float)
    else: data['gene'].x = torch.empty((0, GENE_FEAT_DIM), dtype=torch.float)

    print("  -> 注入网络边...")
    if dd_df is not None and not dd_df.empty:
        dd_edges = [[d2i[r['DrugBank_ID']], dis2i[r['DiseaseID']]] for _, r in dd_df.iterrows() if r['DrugBank_ID'] in d2i and r['DiseaseID'] in dis2i]
        data['drug', 'treats', 'disease'].edge_index = torch.tensor(dd_edges).t().contiguous()

    if dg_df is not None and not dg_df.empty:
        dg_edges = [[dis2i[r['DiseaseID']], g2i[str(r['GeneID'])]] for _, r in dg_df.iterrows() if r['DiseaseID'] in dis2i and str(r['GeneID']) in g2i]
        data['disease', 'associated_with', 'gene'].edge_index = torch.tensor(dg_edges).t().contiguous()

    h_idx = np.nonzero(sim_hpo)
    data['disease', 'sim_h', 'disease'].edge_index = torch.tensor(np.stack(h_idx), dtype=torch.long)
    data['disease', 'sim_h', 'disease'].edge_attr = torch.tensor(sim_hpo[h_idx], dtype=torch.float).unsqueeze(1)

    g_idx = np.nonzero(sim_gene)
    data['disease', 'sim_g', 'disease'].edge_index = torch.tensor(np.stack(g_idx), dtype=torch.long)
    data['disease', 'sim_g', 'disease'].edge_attr = torch.tensor(sim_gene[g_idx], dtype=torch.float).unsqueeze(1)

    save_path = os.path.join(OUTPUT_DIR, "final_hetero_data_strict.pt")
    torch.save(data, save_path)
    print(f"\n🎉 完美收官！异构图已成功保存至: {save_path}")
    print("\n📊 你的异构图长这样:\n", data)

# ==================== 主程序 ====================
def main():
    print("="*60)
    print("🚀 开始多源异构图全栈数据构建 (从 0 到 PyG 封装)")
    print("="*60)
    
    req_files = [CTD_DISEASES, DRUGBANK_XML, CTD_CHEM_DISEASE, CTD_GENE_DISEASE, STRING_FILE, STRING_ALIAS]
    if not check_files(req_files): return

    # --- Phase 1: 节点与边解析 ---
    disease_df = filter_diseases()
    drug_df = parse_drugbank_xml()
    u2e_map = load_uniprot_entrez_map(UNIPROT2ENTREZ)
    drug_df = map_drug_targets(drug_df, u2e_map)
    dd_df = load_ctd_chem_disease(disease_df, drug_df)
    dg_df = load_ctd_gene_disease(disease_df)
    string_edges = load_string_links(STRING_FILE, STRING_ALIAS)
    
    # --- Phase 2: 相似度网络构建 ---
    d2hpo, ic, anc, omim_map = prepare_hpo(disease_df)
    disease_ids = disease_df['DiseaseID'].tolist()
    sim_hpo, sim_gene = compute_similarities(disease_ids, omim_map, d2hpo, anc, ic, dg_df, string_edges)
    
    # --- Phase 3: PyG 图封装 ---
    assemble_heterodata(drug_df, disease_df, dd_df, dg_df, sim_hpo, sim_gene)

if __name__ == "__main__":
    main()