import torch
from torch_geometric.data import HeteroData
import os

def validate_graph(pt_path):
    print("="*50)
    print(f"🔍 开始对异构图进行全身体检: {pt_path}")
    print("="*50)
    
    if not os.path.exists(pt_path):
        print("❌ 文件不存在！")
        return
        
    data = torch.load(pt_path)
    
    print("\n📦 1. 节点特征合法性检查:")
    for node_type in data.node_types:
        x = data[node_type].x
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        status = "❌ 包含无效值!" if has_nan or has_inf else "✅ 正常"
        print(f"  - [{node_type}] 数量: {data[node_type].num_nodes}, 特征维度: {x.shape[1]} -> {status}")

    print("\n🔗 2. 边索引越界与连通性检查:")
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        edge_index = data[edge_type].edge_index
        num_edges = edge_index.shape[1]
        
        if num_edges == 0:
            print(f"  - [{rel}] ⚠️ 警告: 边数量为 0，可能发生图断裂！")
            continue
            
        max_src = edge_index[0].max().item()
        max_dst = edge_index[1].max().item()
        valid_src = max_src < data[src].num_nodes
        valid_dst = max_dst < data[dst].num_nodes
        
        status = "✅ 正常"
        if not valid_src or not valid_dst:
            status = f"❌ 越界! (src_max:{max_src}, dst_max:{max_dst})"
            
        print(f"  - [{rel}] 边数: {num_edges} -> {status}")

    print("\n" + "="*50)
    print("体检结束！")

if __name__ == "__main__":
    validate_graph("/hy-tmp/else/processed/final_hetero_data_strict.pt")