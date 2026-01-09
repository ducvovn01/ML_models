import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def simulate_bac_lieu_wind_farm():
    # 1. Khởi tạo Đồ thị
    G = nx.DiGraph() # Đồ thị có hướng (Directed Graph) để thể hiện hướng gió

    # 2. Cấu hình dựa trên quan sát từ ảnh vệ tinh
    # Giả sử trong khung hình có khoảng 5 hàng (lines), mỗi hàng khoảng 8-10 tuabin
    num_rows = 5      # Số hàng (Lines) chạy ra biển
    turbines_per_row = 10 
    
    # Khoảng cách giả lập (đơn vị: mét)
    dist_between_turbines = 500  # Khoảng cách giữa các trụ trong 1 hàng
    dist_between_rows = 1000     # Khoảng cách giữa các hàng
    
    # Góc nghiêng của hàng so với phương Bắc (giả sử dựa trên ảnh)
    angle_deg = -15 # Nghiêng nhẹ
    angle_rad = np.radians(angle_deg)

    # 3. Tạo Nodes (Vị trí Tuabin)
    pos = {}
    node_colors = []
    
    for r in range(num_rows):
        for t in range(turbines_per_row):
            node_id = f"R{r+1}_T{t+1}" # Ví dụ: R1_T1 (Hàng 1, Trụ 1)
            
            # Tính tọa độ (x, y) giả lập
            # X tịnh tiến theo hàng, Y tịnh tiến ra biển
            base_x = r * dist_between_rows
            base_y = t * dist_between_turbines
            
            # Xoay tọa độ để giống thực tế
            x = base_x * np.cos(angle_rad) - base_y * np.sin(angle_rad)
            y = base_x * np.sin(angle_rad) + base_y * np.cos(angle_rad)
            
            G.add_node(node_id, pos=(x, y))
            pos[node_id] = np.array([x, y])
            
            # Tô màu: Trụ trong bờ (đậm) -> Trụ ngoài biển (nhạt) để dễ nhìn
            node_colors.append(t)

    # 4. Tạo Edges (Mô phỏng Wake Effect)
    # Giả sử gió đang thổi từ đất liền ra biển (hoặc ngược lại) dọc theo hàng
    # Ta nối các cạnh giữa các tuabin liền kề
    for r in range(num_rows):
        for t in range(turbines_per_row - 1):
            src = f"R{r+1}_T{t+1}"
            dst = f"R{r+1}_T{t+2}"
            
            # Thêm cạnh có hướng: Src -> Dst (Wake effect lan truyền)
            G.add_edge(src, dst, type='wake_link')

    # Thêm các cạnh chéo (Spatial correlation giữa các hàng) - Optional
    for r in range(num_rows - 1):
        for t in range(turbines_per_row):
             src = f"R{r+1}_T{t+1}"
             dst = f"R{r+2}_T{t+1}"
             G.add_edge(src, dst, type='spatial_link', style='dashed')

    # 5. Vẽ Đồ thị
    plt.figure(figsize=(12, 8))
    
    # Vẽ Nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, cmap=plt.cm.Blues, edgecolors='black')
    
    # Vẽ Edges (Wake Links - Mũi tên liền)
    wake_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'wake_link']
    nx.draw_networkx_edges(G, pos, edgelist=wake_edges, width=2, edge_color='gray', arrowstyle='->', arrowsize=20)
    
    # Vẽ Edges (Spatial Links - Nét đứt)
    spatial_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'spatial_link']
    nx.draw_networkx_edges(G, pos, edgelist=spatial_edges, width=1, edge_color='orange', style='dashed', alpha=0.5)

    # Vẽ Labels
    # nx.draw_networkx_labels(G, pos, font_size=8) # Bỏ comment nếu muốn hiện tên trụ

    plt.title("Simulation of Bac Lieu Wind Farm Graph Topology\n(Nodes: Turbines, Solid Edges: Wake Effect Flow)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

simulate_bac_lieu_wind_farm()