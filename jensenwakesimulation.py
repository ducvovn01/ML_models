import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- 1. CẤU HÌNH VẬT LÝ ---
FARM_SIZE = 4000
RESOLUTION = 150       # Độ phân giải lưới (càng cao càng mịn)
WIND_SPEED_0 = 12.0    # Gió tự do (m/s)
ROTOR_D = 80.0
WAKE_K = 0.075
CT = 0.8

# --- 2. TẠO LAYOUT ---
def get_turbine_coords():
    coords = []
    # Cụm 1 (Mật độ cao)
    for r in range(7):
        for c in range(8): coords.append([500 + c * 350, 500 + r * 350])
    # Cụm 2 (Xa hơn)
    off_x, off_y = 3000, 800
    for r in range(4):
        for c in range(4): coords.append([off_x + c * 350, off_y + r * 350])
    return np.array(coords)

turbines = get_turbine_coords()
N_TURBINES = len(turbines)

# --- 3. ĐỘNG CƠ TÍNH TOÁN (PHYSICS ENGINE) ---
def calculate_physics_frame(wind_dir, grid_x, grid_y, turbine_coords):
    """
    Tính toán đồng thời:
    1. Trường gió nền (Grid) để vẽ sóng.
    2. Tốc độ gió tại từng tâm Turbine để tô màu.
    """
    # --- A. CHUẨN BỊ TỌA ĐỘ XOAY ---
    theta = np.radians(270 - wind_dir)
    rot_mat = np.array([[np.cos(theta), np.sin(theta)], 
                        [-np.sin(theta), np.cos(theta)]])
    
    # Xoay lưới (Grid)
    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    grid_rot = rot_mat @ grid_coords
    gx_rot = grid_rot[0].reshape(grid_x.shape)
    gy_rot = grid_rot[1].reshape(grid_x.shape)
    
    # Xoay Turbine
    turb_rot = turbine_coords @ rot_mat.T
    
    # --- B. TÍNH TOÁN TRƯỜNG GIÓ NỀN (GRID) ---
    grid_deficit_sq = np.zeros_like(grid_x)
    
    for tx, ty in turb_rot:
        dx = gx_rot - tx
        dy = gy_rot - ty
        # Điều kiện vùng Wake
        r_wake = (ROTOR_D / 2) + WAKE_K * dx
        mask = (dx > 0) & (np.abs(dy) < r_wake)
        
        if np.any(mask):
            term = (1 - np.sqrt(1 - CT)) / (1 + (2 * WAKE_K * dx[mask]) / ROTOR_D)**2
            grid_deficit_sq[mask] += term**2
            
    grid_speed = WIND_SPEED_0 * (1 - np.sqrt(grid_deficit_sq))
    
    # --- C. TÍNH TOÁN TỐC ĐỘ TẠI TỪNG TURBINE ---
    turbine_speeds = np.full(N_TURBINES, WIND_SPEED_0)
    
    # Duyệt qua từng cặp turbine để tính ảnh hưởng
    # Ma trận khoảng cách đã xoay
    n = N_TURBINES
    tx_rot = turb_rot[:, 0]
    ty_rot = turb_rot[:, 1]
    
    # Dùng broadcasting để tính nhanh
    # dx_mat[j, i] = khoảng cách x từ i đến j
    dx_mat = tx_rot[:, np.newaxis] - tx_rot[np.newaxis, :]
    dy_mat = ty_rot[:, np.newaxis] - ty_rot[np.newaxis, :]
    
    # Mask: j nằm sau i (dx > 0)
    valid_mask = dx_mat > 0
    
    # Bán kính wake tại vị trí j do i gây ra
    r_wake_mat = (ROTOR_D / 2) + WAKE_K * dx_mat
    
    # Mask: j nằm trong nón wake của i
    wake_mask = valid_mask & (np.abs(dy_mat) < r_wake_mat)
    
    # Tính thâm hụt
    deficits = np.zeros((n, n))
    term1 = (1 - np.sqrt(1 - CT))
    term2 = (1 + (2 * WAKE_K * dx_mat) / ROTOR_D)**2
    
    # Chỉ tính tại những nơi có wake
    np.divide(term1, term2, out=deficits, where=wake_mask)
    
    # Tổng hợp thâm hụt cho từng turbine
    total_deficit = np.sqrt(np.sum(deficits**2, axis=1))
    turbine_speeds = WIND_SPEED_0 * (1 - total_deficit)
    
    # Tính Vector dòng chảy (U, V) để vẽ mũi tên sóng
    math_angle = np.radians(90 - wind_dir)
    U = grid_speed * np.cos(math_angle)
    V = grid_speed * np.sin(math_angle)
    
    return grid_speed, turbine_speeds, U, V

# --- 4. THIẾT LẬP ĐỒ HỌA (VISUALIZATION) ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8))

# Tạo lưới
x = np.linspace(0, FARM_SIZE, RESOLUTION)
y = np.linspace(0, FARM_SIZE, RESOLUTION)
X, Y = np.meshgrid(x, y)

# Tính toán khung hình đầu tiên
initial_grid, initial_turb_speeds, U, V = calculate_physics_frame(0, X, Y, turbines)

# A. Vẽ Nền Sóng (Heatmap Mịn)
# cmap='coolwarm': Đỏ=Gió mạnh, Xanh=Gió yếu (Độ tương phản cao)
im = ax.imshow(initial_grid, extent=[0, FARM_SIZE, 0, FARM_SIZE], origin='lower',
               cmap='coolwarm', interpolation='bicubic', vmin=4, vmax=12, alpha=0.8)

# B. Vẽ Mũi tên Sóng (Quiver - Flow lines)
skip = 8 # Vẽ thưa để đỡ rối
Q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip],
              color='white', alpha=0.15, scale=400, width=0.002)

# C. Vẽ Turbine đổi màu (Scatter)
# c=initial_turb_speeds: Màu dựa theo tốc độ gió tại đó
# cmap='coolwarm': Đồng bộ màu với nền
scat = ax.scatter(turbines[:, 0], turbines[:, 1], c=initial_turb_speeds, 
                  cmap='coolwarm', vmin=4, vmax=12, s=60, edgecolors='white', linewidth=1.5, zorder=10)

# Trang trí
ax.set_title("Mô phỏng Dòng Chảy & Hiệu suất Turbine", fontsize=16, color='white', fontweight='bold')
ax.set_xlabel("Khoảng cách (m)")
ax.set_ylabel("Khoảng cách (m)")
cbar = plt.colorbar(im, ax=ax, label='Tốc độ gió (m/s)')
cbar.set_label('Tốc độ gió (m/s)', color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Text thông tin
info_box = ax.text(0.02, 0.95, "", transform=ax.transAxes, color='cyan', fontsize=12, fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor='cyan'))

# --- 5. ANIMATION LOOP ---
def update(frame):
    wind_dir = frame % 360
    
    # 1. Tính toán Vật lý
    grid_speed, turb_speeds, U_new, V_new = calculate_physics_frame(wind_dir, X, Y, turbines)
    
    # 2. Cập nhật nền sóng
    im.set_data(grid_speed)
    
    # 3. Cập nhật dòng chảy (mũi tên)
    Q.set_UVC(U_new[::skip, ::skip], V_new[::skip, ::skip])
    
    # 4. Cập nhật MÀU SẮC TURBINE (Quan trọng nhất)
    # Turbine tự động đổi màu: Đỏ (Mạnh) -> Xanh (Yếu)
    scat.set_array(turb_speeds)
    
    # 5. Cập nhật thông tin
    # Tính công suất ước tính toàn trang trại
    p_total = np.sum((turb_speeds/12.0)**3) * 1.5
    info_box.set_text(f"Góc gió: {wind_dir}°\nCông suất: {p_total:.1f} MW")
    
    return im, Q, scat, info_box

# Chạy Animation (30ms/frame -> Mượt mà)
ani = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=30, blit=False)

plt.tight_layout()
plt.show()