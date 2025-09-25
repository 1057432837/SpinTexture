import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

# 读取数据
data = pd.read_csv('SPINTEXTURE_2D_GAMMA_CENTER.dat', skiprows=3, sep=r'\s+', header=None, names=['kx', 'ky', 'Energy', 'Sx', 'Sy', 'Sz'])

kx     = data['kx']    .values
ky     = data['ky']    .values
energy = data['Energy'].values
Sx     = data['Sx']    .values
Sy     = data['Sy']    .values
Sz     = data['Sz']    .values

# 计算自旋相关参数
inplane_spin   = np.sqrt(Sx**2 + Sy**2)
spin_magnitude = np.sqrt(Sx**2 + Sy**2 + Sz**2)
Sx_norm = Sx / (spin_magnitude + 1e-10)
Sy_norm = Sy / (spin_magnitude + 1e-10)
Sz_norm = Sz / (spin_magnitude + 1e-10)

# 分别设置面内和面外自旋的缩放因子
# 面内自旋(Sx, Sy)保持原缩放比例
scale_factor_inplane = 0.06
# 面外自旋(Sz)使用更小的缩放因子，缩短箭头长度
scale_factor_outofplane = 0.03  # 这里减小了面外自旋的缩放比例

Sx_scaled = Sx_norm * scale_factor_inplane
Sy_scaled = Sy_norm * scale_factor_inplane
Sz_scaled = Sz_norm * scale_factor_outofplane  # 应用面外缩放因子

# 确定坐标范围
max_kx = np.max(np.abs(kx))
max_ky = np.max(np.abs(ky))
scope = max(max_kx, max_ky) * 1.1

# 生成能量面网格
grid_x, grid_y = np.mgrid[-scope:scope:300j, -scope:scope:300j]
grid_z = griddata((kx, ky), energy, (grid_x, grid_y), method='cubic')

# 筛选围绕Γ点的同心圆采样点
distances = np.sqrt(kx**2 + ky**2)

num_circles = 6  # 同心圆数量
radii = np.linspace(0.1 * scope, 0.9 * scope, num_circles)
tol = 0.05 * scope  # 半径容差

selected_indices = []
for r in radii:
    circle_indices = np.where((distances >= r - tol) & (distances <= r + tol))[0]
    step = max(1, len(circle_indices) // 30)  # 每个圆约30个点
    selected_indices.extend(circle_indices[::step])

selected_indices = np.unique(selected_indices)
sub_kx = kx[selected_indices]
sub_ky = ky[selected_indices]
sub_energy = energy[selected_indices]
sub_Sx = Sx_scaled[selected_indices]
sub_Sy = Sy_scaled[selected_indices]
sub_Sz = Sz_scaled[selected_indices]
sub_inplane = inplane_spin[selected_indices]

# 绘制图形
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# 绘制能量面
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none', alpha=0.5, antialiased=True)

# 设置箭头颜色
norm = Normalize(vmin=sub_inplane.min(), vmax=sub_inplane.max())
arrow_colors = cm.gray(norm(sub_inplane))

# 绘制自旋箭头（同心圆分布）
quiver = ax.quiver(sub_kx, sub_ky, sub_energy, sub_Sx, sub_Sy, sub_Sz, color=arrow_colors, linewidths=1.0)

# 设置坐标轴
ax.set_xlim(-scope, scope)
ax.set_ylim(-scope, scope)
ax.set_zlim(-2, -1.25)
ax.set_box_aspect([1, 1, 0.25])
ax.zaxis.set_major_locator(ticker.MultipleLocator(0.2))

ax.set_xlabel('kx', fontsize=14, labelpad=10)
ax.set_ylabel('ky', fontsize=14, labelpad=10)
ax.set_zlabel('Energy (eV)', fontsize=14, labelpad=10)
ax.set_title('Lower Band', fontsize=16, pad=5, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.view_init(elev=30, azim=120)

plt.tight_layout()
plt.savefig('spin_texture.png', dpi=1200, transparent=True)
plt.show()
