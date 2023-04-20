import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.misc.pilutil import imread

f = open(r'C:\Users\LENOVO\Desktop\pubg\PUBG_MatchData_Flattened.tsv')  # 添加路径
df = pd.read_csv(f, sep='\t')

# edf和mdf是两个地图，下面把两张地图分开进行处理
edf = df.loc[df['map_id'] == 'ERANGEL']
mdf = df.loc[df['map_id'] == 'MIRAMAR']


# #print(edf.head())
def killer_victim_df_maker(df):
    # 挑出地图中击杀和被杀玩家的坐标
    df = edf
    victim_x_df = df.filter(regex='victim_position_x')
    victim_y_df = df.filter(regex='victim_position_y')
    killer_x_df = df.filter(regex='killer_position_x')
    killer_y_df = df.filter(regex='killer_position_y')
    # ravel()将多维矩阵变成一维
    victim_x_s = pd.Series(victim_x_df.values.ravel('F'))
    victim_y_s = pd.Series(victim_y_df.values.ravel('F'))
    killer_x_s = pd.Series(killer_x_df.values.ravel('F'))
    killer_y_s = pd.Series(killer_y_df.values.ravel('F'))

    vdata = {'x': victim_x_s, 'y': victim_y_s}
    kdata = {'x': killer_x_s, 'y': killer_y_s}

    # dropna(how = 'any')删除带nan的行
    # 再留下坐标等于0（在边界上的异常数据）剔除
    victim_df = pd.DataFrame(data=vdata).dropna(how='any')
    victim_df = victim_df[victim_df['x'] > 0]
    killer_df = pd.DataFrame(data=kdata).dropna(how='any')
    killer_df = killer_df[killer_df['x'] > 0]
    return killer_df, victim_df


ekdf, evdf = killer_victim_df_maker(edf)
mkdf, mvdf = killer_victim_df_maker(mdf)

# print(ekdf.head())#在森林击杀的坐标数据
# print(evdf.head())#在森林被杀的坐标数据
# print(mkdf.head())
# print(mvdf.head())
# print(len(ekdf), len(evdf), len(mkdf), len(mvdf))

# 将dataframe转换成numpy array
plot_data_ev = evdf[['x', 'y']].values
plot_data_ek = ekdf[['x', 'y']].values
plot_data_mv = mvdf[['x', 'y']].values
plot_data_mk = mkdf[['x', 'y']].values

# 将获得的坐标数据与地图上的坐标数据进行匹配
plot_data_ev = plot_data_ev * 4040 / 800000
plot_data_ek = plot_data_ek * 4040 / 800000
plot_data_mv = plot_data_mv * 976 / 800000
plot_data_mk = plot_data_mk * 976 / 800000

# 加载模块
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# 热力图函数
def heatmap(x, y, s, bins=100):
    #    x = plot_data_ev[:,0]
    #    y = plot_data_ev[:,1]
    #    s = 1.5
    #    bins = 800
    # np.histogram2d()将两列数值转为矩阵
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    # 高斯锐化模糊对象
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


# 读取森林地图底图
# Normalize归一化
# np.clip(x,a,b)将x中小于a的值设为a，大于b的值设为b
# cm.bwr 蓝白红
bg = imread('erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:, 0], plot_data_ev[:, 1], 1.5, bins=800)
alphas = np.clip(Normalize(0, hmap.max() / 100, clip=True)(hmap) * 1.5, 0.0, 1.)
colors = Normalize(hmap.max() / 100, hmap.max() / 20, clip=True)(hmap)
colors = cm.bwr(colors)
colors[..., -1] = alphas

hmap2, extent2 = heatmap(plot_data_ek[:, 0], plot_data_ek[:, 1], 1.5, bins=800)
alphas2 = np.clip(Normalize(0, hmap2.max() / 100, clip=True)(hmap2) * 1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max() / 100, hmap2.max() / 20, clip=True)(hmap2)
colors2 = cm.RdBu(colors2)
colors2[..., -1] = alphas2

# '森林死亡率图'
fig, ax = plt.subplots(figsize=(24, 24))
ax.set_xlim(0, 4096);
ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.bwr, alpha=1)
# ax.imshow(colors2, extent = extent2, origin = 'lower', cmap = cm.RdBu, alpha = 0.5)
plt.gca().invert_yaxis()
plt.title('森林地图死亡率图')