# -*- coding: utf-8 -*-
#数据的探索性分析,数据准备和预处理

#打算探究的问题：
#击杀人数分布与吃鸡的关系（苟还是冲）
#是否搭乘载具与吃鸡的关系（跑毒）
#助攻次数与吃鸡概率的关系（队友合作）
#哪种武器淘汰人数多：近战挑选武器，远距离挑选武器（这里其实不太严谨，因为每种武器的刷新率也不一样）（这个数据可以在官网查到来做参考）


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import stats
import seaborn as sns
# from scipy.stats import norm
#from scipy.misc.pilutil import imread

plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.style.use('ggplot') # 使用ggplot风格
# X，Y坐标全部在游戏内坐标中，需要线性缩放以在方形erangel和miramar地图上绘制。最小，最大坐标分别为0, 800, 000。

# 读入数据
#因为match文件过大，处理比较费时间，所以进行了分隔，只取了其中两部分
match1 = pd.read_csv("match1.csv")
match2 = pd.read_csv("match2.csv")
match3 = pd.read_csv("match3.csv")
match = pd.concat([match1, match2, match3])
death = pd.read_csv('death.csv')
# 预览数据
print('数据前五行')
print(match.head(5))
print('数据列项')
print(match.columns)
print('数据形状')
print(match.shape)
print('数据信息')
print(match.info)
print('---------------------------')



# 查看共有多少场比赛
# 去除重复比赛id
all_match_data = match[
        ['match_id', 'game_size', 'match_mode', 'party_size', 'team_id', 'team_placement', 'player_kills',
         'player_dbno', 'player_assists', 'player_dmg', 'player_dist_ride', 'player_dist_walk', 'player_survive_time']]
match_data = all_match_data.drop_duplicates('match_id')
match_counts = pd.value_counts(match_data['match_id']).count()
print(r'共有%d场比赛' % match_counts)

# miramar和erangel是游戏中两种地图
miramar = death[death["map"] == "MIRAMAR"]
erangel = death[death["map"] == "ERANGEL"]


# 是否获得胜利
def is_win(rank):
    label = 0
    if rank == 1:
        label = 1
    return label
# 是否使用过车辆
def is_drive(distance):
    label = 0
    if distance != 0:
        label = 1
    return label

# 添加是否获得胜利的列
new_match_data = match_data.copy()
new_match_data['win_victory'] = match_data['team_placement'].apply(is_win)

def getsafety():
    #找出最后三人死亡的位置
    team_win = match[match["team_placement"]==1] #排名第一的队伍
    #找出每次比赛第一名队伍活的最久的那个player
    grouped = team_win.groupby('match_id').apply(lambda t: t[t.player_survive_time==t.player_survive_time.max()])
    deaths_solo = death[death['match_id'].isin(grouped['match_id'].values)]
    deaths_solo_er = deaths_solo[deaths_solo['map'] == 'ERANGEL']
    deaths_solo_mr = deaths_solo[deaths_solo['map'] == 'MIRAMAR']
    df_second_er = deaths_solo_er[(deaths_solo_er['victim_placement'] == 2)].dropna()
    df_second_mr = deaths_solo_mr[(deaths_solo_mr['victim_placement'] == 2)].dropna()
    print(df_second_er)

    position_data = ["killer_position_x","killer_position_y","victim_position_x","victim_position_y"]
    for position in position_data:
        df_second_mr[position] = df_second_mr[position].apply(lambda x: x*1000/800000)
        df_second_mr = df_second_mr[df_second_mr[position] != 0]
        df_second_er[position] = df_second_er[position].apply(lambda x: x*4096/800000)
        df_second_er = df_second_er[df_second_er[position] != 0]

    df_second_er = df_second_er
    # erangel热力图
    sns.set_context('talk')
    bg = imread("erangel.jpg")
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    ax.imshow(bg)
    sns.kdeplot(df_second_er["victim_position_x"], df_second_er["victim_position_y"], cmap=cm.Blues, alpha=0.7,shade=True)

    # miramar热力图
    bg = imread("miramar.jpg")
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    ax.imshow(bg)
    sns.kdeplot(df_second_mr["victim_position_x"], df_second_mr["victim_position_y"], cmap=cm.Blues,alpha=0.8,shade=True)


def get_drive():
    # 玩家驾驶车辆行驶距离数据
    #绘图
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    # 去除掉车辆行驶距离为0即不开车的数据
    x = match_data[match_data['player_dist_ride'] != 0]['player_dist_ride']
    x[x >= 13000].replace(13000)  # 超过行驶距离一定值的向下归到一起
    plt.hist(x, edgecolor='k', density=0, bins=150, facecolor='#1C7ECE', lw=1, alpha=.8)  # 玩家驾驶车辆行驶距离分布
    plt.xlim(0, 14000)
    plt.title('玩家驾驶车辆行驶距离分布', fontsize=14)
    plt.xlabel('行驶距离', fontsize=11)
    plt.ylabel('玩家人数', fontsize=11)
    plt.text(10000, 550, '13000以上计入13000', style='italic', bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 7}, fontsize=9)
    plt.grid(True, linestyle='--', linewidth=1, axis='y', alpha=0.4)
    #不用载具的玩家
    nodrive_player = match_data['player_dist_ride'].value_counts()[0] / match_data.shape[0]
    print('游戏中没有驾驶过车辆玩家的占比为%.2f%%' % (nodrive_player * 100))
    print('驾驶过车辆玩家的占比为%.2f%%' % ((1 - nodrive_player) * 100))
    print('')

    plt.subplot(2, 2, 3)
    plt.title('玩家车辆使用状况')
    plt.pie([nodrive_player, 1 - nodrive_player], labels=['没有驾驶过车辆', '驾驶过车辆'], autopct='%f%%',
            colors=['#FF1207', '#0FFF4C'], startangle=120)
    plt.savefig('.drive2.png', dpi=300)
    plt.show()

    # 是否驾驶过车辆对获得胜利的影响
    # 添加是否使用过车辆的列
    new_match_data['has_drive_player'] = match_data['player_dist_ride'].apply(is_drive)

    plt.subplot(2, 2, 4)
    ct = pd.crosstab(index=new_match_data['win_victory'], columns=new_match_data['has_drive_player'])
    ct_df = pd.DataFrame(ct)
    plt.bar(ct_df.index, ct_df.loc[0] - ct_df.loc[1], label='没获胜', width=0.4, color='#FF1207', alpha=.8)
    plt.bar(ct_df.index, ct_df.loc[1], bottom=(ct_df.loc[0] - ct_df.loc[1]), label='获胜', width=0.4, color='#0FFF4C',
            alpha=.8)
    plt.xlim(-1.1, 2)
    plt.xticks(ct_df.index)
    plt.xlabel('是否驾驶过车辆')
    plt.legend()
    plt.title('是否搭乘车辆与吃鸡概率的关系')
    plt.grid(True, linestyle='--', linewidth=1, axis='y', alpha=0.4)
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.tight_layout()
    plt.savefig('picture/drive1.png', dpi=300)
    plt.show()


# 击杀人数和造成伤害与获得胜利的关系
def get_kills():
    fig = plt.figure(figsize=(9, 5))
    new_match_data['获胜'] = new_match_data['win_victory']
    g = sns.stripplot(data=new_match_data[['获胜', 'player_dmg', 'player_kills']], x='player_kills',y='player_dmg', hue='获胜')
    g.set(title='击杀人数和伤害量与获胜的关系分布', xlabel='击杀人数', ylabel='造成的伤害')
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.savefig('picture/kill_dmg_win.png',dpi = 300)
    plt.show()
#队友助攻与获得胜利的关系
    g = sns.barplot(data=new_match_data[['win_victory', 'player_assists']], x='player_assists', y='win_victory',color='#1C7ECE')
    g.set(title='助攻次数与吃鸡概率的关系', xlabel='助攻次数', ylabel='吃鸡概率')
    plt.grid(True, linestyle='--', linewidth=1, axis='y', alpha=0.4)
    plt.savefig('picture/assists_win.png', dpi=300)
    plt.show()

def get_party():
    #不同队伍规模的玩家生存时间
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('不同队伍规模下的表现',fontsize=15)
    plt.subplot(1,2,1)
    survive_time_group_by_size = new_match_data.groupby('party_size')['player_survive_time'].mean()
    survive_time_group_by_size_df = pd.DataFrame(survive_time_group_by_size).reset_index()
    plt.title('玩家生存时间')
    plt.xlabel(r'队伍规模(人)')
    plt.ylabel('生存时间')
    plt.bar(survive_time_group_by_size_df['party_size'],survive_time_group_by_size_df['player_survive_time'],edgecolor='k',width = 0.3,facecolor ='#1C7ECE',alpha=.8)
    plt.xticks(survive_time_group_by_size.index)
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.savefig('picture/party_win.png', dpi=300)
    plt.show()

    # 取得单人模式比赛数据
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('各 模 式 下 击 杀 人 数 分 布',x=0.53,y=1)
    plt.subplot(2, 1, 1)
    single_player_match = new_match_data.loc[new_match_data['party_size'] == 1]


    # 单人模式下击杀统计
    x = single_player_match['player_kills']
    plt.title('单人模式击杀人数分布')
    plt.bar(x.value_counts().index.values, x.value_counts(), edgecolor='k', width=0.7, color='#1C7ECE', alpha=.8)
    # plt.xticks(x.value_counts().index.values)
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.xlim(0,20)
    plt.xlabel(r'击杀人数')
    plt.ylabel('击杀频次')
    plt.savefig('picture/single_player_statusn.png', dpi=300)
    plt.show()


    # 组队模式下击杀统计
    plt.subplot(2, 1, 2)
    team_player_match = new_match_data.loc[new_match_data['party_size'] != 1]
    x = team_player_match['player_kills']
    # sns.distplot(x, hist=True)
    plt.bar(x.value_counts().index.values, x.value_counts(), edgecolor='k', width=0.7, color='#1C7ECE', alpha=.8)
    plt.xlim(0, 20)
    plt.xlabel(r'击杀人数')
    plt.ylabel('击杀频次')
    plt.title('组队模式击杀人数分布')
    plt.grid(True,linestyle='--', linewidth=1 ,axis='y',alpha=0.4)
    plt.tight_layout()
    plt.savefig('picture/team_player_status.png',dpi = 300)
    plt.show()

if __name__=='__main__':
   #get_drive()
   #get_kills()
   #get_party()
   #getsafety()
   print("")

