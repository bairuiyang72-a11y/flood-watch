# -*- coding: utf-8 -*-
"""
全国城市内涝监测站点配置(演示用模拟态势)。
覆盖 31 个省级行政区的代表城市, 每项含 province 字段供前端地图按省检索。
接入真实摄像头/视频后替换 source 参数即可。
"""

CITIES = [
    # name, province, lng, lat, 站点标签, 积水基线占比, 波动幅度, 周期s, 有涉水人员, 在线站点数
    {"name": "武汉", "province": "湖北", "lng": 114.31, "lat": 30.59, "cam": "WH-01 珞喻路下穿段",   "base": 0.24, "amp": 0.07, "period": 22, "wading": True,  "cams_online": 18},
    {"name": "广州", "province": "广东", "lng": 113.26, "lat": 23.13, "cam": "GZ-01 大道中隧道路段", "base": 0.21, "amp": 0.08, "period": 26, "wading": True,  "cams_online": 24},
    {"name": "长沙", "province": "湖南", "lng": 112.94, "lat": 28.23, "cam": "CS-01 五一路下穿段",   "base": 0.16, "amp": 0.06, "period": 30, "wading": False, "cams_online": 12},
    {"name": "南宁", "province": "广西", "lng": 108.37, "lat": 22.82, "cam": "NN-01 民族大道涵洞",   "base": 0.14, "amp": 0.07, "period": 24, "wading": False, "cams_online": 9},
    {"name": "郑州", "province": "河南", "lng": 113.63, "lat": 34.75, "cam": "ZZ-01 京广路下穿段",   "base": 0.18, "amp": 0.09, "period": 20, "wading": False, "cams_online": 15},
    {"name": "上海", "province": "上海", "lng": 121.47, "lat": 31.23, "cam": "SH-01 延安路下立交",   "base": 0.10, "amp": 0.05, "period": 34, "wading": False, "cams_online": 30},
    {"name": "重庆", "province": "重庆", "lng": 106.55, "lat": 29.56, "cam": "CQ-01 石黄隧道口",     "base": 0.12, "amp": 0.06, "period": 28, "wading": False, "cams_online": 16},
    {"name": "杭州", "province": "浙江", "lng": 120.16, "lat": 30.28, "cam": "HZ-01 德胜路下穿段",   "base": 0.08, "amp": 0.05, "period": 32, "wading": False, "cams_online": 14},
    {"name": "南京", "province": "江苏", "lng": 118.80, "lat": 32.06, "cam": "NJ-01 应天大街涵洞",   "base": 0.09, "amp": 0.04, "period": 36, "wading": False, "cams_online": 11},
    {"name": "北京", "province": "北京", "lng": 116.41, "lat": 39.90, "cam": "BJ-01 莲花池南路下穿", "base": 0.06, "amp": 0.04, "period": 40, "wading": False, "cams_online": 22},
    {"name": "成都", "province": "四川", "lng": 104.07, "lat": 30.67, "cam": "CD-01 天府大道段",     "base": 0.07, "amp": 0.04, "period": 38, "wading": False, "cams_online": 13},
    {"name": "沈阳", "province": "辽宁", "lng": 123.43, "lat": 41.80, "cam": "SY-01 青年大街下穿",   "base": 0.05, "amp": 0.03, "period": 42, "wading": False, "cams_online": 8},
    {"name": "天津", "province": "天津", "lng": 117.20, "lat": 39.13, "cam": "TJ-01 大沽桥下穿段",   "base": 0.07, "amp": 0.04, "period": 38, "wading": False, "cams_online": 15},
    {"name": "石家庄", "province": "河北", "lng": 114.51, "lat": 38.04, "cam": "SJZ-01 槐安路下穿段", "base": 0.09, "amp": 0.05, "period": 34, "wading": False, "cams_online": 10},
    {"name": "太原", "province": "山西", "lng": 112.55, "lat": 37.87, "cam": "TY-01 迎泽大街涵洞",   "base": 0.06, "amp": 0.04, "period": 40, "wading": False, "cams_online": 8},
    {"name": "呼和浩特", "province": "内蒙古", "lng": 111.75, "lat": 40.84, "cam": "HHHT-01 新华东街下穿段", "base": 0.05, "amp": 0.03, "period": 44, "wading": False, "cams_online": 6},
    {"name": "长春", "province": "吉林", "lng": 125.32, "lat": 43.90, "cam": "CC-01 人民街下穿段",   "base": 0.05, "amp": 0.03, "period": 44, "wading": False, "cams_online": 7},
    {"name": "哈尔滨", "province": "黑龙江", "lng": 126.53, "lat": 45.80, "cam": "HEB-01 经纬街段",  "base": 0.06, "amp": 0.04, "period": 42, "wading": False, "cams_online": 9},
    {"name": "合肥", "province": "安徽", "lng": 117.23, "lat": 31.82, "cam": "HF-01 望江路涵洞",     "base": 0.12, "amp": 0.06, "period": 30, "wading": False, "cams_online": 11},
    {"name": "福州", "province": "福建", "lng": 119.30, "lat": 26.08, "cam": "FZ-01 五四路下穿段",   "base": 0.13, "amp": 0.07, "period": 26, "wading": False, "cams_online": 10},
    {"name": "南昌", "province": "江西", "lng": 115.86, "lat": 28.68, "cam": "NC-01 迎宾大道段",     "base": 0.15, "amp": 0.07, "period": 26, "wading": True,  "cams_online": 9},
    {"name": "济南", "province": "山东", "lng": 117.00, "lat": 36.65, "cam": "JN-01 经四路下穿段",   "base": 0.08, "amp": 0.05, "period": 36, "wading": False, "cams_online": 12},
    {"name": "海口", "province": "海南", "lng": 110.20, "lat": 20.06, "cam": "HK-01 国兴大道涵洞",   "base": 0.16, "amp": 0.08, "period": 22, "wading": False, "cams_online": 7},
    {"name": "贵阳", "province": "贵州", "lng": 106.63, "lat": 26.65, "cam": "GY-01 中华北路下穿段", "base": 0.11, "amp": 0.06, "period": 28, "wading": False, "cams_online": 8},
    {"name": "昆明", "province": "云南", "lng": 102.83, "lat": 24.88, "cam": "KM-01 北京路段",       "base": 0.10, "amp": 0.05, "period": 30, "wading": False, "cams_online": 9},
    {"name": "拉萨", "province": "西藏", "lng": 91.14,  "lat": 29.66, "cam": "LS-01 北京中路段",     "base": 0.04, "amp": 0.03, "period": 46, "wading": False, "cams_online": 4},
    {"name": "西安", "province": "陕西", "lng": 108.94, "lat": 34.34, "cam": "XA-01 解放路下穿段",   "base": 0.07, "amp": 0.05, "period": 36, "wading": False, "cams_online": 13},
    {"name": "兰州", "province": "甘肃", "lng": 103.83, "lat": 36.06, "cam": "LZ-01 南滨路段",       "base": 0.05, "amp": 0.04, "period": 42, "wading": False, "cams_online": 6},
    {"name": "西宁", "province": "青海", "lng": 101.78, "lat": 36.62, "cam": "XN-01 黄河路段",       "base": 0.04, "amp": 0.03, "period": 46, "wading": False, "cams_online": 5},
    {"name": "银川", "province": "宁夏", "lng": 106.23, "lat": 38.49, "cam": "YC-01 解放街段",       "base": 0.04, "amp": 0.03, "period": 46, "wading": False, "cams_online": 5},
    {"name": "乌鲁木齐", "province": "新疆", "lng": 87.62, "lat": 43.82, "cam": "WLMQ-01 南湖路段",  "base": 0.04, "amp": 0.03, "period": 48, "wading": False, "cams_online": 6},
]
