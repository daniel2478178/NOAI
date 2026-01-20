# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 08:55:17 2024

@author: NING MEI
"""

import csv

# 假设.data文件的字段由逗号分隔
input_file = 'wine.data'
output_file = 'wine.csv'

# 打开.data文件和要写入的CSV文件
with open(input_file, 'r') as data_file, open(output_file, 'w', newline='') as csv_file:
    # 创建CSV写入器
    csv_writer = csv.writer(csv_file)
    
    # 读取.data文件的每一行
    for line in data_file:
        # 假设每行由逗号分隔的字段组成
        fields = line.strip().split(',')
        
        # 写入CSV文件
        csv_writer.writerow(fields)