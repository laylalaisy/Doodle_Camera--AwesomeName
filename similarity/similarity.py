import os
import csv
import numpy as np
from scipy.spatial import distance



if __name__ == "__main__":
    log_path = "/home/logs"
    for root, dirs, files in os.walk(log_path):
        log_path_dict = dict()
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            log_path = dir_path + "/public.log"
            log_path_dict[dir_name] = log_path
        return log_path_dict

	print(get_log_path_dict())


	with open('./dataset/photos/edge_matrix/dog/1.csv','r') as photo_file:
	    reader = csv.reader(photo_file)
	    rows= [row for row in reader]
	data=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理
	# print(type(data),data.shape)

	with open('./dataset/doodles/edge_matrix/dog/0.csv','r') as f0:
	    reader = csv.reader(f0)
	    rows= [row for row in reader]
	data0=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/1.csv','r') as f1:
	    reader = csv.reader(f1)
	    rows= [row for row in reader]
	data1=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/2.csv','r') as f2:
	    reader = csv.reader(f2)
	    rows= [row for row in reader]
	data2=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/3.csv','r') as f3:
	    reader = csv.reader(f3)
	    rows= [row for row in reader]
	data3=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/4.csv','r') as f4:
	    reader = csv.reader(f4)
	    rows= [row for row in reader]
	data4=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/5.csv','r') as f5:
	    reader = csv.reader(f5)
	    rows= [row for row in reader]
	data5=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/6.csv','r') as f6:
	    reader = csv.reader(f6)
	    rows= [row for row in reader]
	data6=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/7.csv','r') as f7:
	    reader = csv.reader(f7)
	    rows= [row for row in reader]
	data7=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/8.csv','r') as f8:
	    reader = csv.reader(f8)
	    rows= [row for row in reader]
	data8=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	with open('./dataset/doodles/edge_matrix/dog/9.csv','r') as f9:
	    reader = csv.reader(f9)
	    rows= [row for row in reader]
	data9=np.array(rows, dtype=float)#rows是数据类型是‘list',转化为数组类型好处理

	dist0 = np.linalg.norm(data0-data)
	dist1 = np.linalg.norm(data1-data)
	dist2 = np.linalg.norm(data2-data)
	dist3 = np.linalg.norm(data3-data)
	dist4 = np.linalg.norm(data4-data)
	dist5 = np.linalg.norm(data5-data)
	dist6 = np.linalg.norm(data6-data)
	dist7 = np.linalg.norm(data7-data)
	dist8 = np.linalg.norm(data8-data)
	dist9 = np.linalg.norm(data9-data)

	print(dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9)

	
	# print("out0=",type(data),data.shape)
	# print("out1=",data)
