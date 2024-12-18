from collections import defaultdict

# 创建一个默认值为0的defaultdict字典
default_dict = defaultdict(int)

# 向字典中添加一些键值对
default_dict["key1"] = 1
default_dict["key2"] = 2
default_dict["key3"] = 3

# 遍历键的方法一：使用keys()方法
for key in default_dict.keys():
    print(key)

# 遍历键的方法二：使用for循环
for key in default_dict:
    print(key)

for a,b in default_dict.items():
    print(a,b)
