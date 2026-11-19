"""
MESM 数据加载入口模块 (向后兼容与快捷导入)
提供与 read_and_feature 模块中一致的 load_data 接口。
"""
from read_and_feature import load_data, process_mesm_dataset

__all__ = ["load_data", "process_mesm_dataset"]

if __name__ == "__main__":
    data = load_data()
    print("load_data.py 测试运行成功！")
