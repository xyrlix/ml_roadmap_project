
# 可以根据需要调用不同的机器学习模块

import importlib
import os

def run_model(model_path):
    """运行指定的模型"""
    try:
        # 将路径转换为模块名
        module_name = model_path.replace(os.sep, '.')
        module = importlib.import_module(module_name)
        # 调用模型函数
        model_func = getattr(module, os.path.basename(model_path))
        model_func()
        print(f"模型 {model_path} 运行成功！")
    except Exception as e:
        print(f"运行模型 {model_path} 时出错: {e}")

def list_available_models():
    """列出所有可用的模型"""
    models = []
    # 遍历models目录
    models_dir = 'models'
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            # 跳过隐藏目录和文件
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith('.py') and not file.startswith('__init__'):
                    # 构建模型路径
                    model_path = os.path.join(root, file[:-3]).lstrip('.').lstrip(os.sep)
                    models.append(model_path)
    return models

def main():
    """主函数"""
    print("=== 机器学习模型运行器 ===")
    print("可用的模型:")
    
    models = list_available_models()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    print("\n请选择要运行的模型编号 (输入 0 运行所有模型):")
    choice = input("输入编号: ")
    
    try:
        choice = int(choice)
        if choice == 0:
            # 运行所有模型
            print("\n运行所有模型...")
            for model in models:
                run_model(model)
        elif 1 <= choice <= len(models):
            # 运行选择的模型
            model = models[choice - 1]
            print(f"\n运行模型: {model}")
            run_model(model)
        else:
            print("无效的选择！")
    except ValueError:
        print("请输入有效的数字！")

if __name__ == "__main__":
    main()