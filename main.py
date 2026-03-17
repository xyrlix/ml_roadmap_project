
# 机器学习模型运行器
# 支持交互式选择并运行各类ML模型

import importlib.util
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 切换工作目录到项目根，保证 results/ 相对路径生效
os.chdir(PROJECT_ROOT)


def run_model(model_path: str):
    """动态加载并运行指定模型文件中的入口函数"""
    try:
        abs_path = PROJECT_ROOT / model_path
        if not abs_path.exists():
            print(f"[错误] 文件不存在: {abs_path}")
            return

        # 用文件名（不含.py）作为函数名
        func_name = abs_path.stem  # e.g. linear_regression

        # 动态加载模块
        spec = importlib.util.spec_from_file_location(func_name, abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 获取入口函数
        if hasattr(module, func_name):
            model_func = getattr(module, func_name)
            print(f"\n{'='*60}")
            print(f"▶  运行模型: {model_path}")
            print(f"{'='*60}")
            model_func()
            print(f"\n✅ 模型 [{func_name}] 运行成功！")
        else:
            # 兼容函数名与文件名不一致的情况（如 xgboost_model.py）
            candidates = [name for name in dir(module)
                          if callable(getattr(module, name)) and not name.startswith('_')]
            if candidates:
                fn = getattr(module, candidates[0])
                print(f"\n{'='*60}")
                print(f"▶  运行模型: {model_path}  (函数: {candidates[0]})")
                print(f"{'='*60}")
                fn()
                print(f"\n✅ 模型 [{candidates[0]}] 运行成功！")
            else:
                print(f"[警告] 未找到可调用的入口函数: {model_path}")

    except Exception as e:
        print(f"\n❌ 运行模型 [{model_path}] 时出错: {e}")
        import traceback
        traceback.print_exc()


def list_available_models():
    """遍历 models/ 目录，列出所有非 __init__ 的模型文件"""
    models = []
    models_dir = PROJECT_ROOT / 'models'
    if not models_dir.exists():
        return models

    for root, dirs, files in os.walk(models_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__pycache__')
        for file in sorted(files):
            if file.endswith('.py') and not file.startswith('__'):
                rel = Path(root).relative_to(PROJECT_ROOT) / file
                models.append(str(rel))
    return models


def main():
    """主函数：交互式模型选择与运行"""
    print("\n" + "="*60)
    print("       🤖  机器学习路线图 · 模型运行器")
    print("="*60)
    print("\n📋 可用模型列表：\n")

    models = list_available_models()

    # 按目录分组展示
    current_dir = None
    dir_map = {}  # 目录 -> 模型列表
    for model in models:
        d = str(Path(model).parent)
        dir_map.setdefault(d, []).append(model)

    idx = 1
    model_index = {}
    for d, model_list in dir_map.items():
        category = d.replace('models' + os.sep, '').replace('models/', '')
        print(f"  📁 {category}")
        for m in model_list:
            print(f"     {idx:>3}. {Path(m).stem}")
            model_index[idx] = m
            idx += 1
        print()

    print(f"  共 {len(models)} 个模型\n")
    print("请输入模型编号运行单个模型，输入 0 运行所有模型，输入 q 退出：")

    while True:
        try:
            choice = input("\n>>> 输入编号: ").strip()
            if choice.lower() == 'q':
                print("再见！👋")
                break

            choice_int = int(choice)

            if choice_int == 0:
                print(f"\n将依次运行全部 {len(models)} 个模型...\n")
                for m in models:
                    run_model(m)
                print("\n🎉 所有模型运行完毕！")
                break
            elif choice_int in model_index:
                run_model(model_index[choice_int])
                cont = input("\n继续运行其他模型？(y/n): ").strip().lower()
                if cont != 'y':
                    break
            else:
                print(f"[提示] 请输入 0~{len(models)} 之间的编号")

        except ValueError:
            print("[提示] 请输入有效数字或 q 退出")
        except KeyboardInterrupt:
            print("\n\n已中断，再见！")
            break


if __name__ == "__main__":
    main()
