import os
import re
import argparse


def extract_number(s: str):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def find_bmp_files(path: str):
    files = [f for f in os.listdir(path) if f.lower().endswith(".bmp")]

    # 以文件名中第一个数字为主键进行排序，若无数字则按文件名排序
    def keyfn(name):
        num = extract_number(name)
        return (num if num is not None else float("inf"), name.lower())

    files.sort(key=keyfn)
    return files


def safe_rename(files, dry_run=False, min_width=2):
    # 根据文件数量决定位数，但至少 min_width 位
    count = len(files)
    width = max(min_width, len(str(count)))

    # 先生成临时名，避免直接命名冲突
    tmp_names = []
    for i, fname in enumerate(files):
        tmp = f"__tmp_rename_{i:04d}__.bmp"
        tmp_names.append(tmp)

    mapping_tmp = list(zip(files, tmp_names))
    mapping_final = []
    for i, orig in enumerate(files, 1):
        final = f"{i:0{width}d}.bmp"
        mapping_final.append((orig, final))

    # Dry run: just print mappings
    if dry_run:
        print("DRY RUN: 将进行以下重命名（两步以避免覆盖）:")
        for (o, t), (_, f) in zip(mapping_tmp, mapping_final):
            print(f"{o} -> {t} -> {f}")
        return

    # 执行第一步：old -> tmp
    for old, tmp in mapping_tmp:
        if os.path.exists(tmp):
            raise FileExistsError(f"临时文件已存在：{tmp}")
        os.rename(old, tmp)

    # 执行第二步：tmp -> final
    for (_, tmp), (_, final) in zip(mapping_tmp, mapping_final):
        if os.path.exists(final):
            # 如果目标名意外存在，回滚并报错
            for o, t in mapping_tmp:
                if os.path.exists(t):
                    os.rename(t, o)
            raise FileExistsError(f"目标文件已存在，已回滚：{final}")
        os.rename(tmp, final)


def main():
    parser = argparse.ArgumentParser(
        description="按顺序将目录下的 bmp 图片重命名为 01.bmp 格式"
    )
    parser.add_argument(
        "--path", "-p", default=None, help="要重命名的目录，默认为脚本所在目录"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="仅预览重命名，不做实际操作"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = args.path if args.path else script_dir

    # 切换到目标目录，方便使用相对路径和避免路径问题
    os.chdir(target_dir)

    files = find_bmp_files(".")
    if not files:
        print("未找到任何 .bmp 文件。")
        return

    safe_rename(files, dry_run=args.dry_run, min_width=2)
    if args.dry_run:
        print("\nDRY RUN 完成，未修改任何文件。")
    else:
        print("\n重命名完成。")


if __name__ == "__main__":
    main()
