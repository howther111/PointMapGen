from pathlib import Path

old_file = Path("output.png")
new_file = Path("input.png")

if old_file.exists():
    # input.png が既に存在する場合は削除（上書きするため）
    if new_file.exists():
        new_file.unlink()

    old_file.rename(new_file)
    print("output.png を input.png にリネームしました。")
else:
    print("output.png が見つかりません。")