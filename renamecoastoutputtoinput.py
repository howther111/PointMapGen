from pathlib import Path

old_file = Path("output_coastline.png")
new_file = Path("input_coastline.png")

if old_file.exists():
    # input.png が既に存在する場合は削除（上書きするため）
    if new_file.exists():
        new_file.unlink()

    old_file.rename(new_file)
    print("output_coastline.png を input_coastline.png にリネームしました。")
else:
    print("output_coastline.png が見つかりません。")