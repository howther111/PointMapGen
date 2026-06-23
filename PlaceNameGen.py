import tkinter as tk
from tkinter import messagebox
import random


OUTPUT_FILE = "placenames.txt"


VOWELS = ["ア", "イ", "ウ", "エ", "オ"]

CONSONANT_MORA = [
    "カ", "キ", "ク", "ケ", "コ",
    "サ", "シ", "ス", "セ", "ソ",
    "タ", "チ", "ツ", "テ", "ト",
    "ナ", "ニ", "ヌ", "ネ", "ノ",
    "ハ", "ヒ", "フ", "ヘ", "ホ",
    "マ", "ミ", "ム", "メ", "モ",
    "ヤ", "ユ", "ヨ",
    "ラ", "リ", "ル", "レ", "ロ",
    "ワ",
    "ガ", "ギ", "グ", "ゲ", "ゴ",
    "ザ", "ジ", "ズ", "ゼ", "ゾ",
    "ダ", "デ", "ド",
    "バ", "ビ", "ブ", "ベ", "ボ",
    "パ", "ピ", "プ", "ペ", "ポ",
    "ヴ"
]

YOON_MORA = [
    "キャ", "キュ", "キョ",
    "シャ", "シュ", "ショ",
    "チャ", "チュ", "チョ",
    "ニャ", "ニュ", "ニョ",
    "ヒャ", "ヒュ", "ヒョ",
    "ミャ", "ミュ", "ミョ",
    "リャ", "リュ", "リョ",
    "ギャ", "ギュ", "ギョ",
    "ジャ", "ジュ", "ジョ",
    "ビャ", "ビュ", "ビョ",
    "ピャ", "ピュ", "ピョ",
    "ウィ", "ヴィ", "ウェ", "ヴェ",
    "フィ", "フェ", "ファ", "フォ",
    "ヴァ", "ヴォ", "ウォ"
]

ENDING_MORA = [
    "ア", "イ", "ウ", "エ", "オ",
    "カ", "キ", "ク", "ケ", "コ",
    "ナ", "ニ", "ヌ", "ネ", "ノ",
    "マ", "ミ", "ム", "メ", "モ",
    "ラ", "リ", "ル", "レ", "ロ",
    "ス", "ト", "ン"
]


def weighted_choice(items):
    values = [item[0] for item in items]
    weights = [item[1] for item in items]
    return random.choices(values, weights=weights, k=1)[0]


def generate_syllable(is_first=False, is_last=False):
    if is_last:
        syllable_type = weighted_choice([
            ("normal", 80),
            ("vowel", 8),
            ("n", 8),
            ("long", 4),
        ])
    elif is_first:
        syllable_type = weighted_choice([
            ("normal", 85),
            ("vowel", 10),
            ("yoon", 5),
        ])
    else:
        syllable_type = weighted_choice([
            ("normal", 75),
            ("vowel", 8),
            ("yoon", 7),
            ("long", 5),
            ("small_tsu", 5),
        ])

    if syllable_type == "vowel":
        return random.choice(VOWELS)

    if syllable_type == "yoon":
        return random.choice(YOON_MORA)

    if syllable_type == "long":
        return "ー"

    if syllable_type == "small_tsu":
        return "ッ"

    if syllable_type == "n":
        return "ン"

    if is_last:
        return random.choice(ENDING_MORA)

    return random.choice(CONSONANT_MORA)


def is_bad_name(name):
    bad_patterns = [
        "ーー", "ッッ", "ンン",
        "ーッ", "ッー",
        "ンー", "ンッ",
    ]

    for pattern in bad_patterns:
        if pattern in name:
            return True

    if name[0] in ["ー", "ッ", "ン"]:
        return True

    if name[-1] in ["ー", "ッ"]:
        return True

    return False


def generate_place_name(min_len, max_len):
    for _ in range(2000):
        name = ""

        while len(name) < max_len:
            is_first = len(name) == 0
            remaining = max_len - len(name)

            if len(name) >= min_len and random.random() < 0.35:
                break

            is_last = remaining <= 1

            syllable = generate_syllable(
                is_first=is_first,
                is_last=is_last
            )

            if len(name) + len(syllable) > max_len:
                continue

            name += syllable

        if min_len <= len(name) <= max_len and not is_bad_name(name):
            return name

    return "ナナシ"


def generate_file():
    try:
        min_len = int(min_entry.get())
        max_len = int(max_entry.get())
        count = int(count_entry.get())

        if min_len < 2:
            messagebox.showerror("エラー", "文字数下限は2以上にしてください。")
            return

        if max_len < min_len:
            messagebox.showerror("エラー", "文字数上限は下限以上にしてください。")
            return

        if count < 1:
            messagebox.showerror("エラー", "生成数は1以上にしてください。")
            return

        names = []
        used = set()

        max_attempts = count * 100

        while len(names) < count and max_attempts > 0:
            name = generate_place_name(min_len, max_len)

            if name not in used:
                used.add(name)
                names.append(name)

            max_attempts -= 1

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for name in names:
                f.write(name + "\n")

        messagebox.showinfo(
            "完了",
            f"{OUTPUT_FILE} をUTF-8形式で出力しました。\n"
            f"生成数: {len(names)}"
        )

    except ValueError:
        messagebox.showerror("エラー", "数値を正しく入力してください。")
    except Exception as e:
        messagebox.showerror("エラー", str(e))


def close_app():
    root.destroy()


root = tk.Tk()
root.title("地名ジェネレータ")
root.geometry("320x250")
root.resizable(False, False)

main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill="both", expand=True)

tk.Label(main_frame, text="文字数下限").grid(row=0, column=0, sticky="w")
min_entry = tk.Entry(main_frame, width=10)
min_entry.insert(0, "3")
min_entry.grid(row=0, column=1, sticky="w", pady=5)

tk.Label(main_frame, text="文字数上限").grid(row=1, column=0, sticky="w")
max_entry = tk.Entry(main_frame, width=10)
max_entry.insert(0, "5")
max_entry.grid(row=1, column=1, sticky="w", pady=5)

tk.Label(main_frame, text="生成数").grid(row=2, column=0, sticky="w")
count_entry = tk.Entry(main_frame, width=10)
count_entry.insert(0, "100")
count_entry.grid(row=2, column=1, sticky="w", pady=5)

button_frame = tk.Frame(main_frame)
button_frame.grid(row=3, column=0, columnspan=2, pady=25)

tk.Button(
    button_frame,
    text="生成",
    width=10,
    command=generate_file
).pack(side="left", padx=5)

tk.Button(
    button_frame,
    text="終了",
    width=10,
    command=close_app
).pack(side="left", padx=5)

root.mainloop()