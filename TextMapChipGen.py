import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont
import math
import os


def get_font(size):
    candidates = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def generate_image():
    text = text_box.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("警告", "文字を入力してください。")
        return

    char_count = len(text)
    canvas_size = max(20, int(math.sqrt(char_count) * 20))

    if bg_mode.get() == "white":
        image = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    else:
        image = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))

    draw = ImageDraw.Draw(image)

    cols = math.ceil(math.sqrt(char_count))
    rows = math.ceil(char_count / cols)

    cell_w = canvas_size / cols
    cell_h = canvas_size / rows

    font_size = int(min(cell_w, cell_h) * 0.9)
    font = get_font(font_size)

    for i, ch in enumerate(text):
        col = i % cols
        row = i // cols

        x = col * cell_w
        y = row * cell_h

        bbox = draw.textbbox((0, 0), ch, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tx = x + (cell_w - tw) / 2 - bbox[0]
        ty = y + (cell_h - th) / 2 - bbox[1]

        draw.text((tx, ty), ch, fill=(0, 0, 0, 255), font=font)

    image.save("text_map_chip.png")
    messagebox.showinfo("完了", "text_map_chip.png を生成しました。")


root = tk.Tk()
root.title("テキストマップチップジェネレータ")
root.geometry("420x320")

text_box = tk.Text(root, width=45, height=8)
text_box.pack(pady=10)

bg_mode = tk.StringVar(value="white")

radio_frame = tk.Frame(root)
radio_frame.pack()

tk.Radiobutton(
    radio_frame,
    text="白地",
    variable=bg_mode,
    value="white"
).pack(side=tk.LEFT, padx=10)

tk.Radiobutton(
    radio_frame,
    text="透過",
    variable=bg_mode,
    value="transparent"
).pack(side=tk.LEFT, padx=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=20)

tk.Button(
    button_frame,
    text="生成",
    width=12,
    command=generate_image
).pack(side=tk.LEFT, padx=10)

tk.Button(
    button_frame,
    text="終了",
    width=12,
    command=root.destroy
).pack(side=tk.LEFT, padx=10)

root.mainloop()