import tkinter as tk
from tkinter import messagebox
import random
import math
import os
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont


NODE_RADIUS = 50
MIN_DISTANCE = 200
OUTPUT_FILE = "output.png"
CSV_OUTPUT_FILE = "placenames.csv"

PATH_WIDTH = 6
NODE_OUTLINE_WIDTH = 6

PLATE_WIDTH = 160
PLATE_HEIGHT = 30
PLATE_OFFSET_Y = 10
PLATE_OUTLINE_WIDTH = 2


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
    "フィ", "フェ"
]

ENDING_MORA = [
    "ア", "イ", "ウ", "エ", "オ",
    "カ", "キ", "ク", "ケ", "コ",
    "ナ", "ニ", "ヌ", "ネ", "ノ",
    "マ", "ミ", "ム", "メ", "モ",
    "ラ", "リ", "ル", "レ", "ロ",
    "ス", "ト", "ン"
]


def save_csv_names(node_names, city_sizes=None):
    with open(CSV_OUTPUT_FILE, "w", encoding="shift_jis", errors="ignore", newline="") as f:
        if city_sizes is None:
            names = list(node_names.values())
            f.write(",".join(names))
        else:
            for node, name in node_names.items():
                size = city_sizes[node]
                f.write(f"{name},{size}\n")

def load_font(size):
    font_paths = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]

    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(draw, box, text, font, fill):
    x1, y1, x2, y2 = box
    w, h = text_size(draw, text, font)

    x = x1 + ((x2 - x1) - w) / 2
    y = y1 + ((y2 - y1) - h) / 2 - 1

    draw.text((x, y), text, font=font, fill=fill)


def weighted_choice(items):
    values = [item[0] for item in items]
    weights = [item[1] for item in items]
    return random.choices(values, weights=weights, k=1)[0]


def generate_syllable(is_first=False, is_last=False):
    """
    自然な語感になるよう、音節単位で生成する。
    拗音は低確率、長音・促音も低確率。
    """

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


def generate_place_name(min_len=2, max_len=8):
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


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def distance_point_to_segment(point, start, end):
    px, py = point
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return distance(point, start)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))

    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    return distance(point, (nearest_x, nearest_y))


def path_crosses_unrelated_node(start, end, nodes):
    for node in nodes:
        if node == start or node == end:
            continue

        if distance_point_to_segment(node, start, end) < NODE_RADIUS:
            return True

    return False


def generate_nodes(node_count, canvas_width, canvas_height):
    nodes = []

    margin_x = PLATE_WIDTH // 2 + 20
    margin_y = NODE_RADIUS + PLATE_OFFSET_Y + PLATE_HEIGHT + 30
    margin = max(margin_x, margin_y)

    if canvas_width <= margin * 2 or canvas_height <= margin * 2:
        raise RuntimeError(
            "キャンバスサイズが小さすぎます。\n"
            "縦横ともにもう少し大きい値を指定してください。"
        )

    max_attempts = 100000

    for _ in range(node_count):
        for _ in range(max_attempts):
            x = random.randint(margin, canvas_width - margin)
            y = random.randint(margin, canvas_height - margin)

            if all(distance((x, y), n) >= MIN_DISTANCE for n in nodes):
                nodes.append((x, y))
                break
        else:
            raise RuntimeError(
                "ノードを十分に離して配置できませんでした。\n"
                "ノード数を減らすか、キャンバスサイズを大きくしてください。"
            )

    return nodes


def generate_all_valid_pairs(nodes):
    pairs = []
    weights = []

    for n1, n2 in combinations(nodes, 2):
        if path_crosses_unrelated_node(n1, n2, nodes):
            continue

        d = distance(n1, n2)
        weight = 1.0 / (d * d)

        pairs.append((n1, n2))
        weights.append(weight)

    return pairs, weights


def is_connected(nodes, paths):
    graph = {node: [] for node in nodes}

    for a, b in paths:
        graph[a].append(b)
        graph[b].append(a)

    visited = set()
    stack = [nodes[0]]

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        for next_node in graph[node]:
            if next_node not in visited:
                stack.append(next_node)

    return len(visited) == len(nodes)


def generate_connected_paths(nodes, path_count):
    all_pairs, all_weights = generate_all_valid_pairs(nodes)

    if not all_pairs:
        raise RuntimeError("有効なパス候補がありません。")

    if path_count < len(nodes) - 1:
        path_count = len(nodes) - 1

    connected = [random.choice(nodes)]
    unconnected = [n for n in nodes if n not in connected]

    selected_paths = []
    used_pairs = set()

    while unconnected:
        candidates = []
        candidate_weights = []

        for a in connected:
            for b in unconnected:
                if (a, b) not in all_pairs and (b, a) not in all_pairs:
                    continue

                d = distance(a, b)
                candidates.append((a, b))
                candidate_weights.append(1.0 / (d * d))

        if not candidates:
            raise RuntimeError(
                "全ノードを1つに接続できませんでした。\n"
                "もう一度生成してください。"
            )

        a, b = random.choices(candidates, weights=candidate_weights, k=1)[0]

        selected_paths.append((a, b))
        used_pairs.add(tuple(sorted((a, b))))

        connected.append(b)
        unconnected.remove(b)

    remaining_pairs = []
    remaining_weights = []

    for pair, weight in zip(all_pairs, all_weights):
        key = tuple(sorted(pair))

        if key not in used_pairs:
            remaining_pairs.append(pair)
            remaining_weights.append(weight)

    while len(selected_paths) < path_count and remaining_pairs:
        pair = random.choices(remaining_pairs, weights=remaining_weights, k=1)[0]
        index = remaining_pairs.index(pair)

        selected_paths.append(pair)

        remaining_pairs.pop(index)
        remaining_weights.pop(index)

    if not is_connected(nodes, selected_paths):
        raise RuntimeError("パス生成に失敗しました。もう一度生成してください。")

    return selected_paths


def get_plate_box(x, y):
    plate_x1 = x - PLATE_WIDTH // 2
    plate_y1 = y + NODE_RADIUS + PLATE_OFFSET_Y
    plate_x2 = plate_x1 + PLATE_WIDTH
    plate_y2 = plate_y1 + PLATE_HEIGHT

    return [plate_x1, plate_y1, plate_x2, plate_y2]


def generate_image():
    try:
        canvas_height = int(canvas_height_entry.get())
        canvas_width = int(canvas_width_entry.get())
        node_count = int(node_entry.get())
        path_count = int(path_entry.get())

        if canvas_width < 400:
            messagebox.showerror("エラー", "キャンバス横サイズは400以上にしてください。")
            return

        if canvas_height < 400:
            messagebox.showerror("エラー", "キャンバス縦サイズは400以上にしてください。")
            return

        if node_count < 2:
            messagebox.showerror("エラー", "ノード数は2以上にしてください。")
            return

        if path_count < 1:
            messagebox.showerror("エラー", "パス数は1以上にしてください。")
            return

        transparent = bg_var.get() == "transparent"

        if transparent:
            image = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
            bg_color = (255, 255, 255, 0)
        else:
            image = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))
            bg_color = (255, 255, 255, 255)

        draw = ImageDraw.Draw(image)

        nameplate_enabled = nameplate_var.get() == "yes"
        random_name_enabled = random_name_var.get() == "yes"
        city_size_enabled = city_size_var.get() == "yes"

        name_font = load_font(16)
        size_font = load_font(34)

        nodes = generate_nodes(node_count, canvas_width, canvas_height)
        paths = generate_connected_paths(nodes, path_count)

        node_names = {
            node: generate_place_name(2, 8)
            for node in nodes
        }

        city_sizes = {
            node: str(random.randint(1, 10))
            for node in nodes
        }

        for start, end in paths:
            draw.line([start, end], fill=(0, 0, 0, 255), width=PATH_WIDTH)

        for x, y in nodes:
            node_box = [
                x - NODE_RADIUS,
                y - NODE_RADIUS,
                x + NODE_RADIUS,
                y + NODE_RADIUS
            ]

            draw.ellipse(node_box, fill=bg_color)
            draw.ellipse(
                node_box,
                outline=(0, 0, 0, 255),
                width=NODE_OUTLINE_WIDTH
            )

            if city_size_enabled:
                draw_centered_text(
                    draw,
                    node_box,
                    city_sizes[(x, y)],
                    size_font,
                    (0, 0, 0, 255)
                )

        for x, y in nodes:
            plate_box = get_plate_box(x, y)

            if nameplate_enabled:
                draw.rectangle(
                    plate_box,
                    fill=bg_color,
                    outline=(0, 0, 0, 255),
                    width=PLATE_OUTLINE_WIDTH
                )

            if random_name_enabled:
                draw_centered_text(
                    draw,
                    plate_box,
                    node_names[(x, y)],
                    name_font,
                    (0, 0, 0, 255)
                )

        image.save(OUTPUT_FILE, "PNG")
        if random_name_enabled:
            if city_size_enabled:
                save_csv_names(node_names, city_sizes)
            else:
                save_csv_names(node_names)

        message = (
            f"{OUTPUT_FILE} を出力しました。\n"
            f"サイズ: {canvas_width} × {canvas_height}px\n"
            f"生成パス数: {len(paths)}"
        )

        if random_name_enabled:
            message += f"\n{CSV_OUTPUT_FILE} をShift_JIS形式で出力しました。"

        messagebox.showinfo("完了", message)

    except ValueError:
        messagebox.showerror("エラー", "数値を正しく入力してください。")
    except Exception as e:
        messagebox.showerror("エラー", str(e))


def close_app():
    root.destroy()


root = tk.Tk()
root.title("ポイントマップジェネレータ")
root.geometry("440x500")
root.resizable(False, False)

main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill="both", expand=True)

tk.Label(main_frame, text="キャンバス縦ピクセル数").grid(row=0, column=0, sticky="w")
canvas_height_entry = tk.Entry(main_frame, width=10)
canvas_height_entry.insert(0, "1200")
canvas_height_entry.grid(row=0, column=1, sticky="w", pady=5)

tk.Label(main_frame, text="キャンバス横ピクセル数").grid(row=1, column=0, sticky="w")
canvas_width_entry = tk.Entry(main_frame, width=10)
canvas_width_entry.insert(0, "1200")
canvas_width_entry.grid(row=1, column=1, sticky="w", pady=5)

tk.Label(main_frame, text="ノード数").grid(row=2, column=0, sticky="w")
node_entry = tk.Entry(main_frame, width=10)
node_entry.insert(0, "8")
node_entry.grid(row=2, column=1, sticky="w", pady=5)

tk.Label(main_frame, text="パス数").grid(row=3, column=0, sticky="w")
path_entry = tk.Entry(main_frame, width=10)
path_entry.insert(0, "12")
path_entry.grid(row=3, column=1, sticky="w", pady=5)

bg_var = tk.StringVar(value="white")

tk.Label(main_frame, text="背景").grid(row=4, column=0, sticky="w", pady=(10, 0))
tk.Radiobutton(main_frame, text="白地", variable=bg_var, value="white").grid(row=4, column=1, sticky="w", pady=(10, 0))
tk.Radiobutton(main_frame, text="透過", variable=bg_var, value="transparent").grid(row=4, column=2, sticky="w", pady=(10, 0))

nameplate_var = tk.StringVar(value="no")

tk.Label(main_frame, text="ネームプレート").grid(row=5, column=0, sticky="w", pady=(10, 0))
tk.Radiobutton(main_frame, text="なし", variable=nameplate_var, value="no").grid(row=5, column=1, sticky="w", pady=(10, 0))
tk.Radiobutton(main_frame, text="あり", variable=nameplate_var, value="yes").grid(row=5, column=2, sticky="w", pady=(10, 0))

random_name_var = tk.StringVar(value="no")

tk.Label(main_frame, text="ランダム名").grid(row=6, column=0, sticky="w")
tk.Radiobutton(main_frame, text="なし", variable=random_name_var, value="no").grid(row=6, column=1, sticky="w")
tk.Radiobutton(main_frame, text="あり", variable=random_name_var, value="yes").grid(row=6, column=2, sticky="w")

city_size_var = tk.StringVar(value="no")

tk.Label(main_frame, text="規模を設定").grid(row=7, column=0, sticky="w", pady=(10, 0))
tk.Radiobutton(main_frame, text="なし", variable=city_size_var, value="no").grid(row=7, column=1, sticky="w", pady=(10, 0))
tk.Radiobutton(main_frame, text="あり", variable=city_size_var, value="yes").grid(row=7, column=2, sticky="w", pady=(10, 0))

button_frame = tk.Frame(main_frame)
button_frame.grid(row=8, column=0, columnspan=3, pady=30)

tk.Button(button_frame, text="生成", width=12, command=generate_image).pack(side="left", padx=8)
tk.Button(button_frame, text="終了", width=12, command=close_app).pack(side="left", padx=8)

root.mainloop()