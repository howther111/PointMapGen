import math
import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont

# =========================================================
# 設定
# =========================================================

WINDOW_TITLE = "トーナメント表ジェネレーター"

DEFAULT_OUTPUT_NAME = "tournament.png"

BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"
LINE_COLOR = "black"

# トーナメント表の余白
MARGIN_LEFT = 40
MARGIN_RIGHT = 60
MARGIN_TOP = 70
MARGIN_BOTTOM = 50

# 各対戦枠のサイズ
NAME_BOX_WIDTH = 220
NAME_BOX_HEIGHT = 34

# ラウンド間の間隔
ROUND_GAP = 90

# 1回戦の参加者枠同士の縦間隔
FIRST_ROUND_VERTICAL_GAP = 14

# 線の太さ
LINE_WIDTH = 2

# 画像の最大サイズ
MAX_IMAGE_WIDTH = 30000
MAX_IMAGE_HEIGHT = 30000


# =========================================================
# フォント関連
# =========================================================

def get_japanese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    日本語対応フォントを探して読み込む。
    見つからない場合はPillowの標準フォントを使用する。
    """

    font_candidates = [
        # Windows
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/meiryob.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",

        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",

        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue

    return ImageFont.load_default()


# =========================================================
# トーナメント表作成処理
# =========================================================

def next_power_of_two(number: int) -> int:
    """
    number以上で最小となる2の累乗を返す。
    例:
        5 → 8
        9 → 16
    """
    return 1 << math.ceil(math.log2(number))


def shorten_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int
) -> str:
    """
    名前が枠の幅を超える場合、末尾を「…」で省略する。
    """

    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text

    ellipsis = "…"
    shortened = text

    while shortened:
        candidate = shortened + ellipsis
        text_width = draw.textbbox((0, 0), candidate, font=font)[2]

        if text_width <= max_width:
            return candidate

        shortened = shortened[:-1]

    return ellipsis


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont
) -> None:
    """
    指定された枠の中央に文字を描画する。
    """

    left, top, right, bottom = box

    text = shorten_text(
        draw=draw,
        text=text,
        font=font,
        max_width=(right - left) - 16
    )

    text_bbox = draw.textbbox((0, 0), text, font=font)

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_x = left + ((right - left) - text_width) / 2
    text_y = top + ((bottom - top) - text_height) / 2 - text_bbox[1]

    draw.text(
        (text_x, text_y),
        text,
        fill=TEXT_COLOR,
        font=font
    )


def create_tournament_image(
    participants: list[str],
    output_path: str,
    shuffle_enabled: bool
) -> None:
    """
    参加者一覧からトーナメント表を作成し、PNG形式で保存する。
    """

    if len(participants) < 2:
        raise ValueError("参加者を2名以上入力してください。")

    participants = participants.copy()

    if shuffle_enabled:
        random.shuffle(participants)

    bracket_size = next_power_of_two(len(participants))
    bye_count = bracket_size - len(participants)

    # BYEを一か所に偏らせないため、一定間隔で配置する
    slots: list[str] = []
    participant_index = 0

    for slot_index in range(bracket_size):
        remaining_slots = bracket_size - slot_index
        remaining_participants = len(participants) - participant_index

        if remaining_participants <= 0:
            slots.append("BYE")
            continue

        # 残り枠に対して参加者をなるべく均等に配置
        if bye_count > 0 and remaining_slots > remaining_participants:
            expected_participants = (
                (slot_index + 1) * len(participants) / bracket_size
            )

            if participant_index >= expected_participants:
                slots.append("BYE")
                bye_count -= 1
                continue

        slots.append(participants[participant_index])
        participant_index += 1

    # 万一、参加者が入りきらなかった場合の補正
    while participant_index < len(participants):
        try:
            bye_position = slots.index("BYE")
            slots[bye_position] = participants[participant_index]
            participant_index += 1
        except ValueError:
            break

    round_count = int(math.log2(bracket_size))

    first_round_pitch = NAME_BOX_HEIGHT + FIRST_ROUND_VERTICAL_GAP

    content_height = bracket_size * first_round_pitch
    image_height = MARGIN_TOP + content_height + MARGIN_BOTTOM

    # 1回戦＋以降のラウンド＋優勝者欄
    column_count = round_count + 1

    image_width = (
        MARGIN_LEFT
        + column_count * NAME_BOX_WIDTH
        + round_count * ROUND_GAP
        + MARGIN_RIGHT
    )

    if image_width > MAX_IMAGE_WIDTH or image_height > MAX_IMAGE_HEIGHT:
        raise ValueError(
            "参加者数が多すぎるため、画像サイズが上限を超えます。"
        )

    image = Image.new(
        "RGB",
        (image_width, image_height),
        BACKGROUND_COLOR
    )

    draw = ImageDraw.Draw(image)

    title_font = get_japanese_font(28)
    round_font = get_japanese_font(18)
    name_font = get_japanese_font(16)

    # タイトル

    # 各ラウンドの枠の中心Y座標
    round_centers: list[list[float]] = []

    first_round_centers = [
        MARGIN_TOP + i * first_round_pitch + NAME_BOX_HEIGHT / 2
        for i in range(bracket_size)
    ]

    round_centers.append(first_round_centers)

    for round_index in range(1, round_count + 1):
        previous_centers = round_centers[round_index - 1]

        current_centers = [
            (previous_centers[i] + previous_centers[i + 1]) / 2
            for i in range(0, len(previous_centers), 2)
        ]

        round_centers.append(current_centers)

    # ラウンド名
    round_names = []

    for round_index in range(round_count):
        remaining_players = bracket_size // (2 ** round_index)

        if remaining_players == 2:
            round_name = "決勝"
        elif remaining_players == 4:
            round_name = "準決勝"
        elif remaining_players == 8:
            round_name = "準々決勝"
        else:
            round_name = f"{round_index + 1}回戦"

        round_names.append(round_name)

    round_names.append("優勝")

    # 各ラウンドのX座標
    round_x_positions = [
        MARGIN_LEFT + i * (NAME_BOX_WIDTH + ROUND_GAP)
        for i in range(column_count)
    ]

    # ラウンド名を描画
    for round_index, round_name in enumerate(round_names):
        x = round_x_positions[round_index]

        round_bbox = draw.textbbox(
            (0, 0),
            round_name,
            font=round_font
        )

        round_text_width = round_bbox[2] - round_bbox[0]

        draw.text(
            (
                x + (NAME_BOX_WIDTH - round_text_width) / 2,
                MARGIN_TOP - 35
            ),
            round_name,
            fill=TEXT_COLOR,
            font=round_font
        )

    # 1回戦の参加者枠を描画
    first_x = round_x_positions[0]

    for participant, center_y in zip(slots, round_centers[0]):
        top = int(center_y - NAME_BOX_HEIGHT / 2)
        bottom = top + NAME_BOX_HEIGHT

        box = (
            first_x,
            top,
            first_x + NAME_BOX_WIDTH,
            bottom
        )

        draw.rectangle(
            box,
            outline=LINE_COLOR,
            width=LINE_WIDTH
        )

        draw_centered_text(
            draw=draw,
            box=box,
            text=participant,
            font=name_font
        )

    # 2回戦以降の空欄枠と接続線を描画
    for round_index in range(1, round_count + 1):
        previous_x = round_x_positions[round_index - 1]
        current_x = round_x_positions[round_index]

        previous_centers = round_centers[round_index - 1]
        current_centers = round_centers[round_index]

        for match_index, current_center_y in enumerate(current_centers):
            upper_center_y = previous_centers[match_index * 2]
            lower_center_y = previous_centers[match_index * 2 + 1]

            previous_right = previous_x + NAME_BOX_WIDTH
            middle_x = previous_right + ROUND_GAP // 2

            # 上側の枠から中央まで
            draw.line(
                (
                    previous_right,
                    upper_center_y,
                    middle_x,
                    upper_center_y
                ),
                fill=LINE_COLOR,
                width=LINE_WIDTH
            )

            # 下側の枠から中央まで
            draw.line(
                (
                    previous_right,
                    lower_center_y,
                    middle_x,
                    lower_center_y
                ),
                fill=LINE_COLOR,
                width=LINE_WIDTH
            )

            # 上下をつなぐ縦線
            draw.line(
                (
                    middle_x,
                    upper_center_y,
                    middle_x,
                    lower_center_y
                ),
                fill=LINE_COLOR,
                width=LINE_WIDTH
            )

            # 中央から次ラウンドの枠まで
            draw.line(
                (
                    middle_x,
                    current_center_y,
                    current_x,
                    current_center_y
                ),
                fill=LINE_COLOR,
                width=LINE_WIDTH
            )

            top = int(current_center_y - NAME_BOX_HEIGHT / 2)
            bottom = top + NAME_BOX_HEIGHT

            box = (
                current_x,
                top,
                current_x + NAME_BOX_WIDTH,
                bottom
            )

            draw.rectangle(
                box,
                outline=LINE_COLOR,
                width=LINE_WIDTH
            )

    image.save(output_path, "PNG")


# =========================================================
# GUI
# =========================================================

class TournamentApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title(WINDOW_TITLE)
        self.geometry("680x650")
        self.minsize(560, 500)

        self.shuffle_var = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self) -> None:
        main_frame = ttk.Frame(self, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        title_label = ttk.Label(
            main_frame,
            text="トーナメント表ジェネレーター",
            font=("", 18, "bold")
        )
        title_label.grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 12)
        )

        explanation_label = ttk.Label(
            main_frame,
            text=(
                "参加者の名前を1行につき1名入力してください。\n"
                "参加人数が2の累乗でない場合は、BYE枠が自動追加されます。"
            ),
            justify=tk.LEFT
        )
        explanation_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(0, 8)
        )

        text_frame = ttk.Frame(main_frame)
        text_frame.grid(
            row=2,
            column=0,
            sticky="nsew"
        )

        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.participant_text = tk.Text(
            text_frame,
            wrap=tk.NONE,
            font=("", 12),
            undo=True
        )
        self.participant_text.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        vertical_scrollbar = ttk.Scrollbar(
            text_frame,
            orient=tk.VERTICAL,
            command=self.participant_text.yview
        )
        vertical_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        horizontal_scrollbar = ttk.Scrollbar(
            text_frame,
            orient=tk.HORIZONTAL,
            command=self.participant_text.xview
        )
        horizontal_scrollbar.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        self.participant_text.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set
        )

        # 入力例
        self.participant_text.insert(
            "1.0",
            "山田 太郎\n"
            "佐藤 花子\n"
            "鈴木 一郎\n"
            "高橋 美咲\n"
            "田中 健太\n"
            "伊藤 さくら\n"
            "渡辺 大輔\n"
            "中村 あかり"
        )

        option_frame = ttk.Frame(main_frame)
        option_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(12, 8)
        )

        shuffle_checkbox = ttk.Checkbutton(
            option_frame,
            text="シャッフル",
            variable=self.shuffle_var
        )
        shuffle_checkbox.pack(side=tk.LEFT)

        self.count_label = ttk.Label(
            option_frame,
            text="参加者数：8名"
        )
        self.count_label.pack(side=tk.RIGHT)

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(8, 0)
        )

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        clear_button = ttk.Button(
            button_frame,
            text="入力内容をクリア",
            command=self.clear_participants
        )
        clear_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 5)
        )

        output_button = ttk.Button(
            button_frame,
            text="PNG形式で出力",
            command=self.output_tournament
        )
        output_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(5, 0)
        )

        self.status_label = ttk.Label(
            main_frame,
            text="参加者名を入力してください。",
            anchor="w"
        )
        self.status_label.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(12, 0)
        )

        self.participant_text.bind(
            "<<Modified>>",
            self.on_text_modified
        )

        self.participant_text.edit_modified(False)

    def get_participants(self) -> list[str]:
        """
        テキストボックスから空行を除いた参加者名を取得する。
        """

        text = self.participant_text.get("1.0", tk.END)

        participants = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return participants

    def on_text_modified(self, _event=None) -> None:
        if not self.participant_text.edit_modified():
            return

        participants = self.get_participants()
        self.count_label.configure(
            text=f"参加者数：{len(participants)}名"
        )

        self.participant_text.edit_modified(False)

    def clear_participants(self) -> None:
        self.participant_text.delete("1.0", tk.END)
        self.participant_text.focus_set()

        self.status_label.configure(
            text="入力内容をクリアしました。"
        )

    def output_tournament(self) -> None:
        participants = self.get_participants()

        if len(participants) < 2:
            messagebox.showwarning(
                "参加者不足",
                "参加者を2名以上入力してください。"
            )
            return

        output_path = filedialog.asksaveasfilename(
            title="トーナメント表の保存先",
            defaultextension=".png",
            initialfile=DEFAULT_OUTPUT_NAME,
            filetypes=[
                ("PNG画像", "*.png"),
                ("すべてのファイル", "*.*")
            ]
        )

        if not output_path:
            return

        try:
            self.status_label.configure(
                text="トーナメント表を作成しています..."
            )
            self.update_idletasks()

            create_tournament_image(
                participants=participants,
                output_path=output_path,
                shuffle_enabled=self.shuffle_var.get()
            )

            self.status_label.configure(
                text=f"出力しました：{output_path}"
            )

            messagebox.showinfo(
                "出力完了",
                "トーナメント表をPNG形式で出力しました。\n\n"
                f"{output_path}"
            )

        except (OSError, ValueError) as error:
            self.status_label.configure(
                text="出力に失敗しました。"
            )

            messagebox.showerror(
                "エラー",
                f"トーナメント表を出力できませんでした。\n\n{error}"
            )


# =========================================================
# 起動
# =========================================================

if __name__ == "__main__":
    app = TournamentApp()
    app.mainloop()