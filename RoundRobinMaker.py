import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont


# =========================================================
# 基本設定
# =========================================================

WINDOW_TITLE = "ラウンドロビンメイカー"
DEFAULT_OUTPUT_NAME = "round_robin_table.png"

BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"
LINE_COLOR = "black"

# 表の外側の余白
MARGIN_LEFT = 40
MARGIN_RIGHT = 40
MARGIN_TOP = 40
MARGIN_BOTTOM = 40

# 表のセルサイズ
NAME_COLUMN_WIDTH = 220
HEADER_HEIGHT = 100
CELL_WIDTH = 76
CELL_HEIGHT = 54

LINE_WIDTH = 2

# 極端に大きな画像の生成を防ぐ上限
MAX_IMAGE_WIDTH = 30000
MAX_IMAGE_HEIGHT = 30000


# =========================================================
# フォント処理
# =========================================================

def get_japanese_font(
    size: int,
    bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    日本語を表示できるフォントを読み込む。
    """

    if bold:
        font_candidates = [
            # Windows
            "C:/Windows/Fonts/meiryob.ttc",
            "C:/Windows/Fonts/YuGothB.ttc",
            "C:/Windows/Fonts/msgothic.ttc",

            # macOS
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",

            # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Bold.otf",
        ]
    else:
        font_candidates = [
            # Windows
            "C:/Windows/Fonts/meiryo.ttc",
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
# 文字描画処理
# =========================================================

def get_text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont
) -> tuple[int, int]:
    """
    文字列の幅と高さを取得する。
    """

    bbox = draw.textbbox((0, 0), text, font=font)

    return (
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    )


def shorten_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int
) -> str:
    """
    文字列が枠幅を超える場合は末尾を「…」で省略する。
    """

    text_width, _ = get_text_size(draw, text, font)

    if text_width <= max_width:
        return text

    ellipsis = "…"
    current_text = text

    while current_text:
        candidate = current_text + ellipsis
        candidate_width, _ = get_text_size(
            draw,
            candidate,
            font
        )

        if candidate_width <= max_width:
            return candidate

        current_text = current_text[:-1]

    return ellipsis


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    padding: int = 8
) -> None:
    """
    指定した枠の中央に文字列を描画する。
    """

    left, top, right, bottom = box

    available_width = max(
        1,
        right - left - padding * 2
    )

    display_text = shorten_text(
        draw=draw,
        text=text,
        font=font,
        max_width=available_width
    )

    bbox = draw.textbbox(
        (0, 0),
        display_text,
        font=font
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = left + ((right - left) - text_width) / 2
    y = top + ((bottom - top) - text_height) / 2 - bbox[1]

    draw.text(
        (x, y),
        display_text,
        fill=TEXT_COLOR,
        font=font
    )


def draw_vertical_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont
) -> None:
    """
    上側の参加者名を1文字ずつ縦方向に描画する。
    """

    left, top, right, bottom = box

    available_height = bottom - top - 8
    center_x = (left + right) / 2

    characters = list(text)

    sample_bbox = draw.textbbox(
        (0, 0),
        "あ",
        font=font
    )

    character_height = sample_bbox[3] - sample_bbox[1]
    line_pitch = max(character_height + 2, 18)

    max_characters = max(
        1,
        available_height // line_pitch
    )

    if len(characters) > max_characters:
        if max_characters >= 2:
            characters = (
                characters[:max_characters - 1]
                + ["…"]
            )
        else:
            characters = ["…"]

    total_height = len(characters) * line_pitch
    start_y = top + (
        (bottom - top - total_height) / 2
    )

    for index, character in enumerate(characters):
        bbox = draw.textbbox(
            (0, 0),
            character,
            font=font
        )

        text_width = bbox[2] - bbox[0]

        x = center_x - text_width / 2
        y = (
            start_y
            + index * line_pitch
            - bbox[1]
        )

        draw.text(
            (x, y),
            character,
            fill=TEXT_COLOR,
            font=font
        )


# =========================================================
# 総当たり戦表生成処理
# =========================================================

def create_round_robin_image(
    participants: list[str],
    output_path: str,
    shuffle_enabled: bool
) -> list[str]:
    """
    参加者名と対戦記入欄だけの総当たり戦表を作成する。

    戻り値:
        実際に表へ配置された参加者の順番
    """

    if len(participants) < 2:
        raise ValueError(
            "参加者を2名以上入力してください。"
        )

    table_participants = participants.copy()

    if shuffle_enabled:
        random.shuffle(table_participants)

    participant_count = len(table_participants)

    table_width = (
        NAME_COLUMN_WIDTH
        + participant_count * CELL_WIDTH
    )

    table_height = (
        HEADER_HEIGHT
        + participant_count * CELL_HEIGHT
    )

    image_width = (
        MARGIN_LEFT
        + table_width
        + MARGIN_RIGHT
    )

    image_height = (
        MARGIN_TOP
        + table_height
        + MARGIN_BOTTOM
    )

    if image_width > MAX_IMAGE_WIDTH:
        raise ValueError(
            "参加者数が多すぎるため、"
            "画像の横幅が上限を超えます。"
        )

    if image_height > MAX_IMAGE_HEIGHT:
        raise ValueError(
            "参加者数が多すぎるため、"
            "画像の高さが上限を超えます。"
        )

    image = Image.new(
        "RGB",
        (image_width, image_height),
        BACKGROUND_COLOR
    )

    draw = ImageDraw.Draw(image)

    name_font = get_japanese_font(17)
    header_font = get_japanese_font(16, bold=True)
    cell_font = get_japanese_font(20, bold=True)
    number_font = get_japanese_font(15)

    table_left = MARGIN_LEFT
    table_top = MARGIN_TOP

    table_right = table_left + table_width
    table_bottom = table_top + table_height

    participant_area_left = (
        table_left + NAME_COLUMN_WIDTH
    )

    body_top = table_top + HEADER_HEIGHT

    # -----------------------------------------------------
    # 表全体の外枠
    # -----------------------------------------------------

    draw.rectangle(
        (
            table_left,
            table_top,
            table_right,
            table_bottom
        ),
        outline=LINE_COLOR,
        width=LINE_WIDTH
    )

    # -----------------------------------------------------
    # 左上セル
    # -----------------------------------------------------

    corner_box = (
        table_left,
        table_top,
        participant_area_left,
        body_top
    )

    draw.rectangle(
        corner_box,
        outline=LINE_COLOR,
        width=LINE_WIDTH
    )

    # 左上セルに斜線
    draw.line(
        (
            table_left,
            table_top,
            participant_area_left,
            body_top
        ),
        fill=LINE_COLOR,
        width=LINE_WIDTH
    )

    # 左上セルの補助見出し
    draw.text(
        (
            table_left + 12,
            body_top - 30
        ),
        "参加者",
        fill=TEXT_COLOR,
        font=number_font
    )

    draw.text(
        (
            participant_area_left - 55,
            table_top + 10
        ),
        "対戦",
        fill=TEXT_COLOR,
        font=number_font
    )

    # -----------------------------------------------------
    # 上側の参加者名
    # -----------------------------------------------------

    for column_index, participant in enumerate(
        table_participants
    ):
        cell_left = (
            participant_area_left
            + column_index * CELL_WIDTH
        )

        cell_right = cell_left + CELL_WIDTH

        header_box = (
            cell_left,
            table_top,
            cell_right,
            body_top
        )

        draw.rectangle(
            header_box,
            outline=LINE_COLOR,
            width=LINE_WIDTH
        )

        # 参加者番号
        number_text = str(column_index + 1)

        number_width, _ = get_text_size(
            draw,
            number_text,
            number_font
        )

        draw.text(
            (
                cell_left
                + (CELL_WIDTH - number_width) / 2,
                table_top + 5
            ),
            number_text,
            fill=TEXT_COLOR,
            font=number_font
        )

        # 参加者名を縦書き
        name_box = (
            cell_left,
            table_top + 24,
            cell_right,
            body_top
        )

        draw_vertical_text(
            draw=draw,
            box=name_box,
            text=participant,
            font=header_font
        )

    # -----------------------------------------------------
    # 左側の参加者名と対戦記入欄
    # -----------------------------------------------------

    for row_index, participant in enumerate(
        table_participants
    ):
        row_top = (
            body_top
            + row_index * CELL_HEIGHT
        )

        row_bottom = row_top + CELL_HEIGHT

        # 左側の参加者名欄
        name_box = (
            table_left,
            row_top,
            participant_area_left,
            row_bottom
        )

        draw.rectangle(
            name_box,
            outline=LINE_COLOR,
            width=LINE_WIDTH
        )

        number_column_width = 42

        # 番号欄との区切り線
        draw.line(
            (
                table_left + number_column_width,
                row_top,
                table_left + number_column_width,
                row_bottom
            ),
            fill=LINE_COLOR,
            width=LINE_WIDTH
        )

        # 参加者番号
        draw_centered_text(
            draw=draw,
            box=(
                table_left,
                row_top,
                table_left + number_column_width,
                row_bottom
            ),
            text=str(row_index + 1),
            font=number_font
        )

        # 参加者名
        draw_centered_text(
            draw=draw,
            box=(
                table_left + number_column_width,
                row_top,
                participant_area_left,
                row_bottom
            ),
            text=participant,
            font=name_font
        )

        # 対戦記入欄
        for column_index in range(
            participant_count
        ):
            cell_left = (
                participant_area_left
                + column_index * CELL_WIDTH
            )

            cell_right = cell_left + CELL_WIDTH

            result_box = (
                cell_left,
                row_top,
                cell_right,
                row_bottom
            )

            draw.rectangle(
                result_box,
                outline=LINE_COLOR,
                width=LINE_WIDTH
            )

            # 自分自身との対戦欄だけ横線を表示
            if row_index == column_index:
                draw_centered_text(
                    draw=draw,
                    box=result_box,
                    text="―",
                    font=cell_font
                )

    image.save(output_path, "PNG")

    return table_participants


# =========================================================
# GUIアプリ
# =========================================================

class RoundRobinApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title(WINDOW_TITLE)
        self.geometry("700x680")
        self.minsize(560, 500)

        self.shuffle_var = tk.BooleanVar(
            value=False
        )

        self.create_widgets()

    def create_widgets(self) -> None:
        main_frame = ttk.Frame(
            self,
            padding=16
        )

        main_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        main_frame.columnconfigure(
            0,
            weight=1
        )

        main_frame.rowconfigure(
            2,
            weight=1
        )

        app_title_label = ttk.Label(
            main_frame,
            text="ラウンドロビンメイカー",
            font=("", 18, "bold")
        )

        app_title_label.grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 12)
        )

        explanation_label = ttk.Label(
            main_frame,
            text=(
                "参加者の名前を1行につき1名入力してください。\n"
                "参加者名と対戦記入欄だけの表をPNG形式で出力します。"
            ),
            justify=tk.LEFT
        )

        explanation_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(0, 8)
        )

        # -------------------------------------------------
        # 参加者入力欄
        # -------------------------------------------------

        text_frame = ttk.Frame(main_frame)

        text_frame.grid(
            row=2,
            column=0,
            sticky="nsew"
        )

        text_frame.columnconfigure(
            0,
            weight=1
        )

        text_frame.rowconfigure(
            0,
            weight=1
        )

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

        self.participant_text.insert(
            "1.0",
            "山田 太郎\n"
            "佐藤 花子\n"
            "鈴木 一郎\n"
            "高橋 美咲\n"
            "田中 健太\n"
            "伊藤 さくら"
        )

        # -------------------------------------------------
        # オプション
        # -------------------------------------------------

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
            text="参加者数：6名"
        )

        self.count_label.pack(side=tk.RIGHT)

        # -------------------------------------------------
        # 操作ボタン
        # -------------------------------------------------

        button_frame = ttk.Frame(main_frame)

        button_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(8, 0)
        )

        button_frame.columnconfigure(
            0,
            weight=1
        )

        button_frame.columnconfigure(
            1,
            weight=1
        )

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
            command=self.output_round_robin_table
        )

        output_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(5, 0)
        )

        # -------------------------------------------------
        # ステータス表示
        # -------------------------------------------------

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
        入力欄から空行を除外して参加者名を取得する。
        """

        text = self.participant_text.get(
            "1.0",
            tk.END
        )

        return [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    def find_duplicate_names(
        self,
        participants: list[str]
    ) -> list[str]:
        """
        重複している参加者名を取得する。
        """

        seen = set()
        duplicates = []

        for participant in participants:
            if (
                participant in seen
                and participant not in duplicates
            ):
                duplicates.append(participant)

            seen.add(participant)

        return duplicates

    def on_text_modified(
        self,
        _event=None
    ) -> None:
        """
        参加者数の表示を更新する。
        """

        if not self.participant_text.edit_modified():
            return

        participants = self.get_participants()

        self.count_label.configure(
            text=f"参加者数：{len(participants)}名"
        )

        self.participant_text.edit_modified(False)

    def clear_participants(self) -> None:
        """
        入力欄を空にする。
        """

        self.participant_text.delete(
            "1.0",
            tk.END
        )

        self.participant_text.focus_set()

        self.status_label.configure(
            text="入力内容をクリアしました。"
        )

    def output_round_robin_table(self) -> None:
        """
        総当たり戦表をPNG形式で出力する。
        """

        participants = self.get_participants()

        if len(participants) < 2:
            messagebox.showwarning(
                "参加者不足",
                "参加者を2名以上入力してください。"
            )
            return

        duplicate_names = self.find_duplicate_names(
            participants
        )

        if duplicate_names:
            duplicate_text = "\n".join(
                duplicate_names
            )

            continue_output = messagebox.askyesno(
                "参加者名の重複",
                "同じ参加者名が複数入力されています。\n\n"
                f"{duplicate_text}\n\n"
                "このまま出力しますか？"
            )

            if not continue_output:
                return

        output_path = filedialog.asksaveasfilename(
            title="総当たり戦表の保存先",
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
                text="総当たり戦表を作成しています..."
            )

            self.update_idletasks()

            arranged_participants = (
                create_round_robin_image(
                    participants=participants,
                    output_path=output_path,
                    shuffle_enabled=self.shuffle_var.get()
                )
            )

            self.status_label.configure(
                text=f"出力しました：{output_path}"
            )

            if self.shuffle_var.get():
                arranged_text = "\n".join(
                    f"{index + 1}. {name}"
                    for index, name
                    in enumerate(arranged_participants)
                )

                messagebox.showinfo(
                    "出力完了",
                    "総当たり戦表をPNG形式で出力しました。\n\n"
                    f"{output_path}\n\n"
                    "シャッフル後の順番：\n"
                    f"{arranged_text}"
                )
            else:
                messagebox.showinfo(
                    "出力完了",
                    "総当たり戦表をPNG形式で出力しました。\n\n"
                    f"{output_path}"
                )

        except (OSError, ValueError) as error:
            self.status_label.configure(
                text="出力に失敗しました。"
            )

            messagebox.showerror(
                "エラー",
                "総当たり戦表を出力できませんでした。\n\n"
                f"{error}"
            )


# =========================================================
# アプリ起動
# =========================================================

if __name__ == "__main__":
    app = RoundRobinApp()
    app.mainloop()