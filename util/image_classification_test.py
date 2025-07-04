import pickle
import tkinter as tk
from collections import deque
from enum import Enum, auto
from pathlib import Path
from tkinter import filedialog
from typing import Callable, List, Optional, Union

import h5py
import numpy as np
import pygame as pg
from numpy.typing import NDArray

from nn_core import Network
from nn_core.util import softmax


def open_file_browser(filetypes=None):
    filetypes = filetypes or [("All files", "*.*")]
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(title="Select a file", filetypes=filetypes)

    root.destroy()

    return file_path


class Commons:
    path = None

    clock: Optional[pg.time.Clock] = None
    font: Optional[pg.font.Font] = None

    mouse_pos: List[int] = [0, 0]
    mouse_btns: List[bool] = [False, False, False]
    prev_mouse_btns: List[bool] = [False, False, False]
    mouse_btns_triggered: List[bool] = [False, False, False]

    t: int = 0  # total time elapsed in ms
    dt: int = 0  # time since last update in ms

    @classmethod
    def init(cls):
        cls.path = Path(__file__).parent
        cls.clock = pg.time.Clock()
        cls.font = pg.font.Font(cls.path / "res/JetBrainsMonoNL-Regular.ttf", 15)

    @classmethod
    def update(cls):
        cls.dt = cls.clock.tick()
        cls.t += cls.dt

        cls.prev_mouse_btns = cls.mouse_btns
        cls.mouse_btns = pg.mouse.get_pressed()
        cls.mouse_btns_triggered = [
            n and not p for (n, p) in zip(cls.mouse_btns, cls.prev_mouse_btns)
        ]

        cls.mouse_pos = pg.mouse.get_pos()

    DEFAULT_COLOR = (40, 40, 40)
    HOVERED_COLOR = (70, 70, 70)
    CLICKED_COLOR = (90, 90, 90)


class ButtonStates(Enum):
    DEFAULT = auto()
    HOVERED = auto()
    CLICKED = auto()
    TRIGGERED = auto()


class Button:
    def __init__(self, rect: pg.Rect, label: str, callback: Optional[Callable] = None):
        self.rect = rect
        self.label = label
        self._callback = callback

        self.state = ButtonStates.DEFAULT

        self.last_click_ts = 0
        self.trigger_interval = 500

        self.style = None
        self.init_style()

        self.rendered_label = None
        self.label_rect = None
        self.render_label()

    def init_style(self):
        self.style = {
            ButtonStates.DEFAULT: {
                "bg color": (40, 40, 40),
                "label color": (255, 255, 255),
            },
            ButtonStates.HOVERED: {
                "bg color": (50, 50, 50),
            },
            ButtonStates.CLICKED: {
                "bg color": (70, 70, 70),
            },
        }
        self.style[ButtonStates.TRIGGERED] = self.style[ButtonStates.CLICKED]

    def get_style_prop(self, prop: str):
        return self.style[self.state].get(prop, self.style[ButtonStates.DEFAULT][prop])

    def set_callback(self, callback: callable):
        self._callback = callback

    def update_label(self, new_label: str):
        self.label = new_label
        self.render_label()

    def render_label(self):
        label_color = self.get_style_prop("label color")
        self.rendered_label = Commons.font.render(self.label, True, label_color)
        self.label_rect = self.rendered_label.get_rect()
        self.label_rect.x = int(self.rect.x + (self.rect.w - self.label_rect.w) / 2)
        self.label_rect.y = int(self.rect.y + (self.rect.h - self.label_rect.h) / 2)

    def update(self):
        previous_state = self.state

        if self.rect.collidepoint(Commons.mouse_pos):
            if Commons.mouse_btns[0]:
                if (
                    Commons.mouse_btns_triggered[0]
                    or Commons.t - self.last_click_ts > self.trigger_interval
                ):
                    self.last_click_ts = Commons.t
                    self.state = ButtonStates.TRIGGERED
                else:
                    self.state = ButtonStates.CLICKED
            else:
                self.state = ButtonStates.HOVERED
        else:
            self.state = ButtonStates.DEFAULT

        if previous_state != self.state:
            self.render_label()

        if self.state == ButtonStates.TRIGGERED and callable(self._callback):
            self._callback()

    def draw(self, surface: pg.Surface):
        pg.draw.rect(surface, self.get_style_prop("bg color"), self.rect)
        surface.blit(self.rendered_label, self.label_rect)


class ClassificationDataset:
    def __init__(
        self, x: NDArray, y: NDArray, label_names=None, preprocessing: callable = None
    ):
        """This class is a container for a classification dataset. It expects the output of the neural networks to be one hot encoded

        Args:
            x (NDArray): The input for the neural network
            y (NDArray): The outpu for the nerual network.
        """

        self.x = x
        self.y = y

        num_classes = len(np.unique(y))
        self.y_onehot = np.eye(num_classes)[y]
        self.label_names = (
            label_names if label_names is not None else np.arange(num_classes)
        )

    @property
    def length(self):
        return len(self.x)

    def compute_network_accuracy(self, nn: Network):
        """Computes the networks accuracy over the entire dataset

        Args:
            nn (Network): The network to be tested

        Returns:
            float: scalar describing the accuracy of the network 0 -> bad 1 -> good
        """
        y_pred = nn.forward(self.x, training=False)
        return np.mean(np.argmax(y_pred, axis=1) == self.y)

    def get_label_name(self, label: int):
        return self.label_names[label]

    def get_label_name_by_idx(self, idx: int):
        return self.label_names[self.y[idx]]

    def get_label_name_by_prediction(self, pred: NDArray):
        return self.label_names[np.argmax(pred)]

    def get_x_at_index(self, idx: int) -> NDArray | float:
        """Get the input at index

        Args:
            idx (int): index of the input that is desired

        Returns:
            NDArray | float: the input at index
        """
        return self.x[idx][None, :]

    def verify(self, y_pred: NDArray, idx: int) -> bool:
        return np.argmax(y_pred) == self.y[idx]


# Color constants
MENU_BG = (30, 30, 30)
PANEL_BG = (50, 50, 100)
TEXT_COLOR = (230, 230, 230)
CORRECT_COLOR = (0, 200, 0)
WRONG_COLOR = (200, 0, 0)
BORDER_WIDTH = 4


class Tester:
    """
    Manages multiple ClassificationDatasets and a trained Network, handles accuracy updates,
    image selection, and dataset rotation.
    """

    def __init__(self):
        self.network: Optional[Network] = None
        self.network_name: Optional[str] = None
        self.datasets: deque[dict] = deque()
        self.current_index: int = 0
        self.prediction: Optional[np.ndarray] = None
        self.correct: Optional[bool] = None

    def _safe_label_names(self, labels, num_classes: int) -> np.ndarray:
        if labels is None:
            return np.array([str(i) for i in range(num_classes)], dtype=str)
        arr = np.array(labels, dtype=str)
        if arr.size != num_classes:
            raise ValueError(f"Expected {num_classes} label names, got {arr.size}")
        return arr

    def update_accuracy(self) -> None:
        if not self.network:
            return
        for entry in self.datasets:
            entry["accuracy"] = entry["dataset"].compute_network_accuracy(self.network)

    def load_network(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Network file not found: {path}")
        with path.open("rb") as f:
            self.network = pickle.load(f)
        self.network_name = path.stem
        self.update_accuracy()

    def load_dataset(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.datasets = deque()
        with h5py.File(path, "r") as f:
            train_x = f["train/images"][:] / 255.0
            train_y = f["train/labels"][:]
            test_x = f["test/images"][:] / 255.0
            test_y = f["test/labels"][:]
            raw_labels = f.attrs.get("label_names", None)
        num_classes = len(np.unique(train_y))
        labels = self._safe_label_names(raw_labels, num_classes)
        for split, x, y in (("train", train_x, train_y), ("test", test_x, test_y)):
            ds = ClassificationDataset(x, y, labels)
            self.datasets.append(
                {
                    "name": f"{path.stem}/{split}",
                    "path": path,
                    "dataset": ds,
                    "accuracy": None,
                }
            )
        self.update_accuracy()

    def randomize_image(self) -> None:
        if not self.datasets:
            return
        ds = self.datasets[0]["dataset"]
        self.current_index = np.random.randint(0, ds.length)
        self._update_prediction()

    def rotate_datasets(self) -> None:
        self.datasets.rotate(1)
        self.randomize_image()

    def _update_prediction(self) -> None:
        if not self.network or not self.datasets:
            self.prediction, self.correct = None, None
            return
        ds = self.datasets[0]["dataset"]
        x = ds.get_x_at_index(self.current_index)
        probs = self.network.predict_probabilities(x)
        self.prediction = probs[0] if probs.ndim > 1 else probs
        self.correct = ds.verify(self.prediction, self.current_index)

    def get_image_surface(self, size: tuple[int, int]) -> pg.Surface:
        if not self.datasets:
            return pg.Surface((0, 0))
        ds = self.datasets[0]["dataset"]
        arr = ds.get_x_at_index(self.current_index)
        arr = (arr.reshape(28, 28).T[..., None] * 255).repeat(3, axis=2)
        surf = pg.surfarray.make_surface(arr.astype("uint8"))
        return pg.transform.scale(surf, size)


class App:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((1000, 700))
        Commons.init()
        self.tester = Tester()
        self.menu_rect = pg.Rect(0, 0, 300, 700)
        self.image_rect = pg.Rect(400, 100, 500, 500)
        self.buttons: List[Button] = []
        self._setup_buttons()
        self.running = True

    def _setup_buttons(self):
        padding = 20
        btn_w, btn_h = 260, 50
        y = padding

        def add_btn(label: str, cb: Callable):
            nonlocal y
            btn = Button(pg.Rect(padding, y, btn_w, btn_h), label, cb)
            self.buttons.append(btn)
            y += btn_h + padding

        add_btn("Load Network", self._on_load_network)
        add_btn("Load Dataset", self._on_load_dataset)
        y = 700 - 2 * (btn_h + padding)
        add_btn("Random Image", self.tester.randomize_image)
        add_btn("Rotate Datasets", self.tester.rotate_datasets)

    def _on_load_network(self):
        path = open_file_browser(filetypes=[("Pickle files", ".pkl")])
        if path:
            self.tester.load_network(path)

    def _on_load_dataset(self):
        path = open_file_browser(filetypes=[("HDF5 files", ".h5")])
        if path:
            self.tester.load_dataset(path)

    def run(self):
        while self.running:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    self.running = False
            Commons.update()
            for btn in self.buttons:
                btn.update()
            self._draw()
            pg.display.flip()
        pg.quit()

    def _draw(self):
        self.screen.fill(PANEL_BG)
        pg.draw.rect(self.screen, MENU_BG, self.menu_rect)

        for btn in self.buttons:
            btn.draw(self.screen)

        x0 = self.menu_rect.left + 20
        y0 = self.buttons[1].rect.bottom + 20
        net_txt = f"Network: {self.tester.network_name or 'None'}"
        self.screen.blit(Commons.font.render(net_txt, True, TEXT_COLOR), (x0, y0))

        y1 = y0 + 35
        self.screen.blit(
            Commons.font.render(
                "Datasets:" + (" None" if len(self.tester.datasets) == 0 else ""),
                True,
                TEXT_COLOR,
            ),
            (x0, y1),
        )
        y1 += 20
        for i, entry in enumerate(self.tester.datasets):
            name = entry["name"]
            acc = entry.get("accuracy", 0.0) or 0.0
            prefix = "-> " if i == 0 else "   "

            name_surf = Commons.font.render(f"{prefix}{name}", True, TEXT_COLOR)
            self.screen.blit(name_surf, (x0, y1))

            acc_surf = Commons.font.render(f"{acc*100:.2f}%", True, TEXT_COLOR)
            acc_x = self.menu_rect.right - acc_surf.get_width() - 10
            self.screen.blit(acc_surf, (acc_x, y1))
            y1 += 20

        surf = self.tester.get_image_surface((self.image_rect.w, self.image_rect.h))
        self.screen.blit(surf, self.image_rect.topleft)
        if self.tester.correct is not None:
            color = CORRECT_COLOR if self.tester.correct else WRONG_COLOR
            border_rect = self.image_rect.inflate(BORDER_WIDTH, BORDER_WIDTH)
            pg.draw.rect(self.screen, color, border_rect, BORDER_WIDTH)

        if self.tester.prediction is not None and self.tester.datasets:
            ds = self.tester.datasets[0]["dataset"]
            true_lbl = ds.get_label_name_by_idx(self.tester.current_index)
            pred_lbl = ds.get_label_name_by_prediction(self.tester.prediction)
            label_txt = f"{true_lbl} -> {pred_lbl}"
            lbl_surf = Commons.font.render(label_txt, True, TEXT_COLOR)
            self.screen.blit(lbl_surf, (self.image_rect.x, self.image_rect.bottom + 10))

            prob_title = Commons.font.render("Prediction:", True, (255, 255, 255))
            self.screen.blit(prob_title, (20, y1 + 30))
            y1 += 30
            px = 20
            py = y1 + 20
            for i, (lbl, p) in enumerate(
                sorted(
                    zip(ds.label_names, self.tester.prediction),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ):
                lbl_txt = f"   {lbl}:"
                self.screen.blit(
                    Commons.font.render(lbl_txt, True, TEXT_COLOR), (px, py + i * 20)
                )

                prob_txt = f"{p*100:.2f}%"
                prob_surf = Commons.font.render(prob_txt, True, TEXT_COLOR)
                self.screen.blit(
                    prob_surf,
                    (self.menu_rect.right - 10 - prob_surf.get_width(), py + i * 20),
                )


if __name__ == "__main__":
    app = App()
    app.run()
