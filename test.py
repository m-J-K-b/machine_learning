import pickle

import numpy as np
import pygame as pg

from nn_core.loss import CrossEntropyLoss
from util import compute_accuracy, load_mnist_dataset


def main():
    ### load trained model
    model_path = "bert.pkl"
    with open(model_path, "rb") as f:
        nn = pickle.load(f)

    ### load dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()

    final_accuracy = compute_accuracy(y_test, nn.forward(x_test))

    ### pygame setup
    pg.init()
    RES = WIDTH, HEIGHT = 1000, 600
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()
    font = pg.font.SysFont("Arial", 20)
    pg.display.set_caption(
        f"using: {model_path} with a final accuracy of: {final_accuracy}"
    )

    ### drawing canvas setup
    drawing_surf = pg.Surface((28, 28))
    drawing_surf.fill((0, 0, 0))
    drawing_rect = drawing_surf.get_rect()
    scaled_drawing_rect = pg.Rect(50, 50, 500, 500)
    scaled_drawing_surf = pg.transform.scale(drawing_surf, scaled_drawing_rect.size)
    scaled_drawing_surf_hovered = False

    ### menu setup
    menu_surf = pg.Surface((400, 600))
    menu_rect = menu_surf.get_rect(topleft=(600, 0))

    predictions = []
    predicted_digit = None
    label_color = (255, 255, 255)  # white by default

    def _predict_button_callback():
        nonlocal predictions, predicted_digit

        drawing_surf24 = drawing_surf.convert(24)
        rgb = pg.surfarray.array3d(drawing_surf24)
        gray = np.mean(rgb, axis=-1)
        gray = gray.astype(np.float32) / 255.0
        x = gray.T.reshape(1, -1)

        all_probs = CrossEntropyLoss.softmax(nn.forward(x, training=False))
        predictions = list(all_probs.mean(axis=0) * 100)
        predicted_digit = int(np.argmax(predictions))

        if img_label is not None:
            true_label = int(img_label.split(":")[1].strip())
            _update_img_label(true_label, predicted_digit)

    ### Buttons
    predict_button_rect = pg.Rect((20, 20, 170, 50))
    predict_button_text = font.render("Predict", True, (255, 255, 255))
    predict_button_text_rect = predict_button_text.get_rect()
    predict_button_text_rect.x = (
        int((predict_button_rect.w - predict_button_text_rect.w) / 2)
        + predict_button_rect.x
    )
    predict_button_text_rect.y = (
        int((predict_button_rect.h - predict_button_text_rect.h) / 2)
        + predict_button_rect.y
    )
    predict_button_hovered = False
    predict_button_callback = _predict_button_callback

    def _clear_button_callback():
        _reset_img_label()
        drawing_surf.fill((0, 0, 0))

    clear_button_rect = pg.Rect((210, 20, 170, 50))
    clear_button_text = font.render("clear", True, (255, 255, 255))
    clear_button_text_rect = clear_button_text.get_rect()
    clear_button_text_rect.x = (
        int((clear_button_rect.w - clear_button_text_rect.w) / 2) + clear_button_rect.x
    )
    clear_button_text_rect.y = (
        int((clear_button_rect.h - clear_button_text_rect.h) / 2) + clear_button_rect.y
    )
    clear_button_hovered = False
    clear_button_callback = _clear_button_callback

    def _update_img_label(new_label, predicted=None):
        nonlocal img_label, img_label_text, img_label_rect, label_color

        img_label = f"label: {new_label}"
        if predicted is None:
            label_color = (255, 255, 255)  # white
        elif predicted == new_label:
            label_color = (0, 255, 0)  # green
        else:
            label_color = (255, 0, 0)  # red

        img_label_text = font.render(img_label, True, label_color)
        img_label_rect = img_label_text.get_rect()
        img_label_rect.x = scaled_drawing_rect.x
        img_label_rect.y = scaled_drawing_rect.y - img_label_rect.h - 5

    def _reset_img_label():
        nonlocal img_label, img_label_text, img_label_rect, label_color
        img_label = None
        img_label_text = None
        img_label_rect = None
        label_color = (255, 255, 255)

    img_label = None
    img_label_text = None
    img_label_rect = None

    def _random_img_button_callback():
        nonlocal img_label
        idx = np.random.randint(len(x_test))
        true_label = int(np.argmax(y_test[idx]))
        img = x_test[idx].reshape(28, 28).T
        gray = (img * 255).astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=2)
        pg.surfarray.blit_array(drawing_surf, rgb)

        _predict_button_callback()
        _update_img_label(true_label, predicted_digit)

    random_img_button_rect = pg.Rect((20, predict_button_rect.bottom + 20, 360, 50))
    random_img_button_text = font.render("random img", True, (255, 255, 255))
    random_img_button_text_rect = random_img_button_text.get_rect()
    random_img_button_text_rect.x = (
        int((random_img_button_rect.w - random_img_button_text_rect.w) / 2)
        + random_img_button_rect.x
    )
    random_img_button_text_rect.y = (
        int((random_img_button_rect.h - random_img_button_text_rect.h) / 2)
        + random_img_button_rect.y
    )
    random_img_button_hovered = False
    random_img_button_callback = _random_img_button_callback

    mouse_pos = [0, 0]
    mouse_btns = [False, False, False]
    prev_mouse_btns = mouse_btns
    mouse_btn_triggered = [False, False, False]

    _predict_button_callback()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        mouse_pos = pg.mouse.get_pos()
        prev_mouse_btns = mouse_btns
        mouse_btns = pg.mouse.get_pressed()
        mouse_btn_triggered = [c and not p for c, p in zip(mouse_btns, prev_mouse_btns)]

        scaled_drawing_surf_hovered = scaled_drawing_rect.collidepoint(*mouse_pos)
        predict_button_hovered = predict_button_rect.collidepoint(
            mouse_pos[0] - menu_rect.x, mouse_pos[1] - menu_rect.y
        )
        clear_button_hovered = clear_button_rect.collidepoint(
            mouse_pos[0] - menu_rect.x, mouse_pos[1] - menu_rect.y
        )
        random_img_button_hovered = random_img_button_rect.collidepoint(
            mouse_pos[0] - menu_rect.x, mouse_pos[1] - menu_rect.y
        )

        if mouse_btn_triggered[0] and predict_button_hovered:
            predict_button_callback()
        if mouse_btn_triggered[0] and clear_button_hovered:
            clear_button_callback()
        if mouse_btn_triggered[0] and random_img_button_hovered:
            random_img_button_callback()

        if mouse_btns[0] and scaled_drawing_surf_hovered:
            _reset_img_label()
            x = int(
                (mouse_pos[0] - scaled_drawing_rect.x)
                / scaled_drawing_rect.w
                * drawing_rect.w
            )
            y = int(
                (mouse_pos[1] - scaled_drawing_rect.y)
                / scaled_drawing_rect.h
                * drawing_rect.h
            )
            drawing_surf.set_at((x, y), (255, 255, 255))
            predict_button_callback()

        scaled_drawing_surf = pg.transform.scale(drawing_surf, scaled_drawing_rect.size)

        ### draw
        screen.fill((0, 0, 0))
        menu_surf.fill((30, 30, 30))

        pg.draw.rect(menu_surf, (80, 80, 80), predict_button_rect)
        pg.draw.rect(menu_surf, (80, 80, 80), clear_button_rect)
        pg.draw.rect(menu_surf, (80, 80, 80), random_img_button_rect)

        menu_surf.blit(predict_button_text, predict_button_text_rect)
        menu_surf.blit(clear_button_text, clear_button_text_rect)
        menu_surf.blit(random_img_button_text, random_img_button_text_rect)

        y = random_img_button_rect.bottom + 20
        for i, p in sorted(
            zip(range(10), predictions), key=lambda x: x[1], reverse=True
        ):
            text = font.render(f"{i}: {p:.3f}", True, (255, 255, 255))
            menu_surf.blit(text, (20, y))
            y += text.get_height()

        screen.blit(menu_surf, menu_rect)

        p = 4
        pg.draw.rect(
            screen,
            (100, 100, 100),
            (
                scaled_drawing_rect.x - p,
                scaled_drawing_rect.y - p,
                scaled_drawing_rect.w + 2 * p,
                scaled_drawing_rect.h + 2 * p,
            ),
            width=p,
        )

        screen.blit(scaled_drawing_surf, scaled_drawing_rect)

        if (
            img_label is not None
            and img_label_text is not None
            and img_label_rect is not None
        ):
            screen.blit(img_label_text, img_label_rect)

        pg.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
