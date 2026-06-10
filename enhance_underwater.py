import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading


def white_balance(img):
    try:
        wb = cv2.xphoto.createSimpleWB()
        return wb.balanceWhite(img)
    except:
        return img


def dehaze_simple(img, strength):
    if strength == 0:
        return img

    img = img.astype(np.float32)
    kernel = np.ones((15, 15), np.uint8)
    dark = cv2.erode(np.min(img, axis=2), kernel)

    transmission = 1 - strength * (dark / 255.0)
    transmission = np.clip(transmission, 0.2, 1)

    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = img[:, :, c] / transmission

    return np.clip(result, 0, 255).astype(np.uint8)


def stretch_channels(img):
    result = np.zeros_like(img)
    for i in range(3):
        channel = img[:, :, i]
        min_val = np.percentile(channel, 1)
        max_val = np.percentile(channel, 99)
        result[:, :, i] = np.clip(
            (channel - min_val) * 255 / (max_val - min_val + 1e-6),
            0, 255
        )
    return result.astype(np.uint8)


def denoise(img, strength):
    if strength == 0:
        return img
    return cv2.fastNlMeansDenoisingColored(
        img, None,
        h=10 * strength,
        hColor=10 * strength,
        templateWindowSize=7,
        searchWindowSize=21
    )


def sharpen(img, strength):
    if strength == 0:
        return img
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + strength, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def enhance_underwater(img, red_boost, brightness, clip_limit,
                       dehaze_strength, sharp_strength, noise_strength):

    if (red_boost == 1.0 and brightness == 1.0 and clip_limit == 0.0 and
        dehaze_strength == 0.0 and sharp_strength == 0 and noise_strength == 0):
        return img

    img = white_balance(img)
    img = dehaze_simple(img, dehaze_strength)

    img_float = img.astype(np.float32) / 255.0

    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * red_boost, 0, 1)
    img_float = np.clip(img_float * brightness, 0, 1)

    img = (img_float * 255).astype(np.uint8)

    if clip_limit > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    img = stretch_channels(img)
    img = denoise(img, noise_strength)
    img = sharpen(img, sharp_strength)

    return img


update_job = None
preview_original = None


def load_preview():
    global preview_original

    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    preview_original = cv2.imread(file_path)

    red_slider.set(1.0)
    bright_slider.set(1.0)
    clahe_slider.set(0.0)
    dehaze_slider.set(0.0)
    sharp_slider.set(0)
    noise_slider.set(0)

    update_preview()


def update_preview(event=None):
    global update_job

    if preview_original is None:
        return

    if update_job is not None:
        root.after_cancel(update_job)

    update_job = root.after(120, start_preview_thread)


def start_preview_thread():
    threading.Thread(target=render_preview, daemon=True).start()


def render_preview():
    if preview_original is None:
        return

    preview_small = cv2.resize(preview_original, (500, 400))

    enhanced = enhance_underwater(
        preview_small,
        red_slider.get(),
        bright_slider.get(),
        clahe_slider.get(),
        dehaze_slider.get(),
        sharp_slider.get(),
        noise_slider.get()
    )

    display = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display)
    img_tk = ImageTk.PhotoImage(img_pil)

    root.after(0, update_image_label, img_tk)


def update_image_label(img_tk):
    preview_label.config(image=img_tk)
    preview_label.image = img_tk


def process_folder():
    threading.Thread(target=process_folder_worker, daemon=True).start()


def process_folder_worker():
    input_folder = input_var.get()
    output_folder = output_var.get()

    if not input_folder or not output_folder:
        root.after(0, lambda: messagebox.showerror("Error", "Select folders"))
        return

    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

    root.after(0, lambda: progress.config(maximum=len(files)))

    for i, file in enumerate(files, 1):
        img = cv2.imread(os.path.join(input_folder, file))
        if img is None:
            continue

        enhanced = enhance_underwater(
            img,
            red_slider.get(),
            bright_slider.get(),
            clahe_slider.get(),
            dehaze_slider.get(),
            sharp_slider.get(),
            noise_slider.get()
        )

        cv2.imwrite(os.path.join(output_folder, file), enhanced)

        root.after(0, update_progress_ui, i, file, len(files))

    root.after(0, lambda: messagebox.showinfo("Done", "Processing complete!"))


def update_progress_ui(i, file, total):
    progress["value"] = i
    status_label.config(text=f"{i}/{total} - {file}")


def select_input():
    folder = filedialog.askdirectory()
    if folder:
        input_var.set(folder)
        input_label.config(text=folder)


def select_output():
    folder = filedialog.askdirectory()
    if folder:
        output_var.set(folder)
        output_label.config(text=folder)



root = tk.Tk()
root.title("Underwater Enhancer (Threaded)")
root.geometry("700x900")

input_var = tk.StringVar()
output_var = tk.StringVar()


folder_frame = tk.Frame(root)
folder_frame.pack(pady=10)

tk.Button(folder_frame, text=" Input", command=select_input).grid(row=0, column=0, padx=15)

tk.Button(folder_frame, text=" Output", command=select_output).grid(row=0, column=2, padx=15)


input_label = tk.Label(root, text="No input selected", wraplength=600, fg="gray")
input_label.pack()

output_label = tk.Label(root, text="No output selected", wraplength=600, fg="gray")
output_label.pack(pady=(0, 10))


tk.Button(root, text="Load Preview Image", command=load_preview).pack(pady=10)

preview_label = tk.Label(root)
preview_label.pack()


slider_frame = tk.Frame(root)
slider_frame.pack(pady=10)

left_frame = tk.Frame(slider_frame)
left_frame.grid(row=0, column=0, padx=20)

right_frame = tk.Frame(slider_frame)
right_frame.grid(row=0, column=1, padx=20)

tk.Label(left_frame, text="Red Boost").pack()
red_slider = tk.Scale(left_frame, from_=0.5, to=2.5, resolution=0.1,
                      orient="horizontal", command=update_preview)
red_slider.set(1.0)
red_slider.pack()

tk.Label(left_frame, text="CLAHE Contrast").pack()
clahe_slider = tk.Scale(left_frame, from_=0.0, to=6.0, resolution=0.5,
                        orient="horizontal", command=update_preview)
clahe_slider.set(0.0)
clahe_slider.pack()

tk.Label(left_frame, text="Sharpen Strength").pack()
sharp_slider = tk.Scale(left_frame, from_=0, to=5, resolution=1,
                        orient="horizontal", command=update_preview)
sharp_slider.set(0)
sharp_slider.pack()

tk.Label(right_frame, text="Brightness").pack()
bright_slider = tk.Scale(right_frame, from_=0.5, to=2.0, resolution=0.1,
                         orient="horizontal", command=update_preview)
bright_slider.set(1.0)
bright_slider.pack()

tk.Label(right_frame, text="Dehaze Strength").pack()
dehaze_slider = tk.Scale(right_frame, from_=0.0, to=1.0, resolution=0.1,
                         orient="horizontal", command=update_preview)
dehaze_slider.set(0.0)
dehaze_slider.pack()

tk.Label(right_frame, text="Noise Reduction").pack()
noise_slider = tk.Scale(right_frame, from_=0, to=3, resolution=1,
                        orient="horizontal", command=update_preview)
noise_slider.set(0)
noise_slider.pack()


progress = ttk.Progressbar(root, length=400)
progress.pack(pady=10)

status_label = tk.Label(root, text="Idle")
status_label.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=15)

tk.Button(button_frame, text="Start",
          command=process_folder,
          bg="green", fg="white", width=12).grid(row=0, column=0, padx=10)

tk.Button(button_frame, text="Quit",
          command=root.destroy,
          bg="red", fg="white", width=12).grid(row=0, column=1, padx=10)

root.mainloop()
