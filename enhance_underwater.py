import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk



def enhance_underwater(img, red_boost, brightness, clip_limit):
    img_float = img.astype(np.float32) / 255.0

    
    avg_r = np.mean(img_float[:, :, 2])
    avg_g = np.mean(img_float[:, :, 1])
    avg_b = np.mean(img_float[:, :, 0])
    avg_gray = (avg_r + avg_g + avg_b) / 3

    img_float[:, :, 2] *= (avg_gray / (avg_r + 1e-6))
    img_float[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
    img_float[:, :, 0] *= (avg_gray / (avg_b + 1e-6))

    
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * red_boost, 0, 1)

    
    img_float = np.clip(img_float * brightness, 0, 1)

    img_uint8 = (img_float * 255).astype(np.uint8)

    
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced



def load_preview():
    global preview_img, preview_original

    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    preview_original = cv2.imread(file_path)

    update_preview()



def update_preview(event=None):
    if preview_original is None:
        return

    red_val = red_slider.get()
    bright_val = bright_slider.get()
    clahe_val = clahe_slider.get()

    enhanced = enhance_underwater(preview_original.copy(), red_val, bright_val, clahe_val)

    
    display = cv2.resize(enhanced, (400, 300))

    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(display)
    img_tk = ImageTk.PhotoImage(img_pil)

    preview_label.config(image=img_tk)
    preview_label.image = img_tk


def process_folder():
    input_folder = input_var.get()
    output_folder = output_var.get()

    if not input_folder or not output_folder:
        messagebox.showerror("Error", "Please select folders")
        return

    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

    if not files:
        messagebox.showerror("Error", "No images found")
        return

    progress["maximum"] = len(files)

    red_val = red_slider.get()
    bright_val = bright_slider.get()
    clahe_val = clahe_slider.get()

    for i, file in enumerate(files, 1):
        try:
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            img = cv2.imread(input_path)
            if img is None:
                continue

            enhanced = enhance_underwater(img, red_val, bright_val, clahe_val)
            cv2.imwrite(output_path, enhanced)

            progress["value"] = i
            status_label.config(text=f"{i}/{len(files)} - {file}")
            root.update_idletasks()

        except Exception as e:
            print(f"Error processing {file}: {e}")

    messagebox.showinfo("Done", "Processing complete!")
    status_label.config(text=" Complete")



def select_input():
    input_var.set(filedialog.askdirectory())


def select_output():
    output_var.set(filedialog.askdirectory())


root = tk.Tk()
root.title("Underwater Enhancer")
root.geometry("700x650")

input_var = tk.StringVar()
output_var = tk.StringVar()

preview_original = None
preview_img = None



tk.Label(root, text="Input Folder").pack()
tk.Entry(root, textvariable=input_var, width=60).pack()
tk.Button(root, text="Browse Input", command=select_input).pack(pady=5)


tk.Label(root, text="Output Folder").pack()
tk.Entry(root, textvariable=output_var, width=60).pack()
tk.Button(root, text="Browse Output", command=select_output).pack(pady=5)



tk.Button(root, text="Load Preview Image", command=load_preview).pack(pady=10)

preview_label = tk.Label(root)
preview_label.pack()



tk.Label(root, text="Red Boost (0.5–2.5)").pack()
red_slider = tk.Scale(root, from_=0.5, to=2.5, resolution=0.1,
                      orient="horizontal", command=update_preview)
red_slider.set(1.2)
red_slider.pack()

tk.Label(root, text="Brightness (0.5–2.0)").pack()
bright_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1,
                         orient="horizontal", command=update_preview)
bright_slider.set(1.0)
bright_slider.pack()

tk.Label(root, text="Contrast (CLAHE 0.5–6.0)").pack()
clahe_slider = tk.Scale(root, from_=0.5, to=6.0, resolution=0.5,
                        orient="horizontal", command=update_preview)
clahe_slider.set(2.0)
clahe_slider.pack()



progress = ttk.Progressbar(root, length=400)
progress.pack(pady=10)

status_label = tk.Label(root, text="Idle")
status_label.pack()



tk.Button(root, text="Start",
          command=process_folder,
          bg="green", fg="white").pack(pady=15)


tk.Button(root, text="Quit",
          command=root.destroy,
          bg="red", fg="white").pack(pady=5)


root.mainloop()

