
def show_text(title: str, text: str, window: bool = True) -> None:
    if not window:
        print(f"\n=== {title} ===\n{text}")
        return
    try:
        import tkinter as tk
        from tkinter import filedialog, font, scrolledtext

        root = tk.Tk()
        root.title(title)
        root.geometry("920x640")
        root.attributes('-topmost', True)
        root.after(100, lambda: root.attributes('-topmost', False))
        mono = font.Font(family="Consolas" if "Consolas" in font.families() else "Courier New", size=10)
        box = scrolledtext.ScrolledText(root, wrap="none", font=mono)
        xbar = tk.Scrollbar(root, orient="horizontal", command=box.xview)
        box.configure(xscrollcommand=xbar.set)
        bar = tk.Frame(root)
        bar.pack(side="bottom", fill="x")
        xbar.pack(side="bottom", fill="x")
        box.pack(side="top", fill="both", expand=True)
        box.insert("1.0", text)
        box.configure(state="disabled")

        def copy():
            root.clipboard_clear()
            root.clipboard_append(text)

        def save():
            path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Текст", "*.txt")], initialfile="result.txt")
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

        tk.Button(bar, text="Копировать", command=copy).pack(side="left", padx=6, pady=4)
        tk.Button(bar, text="Сохранить…", command=save).pack(side="left", pady=4)
        tk.Button(bar, text="Закрыть", command=root.destroy).pack(side="right", padx=6, pady=4)
        root.mainloop()
    except Exception:
        print(f"\n=== {title} ===\n{text}")
