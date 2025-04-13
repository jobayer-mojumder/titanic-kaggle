import time

def handle_viewer(task_name, fn):
    print(f"\n📊 {task_name}")
    start = time.time()
    fn()
    print(f"\n⏱️ Completed in {time.time() - start:.2f}s (viewer)")
    input("🔍 Press Enter to return to menu...")