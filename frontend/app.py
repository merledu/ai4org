import webview
import os

def create_window():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, 'index.html')
    
    window = webview.create_window(
        'Your Personal Intelligence Platform',
        url=html_path,
        width=900,
        height=700,
        resizable=True,
        text_select=True
    )
    
    return window

if __name__ == '__main__':
    window = create_window()
    webview.start()
    