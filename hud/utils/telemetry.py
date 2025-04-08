from __future__ import annotations


def stream(live_url: str | None = None) -> None:
    """
    Display a stream in the HUD system.
    """
    if live_url is None:
        raise ValueError("live_url cannot be None")
    from IPython.display import HTML, display

    html_content = f"""
    <div style="width: 960px; height: 540px; overflow: hidden;">
        <div style="transform: scale(0.5); transform-origin: top left;">
            <iframe src="{live_url}" width="1920" height="1080" style="border: 1px solid #ddd;">
            </iframe>
        </div>
    </div>
    """
    display(HTML(html_content))
