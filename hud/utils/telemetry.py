from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

def stream(live_url: str | None = None) -> str:
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
    try:
        display(HTML(html_content))
    except Exception as e:
        logger.warning(e)
    
    return html_content


def display_screenshot(base64_image: str, width: int = 960, height: int = 540) -> str:
    """
    Display a base64-encoded screenshot image.
    
    Args:
        base64_image: Base64-encoded image string (without the data URI prefix)
        width: Display width in pixels
        height: Display height in pixels
        
    Returns:
        The HTML string used to display the image
        
    Note:
        This function will both display the image in IPython environments
        and return the HTML string for other contexts.
    """
    from IPython.display import HTML, display
    
    # Ensure the base64 image doesn't already have the data URI prefix
    if base64_image.startswith("data:image"):
        img_src = base64_image
    else:
        img_src = f"data:image/png;base64,{base64_image}"
    
    html_content = f"""
    <div style="width: {width}px; height: {height}px; overflow: hidden; margin: 10px 0; border: 1px solid #ddd;">
        <img src="{img_src}" style="max-width: 100%; max-height: 100%;">
    </div>
    """  # noqa: E501
    
    # Display in IPython environments
    try:
        display(HTML(html_content))
    except Exception as e:
        logger.warning(e)
    
    return html_content
