from __future__ import annotations

from hud.utils.telemetry import stream


def test_stream():
    html_content = stream("https://example.com")
    assert html_content is not None
    assert "<div style=" in html_content
    assert 'src="https://example.com"' in html_content


def test_display_screenshot():
    from hud.utils.telemetry import display_screenshot

    # This is a simple 1x1 transparent PNG image in base64 format
    base64_image = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
        "AAABJRU5ErkJggg=="
    )

    html_content = display_screenshot(base64_image)
    assert html_content is not None
    assert "<div style=" in html_content
    assert "width: 960px" in html_content
    assert "height: 540px" in html_content
    assert f"data:image/png;base64,{base64_image}" in html_content

    # Test with custom dimensions
    custom_html = display_screenshot(base64_image, width=800, height=600)
    assert "width: 800px" in custom_html
    assert "height: 600px" in custom_html

    # Test with data URI already included
    data_uri = f"data:image/png;base64,{base64_image}"
    uri_html = display_screenshot(data_uri)
    assert data_uri in uri_html
