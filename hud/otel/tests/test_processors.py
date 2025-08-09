"""Tests for OpenTelemetry processors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hud.otel.processors import HudEnrichmentProcessor


class TestHudEnrichmentProcessor:
    """Test HudEnrichmentProcessor."""

    def test_on_start_with_run_id(self):
        """Test on_start with current task run ID."""

        processor = HudEnrichmentProcessor()

        # Mock span
        span = MagicMock()
        span.set_attribute = MagicMock()
        span.is_recording.return_value = True

        # Mock context with run ID
        with patch("hud.otel.processors.get_current_task_run_id", return_value="test-run-123"):
            processor.on_start(span, parent_context=None)

        # Verify attribute was set
        span.set_attribute.assert_called_with("hud.task_run_id", "test-run-123")

    def test_on_start_no_run_id(self):
        """Test on_start without current task run ID."""

        processor = HudEnrichmentProcessor()

        # Mock span
        span = MagicMock()
        span.set_attribute = MagicMock()
        span.is_recording.return_value = True

        # Mock context without run ID
        with patch("hud.otel.processors.get_current_task_run_id", return_value=None):
            processor.on_start(span, parent_context=None)

        # Verify no task run ID attribute was set
        span.set_attribute.assert_not_called()

    def test_on_end(self):
        """Test on_end does nothing."""

        processor = HudEnrichmentProcessor()
        span = MagicMock()

        # Should not raise
        processor.on_end(span)

    def test_shutdown(self):
        """Test shutdown does nothing."""

        processor = HudEnrichmentProcessor()

        # Should not raise
        processor.shutdown()

    def test_force_flush(self):
        """Test force_flush returns True."""

        processor = HudEnrichmentProcessor()

        # Should return True
        result = processor.force_flush()
        assert result is True

    def test_on_start_with_job_id(self):
        """Test on_start with job ID in baggage."""

        processor = HudEnrichmentProcessor()

        # Mock span
        span = MagicMock()
        span.set_attribute = MagicMock()
        span.is_recording.return_value = True

        # Mock baggage with job ID
        with (
            patch("hud.otel.processors.baggage.get_baggage", return_value="job-123"),
            patch("hud.otel.processors.get_current_task_run_id", return_value=None),
        ):
            processor.on_start(span, parent_context=None)

        # Verify job ID attribute was set
        span.set_attribute.assert_called_with("hud.job_id", "job-123")

    def test_on_start_exception_handling(self):
        """Test on_start handles exceptions gracefully."""

        processor = HudEnrichmentProcessor()

        # Mock span that raises exception
        span = MagicMock()
        span.is_recording.side_effect = Exception("Test error")

        # Should not raise
        processor.on_start(span, parent_context=None)

    def test_on_start_exception_handling_extended(self):
        """Test that exceptions in on_start are caught and logged."""
        from hud.otel.processors import HudEnrichmentProcessor

        processor = HudEnrichmentProcessor()

        # Create a mock span that raises when setting attributes
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.set_attribute.side_effect = RuntimeError("Attribute error")

        parent_context = {}

        # Patch logger and other dependencies to force an exception
        with (
            patch("hud.otel.processors.logger") as mock_logger,
            patch("hud.otel.processors.get_current_task_run_id", return_value="test-id"),
        ):
            # Should not raise, exception should be caught
            processor.on_start(mock_span, parent_context)

            # Verify logger.debug was called with the exception
            mock_logger.debug.assert_called_once()
            args = mock_logger.debug.call_args[0]
            assert "HudEnrichmentProcessor.on_start error" in args[0]
            assert "Attribute error" in str(args[1])

    def test_on_start_with_get_current_task_run_id_exception(self):
        """Test exception handling when get_current_task_run_id fails."""
        processor = HudEnrichmentProcessor()

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        parent_context = {}

        # Make get_current_task_run_id raise an exception
        with (
            patch(
                "hud.otel.processors.get_current_task_run_id",
                side_effect=ValueError("Context error"),
            ),
            patch("hud.otel.processors.logger") as mock_logger,
        ):
            # Should not raise
            processor.on_start(mock_span, parent_context)

            # Verify logger.debug was called
            mock_logger.debug.assert_called_once()
            args = mock_logger.debug.call_args[0]
            assert "Context error" in str(args[1])

    def test_on_start_with_baggage_exception(self):
        """Test exception handling when baggage.get_baggage fails."""
        processor = HudEnrichmentProcessor()

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        parent_context = {}

        # Make baggage.get_baggage raise an exception
        with (
            patch("hud.otel.processors.baggage.get_baggage", side_effect=KeyError("Baggage error")),
            patch("hud.otel.processors.logger") as mock_logger,
        ):
            # Should not raise
            processor.on_start(mock_span, parent_context)

            # Verify logger.debug was called
            mock_logger.debug.assert_called_once()
            args = mock_logger.debug.call_args[0]
            assert "Baggage error" in str(args[1])
