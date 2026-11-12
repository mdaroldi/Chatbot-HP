"""Behavior of the single error-body builder and the domain error base class."""

from app.core.errors import AppError, error_body


def test_error_body_has_fixed_shape_with_empty_details_by_default() -> None:
    assert error_body("not_found", "Not Found", "req-12345678") == {
        "error": {
            "code": "not_found",
            "message": "Not Found",
            "request_id": "req-12345678",
            "details": [],
        }
    }


def test_error_body_includes_given_details() -> None:
    details = [{"loc": ["body", "message"], "msg": "Field required", "type": "missing"}]

    body = error_body("validation_error", "Request validation failed.", None, details)

    assert body["error"]["details"] == details
    assert body["error"]["request_id"] is None


def test_app_error_defaults_to_internal_status_and_keeps_its_message() -> None:
    error = AppError("Something specific went wrong.")

    assert error.status_code == 500
    assert error.code == "app_error"
    assert error.message == "Something specific went wrong."
    assert str(error) == "Something specific went wrong."


def test_app_error_subclasses_declare_status_and_code() -> None:
    class QuotaExceededError(AppError):
        status_code = 429
        code = "quota_exceeded"

    error = QuotaExceededError("Slow down.")

    assert (error.status_code, error.code, error.message) == (429, "quota_exceeded", "Slow down.")
